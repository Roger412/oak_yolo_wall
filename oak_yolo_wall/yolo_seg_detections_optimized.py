'''
# ------------------------------------------------------------------------------
# OakYoloMinimal ROS2 Node Summary
#
# Minimal DepthAI + ROS2 bridge that runs an on-device YOLO instance-segmentation
# network on rectified RGB frames (512x288) and publishes lightweight ROS outputs.
#
# Pipeline / inference:
# - Creates a DepthAI pipeline with the RGB camera (rectified via enableUndistortion=True).
# - Loads a NN archive (MODEL_PATH) and runs ParsingNeuralNetwork on the RGB stream.
# - Uses small, non-blocking host queues (maxSize=1 + tryGet drain) to avoid backlog.
#
# Threading model:
# - A dedicated reader thread performs all DepthAI queue reads and immediately converts
#   packets into ROS messages (so DepthAI objects are not held across threads).
# - A ROS timer publishes only the latest cached messages (RGB optional) at a fixed rate.
#
# Published topics:
# - /yolo/detections (yolo_seg_interfaces/YoloSegDetectionArray):
#     Converts the NN instance mask output into polygons using OpenCV contours and
#     publishes class/score + polygon + bbox-like fields per detection.
# - /oak/rgb/camera_info (sensor_msgs/CameraInfo):
#     Built once from DepthAI calibration intrinsics for the configured resolution and
#     published periodically (independent of RGB publishing).
# - /oak/rgb/image_rect (sensor_msgs/Image) [optional]:
#     Publishes rectified RGB frames when publish_rgb=True.
#
# Debug / robustness:
# - Logs host-side latency (time from DepthAI read to publish) at ~1 Hz.
# - Includes a simple watchdog that warns if the publish timer stops ticking.
# - Enables faulthandler and SIGUSR1-triggered stack dumps for crash debugging.
#
# Non-features by design (compared to the larger node):
# - No depth/stereo pipeline, no depth image, no PointCloud2, no overlay rendering,
#   and no sequence-number matching between RGB and detections (latest-only behavior).
# ------------------------------------------------------------------------------
'''
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork
from depthai_nodes.message.utils import copy_message

from sensor_msgs.msg import Image, CameraInfo

from yolo_seg_interfaces.msg import YoloSegDetection, YoloSegDetectionArray
from geometry_msgs.msg import Point32
from copy import deepcopy


from cv_bridge import CvBridge
import numpy as np
import threading
import signal
import faulthandler
import contextlib
import traceback
import time
import cv2
import os


MODEL_PATH = "/robo_ws/src/oak_yolo_wall/blob_models/yolov8nseg_100e_512x288.rvc2_legacy.rvc2.tar.xz"
DEVICE = None

faulthandler.enable()
# Optional: you can force-dump stack traces with:
#   kill -USR1 <pid>
faulthandler.register(signal.SIGUSR1)

qos = QoSProfile( reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)

class PipelineProxy:
    def __init__(self, pipeline):
        self._p = pipeline

    def create(self, node_type, *a, **kw):
        if isinstance(node_type, type) and issubclass(node_type, dai.node.ThreadedHostNode):
            return node_type()
        return self._p.create(node_type, *a, **kw)

    def __getattr__(self, name):
        return getattr(self._p, name)


class OakYoloMinimal(Node):
    def __init__(self):
        super().__init__("oak_yolo_minimal")

        self.bridge = CvBridge()

        # ----- Parameters -----
        self.declare_parameter("fps", 10.0)
        fps = float(self.get_parameter("fps").value)
        self.declare_parameter("publish_rgb", False)
        self.publish_rgb = bool(self.get_parameter("publish_rgb").value)
        self.declare_parameter("camera_frame", "oak_rgb")
        self.camera_frame = str(self.get_parameter("camera_frame").value)
        self.class_names = ["red_striped_tape", "yellow_striped_tape"]

        # ----- publishers -----
        self.pub_rgb = ( self.create_publisher(Image, "/oak/rgb/image_rect", qos) if self.publish_rgb else None )

        self.pub_info = self.create_publisher( CameraInfo, "/oak/rgb/camera_info", qos )

        self.pub_yolo = self.create_publisher( YoloSegDetectionArray, "/yolo/detections", qos )

        # ----- DepthAI device -----
        self.device = dai.Device(dai.DeviceInfo(DEVICE)) if DEVICE else dai.Device()

        self._cam_info_msg = self._make_camera_info_msg( width=512, height=288, frame_id=self.camera_frame, socket=dai.CameraBoardSocket.RGB, )

        platform = self.device.getPlatform()
        img_type = ( dai.ImgFrame.Type.BGR888i if platform.name == "RVC4" else dai.ImgFrame.Type.BGR888p )

        # ----- Pipeline -----
        self._stack = contextlib.ExitStack()
        self.pipeline = self._stack.enter_context(dai.Pipeline(self.device))

        cam = self.pipeline.create(dai.node.Camera).build()
        rgb_out = cam.requestOutput( (512, 288), type=img_type, fps=fps, enableUndistortion=True, )

        nn_archive = dai.NNArchive(MODEL_PATH)
        self.nn = ParsingNeuralNetwork()

        if hasattr(self.nn, "_pipeline"):
            self.nn._pipeline = PipelineProxy(self.pipeline)

        self.nn.build(rgb_out, nn_archive)

        # ----- Output queues -----
        self.q_rgb = ( rgb_out.createOutputQueue(maxSize=1, blocking=False) if self.publish_rgb else None )

        self.q_det = self.nn.out.createOutputQueue( maxSize=1, blocking=False )

        self.get_logger().info("Starting pipeline…")
        self.pipeline.start()
        self.get_logger().info("Pipeline started.")

        # ----- Shared buffers (reader → ROS timer) -----
        self._latest_rgb = None
        self._latest_det = None
        self._latest_host_time = None

        self._latest_lock = threading.Lock()
        self._stop_evt = threading.Event()

        self.reader_thread = threading.Thread( target=self._reader_loop, daemon=True,)
        self.reader_thread.start()

        # ----- Timers -----
        self._last_tick_time = time.time()
        self.watchdog = self.create_timer(1.0, self._watchdog)
        self.timer = self.create_timer(1.0 / fps, self._publish_tick)



    def _reader_loop(self):
        try:
            os.sched_setaffinity(0, {2})  # specific core to run on
        except Exception:
            pass

        while not self._stop_evt.is_set():
            # with self._latest_lock:
            #     if self._latest_det is not None:
            #         time.sleep(0.001)
            #         continue
            try:        
                # These are DepthAI calls -> keep them OUT of ROS timer thread.
                if self.publish_rgb: 
                    rgb_pkt = self._drain_latest(self.q_rgb)
                else: 
                    rgb_pkt = None
                det_pkt = self._drain_latest(self.q_det)


                if rgb_pkt is None and det_pkt is None:
                    time.sleep(0.001)
                    continue

                stamp = self.get_clock().now().to_msg()

                # Convert detections immediately (so we don't hold DepthAI objects)
                det_arr = None
                if det_pkt is not None:
                    dets_copy = copy_message(det_pkt)
                    dets = list(getattr(det_pkt, "detections", [])) or []
                    masks = np.asarray(getattr(dets_copy, "masks", []))

                    det_arr = self._build_det_array_from_masks(dets, masks, stamp, self.camera_frame)

                # Convert RGB immediately (same reason)
                rgb_msg = None
                if self.publish_rgb and rgb_pkt is not None:
                    img = rgb_pkt.getCvFrame()
                    rgb_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
                    rgb_msg.header.stamp = stamp
                    rgb_msg.header.frame_id = self.camera_frame

                t_read = time.monotonic()

                with self._latest_lock:
                    if rgb_msg is not None:
                        self._latest_rgb = rgb_msg
                    if det_arr is not None:
                        self._latest_det = det_arr

                    self._latest_host_time = t_read     # ← THIS is the important part


            except Exception:
                # don't kill the thread; log and keep going
                self.get_logger().error("Reader loop exception:\n" + traceback.format_exc())
                time.sleep(0.1)


    def _publish_tick(self):
        self._last_tick_time = time.time()

        with self._latest_lock:
            rgb_msg = self._latest_rgb
            det_msg = self._latest_det
            host_t  = self._latest_host_time

            self._latest_rgb = None
            self._latest_det = None

        if self.pub_info is not None and self._cam_info_msg is not None:
            ci = deepcopy(self._cam_info_msg)
            ci.header.stamp = self.get_clock().now().to_msg()   
            ci.header.frame_id = self.camera_frame
            self.pub_info.publish(ci)

        if self.publish_rgb and rgb_msg is not None and self.pub_rgb is not None:
            self.pub_rgb.publish(rgb_msg)

        if det_msg is not None:
            self.pub_yolo.publish(det_msg)

        t_pub = time.monotonic()

        if host_t is not None:
            self.get_logger().info(
                f"Host latency: {(t_pub - host_t)*1000:.1f} ms",
                throttle_duration_sec=1.0
            )


    def _make_camera_info_msg(self, width: int, height: int, frame_id: str, socket):
        """
        Build a sensor_msgs/CameraInfo from DepthAI calibration for the given output size.
        Assumes you're publishing an undistorted/rectified image (enableUndistortion=True).
        """
        try:
            calib = self.device.readCalibration()

            # Intrinsics for the given output resolution
            K = np.array(calib.getCameraIntrinsics(socket, width, height), dtype=np.float64)  # 3x3

            # Distortion: if you are publishing rectified images, this can be zeros.
            # If you publish RAW/distorted, you'd want the real distortion coefficients here.
            # DepthAI calibration supports distortion queries, but APIs vary by version;
            # safest for rectified stream is "no distortion".
            D = [0.0, 0.0, 0.0, 0.0, 0.0]

            ci = CameraInfo()
            ci.header.frame_id = frame_id
            ci.width = width
            ci.height = height

            ci.distortion_model = "plumb_bob"
            ci.d = D

            # K (3x3) row-major
            ci.k = [
                float(K[0,0]), float(K[0,1]), float(K[0,2]),
                float(K[1,0]), float(K[1,1]), float(K[1,2]),
                float(K[2,0]), float(K[2,1]), float(K[2,2]),
            ]

            # Rectification matrix R: identity for rectified stream
            ci.r = [1.0,0.0,0.0,
                    0.0,1.0,0.0,
                    0.0,0.0,1.0]

            # Projection matrix P (3x4): usually [K | 0] for monocular
            ci.p = [
                float(K[0,0]), float(K[0,1]), float(K[0,2]), 0.0,
                float(K[1,0]), float(K[1,1]), float(K[1,2]), 0.0,
                float(K[2,0]), float(K[2,1]), float(K[2,2]), 0.0,
            ]

            return ci

        except Exception as e:
            self.get_logger().error(f"Failed to build CameraInfo from DepthAI calibration: {e}")
            return None


    # ---------------------------
    # Helpers
    # ---------------------------

    def _drain_latest(self, q):
        pkt = None
        while True:
            p = q.tryGet()
            if p is None:
                break
            pkt = p
        return pkt

    def _watchdog(self):
        import time
        age = time.time() - self._last_tick_time
        if age > 2.0:
            self.get_logger().warn(f"WATCHDOG: no tick for {age:.1f}s")

    def _build_det_array_from_masks(self, dets, masks, stamp, frame_id):
        det_arr = YoloSegDetectionArray()
        det_arr.header.stamp = stamp
        det_arr.header.frame_id = frame_id

        MIN_AREA_PX   = 50
        EPS_FRAC      = 0.01
        MAX_VERTICES  = 200

        if masks.size == 0:
            return det_arr

        for inst_id, det in enumerate(dets):
            mask_bin = (masks == inst_id).astype(np.uint8) * 255
            if cv2.countNonZero(mask_bin) < 10:
                continue

            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            cnt = max(contours, key=cv2.contourArea)
            if cv2.contourArea(cnt) < MIN_AREA_PX:
                continue

            perim = cv2.arcLength(cnt, True)
            eps = max(1.0, EPS_FRAC * perim)
            approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)

            if approx.shape[0] > MAX_VERTICES:
                idx = np.linspace(0, approx.shape[0]-1, MAX_VERTICES).astype(np.int32)
                approx = approx[idx]

            if approx.shape[0] < 3:
                continue

            xs = approx[:, 0].astype(np.float32)
            ys = approx[:, 1].astype(np.float32)

            d = YoloSegDetection()
            d.header = det_arr.header
            d.class_id = int(getattr(det, "label", 0))
            label_name = getattr(det, "label_name", "")
            d.class_name = label_name or (
                self.class_names[d.class_id] if 0 <= d.class_id < len(self.class_names) else str(d.class_id)
            )
            d.score = float(getattr(det, "confidence", 0.0))

            x0, x1 = float(xs.min()), float(xs.max())
            y0, y1 = float(ys.min()), float(ys.max())
            d.x = 0.5 * (x0 + x1)
            d.y = 0.5 * (y0 + y1)
            d.w = float(x1 - x0)
            d.h = float(y1 - y0)

            for x, y in approx:
                d.polygon.append(Point32(x=float(x), y=float(y), z=0.0))

            det_arr.detections.append(d)

        return det_arr

    def destroy_node(self):
        self._stop_evt.set()
        try:
            if hasattr(self, "reader_thread"):
                self.reader_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self.timer.cancel()
        except Exception:
            pass
        try:
            self.pipeline.stop()
        except Exception:
            pass
        try:
            self.device.close()
        except Exception:
            pass
        try:
            self._stack.close()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = OakYoloMinimal()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
