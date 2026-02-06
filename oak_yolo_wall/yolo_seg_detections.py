'''
THIS NODE IS NOT OPTIMIZED TO WORK WELL IN THE JETSON 
# ------------------------------------------------------------------------------
# OakYolo ROS2 Node Summary
#
# This node runs a DepthAI pipeline on an OAK device to perform YOLO instance
# segmentation on rectified RGB frames, then publishes both visualization and
# ROS-friendly outputs.
#
# Core outputs:
# - Publishes rectified RGB images on /oak/rgb/image_rect.
# - Runs a YOLO segmentation NN (from MODEL_PATH) and publishes detections as
#   yolo_seg_interfaces/YoloSegDetectionArray on /yolo/detections, where each
#   detection includes class info, score, bbox-like fields, and a polygon
#   extracted from the instance mask.
# - Publishes an annotated overlay image on /oak/overlay/image with polygons,
#   rotated rectangles (if provided), and labels drawn for debugging.
# - Publishes a colored mask visualization (ApplyColormap) on /oak/nn/mask_colormap.
#
# Optional depth / RGB-D outputs (enabled via parameters):
# - If enable_depth_image is true, builds a stereo depth pipeline (CAM_B/C),
#   aligns depth to RGB (StereoDepth + ImageAlign on RVC4), and publishes the
#   aligned depth image on /oak/depth/image_raw.
# - Publishes depth CameraInfo (aligned-to-RGB intrinsics) on /oak/depth/camera_info.
# - If enable_pcl is true, builds an RGBD node and publishes a PointCloud2 on
#   /oak/depth/points (organized when the resolution matches common sizes).
#
# Synchronization:
# - Buffers RGB frames, NN detections, and mask-colormap frames by DepthAI
#   sequence number and processes the most recent RGB<->detection matched pair,
#   which helps keep overlays/detections aligned with the correct image.
# ------------------------------------------------------------------------------
'''


#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import depthai as dai
import depthai_nodes
from depthai_nodes.node import ParsingNeuralNetwork, ApplyColormap, ImgFrameOverlay
from depthai_nodes.message import ImgDetectionsExtended as MsgImgDetExt
from depthai_nodes.message.utils import copy_message
from sensor_msgs.msg import PointCloud2, PointField
import struct

from yolo_seg_interfaces.msg import YoloSegDetection, YoloSegDetectionArray
from geometry_msgs.msg import Point32

import contextlib

import cv2
import numpy as np

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose
from datetime import timedelta

from sensor_msgs.msg import CameraInfo


MODEL_PATH = "/robo_ws/src/oak_yolo_wall/models/blob_models/yolov8nseg_300e_512x288.rvc2_legacy.rvc2.tar.xz"
DEVICE = None

class PipelineProxy:
    def __init__(self, pipeline):
        self._p = pipeline
    def create(self, node_type, *a, **kw):
        if isinstance(node_type, type) and issubclass(node_type, dai.node.ThreadedHostNode):
            return node_type()
        return self._p.create(node_type, *a, **kw)
    def __getattr__(self, name):
        return getattr(self._p, name)


class OakYolo(Node):
    def __init__(self):
        super().__init__("oak_yolo_node")
        self.bridge = CvBridge()

        self.declare_parameter("enable_pcl", False)
        self.declare_parameter("enable_depth_image", True)
        self.declare_parameter("publish_fps", 10.0)
        
        pub_fps = float(self.get_parameter("publish_fps").value)
        self.class_names = ["red_striped_tape", "yellow_striped_tape"]  

        self.pub_rgb = self.create_publisher(Image, "/oak/rgb/image_rect", 10)
        self.pub_overlay = self.create_publisher(Image, "/oak/overlay/image", 10)
        self.pub_caminfo = self.create_publisher(CameraInfo,"/oak/depth/camera_info",10)
        self.enable_pcl = bool(self.get_parameter("enable_pcl").value)
        self.pub_yolo = self.create_publisher(YoloSegDetectionArray, "/yolo/detections", 10)
        self.pub_depth = None
        
        self.enable_depth_image = bool(self.get_parameter("enable_depth_image").value)
        if self.enable_depth_image:
            self.pub_depth = self.create_publisher(Image, "/oak/depth/image_raw", 10)

        # --- Device + pipeline (use the pattern that works for you) ---
        self.device = dai.Device(dai.DeviceInfo(DEVICE)) if DEVICE else dai.Device()
        platform = self.device.getPlatform()
        img_frame_type = dai.ImgFrame.Type.BGR888i if platform.name == "RVC4" else dai.ImgFrame.Type.BGR888p

        self.calib = self.device.readCalibration()
        self._caminfo_sent = False
        self.depth_frame_id = "oak_depth"


        # Keep context managers alive for the lifetime of the node
        self._stack = contextlib.ExitStack()
        self.pipeline = self._stack.enter_context(dai.Pipeline(self.device))
 
            
        cam = self.pipeline.create(dai.node.Camera).build()
        nn_archive = dai.NNArchive(MODEL_PATH)

        # ---- Host node instance ----
        self.nn_with_parser = ParsingNeuralNetwork()
        if hasattr(self.nn_with_parser, "_pipeline"):
            self.nn_with_parser._pipeline = PipelineProxy(self.pipeline)

        # ---- Build the host node ----
        rgb_out = cam.requestOutput((512, 288), type=img_frame_type, fps=pub_fps, enableUndistortion=True)
        self.nn_with_parser.build(rgb_out, nn_archive)

        self.apply_colormap_node = ApplyColormap().build(self.nn_with_parser.out)
        self.q_mask = self.apply_colormap_node.out.createOutputQueue(maxSize=1, blocking=False)
        self.pub_mask = self.create_publisher(Image, "/oak/nn/mask_colormap", 10)

        need_stereo = self.enable_depth_image or self.enable_pcl

        stereo = None
        align = None
        rgbd = None

        if need_stereo:
            left  = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
            right = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

            stereo = self.pipeline.create(dai.node.StereoDepth)
            stereo.setRectifyEdgeFillColor(0)
            stereo.enableDistortionCorrection(True)
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
            stereo.initialConfig.postProcessing.thresholdFilter.maxRange = 10000

            left.requestOutput((512, 288)).link(stereo.left)
            right.requestOutput((512, 288)).link(stereo.right)

            if platform.name == "RVC4":
                align = self.pipeline.create(dai.node.ImageAlign)
                stereo.depth.link(align.input)
                rgb_out.link(align.inputAlignTo)
                # aligned depth is now align.outputAligned
            else:
                # older platforms: stereo can align to rgb_out directly
                rgb_out.link(stereo.inputAlignTo)
                # aligned depth is now stereo.depth


        self.q_depth = None
        if self.enable_depth_image and need_stereo:
            if platform.name == "RVC4":
                self.q_depth = align.outputAligned.createOutputQueue(maxSize=1, blocking=False)
            else:
                self.q_depth = stereo.depth.createOutputQueue(maxSize=1, blocking=False)

        self.q_pcl = None
        self.pub_pcl = None

        if self.enable_pcl and need_stereo:
            self.pub_pcl = self.create_publisher(PointCloud2, "/oak/depth/points", 10)

            rgbd = self.pipeline.create(dai.node.RGBD).build()
            rgbd.setDepthUnits(dai.StereoDepthConfig.AlgorithmControl.DepthUnit.METER)

            if platform.name == "RVC4":
                align.outputAligned.link(rgbd.inDepth)
            else:
                stereo.depth.link(rgbd.inDepth)

            rgb_out.link(rgbd.inColor)
            self.q_pcl = rgbd.pcl.createOutputQueue(maxSize=1, blocking=False)



        # RGBD queue
        self.q_rgb = ( rgb_out.createOutputQueue(maxSize=1, blocking=False) if self.publish_rgb else None )

        # Output queue for detections
        self.q_det = self.nn_with_parser.out.createOutputQueue(maxSize=1, blocking=False)

        self.pipeline.start()
        # self.visualizer.registerPipeline(self.pipeline)

        from collections import OrderedDict

        self._rgb_buf  = OrderedDict()   # seq -> ImgFrame
        self._det_buf  = OrderedDict()   # seq -> ImgDetectionsExtended
        self._mask_buf = OrderedDict()   # seq -> ImgFrame (colormap) or whatever ApplyColormap outputs

        self._buf_max = 30   # ~3 seconds at 10 FPS, adjust as you like
        
        self.timer = self.create_timer(1.0 / pub_fps, self._tick) 


    def _make_depth_caminfo(self, stamp, width, height):
        msg = CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = self.depth_frame_id

        msg.width = int(width)
        msg.height = int(height)

        # CAM_A is RGB, CAM_B/C are stereo
        # Depth is aligned to RGB => use CAM_A intrinsics
        K = self.calib.getCameraIntrinsics(
            dai.CameraBoardSocket.CAM_A,
            width,
            height,
        )

        fx = K[0][0]
        fy = K[1][1]
        cx = K[0][2]
        cy = K[1][2]

        msg.k = [
            fx, 0.0, cx,
            0.0, fy, cy,
            0.0, 0.0, 1.0,
        ]

        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]  # already rectified
        msg.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]
        msg.p = [
            fx, 0.0, cx, 0.0,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]

        msg.distortion_model = "plumb_bob"
        msg.binning_x = 1
        msg.binning_y = 1

        return msg


    def _get_seq(self, pkt):
        # ImgFrame has getSequenceNum()
        if pkt is None:
            return None
        if hasattr(pkt, "getSequenceNum"):
            try:
                return int(pkt.getSequenceNum())
            except Exception:
                pass
        # sometimes DepthAI nodes add .sequenceNum or .sequence_num
        for name in ("sequenceNum", "sequence_num", "seq"):
            if hasattr(pkt, name):
                try:
                    return int(getattr(pkt, name))
                except Exception:
                    pass
        return None

    def _buf_put(self, buf, seq, pkt):
        if seq is None:
            return
        buf[seq] = pkt
        # keep small
        while len(buf) > self._buf_max:
            buf.popitem(last=False)


    def _drain_latest(self, q):
        latest = None
        while True:
            pkt = q.tryGet()
            if pkt is None:
                break
            latest = pkt
        return latest

    def destroy_node(self):
        try:
            self._stack.close()
        finally:
            super().destroy_node()

    def _tick(self):
        if not self.pipeline.isRunning():
            return

        stamp = self.get_clock().now().to_msg()
        frame_id = "oak_rgb"

        # ---- ingest all new packets into buffers ----
        while True:
            p = self.q_rgb.tryGet()
            if p is None: break
            self._buf_put(self._rgb_buf, self._get_seq(p), p)

        while True:
            p = self.q_det.tryGet()
            if p is None: break
            self._buf_put(self._det_buf, self._get_seq(p), p)

        while True:
            p = self.q_mask.tryGet()
            if p is None: break
            self._buf_put(self._mask_buf, self._get_seq(p), p)

        depth_pkt = None
        if self.enable_depth_image and self.q_depth is not None:
            depth_pkt = self._drain_latest(self.q_depth)

        if depth_pkt is not None and self.pub_depth is not None:
            depth = depth_pkt.getCvFrame()  # often uint16 depth in mm
            if depth.dtype == np.uint16:
                dmsg = self.bridge.cv2_to_imgmsg(depth, encoding="16UC1")
            elif depth.dtype == np.float32:
                dmsg = self.bridge.cv2_to_imgmsg(depth, encoding="32FC1")
            else:
                dmsg = self.bridge.cv2_to_imgmsg(depth)

            dmsg.header.stamp = stamp
            dmsg.header.frame_id = "oak_depth"  # better than oak_rgb for depth
            self.pub_depth.publish(dmsg)

        # after publishing depth image (or at least after you got depth)
        if (not self._caminfo_sent) and (depth_pkt is not None):
            depth = depth_pkt.getCvFrame()
            H, W = depth.shape[:2]
            caminfo = self._make_depth_caminfo(stamp, W, H)
            self.pub_caminfo.publish(caminfo)
            self._caminfo_sent = True

        # PCL can stay "latest" because you're not drawing it on image pixels here
        pcl_pkt = None
        if self.enable_pcl and self.q_pcl is not None:
            pcl_pkt = self._drain_latest(self.q_pcl)

        # ---- choose a seq present in rgb+det ----
        common = set(self._rgb_buf.keys()) & set(self._det_buf.keys())
        if not common:
            # self.get_logger().warn("No rgb<->det seq match yet", throttle_duration_sec=5.0)
            return

        seq = max(common)  # most recent matched pair

        rgb = self._rgb_buf.pop(seq)
        dets_pkt = self._det_buf.pop(seq)

        # mask is optional (might arrive slightly later)
        mask_pkt = self._mask_buf.pop(seq, None)

        # self.get_logger().info(f"PAIR seq={seq}", throttle_duration_sec=1.0)

        if rgb is None:
            return

        base = rgb.getCvFrame()
        H, W = base.shape[:2]
        DEPTH_W, DEPTH_H = W, H  
        
        # We’ll draw fills into this, then alpha-blend onto `base`
        overlay = base.copy()
        annot = base.copy()  # final image we publish

        # Publish raw rgb
        msg = self.bridge.cv2_to_imgmsg(base, encoding="bgr8")
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        self.pub_rgb.publish(msg)

        if pcl_pkt is not None and self.pub_pcl is not None:
            pts, cols = pcl_pkt.getPointsRGB()   # pts: (N,3) float32, cols: (N,3) uint8
            pts = np.asarray(pts)
            cols = np.asarray(cols)

            # --- normalize colors to (N,3) uint8 ---
            cols = np.asarray(cols)

            # Case A: already (N,3)
            if cols.ndim == 2 and cols.shape[1] == 3:
                pass

            # Case B: (N,4) -> drop alpha
            elif cols.ndim == 2 and cols.shape[1] == 4:
                cols = cols[:, :3]

            # Case C: flat array length N*4 or N*3
            elif cols.ndim == 1:
                if cols.size == pts.shape[0] * 4:
                    cols = cols.reshape((-1, 4))[:, :3]
                elif cols.size == pts.shape[0] * 3:
                    cols = cols.reshape((-1, 3))
                else:
                    self.get_logger().warn(f"Unexpected cols size={cols.size} for pts N={pts.shape[0]}")
                    return
            else:
                self.get_logger().warn(f"Unexpected cols shape={cols.shape}")
                return


            # self.get_logger().info(f"pcl: points={pts.shape} dtype={pts.dtype} cols={cols.shape} cols_dtype={cols.dtype}", throttle_duration_sec=2.0)

            N = int(pts.shape[0])

            if N == 640 * 400:
                pcl_msg = self._points_to_pointcloud2_organized(pts, cols, 640, 400, stamp, frame_id)
            elif N == 512 * 288:
                pcl_msg = self._points_to_pointcloud2_organized(pts, cols, 512, 288, stamp, frame_id)
            else:
                pcl_msg = self._points_to_pointcloud2(pts, cols, stamp, frame_id)


            self.pub_pcl.publish(pcl_msg)


        # If no detections packet yet, just publish the plain image as overlay
        if dets_pkt is None:
            omsg = self.bridge.cv2_to_imgmsg(annot, encoding="bgr8")
            omsg.header.stamp = stamp
            omsg.header.frame_id = frame_id
            self.pub_overlay.publish(omsg)

            # self.get_logger().info(f"dets_pkt type={type(dets_pkt)}", throttle_duration_sec=5.0)
            # self.get_logger().info(f"has masks={hasattr(dets_pkt,'masks')} has dets={hasattr(dets_pkt,'detections')}", throttle_duration_sec=5.0,)
            return

        # self.get_logger().info(f"type(dets_pkt)={type(dets_pkt)} "f"isinstance ImgDetectionsExtended={isinstance(dets_pkt, MsgImgDetExt)}",throttle_duration_sec=2.0,)

        # Copy message (important: this is what ApplyColormap does)
        dets_copy = copy_message(dets_pkt)

        # --- detections list ---
        dets = list(getattr(dets_pkt, "detections", [])) or []

        # Try to access as in ApplyColormap
        if hasattr(dets_copy, "masks"):
            mask2d = np.asarray(dets_copy.masks)
            # self.get_logger().info(f"copy_message masks: shape={mask2d.shape} dtype={mask2d.dtype} min={mask2d.min()} max={mask2d.max()}",throttle_duration_sec=2.0,)

            u = np.unique(mask2d)
            # self.get_logger().info(f"mask2d unique (first 20)={u[:20]}", throttle_duration_sec=2.0)
            # self.get_logger().info(f"dets count={len(dets)}", throttle_duration_sec=2.0)

            # 🔥 My suggestion: detect mismatch between mask instance ids and det indices
            pos = u[u >= 0]
            if pos.size and pos.max() >= len(dets):
                self.get_logger().warn(
                    f"mask has inst_id max={int(pos.max())} but dets_count={len(dets)} (mismatch)",
                    throttle_duration_sec=2.0,
                )
        else:
            self.get_logger().warn("dets_copy has no .masks attribute", throttle_duration_sec=2.0)
            mask2d = None

        # Colormap output (debug / visualization)
        if mask_pkt is not None:
            mask_img = mask_pkt.getCvFrame()
            
            # self.get_logger().info(f"mask_colormap shape={mask_img.shape} dtype={mask_img.dtype} min={mask_img.min()} max={mask_img.max()}",throttle_duration_sec=2.0,)
            mmsg = self.bridge.cv2_to_imgmsg(mask_img, encoding="bgr8")
            mmsg.header.stamp = stamp
            mmsg.header.frame_id = frame_id
            self.pub_mask.publish(mmsg)

        # --- convert mask2d -> polygons and draw ---
        alpha = 0.35  # <-- translucency strength (0 = invisible, 1 = opaque)
        drew_any = False

        if mask2d is not None and getattr(mask2d, "size", 0) > 0:
            polys_by_id = self._mask2d_instance_polygons(mask2d, W, H)

            det_arr = YoloSegDetectionArray()
            det_arr.header.stamp = stamp
            det_arr.header.frame_id = frame_id

            for inst_id, poly in polys_by_id.items():
                
                if not (0 <= inst_id < len(dets)):
                    continue

                det_src = dets[inst_id]
                cls_id = int(getattr(det_src, "label", 0))
                score = float(getattr(det_src, "confidence", 0.0))
                cls_name = getattr(det_src, "label_name", "") or (
                    self.class_names[cls_id] if 0 <= cls_id < len(self.class_names) else str(cls_id)
                )
                
                poly_i = poly.astype(np.int32).reshape(-1, 1, 2)  # OpenCV wants Nx1x2

                # outline only (no darkening)
                color = self._color_for_label(cls_id)  # BGR
                cv2.polylines(annot, [poly_i], isClosed=True, color=color, thickness=2)

                # bbox (pixels) -> cx,cy,w,h like your PC node does
                bb = self._det_to_bbox_px(det_src, W, H)
                if bb is None:
                    # fallback: bbox from polygon extents
                    x0, y0 = float(np.min(poly[:,0])), float(np.min(poly[:,1]))
                    x1, y1 = float(np.max(poly[:,0])), float(np.max(poly[:,1]))
                else:
                    x0, y0, x1, y1, _ = bb

                cx = 0.5 * (x0 + x1)
                cy = 0.5 * (y0 + y1)
                bw = float(abs(x1 - x0))
                bh = float(abs(y1 - y0))

                det_msg = YoloSegDetection()
                det_msg.header = det_arr.header
                det_msg.class_id = cls_id
                det_msg.class_name = cls_name
                det_msg.score = score
                det_msg.x = float(cx)
                det_msg.y = float(cy)
                det_msg.w = float(bw)
                det_msg.h = float(bh)

                for (u, v) in poly.astype(np.float32):
                    u2 = int(round(u))
                    v2 = int(round(v))
                    u2 = max(0, min(DEPTH_W - 1, u2))
                    v2 = max(0, min(DEPTH_H - 1, v2))
                    det_msg.polygon.append(Point32(x=float(u2), y=float(v2), z=0.0))


                det_arr.detections.append(det_msg)

            self.pub_yolo.publish(det_arr)

        else:
            # keep a log but don't spam too hard
            if mask2d is None:
                self.get_logger().warn("mask2d is None (no masks available yet)", throttle_duration_sec=2.0)
            # elif dets == []:
                # self.get_logger().info("No detections in dets_pkt yet", throttle_duration_sec=2.0)
            else:
                self.get_logger().warn(
                    f"mask2d present but empty? shape={getattr(mask2d,'shape',None)}",
                    throttle_duration_sec=2.0,
                )

        # Optional: draw rotated rects + labels on top
        for det in dets:
            rr = getattr(det, "rotated_rect", None)
            if rr is None:
                continue

            label = int(getattr(det, "label", 0))
            name = getattr(det, "label_name", "") or (
                self.class_names[label] if 0 <= label < len(self.class_names) else str(label)
            )
            text = f"{name} {float(getattr(det, 'confidence', 0.0)):.2f}"
            self._draw_rotated_rect(annot, rr, W, H, text, label)

        # Publish annotated overlay
        omsg = self.bridge.cv2_to_imgmsg(annot, encoding="bgr8")
        omsg.header.stamp = stamp
        omsg.header.frame_id = frame_id
        self.pub_overlay.publish(omsg)
        


    def _det_to_bbox_px(self, det, W, H):
        """
        Returns (x0,y0,x1,y1) in pixels, plus a display label string.
        Works even if det has only keypoints/rotated_rect.
        """
        label = int(getattr(det, "label", 0))
        name = getattr(det, "label_name", "") or (self.class_names[label] if 0 <= label < len(self.class_names) else str(label))
        conf = float(getattr(det, "confidence", 0.0))
        text = f"{name} {conf:.2f}"

        # 1) Prefer keypoints if present (most reliable for seg-style dets)
        kps = getattr(det, "keypoints", None)
        if kps:
            xs, ys = [], []
            for kp in kps:
                # kp may be dict-like or object-like
                x = getattr(kp, "x", None)
                y = getattr(kp, "y", None)
                if x is None and isinstance(kp, dict):
                    x, y = kp.get("x"), kp.get("y")
                if x is None or y is None:
                    continue

                # Heuristic: many APIs store keypoints normalized [0..1]
                if 0.0 <= float(x) <= 1.0 and 0.0 <= float(y) <= 1.0:
                    xs.append(float(x) * W)
                    ys.append(float(y) * H)
                else:
                    xs.append(float(x))
                    ys.append(float(y))

            if xs and ys:
                x0 = int(max(0, min(W - 1, min(xs))))
                x1 = int(max(0, min(W - 1, max(xs))))
                y0 = int(max(0, min(H - 1, min(ys))))
                y1 = int(max(0, min(H - 1, max(ys))))
                return x0, y0, x1, y1, text

        # 2) Fallback: rotated_rect if it contains usable geometry
        rr = getattr(det, "rotated_rect", None)
        if rr is not None:
            # Try common field names (depends on DepthAI message type)
            cx = getattr(rr, "center", None)
            if cx is not None:
                cx_x = getattr(cx, "x", None)
                cx_y = getattr(cx, "y", None)
            else:
                cx_x = getattr(rr, "x", None)
                cx_y = getattr(rr, "y", None)

            w = getattr(rr, "width", None) or getattr(rr, "w", None)
            h = getattr(rr, "height", None) or getattr(rr, "h", None)

            if None not in (cx_x, cx_y, w, h):
                cx_x, cx_y, w, h = float(cx_x), float(cx_y), float(w), float(h)

                # Heuristic normalization check
                if 0.0 <= cx_x <= 1.0 and 0.0 <= cx_y <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0:
                    cx_x *= W
                    cx_y *= H
                    w *= W
                    h *= H

                x0 = int(max(0, min(W - 1, cx_x - w / 2)))
                x1 = int(max(0, min(W - 1, cx_x + w / 2)))
                y0 = int(max(0, min(H - 1, cx_y - h / 2)))
                y1 = int(max(0, min(H - 1, cx_y + h / 2)))
                return x0, y0, x1, y1, text

        return None
        arr = Detection2DArray()
        arr.header.stamp = stamp
        arr.header.frame_id = frame_id

        for det in dets:
            bb = self._det_to_bbox_px(det, W, H)
            if bb is None:
                continue
            x0, y0, x1, y1, _ = bb

            d = Detection2D()
            d.header = arr.header

            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)
            d.bbox.center.position.x = float(cx)
            d.bbox.center.position.y = float(cy)
            d.bbox.size_x = float(abs(x1 - x0))
            d.bbox.size_y = float(abs(y1 - y0))

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(int(getattr(det, "label", 0)))
            hyp.hypothesis.score = float(getattr(det, "confidence", 0.0))
            d.results.append(hyp)

            arr.detections.append(d)

        self.pub_det.publish(arr)

    def _color_for_label(self, label):
        # BGR (OpenCV!)
        if label == 0:      # red tape
            return (0, 0, 255)
        elif label == 1:    # yellow tape
            return (0, 255, 255)
        return (0, 255, 0)


    def _draw_rotated_rect(self, img, rr, W, H, text, label):
        """
        Draw a rotated rectangle from DepthAI rotated_rect.
        """
        import math
        import cv2
        import numpy as np

        cx = float(rr.center.x)
        cy = float(rr.center.y)
        w  = float(rr.size.width)
        h  = float(rr.size.height)
        a  = float(rr.angle)  # radians

        # normalize if needed
        if 0.0 <= cx <= 1.0:
            cx *= W
            cy *= H
            w  *= W
            h  *= H

        rect = ((cx, cy), (w, h), math.degrees(a))
        box = cv2.boxPoints(rect).astype(int)

        color = self._color_for_label(label)

        cv2.polylines(img, [box], True, color, 2)

        top_idx = np.argmin(box[:, 1])
        tx, ty = box[top_idx]
        ty = max(10, ty - 6)

        cv2.putText(
            img,
            text,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,          # ⬅ smaller font
            color,
            1,            # thinner
            cv2.LINE_AA
        )

    def _mask2d_instance_polygons(self, mask2d: np.ndarray, W: int, H: int, min_area_px: int = 50):
        """
        mask2d: (288,512) int16. background=-1, instances=0..N-1
        Returns dict {instance_id: polygon Nx2 float32 in image pixels}
        """
        if mask2d is None or mask2d.size == 0:
            return {}

        ids = np.unique(mask2d)
        ids = ids[ids >= 0]  # drop background
        if ids.size == 0:
            return {}

        inH, inW = mask2d.shape[:2]
        sx = W / float(inW)
        sy = H / float(inH)

        polys = {}
        for inst_id in ids.tolist():
            binmask = (mask2d == inst_id).astype(np.uint8) * 255

            # optional cleanup (helps remove speckles)
            # binmask = cv2.morphologyEx(binmask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

            contours, _ = cv2.findContours(binmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            # take largest contour
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            if area < min_area_px:
                continue

            # simplify contour (tune epsilon if you want fewer points)
            eps = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, eps, True)  # (N,1,2)

            poly = approx.reshape(-1, 2).astype(np.float32)
            # scale from NN mask space -> RGB frame pixels
            poly[:, 0] *= sx
            poly[:, 1] *= sy

            if poly.shape[0] >= 3:
                polys[int(inst_id)] = poly

        return polys

    def _points_to_pointcloud2(self, points_xyz: np.ndarray, colors_rgb: np.ndarray, stamp, frame_id: str) -> PointCloud2:
        """
        points_xyz: (N,3) float32 in meters
        colors_rgb: (N,3) uint8 (0..255)
        Packs fields: x,y,z (float32) + rgb (uint32)
        """
        msg = PointCloud2()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.height = 1
        msg.width = int(points_xyz.shape[0])
        msg.is_bigendian = False
        msg.is_dense = False

        msg.fields = [
            PointField(name="x",   offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name="y",   offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name="z",   offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32,  count=1),
        ]
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width

        # pack rgb -> uint32: 0xRRGGBB
        rgb_u32 = (colors_rgb[:, 0].astype(np.uint32) << 16) | (colors_rgb[:, 1].astype(np.uint32) << 8) | colors_rgb[:, 2].astype(np.uint32)

        # structured array then bytes
        cloud = np.empty((msg.width,), dtype=np.dtype([
            ("x",  np.float32),
            ("y",  np.float32),
            ("z",  np.float32),
            ("rgb", np.uint32),
        ]))
        cloud["x"] = points_xyz[:, 0].astype(np.float32)
        cloud["y"] = points_xyz[:, 1].astype(np.float32)
        cloud["z"] = points_xyz[:, 2].astype(np.float32)
        cloud["rgb"] = rgb_u32

        msg.data = cloud.tobytes()
        return msg

    def _points_to_pointcloud2_organized(self, points_xyz, colors_rgb, W, H, stamp, frame_id):
        msg = PointCloud2()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.height = int(H)
        msg.width  = int(W)
        msg.is_bigendian = False
        msg.is_dense = False

        msg.fields = [
            PointField(name="x",   offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name="y",   offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name="z",   offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32,  count=1),
        ]
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width

        pts = np.asarray(points_xyz, dtype=np.float32).reshape((H * W, 3))
        cols = np.asarray(colors_rgb, dtype=np.uint8).reshape((H * W, 3))

        rgb_u32 = (cols[:, 0].astype(np.uint32) << 16) | (cols[:, 1].astype(np.uint32) << 8) | cols[:, 2].astype(np.uint32)

        cloud = np.empty((H * W,), dtype=np.dtype([("x", np.float32), ("y", np.float32), ("z", np.float32), ("rgb", np.uint32)]))
        cloud["x"] = pts[:, 0]
        cloud["y"] = pts[:, 1]
        cloud["z"] = pts[:, 2]
        cloud["rgb"] = rgb_u32

        msg.data = cloud.tobytes()
        return msg


def main():
    rclpy.init()
    node = OakYolo()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()