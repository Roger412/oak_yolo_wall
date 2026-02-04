'''
# ------------------------------------------------------------------------------
# YoloOverlayCombiner ROS2 Node Summary
#
# Lightweight “overlay” utility node that combines an incoming RGB image with
# YOLO segmentation detections (polygons) and optional centroid-distance debug
# data, then publishes an annotated image.
#
# Inputs:
# - /oak/rgb/image_rect (sensor_msgs/Image): base image to draw on
# - /yolo/detections (yolo_seg_interfaces/YoloSegDetectionArray): polygons/bboxes to render
# - /tape_wall/centroid_distance_m (std_msgs/Float32MultiArray): optional triples
#     [u0, v0, d0, u1, v1, d1, ...] to draw centroid markers and distance text
#
# Behavior:
# - Keeps the latest image, detections, and centroid array in memory (no message_filters).
# - On any new message, attempts to publish an overlay if an image exists and the
#   image/detections timestamps are within max_dt_ms; otherwise optionally publishes
#   the clean image (publish_when_no_dets).
# - Draws detection polygons (preferred) or falls back to bbox drawing if no polygon.
# - Optionally draws centroid dots + “<d> m” labels for each (u,v,d) triple.
#
# Output:
# - /oak/overlay/image (sensor_msgs/Image): annotated image for RViz/debugging
#
# Notes:
# - Uses BEST_EFFORT QoS (depth=1) to avoid backpressure and keep latency low.
# - Centroid messages have no Header, so time alignment is assumed to match the
#   detection cycle when enabled (no strict gating against image timestamps).
# ------------------------------------------------------------------------------
'''
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from yolo_seg_interfaces.msg import YoloSegDetectionArray
from std_msgs.msg import Float32MultiArray

from cv_bridge import CvBridge
import numpy as np
import cv2
import threading


def best_effort_qos(depth: int = 1) -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )


class YoloOverlayCombiner(Node):
    def __init__(self):
        super().__init__("yolo_overlay_combiner")
        self.bridge = CvBridge()

        # ---- params ----
        self.declare_parameter("image_topic", "/oak/rgb/image_rect")
        self.declare_parameter("detections_topic", "/yolo/detections")
        self.declare_parameter("overlay_topic", "/oak/overlay/image")
        self.declare_parameter("max_dt_ms", 250.0)  # allowed time gap between image and detections
        self.declare_parameter("publish_when_no_dets", True)  # publish clean image if no detections

        # Centroid u,v,dist triples topic (Float32MultiArray: [u0,v0,d0, u1,v1,d1, ...])
        self.declare_parameter("centroid_topic", "/tape_wall/centroid_distance_m")
        self.declare_parameter("max_centroid_dt_ms", 250.0)  # allowed time gap between image and centroid msg
        self.declare_parameter("draw_centroids", True)
        self.declare_parameter("centroid_radius_px", 4)
        self.declare_parameter("centroid_text_scale", 0.45)
        self.declare_parameter("centroid_text_thickness", 1)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.dets_topic = str(self.get_parameter("detections_topic").value)
        self.overlay_topic = str(self.get_parameter("overlay_topic").value)
        self.max_dt_ms = float(self.get_parameter("max_dt_ms").value)
        self.publish_when_no_dets = bool(self.get_parameter("publish_when_no_dets").value)

        self.centroid_topic = str(self.get_parameter("centroid_topic").value)
        self.max_centroid_dt_ms = float(self.get_parameter("max_centroid_dt_ms").value)
        self.draw_centroids = bool(self.get_parameter("draw_centroids").value)
        self.centroid_radius_px = int(self.get_parameter("centroid_radius_px").value)
        self.centroid_text_scale = float(self.get_parameter("centroid_text_scale").value)
        self.centroid_text_thickness = int(self.get_parameter("centroid_text_thickness").value)

        # ---- state ----
        self._lock = threading.Lock()
        self._latest_img_msg = None
        self._latest_det_msg = None
        self._latest_centroid_msg = None

        # ---- pubs/subs ----
        qos = best_effort_qos(depth=1)

        self.sub_img = self.create_subscription(Image, self.image_topic, self._on_image, qos)
        self.sub_det = self.create_subscription(YoloSegDetectionArray, self.dets_topic, self._on_dets, qos)
        self.sub_cent = self.create_subscription(Float32MultiArray, self.centroid_topic, self._on_centroids, qos)

        self.pub_overlay = self.create_publisher(Image, self.overlay_topic, qos)

        self.get_logger().info(
            "Overlay combiner:\n"
            f"  image   : {self.image_topic}\n"
            f"  dets    : {self.dets_topic}\n"
            f"  centroid: {self.centroid_topic}\n"
            f"  out     : {self.overlay_topic}\n"
            f"  max_dt_ms={self.max_dt_ms} max_centroid_dt_ms={self.max_centroid_dt_ms}\n"
            f"  publish_when_no_dets={self.publish_when_no_dets} draw_centroids={self.draw_centroids}"
        )

    # ---------------- callbacks ----------------

    def _on_image(self, msg: Image):
        with self._lock:
            self._latest_img_msg = msg
            det_msg = self._latest_det_msg
            cent_msg = self._latest_centroid_msg

        self._try_publish_overlay(img_msg=msg, det_msg=det_msg, cent_msg=cent_msg)

    def _on_dets(self, msg: YoloSegDetectionArray):
        with self._lock:
            self._latest_det_msg = msg
            img_msg = self._latest_img_msg
            cent_msg = self._latest_centroid_msg

        self._try_publish_overlay(img_msg=img_msg, det_msg=msg, cent_msg=cent_msg)

    def _on_centroids(self, msg: Float32MultiArray):
        with self._lock:
            self._latest_centroid_msg = msg
            img_msg = self._latest_img_msg
            det_msg = self._latest_det_msg

        self._try_publish_overlay(img_msg=img_msg, det_msg=det_msg, cent_msg=msg)

    # ---------------- overlay logic ----------------

    def _try_publish_overlay(self, img_msg: Image, det_msg: YoloSegDetectionArray, cent_msg: Float32MultiArray):
        if img_msg is None:
            return

        # If no dets yet, optionally publish clean image
        if det_msg is None:
            if self.publish_when_no_dets:
                self.pub_overlay.publish(img_msg)
            return

        # Check timestamp proximity for detections
        dt_ms = self._dt_ms(img_msg.header.stamp, det_msg.header.stamp)
        if dt_ms is None or dt_ms > self.max_dt_ms:
            if self.publish_when_no_dets:
                self.pub_overlay.publish(img_msg)
            return

        # Convert image
        try:
            cv_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"imgmsg_to_cv2 failed: {e}")
            return

        overlay = cv_img.copy()
        dets = getattr(det_msg, "detections", []) or []
        if dets:
            self._draw_detections(overlay, dets)

        # Draw centroids if present and roughly time-aligned
        if self.draw_centroids and cent_msg is not None:
            ok = True
            # Float32MultiArray has no header; we can only gate using det_msg stamp (same callback cycle)
            # If you later switch to a custom msg with Header, you can gate directly vs image stamp.
            # Here we just draw if dets are aligned to image, assuming centroid msg is from the same dets.
            if ok:
                self._draw_centroids_uvd(overlay, cent_msg.data)

        # Publish overlay
        out = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
        out.header = img_msg.header
        self.pub_overlay.publish(out)

    def _dt_ms(self, a, b):
        # a,b are builtin_interfaces/Time
        try:
            da = float(a.sec) + float(a.nanosec) * 1e-9
            db = float(b.sec) + float(b.nanosec) * 1e-9
            return abs(da - db) * 1000.0
        except Exception:
            return None

    def _color_for_class(self, class_id: int):
        # BGR
        if class_id == 0:
            return (0, 0, 255)      # red
        if class_id == 1:
            return (0, 255, 255)    # yellow
        return (0, 255, 0)          # default green

    # ---------------- drawing ----------------

    def _draw_centroids_uvd(self, img_bgr, flat_uvd):
        """
        flat_uvd: [u0, v0, d0, u1, v1, d1, ...]
        Draws a small dot at (u,v) and text under it: "<d> m"
        """
        if flat_uvd is None:
            return
        if len(flat_uvd) < 3:
            return
        if (len(flat_uvd) % 3) != 0:
            self.get_logger().warn(f"centroid array length {len(flat_uvd)} not multiple of 3; ignoring extras")
        H, W = img_bgr.shape[:2]

        r = max(1, int(self.centroid_radius_px))
        scale = float(self.centroid_text_scale)
        thick = int(self.centroid_text_thickness)

        # White dot + black outline for contrast
        dot_color = (255, 255, 255)
        outline_color = (0, 0, 0)

        for k in range(0, len(flat_uvd) - 2, 3):
            u = float(flat_uvd[k + 0])
            v = float(flat_uvd[k + 1])
            d = float(flat_uvd[k + 2])

            x = int(np.clip(round(u), 0, W - 1))
            y = int(np.clip(round(v), 0, H - 1))

            # Dot (outline then fill)
            cv2.circle(img_bgr, (x, y), r + 1, outline_color, -1, cv2.LINE_AA)
            cv2.circle(img_bgr, (x, y), r, dot_color, -1, cv2.LINE_AA)

            # Text slightly below
            text = f"{d:.2f} m"
            ty = min(H - 2, y + 14)
            tx = min(W - 2, x + 4)

            # Shadow/outline text for readability
            cv2.putText(img_bgr, text, (tx + 1, ty + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, outline_color, thick + 2, cv2.LINE_AA)
            cv2.putText(img_bgr, text, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, dot_color, thick, cv2.LINE_AA)

    def _draw_detections(self, img_bgr, detections):
        H, W = img_bgr.shape[:2]

        for d in detections:
            cls = int(getattr(d, "class_id", 0))
            name = getattr(d, "class_name", str(cls))
            score = float(getattr(d, "score", 0.0))
            color = self._color_for_class(cls)

            poly = getattr(d, "polygon", None)
            if poly is not None and len(poly) >= 3:
                pts = np.array([[p.x, p.y] for p in poly], dtype=np.int32).reshape(-1, 1, 2)

                pts[:, 0, 0] = np.clip(pts[:, 0, 0], 0, W - 1)
                pts[:, 0, 1] = np.clip(pts[:, 0, 1], 0, H - 1)

                cv2.polylines(img_bgr, [pts], isClosed=True, color=color, thickness=2)

                x, y = int(pts[0, 0, 0]), int(pts[0, 0, 1])
                cv2.putText(
                    img_bgr,
                    f"{name} {score:.2f}",
                    (x, max(12, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            else:
                cx = float(getattr(d, "x", 0.0))
                cy = float(getattr(d, "y", 0.0))
                bw = float(getattr(d, "w", 0.0))
                bh = float(getattr(d, "h", 0.0))

                x0 = int(np.clip(cx - bw / 2.0, 0, W - 1))
                y0 = int(np.clip(cy - bh / 2.0, 0, H - 1))
                x1 = int(np.clip(cx + bw / 2.0, 0, W - 1))
                y1 = int(np.clip(cy + bh / 2.0, 0, H - 1))

                cv2.rectangle(img_bgr, (x0, y0), (x1, y1), color, 2)
                cv2.putText(
                    img_bgr,
                    f"{name} {score:.2f}",
                    (x0, max(12, y0 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                    cv2.LINE_AA,
                )


def main():
    rclpy.init()
    node = YoloOverlayCombiner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
