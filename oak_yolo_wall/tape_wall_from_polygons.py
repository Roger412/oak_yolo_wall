#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import numpy as np
import struct

from sensor_msgs.msg import PointCloud2, PointField
from yolo_seg_interfaces.msg import YoloSegDetectionArray

qos = QoSProfile( reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)


class TapeWallFromPolygons(Node):
    """
    Takes polygon pixels from /yolo/detections and maps them into a ground rectangle:
      y (left/right) in [-side_near_m, +side_near_m]
      x (forward)    in [x_near_m, x_far_m]

    Pixel mapping (assuming rectified, flat-ish ground, stable camera):
      u in [0..W-1] -> y in [-side_near_m..+side_near_m]
      v in [0..H-1] -> x in [x_far_m..x_near_m]   (top is far, bottom is near)

    Then extrudes into a "wall" by adding z layers.
    Publishes PointCloud2 in base_frame (default base_link).
    """

    def __init__(self):
        super().__init__("tape_wall_from_polygons")

        # ---- params ----
        self.declare_parameter("detections_topic", "/yolo/detections")
        self.declare_parameter("out_topic", "/tape_wall/points")
        self.declare_parameter("base_frame", "oak_depth")

        # Image size used by polygon coordinates (your NN input / RGB output size)
        self.declare_parameter("img_width", 512)
        self.declare_parameter("img_height", 288)

        # Your empirical ground coverage (meters)
        
        self.declare_parameter("x_near_m", 0.22)   # 22cm
        self.declare_parameter("x_far_m", 1.72)    # 172cm
        self.declare_parameter("side_near_m", 0.19)  # half-width at x_near
        self.declare_parameter("side_far_m", 1.0)   # half-width at x_far 

        # Sampling / density
        self.declare_parameter("poly_stride", 1)      # take every Nth polygon vertex
        self.declare_parameter("densify_edges", True) # interpolate between vertices
        self.declare_parameter("densify_step_px", 3)  # approx pixel spacing along edges if densify_edges

        # Wall geometry
        self.declare_parameter("mode", "wall")   # "wall" or "flat"
        self.declare_parameter("flat_z", 0.25)   # used if mode=="flat"
        self.declare_parameter("wall_thickness", 0.06)   # total thickness in meters
        self.declare_parameter("wall_thickness_step", 0.01)

        self.declare_parameter("wall_z_min", 0.05)
        self.declare_parameter("wall_z_max", 0.5)
        self.declare_parameter("wall_z_step", 0.03)

        # Optional: shrink polygon slightly? (not implemented; leaving hook)
        self.declare_parameter("min_polygon_points", 3)

        det_topic = self.get_parameter("detections_topic").value
        out_topic = self.get_parameter("out_topic").value

        self.sub = self.create_subscription(YoloSegDetectionArray, det_topic, self.on_detections, qos)
        self.pub = self.create_publisher(PointCloud2, out_topic, qos)

        self.get_logger().info(
            f"Listening on {det_topic}, publishing cloud on {out_topic}. "
            f"Mapping: y=±{self.get_parameter('side_near_m').value}m, "
            f"x=[{self.get_parameter('x_near_m').value},{self.get_parameter('x_far_m').value}]m"
        )

    # ---------------------------
    # Core mapping: pixel -> (x,y)
    # ---------------------------
    def pixel_to_ground_xy(self, u: float, v: float, W: int, H: int, side_near: float, side_far: float, x_near: float, x_far: float):

        u = max(0.0, min(float(W - 1), float(u)))
        v = max(0.0, min(float(H - 1), float(v)))

        # v: top->bottom => x: far -> near
        v01 = v / float(H - 1)
        x = x_far + v01 * (x_near - x_far)  # v=0 => far, v=H-1 => near

        # interpolate half-width: near=small, far=big
        # normalize x into t in [0..1] where t=0 at near, t=1 at far
        if abs(x_far - x_near) < 1e-6:
            t = 0.0
        else:
            t = (x - x_near) / (x_far - x_near)
        t = max(0.0, min(1.0, t))

        half = side_near + t * (side_far - side_near)

        # u: left->right => y: -half .. +half
        u01 = u / float(W - 1)
        y = (u01 - 0.5) * 2.0 * half

        return -x, y



    def densify_poly(self, poly_uv: np.ndarray, step_px: float):
        """
        poly_uv: (N,2) float
        returns a new array with interpolated points along edges
        """
        if poly_uv.shape[0] < 2:
            return poly_uv

        out = []
        N = poly_uv.shape[0]
        for i in range(N):
            p0 = poly_uv[i]
            p1 = poly_uv[(i + 1) % N]
            out.append(p0)

            d = np.linalg.norm(p1 - p0)
            if d <= step_px:
                continue

            k = int(np.floor(d / step_px))
            for j in range(1, k):
                t = j / float(k)
                out.append((1.0 - t) * p0 + t * p1)

        return np.array(out, dtype=np.float32)

    # ---------------------------
    # ROS callback
    # ---------------------------
    def on_detections(self, msg: YoloSegDetectionArray):
        W = int(self.get_parameter("img_width").value)
        H = int(self.get_parameter("img_height").value)

        side_near_m = float(self.get_parameter("side_near_m").value)
        x_near = float(self.get_parameter("x_near_m").value)
        x_far  = float(self.get_parameter("x_far_m").value)

        poly_stride = int(self.get_parameter("poly_stride").value)
        densify = bool(self.get_parameter("densify_edges").value)
        densify_step_px = float(self.get_parameter("densify_step_px").value)

        mode = str(self.get_parameter("mode").value).lower().strip()
        flat_z = float(self.get_parameter("flat_z").value)

        zmin = float(self.get_parameter("wall_z_min").value)
        zmax = float(self.get_parameter("wall_z_max").value)
        zstep = float(self.get_parameter("wall_z_step").value)

        min_pts = int(self.get_parameter("min_polygon_points").value)

        # z layers
        if mode == "flat":
            z_layers = np.array([flat_z], dtype=np.float32)
        else:
            z_layers = np.arange(zmin, zmax + 1e-6, zstep, dtype=np.float32)

        pts_xyz = []

        for det in msg.detections:
            if len(det.polygon) < min_pts:
                continue

            # polygon pixels from message
            poly = np.array([[p.x, p.y] for p in det.polygon], dtype=np.float32)

            # downsample vertices
            if poly_stride > 1 and poly.shape[0] > poly_stride:
                poly = poly[::poly_stride, :]

            # densify edges (nice for "continuous wall")
            if densify and poly.shape[0] >= 3:
                poly = self.densify_poly(poly, step_px=densify_step_px)

            # map each sampled pixel to ground and extrude
            thick = float(self.get_parameter("wall_thickness").value)
            step  = float(self.get_parameter("wall_thickness_step").value)

            offsets = np.arange(-thick / 2.0, thick / 2.0 + 1e-6, step)

            for (u, v) in poly:
                side_near = float(self.get_parameter("side_near_m").value)
                side_far  = float(self.get_parameter("side_far_m").value)
                x_near    = float(self.get_parameter("x_near_m").value)
                x_far     = float(self.get_parameter("x_far_m").value)

                x, y = self.pixel_to_ground_xy(u, v, W, H, side_near, side_far, x_near, x_far)


                for dy in offsets:          # <-- thickness left/right
                    for z in z_layers:      # <-- height
                        pts_xyz.append((x, y + dy, float(z)))

        if not pts_xyz:
            return

        pts_xyz = np.asarray(pts_xyz, dtype=np.float32)

        cloud = self.make_cloud_xyz(msg.header.stamp, self.get_parameter("base_frame").value, pts_xyz)
        self.pub.publish(cloud)

    # ---------------------------
    # PointCloud2 helper
    # ---------------------------
    def make_cloud_xyz(self, stamp, frame_id: str, pts_xyz: np.ndarray) -> PointCloud2:
        pc = PointCloud2()
        pc.header.stamp = stamp
        pc.header.frame_id = frame_id

        pc.height = 1
        pc.width = int(pts_xyz.shape[0])
        pc.is_bigendian = False
        pc.is_dense = False

        pc.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        pc.point_step = 12
        pc.row_step = pc.point_step * pc.width

        pc.data = pts_xyz.tobytes()
        return pc


def main():
    rclpy.init()
    node = TapeWallFromPolygons()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
