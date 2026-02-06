'''
# ------------------------------------------------------------------------------
# DepthBandToPointCloud ROS2 Node Summary
#
# Lightweight depth-to-pointcloud utility that converts only a horizontal “band”
# (a configurable range of image rows) from a depth image into a sparse XYZ
# PointCloud2 in the camera optical frame. This is useful when you only care about
# obstacles / ground profile in a particular region (e.g., near the bottom of the
# image) and want something much cheaper than a full dense point cloud.
#
# Inputs:
# - /oak/depth/image_raw (sensor_msgs/Image):
#     Depth image, typically uint16 in millimeters (16UC1), but also supports float depth.
# - /oak/depth/camera_info (sensor_msgs/CameraInfo):
#     Provides intrinsics (fx, fy, cx, cy) used for back-projection.
#
# Processing:
# - Waits until CameraInfo has been received (fx/fy/cx/cy cached from msg.k).
# - Extracts a row band [row_start, row_end) and samples it sparsely using row_stride
#   and col_stride to reduce compute and point count.
# - Converts depth to meters (uint16 mm → meters, otherwise assumes already meters).
# - Filters points by range (min_range_m .. max_range_m) and ignores invalid/zero depth.
# - Back-projects valid pixels (u,v,Z) into 3D camera-optical coordinates:
#     X = (u - cx) * Z / fx
#     Y = (v - cy) * Z / fy
#     Z = Z
#
# Output:
# - /oak/depth/band_points (sensor_msgs/PointCloud2):
#     Uncolored XYZ point cloud (FLOAT32 x/y/z) with one point per sampled valid pixel.
#     Header stamp comes from the depth image; frame_id is configurable (defaults to oak_depth).
#
# Design choices / limits:
# - Publishes nothing when there is no CameraInfo yet, when the band is empty/invalid,
#   or when no valid depth pixels exist in the chosen band.
# - Produces a sparse cloud (not organized) and does not attach RGB color fields.
# - Assumes the depth image and intrinsics correspond to the same resolution and that
#   the chosen frame_id matches the depth optical frame used by TF.
# ------------------------------------------------------------------------------
'''

#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from cv_bridge import CvBridge

qos = QoSProfile( reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)

class DepthBandToPointCloud(Node):
    def __init__(self):
        super().__init__("depth_band_to_pointcloud")
        self.bridge = CvBridge()

        self.declare_parameter("depth_topic", "/oak/depth/image_raw")
        self.declare_parameter("camera_info_topic", "/oak/depth/camera_info")
        self.declare_parameter("cloud_topic", "/oak/depth/band_points")

        self.declare_parameter("row_start", 200)
        self.declare_parameter("row_end", 260)
        self.declare_parameter("col_stride", 2)      # sample every N cols
        self.declare_parameter("row_stride", 2)      # sample every N rows
        self.declare_parameter("min_range_m", 0.2)
        self.declare_parameter("max_range_m", 10.0)
        self.declare_parameter("frame_id", "oak_depth")  # should match your depth optical frame

        self.depth_topic = self.get_parameter("depth_topic").value
        self.caminfo_topic = self.get_parameter("camera_info_topic").value
        self.cloud_topic = self.get_parameter("cloud_topic").value

        self.row_start = int(self.get_parameter("row_start").value)
        self.row_end = int(self.get_parameter("row_end").value)
        self.col_stride = int(self.get_parameter("col_stride").value)
        self.row_stride = int(self.get_parameter("row_stride").value)
        self.min_range = float(self.get_parameter("min_range_m").value)
        self.max_range = float(self.get_parameter("max_range_m").value)
        self.frame_id = self.get_parameter("frame_id").value

        self.fx = self.fy = self.cx = self.cy = None

        self.sub_info = self.create_subscription(CameraInfo, self.caminfo_topic, self.on_info, qos)
        self.sub_depth = self.create_subscription(Image, self.depth_topic, self.on_depth, qos)
        self.pub_cloud = self.create_publisher(PointCloud2, self.cloud_topic, qos)

    def on_info(self, msg: CameraInfo):
        # K = [fx 0 cx; 0 fy cy; 0 0 1]
        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])
        self.cx = float(msg.k[2])
        self.cy = float(msg.k[5])

    def on_depth(self, msg: Image):
        if self.fx is None:
            return

        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        # typical: uint16 in mm
        if depth is None or depth.size == 0:
            return

        H, W = depth.shape[:2]
        r0 = max(0, min(H - 1, self.row_start))
        r1 = max(0, min(H, self.row_end))
        if r1 <= r0:
            return

        # Sample a band
        rows = np.arange(r0, r1, max(1, self.row_stride), dtype=np.int32)
        cols = np.arange(0, W, max(1, self.col_stride), dtype=np.int32)
        vv, uu = np.meshgrid(rows, cols, indexing="ij")  # vv shape (Nr,Nc)

        d = depth[vv, uu]

        # Convert to meters
        if d.dtype == np.uint16:
            Z = d.astype(np.float32) * 0.001
        else:
            Z = d.astype(np.float32)

        valid = (Z > 0.0) & (Z >= self.min_range) & (Z <= self.max_range)
        if not np.any(valid):
            return

        uu = uu[valid].astype(np.float32)
        vv = vv[valid].astype(np.float32)
        Z  = Z[valid].astype(np.float32)

        # Back-project to camera frame (optical)
        X = (uu - self.cx) * Z / self.fx
        Y = (vv - self.cy) * Z / self.fy

        points = np.stack([X, Y, Z], axis=1).astype(np.float32)

        cloud_msg = self._points_to_pointcloud2(points, msg.header.stamp, self.frame_id)
        self.pub_cloud.publish(cloud_msg)

    def _points_to_pointcloud2(self, points_xyz: np.ndarray, stamp, frame_id: str) -> PointCloud2:
        pc = PointCloud2()
        pc.header.stamp = stamp
        pc.header.frame_id = frame_id
        pc.height = 1
        pc.width = int(points_xyz.shape[0])
        pc.is_bigendian = False
        pc.is_dense = False

        pc.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        pc.point_step = 12
        pc.row_step = pc.point_step * pc.width

        pc.data = points_xyz.tobytes()
        return pc


def main():
    rclpy.init()
    node = DepthBandToPointCloud()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
