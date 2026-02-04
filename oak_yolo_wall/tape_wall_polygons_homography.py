'''
# ------------------------------------------------------------------------------
# TapeWallHomography ROS2 Node Summary
#
# Converts YOLO segmentation polygons (image pixels) into a 3D PointCloud2 by
# projecting each polygon pixel (or a sampled set of pixels) onto a flat ground
# plane using camera intrinsics + TF, optionally extruding in Z to form a “wall”.
#
# Inputs:
# - /yolo/detections (yolo_seg_interfaces/YoloSegDetectionArray):
#     Each detection provides a polygon in image pixel coordinates (u,v).
# - /oak/rgb/camera_info (sensor_msgs/CameraInfo):
#     Used to build K and K^-1 for pixel→ray projection.
# - TF: ground_frame <- camera_frame (tf2):
#     Used to transform camera rays into the ground frame and intersect with z=0.
#
# Core math / projection:
# - For each sampled pixel (u,v), computes a camera ray using K^-1.
# - Rotates that ray from optical frame into the robot’s camera convention
#   (oak_rgb: +X forward, +Y left, +Z up) using a fixed quaternion-derived R.
# - Uses TF to express the ray and camera origin in ground_frame, then intersects
#   the ray with the ground plane (z=0) to get (x,y) in meters.
#
# Sampling modes:
# - fill_polygon=True: scanline-fills the polygon and samples interior points on
#   a stride grid (fill_stride_px), capped by max_fill_points.
# - fill_polygon=False: uses only polygon vertices (optionally strided and/or
#   densified along edges via densify_edges + densify_step_px).
#
# Output geometry:
# - mode="flat": emits points at a single z (flat_z).
# - mode="wall": emits points across multiple z layers from wall_z_min..wall_z_max
#   with spacing wall_z_step (a vertical “extrusion”).
# - output_frame_mode:
#     - "base": publishes points in ground_frame (typical for Nav2/costmaps).
#     - "camera": publishes points in camera_frame by transforming back.
#
# Outputs:
# - /tape_wall/points (sensor_msgs/PointCloud2):
#     Uncolored XYZ point cloud representing the projected tape/area.
# - /tape_wall/centroid_distance_m (std_msgs/Float32MultiArray, optional debug):
#     When debug=True, publishes [u,v,dist,...] triples where dist is the planar
#     distance from base_link to the ground-intersection of each detection centroid.
#
# Reliability / robustness:
# - Uses BEST_EFFORT QoS (depth=1) to minimize latency/backpressure.
# - TF lookup tries at message stamp first, falls back to latest TF if needed.
# - Watchdog can publish an empty cloud if detections stop arriving
#   (empty_after_sec) to “clear” downstream consumers.
# ------------------------------------------------------------------------------
'''
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import numpy as np
import math
import tf2_ros
from tf2_ros import TransformException

from sensor_msgs.msg import PointCloud2, PointField, CameraInfo
from yolo_seg_interfaces.msg import YoloSegDetectionArray
from rclpy.duration import Duration
from rclpy.time import Time
from std_msgs.msg import Float32MultiArray

qos = QoSProfile( reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)


class TapeWallHomography(Node):
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

        #### PARAMS ####
        # ---- topic names ----
        self.declare_parameter("detections_topic", "/yolo/detections")      # input YOLO polygons topic
        self.declare_parameter("out_topic", "/tape_wall/points")            # output PointCloud2 topic
        self.declare_parameter("camera_info_topic", "/oak/rgb/camera_info") # CameraInfo source topic

        # ---- frames ----
        self.declare_parameter("camera_frame", "oak_rgb")    # camera pose frame, optional override; empty => use msg.header.frame_id
        self.declare_parameter("ground_frame", "base_link")  # frame where ground plane z=0 is defined
        self.declare_parameter("output_frame_mode", "base")  # base: wall points relative to ground_frame; camera: relative to camera_frame

        # Image size used by polygon coordinates (NN input / RGB output size)
        self.declare_parameter("img_width", 512)
        self.declare_parameter("img_height", 288)

        # ---- Polygon sampling (outline or filled mode) ---- 
        self.declare_parameter("min_polygon_points", 3)     # minimum points in polygon to process
        self.declare_parameter("fill_polygon", True)       # True => filled polygon instead of outline
        self.declare_parameter("fill_stride_px", 2)         # bigger => fewer points (faster)
        self.declare_parameter("max_fill_points", 2000)     # safety cap per detection
            # ---- If filled polygon = false (outline mode) ---- 
        self.declare_parameter("poly_stride", 1)            # take every Nth polygon vertex (useful if polygons have to many vertices, not needed with the striped lines used)
        self.declare_parameter("densify_edges", False)      # interpolate between vertices
        self.declare_parameter("densify_step_px", 3)        # approx pixel spacing along edges if densify_edges

        # ---- Wall extrusion and geometry ----
        self.declare_parameter("mode", "flat")          # "wall" or "flat" (wall extrudes in z, flat is single z plane)
        self.declare_parameter("flat_z", 0.05)          # height of PointCloud2, used if mode=="flat"
        self.declare_parameter("wall_z_min", 0.05)      # minimum z for wall mode
        self.declare_parameter("wall_z_max", 0.5)       # maximum z for wall mode
        self.declare_parameter("wall_z_step", 0.03)     # step between z layers in wall mode

        # Conversion from standard computer vision frame (z forward, x right, y down) to our robot coordinate frame (x forward, y left, z up)
        # Quaternion: x=-0.5, y=+0.5, z=-0.5, w=+0.5
        self._R_oak_opt = self.quat_to_R(-0.5, 0.5, -0.5, 0.5)
        self._logged_initial_camera_pose = False # for debug

        # ---- TF2 ---- to get transform from camera_frame -> ground_frame 
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=3.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Watchdog to publish empty cloud if no detections received for a while
        self.declare_parameter("empty_after_sec", 0.5)                      # timeout in seconds, set 0 to disable
        self._watchdog_timer = self.create_timer(0.1, self._watchdog_cb)    # check at 10Hz
        self._last_det_time = self.get_clock().now()                        # timestamp of last received detections
        self._cleared_once = False                                          # to avoid repeated empty pubs

        # Camera intrinsics
        cam_info_topic = self.get_parameter("camera_info_topic").value
        self._K = None 
        self._Kinv = None
        self._cam_frame = None  # from CameraInfo unless overridden
        self.sub_info = self.create_subscription(CameraInfo, cam_info_topic, self.on_camera_info, qos)
        self.get_logger().info(f"Waiting for CameraInfo on {cam_info_topic} ...")

        # Yolo Detections topic subscription 
        det_topic = self.get_parameter("detections_topic").value
        self.sub = self.create_subscription(YoloSegDetectionArray, det_topic, self.on_detections, qos)

        # Resulting PointCloud2 publisher
        out_topic = self.get_parameter("out_topic").value
        self.pub = self.create_publisher(PointCloud2, out_topic, qos)

        self.get_logger().info(
        "\n".join([
            "=== tape_wall_from_polygons ===",
            f" detections_topic   : {det_topic}",
            f" out_topic          : {out_topic}",
            f" camera_info_topic  : {cam_info_topic}",
            f" ground_frame       : {self.get_parameter('ground_frame').value}",
            f" camera_frame       : {self.get_parameter('camera_frame').value} (override; else CameraInfo frame_id)",
            f" output_frame_mode  : {self.get_parameter('output_frame_mode').value} (base=caster in ground_frame, camera=in camera_frame)",
            f" image size         : {self.get_parameter('img_width').value} x {self.get_parameter('img_height').value}",
            f" Extrusion mode     : {self.get_parameter('mode').value} (flat or wall)",
            "===============================",
        ])
        )

        # Debug logging
        self.declare_parameter("debug", False)
        self.debug = self.get_parameter("debug").value
        # Debug: centroid distance (camera -> ground hit of centroid ray)
        self.declare_parameter("centroid_distance_topic", "/tape_wall/centroid_distance_m")
        self._pub_centroid_dist = self.create_publisher(Float32MultiArray,self.get_parameter("centroid_distance_topic").value,qos)
 

    def on_camera_info(self, msg: CameraInfo):
        # Cache intrinsics once (or update if it changes)
        fx = msg.k[0]
        fy = msg.k[4]
        cx = msg.k[2]
        cy = msg.k[5]

        if fx <= 0 or fy <= 0:
            self.get_logger().warn("CameraInfo fx/fy invalid, ignoring.")
            return

        self._K = np.array([[fx, 0.0, cx],
                            [0.0, fy, cy],
                            [0.0, 0.0, 1.0]], dtype=np.float64)
        self._Kinv = np.linalg.inv(self._K)

        override = str(self.get_parameter("camera_frame").value).strip()
        self._cam_frame = override if override else msg.header.frame_id

        if (self.debug):
            self.get_logger().info(
                f"CameraInfo received. fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f} frame={self._cam_frame}",
                throttle_duration_sec=2.0
            )


    def pixel_to_ground_xy_rayplane(self, u: float, v: float, T_ground_cam) -> tuple | None:
        if self._Kinv is None:
            return None

        # Pixel -> ray in *optical* camera coordinates (what K expects)
        p = np.array([u, v, 1.0], dtype=np.float64)
        ray_opt = self._Kinv @ p

        # Convert optical ray to your intuitive oak_rgb ray:
        # (+X forward, +Y left, +Z up)
        ray_cam = self._R_oak_opt @ ray_opt

        # TF gives ground <- oak_rgb
        t = T_ground_cam.transform.translation
        q = T_ground_cam.transform.rotation
        R_ground_cam = self.quat_to_R(q.x, q.y, q.z, q.w)

        Cg = np.array([t.x, t.y, t.z], dtype=np.float64)     # camera origin in base_link
        ray_g = R_ground_cam @ ray_cam                       # ray direction in base_link

        denom = ray_g[2]
        if abs(denom) < 1e-9:
            return None

        lam = -Cg[2] / denom
        if lam <= 0.0:
            return None

        Pg = Cg + lam * ray_g
        return float(Pg[0]), float(Pg[1])


    def base_to_ground_hit_distance(self, u: float, v: float, T_ground_cam) -> float | None:
        hit = self.pixel_to_ground_xy_rayplane(u, v, T_ground_cam)
        if hit is None:
            return None
        gx, gy = hit
        return math.hypot(gx, gy)  # sqrt(gx^2 + gy^2)


    def quat_to_R(self, x, y, z, w):
        # normalized quaternion -> rotation matrix
        n = x*x + y*y + z*z + w*w
        if n < 1e-12:
            return np.eye(3, dtype=np.float64)
        s = 2.0 / n

        xx, yy, zz = x*x*s, y*y*s, z*z*s
        xy, xz, yz = x*y*s, x*z*s, y*z*s
        wx, wy, wz = w*x*s, w*y*s, w*z*s

        R = np.array([
            [1.0 - (yy + zz),       xy - wz,       xz + wy],
            [      xy + wz, 1.0 - (xx + zz),       yz - wx],
            [      xz - wy,       yz + wx, 1.0 - (xx + yy)],
        ], dtype=np.float64)
        return R

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
        self._last_det_time = self.get_clock().now()
        self._cleared_once = False

        if self._Kinv is None or self._cam_frame is None:
            self.get_logger().warn("No CameraInfo yet -> cannot project.", throttle_duration_sec=2.0)
            return

        ground_frame = str(self.get_parameter("ground_frame").value).strip() or "base_link"
        cam_frame = self._cam_frame  # from CameraInfo (or overridden in on_camera_info)

        out_mode = str(self.get_parameter("output_frame_mode").value).strip().lower()
        if out_mode not in ("base", "camera"):
            out_mode = "base"

        W = int(self.get_parameter("img_width").value)
        H = int(self.get_parameter("img_height").value)

        poly_stride = int(self.get_parameter("poly_stride").value)
        densify = bool(self.get_parameter("densify_edges").value)
        densify_step_px = float(self.get_parameter("densify_step_px").value)
        fill_polygon = bool(self.get_parameter("fill_polygon").value)
        fill_stride = int(self.get_parameter("fill_stride_px").value)
        max_fill = int(self.get_parameter("max_fill_points").value)

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

        # ---- TF lookup (stamp + fallback to latest) ----
        try:
            T_ground_cam = self.tf_buffer.lookup_transform(
                ground_frame, cam_frame,
                Time.from_msg(msg.header.stamp),
                timeout=Duration(seconds=0.05)
            )
        except TransformException as ex1:
            try:
                T_ground_cam = self.tf_buffer.lookup_transform(
                    ground_frame, cam_frame,
                    Time(),  # latest
                    timeout=Duration(seconds=0.05)
                )
                self.get_logger().warn(f"TF at stamp failed, using latest. ex={ex1}", throttle_duration_sec=1.0)
            except TransformException as ex2:
                self.get_logger().error(f"TF lookup failed {ground_frame}<-{cam_frame}: {ex2}")
                return

        if not self._logged_initial_camera_pose:
            self._log_camera_pose(T_ground_cam, ground_frame, cam_frame, every_sec=0.0)
            self._logged_initial_camera_pose = True
            
        tz = T_ground_cam.transform.translation.z

        t = T_ground_cam.transform.translation

        # If output camera, compute inverse transform for points
        if out_mode == "camera":
            try:
                T_cam_ground = self.tf_buffer.lookup_transform(
                    cam_frame, ground_frame,
                    Time.from_msg(msg.header.stamp),
                    timeout=Duration(seconds=0.05)
                )
            except TransformException:
                T_cam_ground = self.tf_buffer.lookup_transform(
                    cam_frame, ground_frame, Time(),
                    timeout=Duration(seconds=0.05)
                )
            t_bg = T_cam_ground.transform.translation
            q_bg = T_cam_ground.transform.rotation
            R_cam_ground = self.quat_to_R(q_bg.x, q_bg.y, q_bg.z, q_bg.w)
            t_cam_ground = np.array([t_bg.x, t_bg.y, t_bg.z], dtype=np.float64)

        # ---- debug counters ----
        n_det = len(msg.detections)
        n_poly_pts_in = 0
        n_poly_pts_used = 0
        rej_none = 0

        centroid_dists = []

        pts_xyz = []

        for i, det in enumerate(msg.detections):
            if len(det.polygon) < min_pts:
                continue

            poly = np.array([[p.x, p.y] for p in det.polygon], dtype=np.float32)
            poly[:, 0] = np.clip(poly[:, 0], 0, W - 1)
            poly[:, 1] = np.clip(poly[:, 1], 0, H - 1)

            if poly_stride > 1 and poly.shape[0] > poly_stride:
                poly = poly[::poly_stride, :]

            if (not fill_polygon) and densify and poly.shape[0] >= 3:
                poly = self.densify_poly(poly, step_px=densify_step_px)

            # after you build `poly` (and maybe stride/densify it if you still want):
            if fill_polygon:
                # IMPORTANT: fill wants the polygon vertices (outline) as input
                filled_uv = self.fill_polygon_scanline(poly, W, H, stride=fill_stride)

                # Safety cap so you don't explode point count
                if filled_uv.shape[0] > max_fill:
                    idx = np.random.choice(filled_uv.shape[0], size=max_fill, replace=False)
                    filled_uv = filled_uv[idx]

                samples_uv = filled_uv.astype(np.float32)  # shape (M,2)
            else:
                samples_uv = poly  # original behavior (outline / densified outline)

            for (u, v) in samples_uv:
                n_poly_pts_in += 1

                hit = self.pixel_to_ground_xy_rayplane(float(u), float(v), T_ground_cam)
                if hit is None:
                    rej_none += 1
                    continue

                gx, gy = hit
                n_poly_pts_used += 1

                for z in z_layers:
                    px, py, pz = gx, gy, float(z)

                    if out_mode == "base":
                        pts_xyz.append((px, py, pz))
                    else:
                        p_base = np.array([px, py, pz], dtype=np.float64)
                        p_cam = R_cam_ground @ p_base + t_cam_ground
                        pts_xyz.append((float(p_cam[0]), float(p_cam[1]), float(p_cam[2])))


            if self.debug:
                uc = float(np.mean(poly[:, 0]))
                vc = float(np.mean(poly[:, 1]))
                dist = self.base_to_ground_hit_distance(uc, vc, T_ground_cam)
                if dist is not None:
                    centroid_dists.extend([uc, vc, float(dist)])


        if self.debug:
            msg_out = Float32MultiArray()
            msg_out.data = centroid_dists
            self._pub_centroid_dist.publish(msg_out)

            if self.debug:
                self.get_logger().info(
                    f"centroid_dists_triples={len(centroid_dists)//3} data={centroid_dists}"
                )

        out_frame = ground_frame if out_mode == "base" else cam_frame

        if (self.debug):
            self.get_logger().info(
                f"dets={n_det} poly_in={n_poly_pts_in} used={n_poly_pts_used} "
                f"rej_none={rej_none} out_pts={len(pts_xyz)} out_frame={out_frame}",
                throttle_duration_sec=1.0
            )

        if len(pts_xyz) == 0:
            empty = self.make_cloud_xyz(msg.header.stamp, out_frame, np.zeros((0, 3), dtype=np.float32))
            self.pub.publish(empty)
            return

        pts_xyz = np.asarray(pts_xyz, dtype=np.float32)
        cloud = self.make_cloud_xyz(msg.header.stamp, out_frame, pts_xyz)
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


    def _watchdog_cb(self):
        empty_after = float(self.get_parameter("empty_after_sec").value)
        if empty_after <= 0.0:
            return

        now = self.get_clock().now()
        dt = (now - self._last_det_time).nanoseconds * 1e-9

        if dt >= empty_after and not self._cleared_once:
            # publish an empty cloud with "now" stamp
            empty = self.make_cloud_xyz(now.to_msg(), self.get_parameter("camera_frame").value,
                                        np.zeros((0, 3), dtype=np.float32))
            self.pub.publish(empty)
            self._cleared_once = True


    def fill_polygon_scanline(self, poly_uv: np.ndarray, W: int, H: int, stride: int = 8) -> np.ndarray:
        """
        poly_uv: (N,2) float32/float64 in pixel coords (u=x, v=y)
        Returns: (M,2) int32 pixels inside polygon sampled every 'stride' pixels.
        No OpenCV. Uses scanline filling.

        Notes:
        - Assumes polygon is closed; will close it if needed.
        - Clips to image bounds.
        """
        if poly_uv is None or len(poly_uv) < 3:
            return np.zeros((0,2), dtype=np.int32)

        poly = np.asarray(poly_uv, dtype=np.float64)

        # Clip rough bounds first to reduce work
        min_u = int(np.floor(np.min(poly[:,0])))
        max_u = int(np.ceil (np.max(poly[:,0])))
        min_v = int(np.floor(np.min(poly[:,1])))
        max_v = int(np.ceil (np.max(poly[:,1])))

        min_u = max(min_u, 0); max_u = min(max_u, W-1)
        min_v = max(min_v, 0); max_v = min(max_v, H-1)

        if min_u > max_u or min_v > max_v:
            return np.zeros((0,2), dtype=np.int32)

        # Ensure closed polygon
        if not np.allclose(poly[0], poly[-1]):
            poly = np.vstack([poly, poly[0]])

        xs = poly[:,0]
        ys = poly[:,1]

        filled = []

        # Scanline step in v
        for v in range(min_v, max_v + 1, stride):
            y = float(v) + 0.5  # center of pixel row

            # Find edge intersections with this scanline
            xints = []
            for i in range(len(poly)-1):
                x0, y0 = xs[i], ys[i]
                x1, y1 = xs[i+1], ys[i+1]

                # Ignore horizontal edges
                if abs(y1 - y0) < 1e-12:
                    continue

                # Check if scanline intersects edge (half-open to avoid double counting vertices)
                y_min = min(y0, y1)
                y_max = max(y0, y1)
                if not (y_min <= y < y_max):
                    continue

                # Compute intersection x
                t = (y - y0) / (y1 - y0)
                x = x0 + t * (x1 - x0)
                xints.append(x)

            if len(xints) < 2:
                continue

            xints.sort()

            # Fill between pairs
            for j in range(0, len(xints) - 1, 2):
                xa = int(np.floor(xints[j]   ))
                xb = int(np.floor(xints[j+1] ))

                # Clip and step in u
                xa = max(xa, min_u)
                xb = min(xb, max_u)
                if xa > xb:
                    continue

                for u in range(xa, xb + 1, stride):
                    filled.append((u, v))

        if not filled:
            return np.zeros((0,2), dtype=np.int32)

        return np.array(filled, dtype=np.int32)



    def _quat_to_rpy_zyx(self, x, y, z, w):
        """
        Returns roll, pitch, yaw in radians using ZYX convention (yaw around Z, pitch around Y, roll around X)
        for the rotation that maps camera-frame vectors into ground-frame vectors.
        """
        # normalize
        n = x*x + y*y + z*z + w*w
        if n < 1e-12:
            return 0.0, 0.0, 0.0
        s = 1.0 / math.sqrt(n)
        x *= s; y *= s; z *= s; w *= s

        # ZYX (yaw-pitch-roll)
        t0 = +2.0 * (w*x + y*z)
        t1 = +1.0 - 2.0 * (x*x + y*y)
        roll = math.atan2(t0, t1)

        t2 = +2.0 * (w*y - z*x)
        t2 = max(-1.0, min(1.0, t2))
        pitch = math.asin(t2)

        t3 = +2.0 * (w*z + x*y)
        t4 = +1.0 - 2.0 * (y*y + z*z)
        yaw = math.atan2(t3, t4)

        return roll, pitch, yaw


    def _log_camera_pose(self, T_ground_cam, ground_frame: str, cam_frame: str, every_sec: float = 1.0):
        """
        Logs camera height and intuitive angles.
        - height = translation.z (camera origin in ground_frame)
        - forward axis = +X of cam_frame (because oak_rgb: +X forward)
        - yaw = heading of forward axis in ground XY
        - pitch_down = how much forward axis points down toward the ground (positive = looking down)
        """
        now = self.get_clock().now()
        if not hasattr(self, "_last_tf_log"):
            self._last_tf_log = now
        dt = (now - self._last_tf_log).nanoseconds * 1e-9
        if dt < every_sec:
            return
        self._last_tf_log = now

        t = T_ground_cam.transform.translation
        q = T_ground_cam.transform.rotation

        R = self.quat_to_R(q.x, q.y, q.z, q.w)

        # camera forward (+X in oak_rgb) expressed in ground
        f = R @ np.array([1.0, 0.0, 0.0], dtype=np.float64)

        # yaw of forward axis in ground frame
        yaw = math.degrees(math.atan2(f[1], f[0]))

        # "tilt down" angle: 0 = parallel to ground, + = looking down
        horiz = math.sqrt(f[0]*f[0] + f[1]*f[1])
        pitch_down = math.degrees(math.atan2(-f[2], horiz))  # if f.z is negative -> looking down -> positive

        # Optional: also show classic rpy from quaternion (can be confusing, but sometimes useful)
        roll, pitch, yaw_rpy = self._quat_to_rpy_zyx(q.x, q.y, q.z, q.w)
        roll_d  = math.degrees(roll)
        pitch_d = math.degrees(pitch)
        yaw_d   = math.degrees(yaw_rpy)

        self.get_logger().info(
            " | ".join([
                f"TF {ground_frame}<-{cam_frame}",
                f"pos[m]=({t.x:+.3f},{t.y:+.3f},{t.z:+.3f})",
                f"height={t.z:.3f}",
                f"fwd=({f[0]:+.3f},{f[1]:+.3f},{f[2]:+.3f})",
                f"yaw_fwd={yaw:+.1f}deg",
                f"pitchDown={pitch_down:+.1f}deg",
                f"rpyZYX=({roll_d:+.1f},{pitch_d:+.1f},{yaw_d:+.1f})deg",
            ])
        )



def main():
    rclpy.init()
    node = TapeWallHomography()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
