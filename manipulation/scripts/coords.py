"""
共用座標轉換：相機像素+深度 -> 世界座標 -> 機器人 base（root_link）座標系。

pixel_to_world() 沿用 g1_scan_python/scenario.py 裡 G1Scenario.scan() 已經驗證過的
做法（camera 外參用 isaacsim.sensors.camera.Camera 的 get_world_pose() 取得，
不是重新手刻 4x4 矩陣）。world_to_robot_base() 是新增的，把世界座標點轉到機器人
root_link 的座標系下，這是 RMPflow set_end_effector_target() 期望的輸入座標系。
"""

import numpy as np
from isaacsim.core.utils.numpy.rotations import quats_to_rot_matrices


def pixel_to_world(u, v, depth_value, intrinsics, cam_pos, cam_quat):
    """(u,v) 像素 + 深度值 -> 世界座標。Isaac Sim 相機本地座標系朝 -Z 看。"""
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    x_c = (u - cx) * depth_value / fx
    y_c = (v - cy) * depth_value / fy
    z_c = -depth_value
    point_camera = np.array([x_c, y_c, z_c])
    rot_mat = quats_to_rot_matrices(cam_quat)
    return rot_mat @ point_camera + cam_pos


def world_to_base(point_world, base_pos, base_quat):
    """世界座標點 -> 機器人 root_link 座標系（base_quat 的逆旋轉 + 平移）。"""
    rot_mat = quats_to_rot_matrices(base_quat)
    return rot_mat.T @ (np.asarray(point_world) - np.asarray(base_pos))


def look_at_rotation(eye, target):
    """給定相機位置+注視點，算出相機本地座標系(local X=right, Y=up, Z=-forward，
    對應標準 USD 相機朝 -Z 看、+Y 朝上)相對世界座標系的旋轉矩陣。

    刻意不依賴 rep.create.camera(look_at=...) 內部怎麼算旋轉 —— 直接把這個矩陣轉成
    euler XYZ 角度餵給 rep.create.camera(rotation=...)，跟這裡拿來算外參轉換用的是
    同一個矩陣，兩邊必定一致，不用去猜 Replicator 內部 look_at 的旋轉慣例對不對。
    """
    eye = np.array(eye, dtype=float)
    target = np.array(target, dtype=float)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    up_hint = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up_hint)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, fwd)
    return np.stack([right, true_up, -fwd], axis=1)


def look_at_quat_and_euler_deg(eye, target):
    """回傳 (wxyz 四元數, XYZ 外部歐拉角/度)。四元數給 coords 座標轉換用，
    歐拉角給 rep.create.camera(rotation=...) 用，確保兩者是同一個旋轉。"""
    from scipy.spatial.transform import Rotation
    rot = look_at_rotation(eye, target)
    r = Rotation.from_matrix(rot)
    xyzw = r.as_quat()
    quat_wxyz = xyzw[[3, 0, 1, 2]]
    euler_xyz_deg = r.as_euler("XYZ", degrees=True)
    return quat_wxyz, euler_xyz_deg
