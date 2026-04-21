from __future__ import absolute_import, division, print_function
import os
import numpy as np
from collections import Counter


def load_velodyne_points(filename):
    """Load 3D point cloud from KITTI file format
    (adapted from https://github.com/hunse/kitti)
    """
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points


def read_calib_file(path):
    """Read KITTI calibration file
    (from https://github.com/hunse/kitti)
    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data


def sub2ind(matrixSize, rowSub, colSub):
    """Convert row, col matrix subscripts to linear indices
    """
    m, n = matrixSize
    return rowSub * (n-1) + colSub - 1


def generate_depth_map(calib_dir, velo_filename, cam=2, vel_depth=False):
    """Generate a depth map from velodyne data
    """
    # load calibration files
    cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
    velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
    # 构建 4x4 的外参矩阵 T_velo_to_cam | 目的：确定雷达和摄像头之间的相对位置。
    # velo2cam：这是一个变换矩阵，负责把点从“雷达坐标系”转到“摄像头坐标系”。
    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    # get image shape
    im_shape = cam2cam["S_rect_02"][::-1].astype(np.int32)

    # compute projection matrix velodyne->image plane | 一步到位地把雷达点直接投影到图像像素坐标
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3) # 矫正旋转矩阵
    P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3, 4) # 相机内参投影矩阵
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)# 最终投影矩阵

    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    velo = load_velodyne_points(velo_filename)
    velo = velo[velo[:, 0] >= 0, :]  # 过滤掉雷达后方的点（x<0的点看不见）

    # project the points to the camera
    # 投影：3D 点 (x, y, z, 1) -> 2D 点 (u*z, v*z, z)

    velo_pts_im = np.dot(P_velo2im, velo.T).T

    # TODO 1. 提取深度值（z轴）
    depth_vals = velo_pts_im[:, 2]
    # 2. 过滤条件：深度值>0（避免除以0）、非nan、非inf
    valid_mask = (depth_vals > 1e-3) & (~np.isnan(depth_vals)) & (~np.isinf(depth_vals))
    # 3. 只保留有效点
    velo_pts_im = velo_pts_im[valid_mask]
    # 4. 若过滤后无有效点，直接返回空深度图（避免后续报错）
    if len(velo_pts_im) == 0:
        return np.zeros(im_shape[:2])

    # 5. 现在执行除法，不会有无效值警告！
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]

    # velo_pts_im = np.dot(P_velo2im, velo.T).T
    # velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]

    if vel_depth:
        velo_pts_im[:, 2] = velo[:, 0]

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]

    # project to image
    depth = np.zeros((im_shape[:2]))
    depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth < 0] = 0

    return depth
