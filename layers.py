# 兼容Python2的绝对导入、除法、打印语法（Python3环境下也可运行）
from __future__ import absolute_import, division, print_function

import numpy as np  # 数值计算库，用于生成网格、矩阵运算等
import torch       # PyTorch核心库，构建神经网络和张量运算
import torch.nn as nn  # 神经网络层封装
import torch.nn.functional as F  # 神经网络函数（如插值、激活）
import math        # 数学函数（如三角函数、开方）


def disp_to_depth(disp, min_depth, max_depth):
    """
    将网络输出的归一化视差(disp)转换为物理意义的深度值(depth)
    原理：视差与深度成反比（depth=1/disp），结合线性缩放平移映射到指定深度范围
    Args:
        disp: 网络输出的视差图，shape=[B,1,H,W]，值域0~1（sigmoid输出）
        min_depth: 深度最小值（物理意义，如0.1米）
        max_depth: 深度最大值（物理意义，如100米）
    Returns:
        scaled_disp: 缩放后的视差图，值域[1/max_depth, 1/min_depth]
        depth: 最终深度图，值域[min_depth, max_depth]
    """
    # 视差最小值 = 1/最大深度（深度越大，视差越小）
    min_disp = 1 / max_depth
    # 视差最大值 = 1/最小深度（深度越小，视差越大）
    max_disp = 1 / min_depth
    # 线性缩放：将disp从[0,1]映射到[min_disp, max_disp]（缩放+平移）
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    # 深度 = 1/视差（视差与深度成反比）
    depth = 1 / scaled_disp
    return scaled_disp, depth


def transformation_from_parameters(axisangle, translation, invert=False):
    """
    将网络输出的位姿参数（轴角旋转+平移向量）转换为4x4相机变换矩阵
    Args:
        axisangle: 轴角表示的旋转向量，shape=[B,1,3]（B=批次大小）
        translation: 平移向量，shape=[B,3]
        invert: 是否求逆（用于反向相机运动）
    Returns:
        M: 4x4相机变换矩阵，shape=[B,4,4]
    """
    # 轴角旋转向量 → 4x4旋转矩阵
    R = rot_from_axisangle(axisangle)
    # 复制平移向量（避免原地修改）
    t = translation.clone()

    # 若求逆：旋转矩阵转置（正交矩阵逆=转置），平移向量取反
    if invert:
        R = R.transpose(1, 2)
        t *= -1

    # 平移向量 → 4x4平移矩阵
    T = get_translation_matrix(t)

    # 组合旋转和平移：逆变换是R*T，正变换是T*R（齐次矩阵乘法规则）
    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """
    将平移向量转换为4x4齐次平移矩阵
    齐次矩阵格式：
        [1 0 0 tx]
        [0 1 0 ty]
        [0 0 1 tz]
        [0 0 0 1]
    Args:
        translation_vector: 平移向量，shape=[B,3]
    Returns:
        T: 4x4平移矩阵，shape=[B,4,4]
    """
    # 初始化4x4全零矩阵，设备与输入一致（CPU/GPU）
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    # 重塑平移向量为[B,3,1]（适配矩阵乘法维度）
    t = translation_vector.contiguous().view(-1, 3, 1)

    # 填充单位矩阵部分（前3x3对角线为1）
    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    # 填充平移分量（第4列前3行）
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """
    将轴角旋转向量转换为4x4齐次旋转矩阵（罗德里格斯公式的矩阵实现）
    轴角原理：vec = 旋转轴 × 旋转角度（||vec||=角度，vec/||vec||=旋转轴）
    Args:
        vec: 轴角旋转向量，shape=[B,1,3]
    Returns:
        rot: 4x4旋转矩阵，shape=[B,4,4]
    """
    # 计算旋转角度（向量的L2范数），shape=[B,1,1]
    angle = torch.norm(vec, 2, 2, True)
    # 计算单位旋转轴（避免除零，加1e-7），shape=[B,1,3]
    axis = vec / (angle + 1e-7)

    # 角度的余弦/正弦值
    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca  # 简化计算的中间变量

    # 提取旋转轴的x/y/z分量（扩展维度适配矩阵运算）
    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    # 预计算中间变量（减少重复计算）
    xs = x * sa  # x*sinθ
    ys = y * sa  # y*sinθ
    zs = z * sa  # z*sinθ
    xC = x * C  # x*(1-cosθ)
    yC = y * C  # y*(1-cosθ)
    zC = z * C  # z*(1-cosθ)
    xyC = x * yC  # x*y*(1-cosθ)
    yzC = y * zC  # y*z*(1-cosθ)
    zxC = z * xC  # z*x*(1-cosθ)

    # 初始化4x4旋转矩阵（设备与输入一致）
    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    # 填充3x3旋转矩阵部分（罗德里格斯公式展开）
    # i 是 “行索引”（0、1、2 分别对应第 1、2、3 行）；
    # j 是 “列索引”（0、1、2 分别对应第 1、2、3 列）

    rot[:, 0, 0] = torch.squeeze(x * xC + ca) # “第 0 行第 0 列” 元素；
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca) # “第 2 行第 2 列” 元素。
    # 齐次矩阵最后一行最后一列设为1
    rot[:, 3, 3] = 1

    return rot


class ConvBlock(nn.Module):
    """标准卷积块：3x3卷积 + ELU激活（基础特征提取单元）"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        # 3x3卷积层（带反射填充，保留边缘信息）
        self.conv = Conv3x3(in_channels, out_channels)
        # ELU激活函数（带inplace=True节省内存）
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        # 卷积 → 激活
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class ConvBlockDepth(nn.Module):
    """轻量化卷积块：深度可分离3x3卷积 + GELU激活（减少参数量）"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlockDepth, self).__init__()
        # 深度可分离3x3卷积层
        self.conv = DepthConv3x3(in_channels, out_channels)
        # GELU激活函数（比ELU更适合轻量化模型）
        self.nonlin = nn.GELU()

    def forward(self, x):
        # 深度可分离卷积 → 激活
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class DepthConv3x3(nn.Module):
    """深度可分离3x3卷积层：先逐通道卷积，再逐点卷积（分组卷积实现）"""
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(DepthConv3x3, self).__init__()
        # 填充方式：反射填充（use_refl=True）/零填充（False），保留边缘特征
        if use_refl:
            self.pad = nn.ReflectionPad2d(1)  # 3x3卷积需填充1圈，保持尺寸不变
        else:
            self.pad = nn.ZeroPad2d(1)
        # 深度可分离卷积：groups=out_channels（逐通道卷积），无偏置（减少参数）
        self.conv = nn.Conv2d(int(in_channels), int(out_channels),
                              kernel_size=3, groups=int(out_channels), bias=False)

    def forward(self, x):
        # 填充 → 深度可分离卷积
        out = self.pad(x)
        out = self.conv(out)
        return out


class Conv3x3(nn.Module):
    """标准3x3卷积层：带填充，保证输入输出尺寸一致"""
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()
        # 填充方式：反射填充（优先，避免边缘信息丢失）/零填充
        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        # 标准3x3卷积（无分组，带偏置）
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        # 填充 → 卷积
        out = self.pad(x)
        out = self.conv(out)
        return out

class ConvBlock1x1(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock1x1, self).__init__()

        self.conv = Conv1x1(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

class ConvBlock1x3_3x1(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock1x3_3x1, self).__init__()

        self.conv = Conv1x3_3x1(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class Conv1x1(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels):
        super(Conv1x1, self).__init__()

        self.conv = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1, stride=1)

    def forward(self, x):
        out = self.conv(x)
        return out

class Conv1x3_3x1(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv1x3_3x1, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv1x3 = nn.Conv2d(int(in_channels), int(out_channels), (1, 3))
        self.conv3x1 = nn.Conv2d(int(out_channels), int(out_channels), (3, 1))
        # self.elu1 = nn.ELU(inplace=True)
        # self.elu2 = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv1x3(out)
        # out = self.elu1(out)
        out = self.conv3x1(out)
        # out = self.elu2(out)
        return out


class Conv3x3_down(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3_down, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3, 2)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out
class BackprojectDepth(nn.Module):
    """
    深度图反投影层：将2D深度图转换为3D点云（相机坐标系）
    原理：cam_points = depth × (K_inv × pix_coords)，其中K_inv是内参逆矩阵
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()
        # 保存批次大小、图像高/宽
        self.batch_size = batch_size
        self.height = height
        self.width = width

        # 生成像素坐标网格：shape=[2, H, W]（0行是x坐标，1行是y坐标）
        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        # 转为PyTorch参数（无需梯度更新）
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        # 生成全1向量：用于构造齐次坐标（x,y,1），shape=[B,1,H*W]
        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        # 构造像素齐次坐标：shape=[B,3,H*W]（3行分别是x,y,1）
        # 1. 展平像素坐标为[2, H*W]，扩展批次维度为[1,2,H*W]
        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        # 2. 复制到整个批次，shape=[B,2,H*W]--复制到整个批次：形状变为[B, 2, H*W]（每个样本共享相同的像素坐标）
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        # 3. 拼接全1向量，得到齐次坐标，转为参数（无需梯度）--拼接全1向量，形成齐次坐标[B, 3, H*W]（3行分别为u, v, 1）
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        """
        前向传播：深度图→3D点云
        Args:
            depth: 深度图，shape=[B,1,H,W]
            inv_K: 相机内参逆矩阵，shape=[B,4,4]
        Returns:
            cam_points: 相机坐标系3D点云，shape=[B,4,H*W]（齐次坐标）
        """
        # 第一步：内参逆 × 像素齐次坐标 → [B,3,H*W]
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        # 第二步：深度值缩放 → 相机坐标系3D点（x,y,z）
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        # 第三步：拼接全1向量，转为齐次坐标（x,y,z,1）
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    """
    3D点云投影层：将相机坐标系3D点云投影到2D像素平面
    原理：pix_coords = (K×T×cam_points)[:2,:] / (K×T×cam_points)[2,:]（透视除法）
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()
        # 保存批次大小、图像高/宽、防除零epsilon
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps  # 避免除以0

    def forward(self, points, K, T):
        """
        前向传播：3D点云→2D像素坐标
        Args:
            points: 3D点云（齐次坐标），shape=[B,4,H*W]
            K: 相机内参矩阵，shape=[B,4,4]
            T: 相机位姿矩阵，shape=[B,4,4]
        Returns:
            pix_coords: 归一化像素坐标，shape=[B,H,W,2]，值域[-1,1]（适配F.grid_sample）
        """
        # 计算投影矩阵P = K×T（取前3行），shape=[B,3,4]|投影矩阵P是内参K和位姿T的组合，负责 “先转换坐标系，再投影到像素平面”；
        P = torch.matmul(K, T)[:, :3, :]

        # 第一步：3D点云投影到相机平面 → [B,3,H*W]
        cam_points = torch.matmul(P, points)

        # 第二步：透视除法（z分量为深度，避免除零加eps）→ [B,2,H*W]|整数索引会消除对应维度，所以要unsqueeze
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)

        # 第三步：重塑为[B,2,H,W]，转置为[B,H,W,2]（适配网格采样）
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)

        # 第四步：归一化到[-1,1]（PyTorch grid_sample要求的坐标范围）
        pix_coords[..., 0] /= self.width - 1    # x坐标归一化到[0,1]
        pix_coords[..., 1] /= self.height - 1   # y坐标归一化到[0,1]
        pix_coords = (pix_coords - 0.5) * 2     # 平移缩放至[-1,1]

        return pix_coords


def upsample(x, scale_factor=2, mode="bilinear"):
    """
    上采样函数：将特征图放大指定倍数（默认2倍）
    Args:
        x: 输入特征图，shape=[B,C,H,W]
        scale_factor: 放大倍数（默认2）
        mode: 插值方式（bilinear=双线性，nearest=最近邻）
    Returns:
        上采样后的特征图，shape=[B,C,H*scale_factor,W*scale_factor]
    """
    return F.interpolate(x, scale_factor=scale_factor, mode=mode)

def updown_sample(x, scale_fac):
    """Upsample input tensor by a factor of scale_fac
    """
    return F.interpolate(x, scale_factor=scale_fac, mode="nearest")


def get_smooth_loss(disp, img):
    """
    计算视差图的平滑损失（边缘感知）
    原理：图像边缘处允许视差突变，平坦区域强制视差平滑
    Args:
        disp: 视差图，shape=[B,1,H,W]
        img: 原始RGB图像，shape=[B,3,H,W]
    Returns:
        smooth_loss: 平滑损失值（标量）
    """
    # 计算视差图x方向梯度（水平方向）：shape=[B,1,H,W-1]
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    # 计算视差图y方向梯度（垂直方向）：shape=[B,1,H-1,W]
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    # 计算RGB图像x方向梯度（取均值，降通道）：shape=[B,1,H,W-1]
    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    # 计算RGB图像y方向梯度（取均值，降通道）：shape=[B,1,H-1,W]
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    # 边缘感知加权：图像梯度越大，视差梯度权重越小（exp(-grad_img)）
    grad_disp_x *= torch.exp(-grad_img_x) # 图像梯度大（边缘地区）,视差梯度大是正常的，权重应趋于0，使不计入损失。
    grad_disp_y *= torch.exp(-grad_img_y) #

    # 总平滑损失 = x方向梯度均值 + y方向梯度均值
    return grad_disp_x.mean() + grad_disp_y.mean()


class SSIM(nn.Module):
    """
    结构相似性（SSIM）损失层：衡量两幅图像的结构相似性，值越小越相似
    原理：从亮度、对比度、结构三个维度计算相似度
    """
    def __init__(self):
        super(SSIM, self).__init__()
        # 均值池化（3x3核，步长1）：计算局部亮度
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        # 方差/协方差池化：计算局部对比度/结构
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        # 反射填充（3x3池化需填充1圈，保持尺寸）
        self.refl = nn.ReflectionPad2d(1)

        # SSIM公式的常数项（避免分母为0）
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        """
        前向传播：计算SSIM损失
        Args:
            x/y: 对比的两幅图像，shape=[B,C,H,W]
        Returns:
            ssim_loss: SSIM损失值，值域[0,1]
        """
        # 填充图像（保持池化后尺寸不变）
        x = self.refl(x)
        y = self.refl(y)

        # 计算局部亮度均值
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        # 计算局部方差（E[X²] - (E[X])²）
        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        # 计算局部协方差（E[XY] - E[X]E[Y]）
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        # SSIM分子：(2μxμy+C1)(2σxy+C2)
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        # SSIM分母：(μx²+μy²+C1)(σx+σy+C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        # SSIM损失：(1 - SSIM)/2，钳位到[0,1]（确保损失非负）
        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def compute_depth_errors(gt, pred):
    """
    计算深度估计的评估指标（行业标准）
    Args:
        gt: 真实深度图，shape=[B,H,W]（已去除无效值）
        pred: 预测深度图，shape=[B,H,W]
    Returns:
        abs_rel: 绝对相对误差
        sq_rel: 平方相对误差
        rmse: 均方根误差
        rmse_log: 对数域均方根误差
        a1/a2/a3: 阈值精度（1.25^1/2/3倍内的像素比例）
    """

    # ========== 兜底过滤：只保留有效点计算 ==========
    valid_mask = (gt > 1e-3) & (gt < 80) & (pred > 1e-3) & (pred < 80)
    valid_mask = valid_mask & (~torch.isinf(gt)) & (~torch.isinf(pred))
    valid_mask = valid_mask & (~torch.isnan(gt)) & (~torch.isnan(pred))
    gt = gt[valid_mask]
    pred = pred[valid_mask]

    # 计算预测/真实深度的比值上限（避免除零）
    thresh = torch.max((gt / pred), (pred / gt))
    # 阈值精度：a1(1.25) > a2(1.25²) > a3(1.25³)，值越高越好
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    # 均方根误差（RMSE）
    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    # 对数域均方根误差（RMSE_log）：减轻尺度影响
    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    # 绝对相对误差（AbsRel）：整体误差比例
    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    # 平方相对误差（SqRel）：对大误差更敏感
    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3