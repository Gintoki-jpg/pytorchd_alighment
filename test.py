import numpy as np
import torch

def get_rotation_matrix(pitch, yaw, roll):
    # Ensure pitch, yaw, roll are tensors with requires_grad=True
    pitch = pitch.unsqueeze(0) if pitch.dim() == 0 else pitch
    yaw = yaw.unsqueeze(0) if yaw.dim() == 0 else yaw
    roll = roll.unsqueeze(0) if roll.dim() == 0 else roll

    # Precompute trigonometric functions
    cos_p = torch.cos(pitch)
    sin_p = torch.sin(pitch)
    cos_y = torch.cos(yaw)
    sin_y = torch.sin(yaw)
    cos_r = torch.cos(roll)
    sin_r = torch.sin(roll)

    # Create 3x3 identity matrices
    ones = torch.ones_like(pitch)
    zeros = torch.zeros_like(pitch)

    # Rotation matrix around the X-axis
    R_x = torch.stack([
     torch.stack([ones, zeros, zeros], dim=-1),
     torch.stack([zeros, cos_p, -sin_p], dim=-1),
     torch.stack([zeros, sin_p, cos_p], dim=-1)
    ], dim=-2)

    # Rotation matrix around the Y-axis
    R_y = torch.stack([
     torch.stack([cos_y, zeros, sin_y], dim=-1),
     torch.stack([zeros, ones, zeros], dim=-1),
     torch.stack([-sin_y, zeros, cos_y], dim=-1)
    ], dim=-2)

    # Rotation matrix around the Z-axis
    R_z = torch.stack([
     torch.stack([cos_r, -sin_r, zeros], dim=-1),
     torch.stack([sin_r, cos_r, zeros], dim=-1),
     torch.stack([zeros, zeros, ones], dim=-1)
    ], dim=-2)

    # Combine the rotation matrices
    R = torch.matmul(R_z, torch.matmul(R_y, R_x))

    # # If the inputs were scalars, remove the extra dimension
    # if R.shape[0] == 1:
    #     R = R.squeeze(0)
    # C2W -> W2C
    R.transpose(1, 2)
    return R

def extract_euler_angles(R):
    # 确保 R 是一个 3x3 矩阵
    assert R.shape == (3, 3)

    # 首先对输入的W2C R执行转置操作获得C2W下的R
    R = R.T

    # 提取旋转矩阵的元素
    r11, r12, r13 = R[0, :]
    r21, r22, r23 = R[1, :]
    r31, r32, r33 = R[2, :]

    # 计算俯仰角（pitch）
    pitch = np.arctan2(-r31, np.sqrt(r32 ** 2 + r33 ** 2))

    # 计算偏航角（yaw）
    yaw = np.arctan2(r32, r33)

    # 计算滚转角（roll）
    roll = np.arctan2(r21, r11)

    return pitch, yaw, roll



# 给定的旋转矩阵（这个旋转矩阵是W2C下的旋转矩阵）
R = np.array([[0.2654, 0.5792, 0.7707],
              [-0.4987, -0.6016, 0.6239],
              [0.8251, -0.5500, 0.1292]])


pitch, yaw, roll = extract_euler_angles(R)


# 使用三个欧拉角计算旋转矩阵
pitch = torch.tensor(pitch, requires_grad=True)
yaw = torch.tensor(yaw, requires_grad=True)
roll = torch.tensor(roll, requires_grad=True)
R = get_rotation_matrix(pitch, yaw, roll)

print("Pitch:", pitch)
print("Yaw:", yaw)
print("Roll:", roll)

