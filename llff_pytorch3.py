# 这份代码主要用于实现相机坐标系的转换和相对RT的生成

import numpy as np
import trimesh
import cv2 as cv
import sys
import os
from glob import glob
import torch
import cv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 计算相对旋转矩阵和相对平移矩阵
def cal_rel_RT(R1, R2, T1, T2):
    # R1 (1,3,3) T1 (1,3)
    # convert tensor to numpy
    # R1 = R1.cpu().detach().numpy()
    # R2 = R2.cpu().detach().numpy()
    # T1 = T1.cpu().detach().numpy()
    # T2 = T2.cpu().detach().numpy()

    R1_inv = np.linalg.inv(R1)  # 计算R1的逆
    R_rel = R2 @ R1_inv  # 计算相对旋转矩阵
    T_rel = T2 - (R_rel @ T1)  # 计算相对平移向量

    # 将numpy转换回tensor
    # R_rel = torch.tensor(R_rel, dtype=torch.float32, device=device)
    # T_rel = torch.tensor(T_rel, dtype=torch.float32, device=device)

    return R_rel, T_rel  # (3,3) (3,)


# 根据相机1的R和T以及相对R和T可以直接计算得到相机2的R和T
# 需要将tensor转换为numpy再计算
def cal_R2_RT(R1, T1, R_rel, T_rel):
    # R1 = R1.cpu().detach().numpy()
    # T1 = T1.cpu().detach().numpy()
    # R_rel = R_rel.cpu().detach().numpy()
    # T_rel = T_rel.cpu().detach().numpy()

    R2_pre = R_rel @ R1
    T2_pre = R_rel @ T1 + T_rel
    # R2_pre = R2_pre[np.newaxis, ...]
    # T2_pre = T2_pre[np.newaxis, ...]  # 增加维度保证维度一致

    # 将numpy转换回tensor
    # R2_pre = torch.tensor(R2_pre, dtype=torch.float32, device=device)
    # T2_pre = torch.tensor(T2_pre, dtype=torch.float32, device=device)

    return R2_pre, T2_pre  # (1,3,3) (1,3)

def get_t_R_matrix(cam_id, cameras):
    world_mat = cameras[f'world_mat_{cam_id}']
    scale_mat = cameras[f'scale_mat_{cam_id}']

    P = world_mat @ scale_mat
    # P = world_mat   # 注释查看无缩放的效果
    P = P[:3, :4]  # 取投影矩阵的前三行和四列

    # 分解投影矩阵，获得旋转矩阵 R 和摄像机中心 c
    R = cv2.decomposeProjectionMatrix(P)[1]  # w2c下的旋转矩阵
    c = cv2.decomposeProjectionMatrix(P)[2]  # 摄像机中心
    c = (c[:3] / c[3])  # 将齐次坐标归一化

    t = -R @ c  # 计算平移向量 t（w2c 下的平移）

    return t, R

# 将旋转矩阵和平移向量构建成 W2C 矩阵
def construct_w2c_matrix(t, R):
    W2C = np.eye(4)
    W2C[:3, :3] = R  # 前 3x3 是旋转矩阵
    W2C[:3, 3] = t.reshape(-1)  # 第四列是平移向量
    return W2C


if __name__ == '__main__':
    # work_dir = sys.argv[1]
    # 原始的pose文件其实并不是很好，因为并没有执行归一化处理，可能导致算出来的相机中心点根本没对着物体
    work_dir = r'D:\Code\self_study\3D_rec\Pro_MVX\Calib_Code\Pytorch3D\Camera_align\data\black_dog\colmap'
    camera_file = os.path.join(work_dir, "preprocessed", 'cameras_sphere.npz')
    cameras = np.load(camera_file, allow_pickle=True)
    print(cameras.files)

    poses_hwf = np.load(os.path.join(work_dir, 'poses.npy')) # n_images, 3, 5
    poses_raw = poses_hwf[:, :, :4]
    hwf = poses_hwf[:, :, 4]
    pose = np.diag([1.0, 1.0, 1.0, 1.0])
    pose[:3, :4] = poses_raw[0]
    pts = []
    pts.append((pose @ np.array([0, 0, 0, 1])[:, None]).squeeze()[:3])
    pts.append((pose @ np.array([1, 0, 0, 1])[:, None]).squeeze()[:3])
    pts.append((pose @ np.array([0, 1, 0, 1])[:, None]).squeeze()[:3])
    pts.append((pose @ np.array([0, 0, 1, 1])[:, None]).squeeze()[:3])
    pts = np.stack(pts, axis=0)
    pcd = trimesh.PointCloud(pts)
    pcd.export(os.path.join(work_dir, 'pose.ply'))
    #

    cam_dict = dict()
    n_images = len(poses_raw)

    # Convert space（llff->colmap->pytorch3d）
    convert_mat = np.zeros([4, 4], dtype=np.float32)
    convert_mat[0, 1] = 1.0
    convert_mat[1, 0] = 1.0
    convert_mat[2, 2] =-1.0
    convert_mat[3, 3] = 1.0

    convert_mat_2 = np.zeros([4, 4], dtype=np.float32)
    convert_mat_2[0, 0] = -1.0
    convert_mat_2[1, 1] = -1.0
    convert_mat_2[2, 2] = 1.0
    convert_mat_2[3, 3] = 1.0

    convert_mat_3 = np.zeros([4, 4], dtype=np.float32)
    convert_mat_3[0, 1] = -1.0
    convert_mat_3[1, 0] = -1.0
    convert_mat_3[2, 2] = -1.0
    convert_mat_3[3, 3] =  1.0

    pose_list = []
    pose_colmap_list = []

    for i in range(n_images):
        pose = np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32)
        pose[:3, :4] = poses_raw[i]
        # pose = pose @ convert_mat  # llff->colmap
        # pose = pose @ convert_mat_2  # colmap->pytorch3d
        pose_colmap = pose

        pose = pose @ convert_mat_3  # llff->pytorch3d

        # now all pose is c2w
        # we don't need to convert pose to w2c,and the scale matrix is not needed
        pose_list.append(pose)
        pose_colmap_list.append(pose_colmap)
        # print the c2w matrix
        # print("c2w_{}:".format(i), pose)

    # extract R and T
    R_list = []
    T_list = []
    for i in range(n_images):
        R = pose_list[i][:3, :3]
        T = pose_list[i][:3, 3]
        R_list.append(R)
        T_list.append(T)
        # print("R_{}:".format(i), R)
        # print("T_{}:".format(i), T)

    # calculate relative R and T(under pytorch3d coordinate system,and c2w matrix)
    R_rel_list = []
    T_rel_list = []
    camera_1 = 0
    camera_2 = 5
    # 存储colmap的第一个相机pose为npy文件
    np.save( './pose_colmap.npy', pose_colmap_list[camera_1])


    camera_1_R = R_list[camera_1]
    camera_1_T = T_list[camera_1]
    print("camera_1_R:", camera_1_R)
    print("camera_1_T:", camera_1_T)

    camera_2_R = R_list[camera_2]
    camera_2_T = T_list[camera_2]
    print("camera_2_R:", camera_2_R)
    print("camera_2_T:", camera_2_T)

    R_rel, T_rel = cal_rel_RT(camera_1_R, camera_2_R, camera_1_T, camera_2_T)
    print("R_rel:", R_rel)
    print("T_rel:", T_rel)

    # cal R2 and T2
    camera_2_R_test,camera_2_T_test = cal_R2_RT(camera_1_R,camera_1_T,R_rel,T_rel)
    print("camera_2_R_test:", camera_2_R_test)
    print("camera_2_T_test:", camera_2_T_test)


























