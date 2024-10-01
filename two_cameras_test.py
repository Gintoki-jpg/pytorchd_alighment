import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
import cv2
import ast
from pytorch3d.renderer.mesh.shader import ShaderBase
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# load_obj从文件中加载3D对象（.obj格式）
from pytorch3d.io import load_obj

# Meshes用于存储和操作3D网格数据结构
from pytorch3d.structures import Meshes


from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
    PerspectiveCameras,FoVOrthographicCameras
)

device = torch.device("cuda:0")


def cal_R2_RT(R1, T1, R_rel, T_rel):
    R1 = R1.cpu().detach().numpy()
    T1 = T1.cpu().detach().numpy()
    R_rel = R_rel.cpu().detach().numpy()
    T_rel = T_rel.cpu().detach().numpy()
    R2_pre = R_rel @ R1[0]
    T2_pre = R_rel @ T1[0] + T_rel
    R2_pre = R2_pre[np.newaxis, ...]
    T2_pre = T2_pre[np.newaxis, ...]
    # 将numpy转换回tensor
    R2_pre = torch.tensor(R2_pre, dtype=torch.float32, device=device)
    T2_pre = torch.tensor(T2_pre, dtype=torch.float32, device=device)
    return R2_pre, T2_pre  # (1,3,3) (1,3)


def parse_tuple(s):
    try:
        return tuple(map(int, s.strip("()").split(',')))
    except ValueError:
        raise argparse.ArgumentTypeError("Tuple must be in the form (x,y,z)")



# 自定义解析函数
def parse_nested_list(string):
    # 使用 ast.literal_eval 安全地解析字符串为列表
    return ast.literal_eval(string)


def load_obj_mesh(obj_path, device):
    verts, faces_idx, _ = load_obj(obj_path, device=device)
    faces = faces_idx.verts_idx

    # # 计算边界框以平移顶点，使边界框中心位于世界坐标系原点
    # min_xyz = verts.min(dim=0)[0]
    # max_xyz = verts.max(dim=0)[0]
    # center = (min_xyz + max_xyz) / 2.0
    # verts_translated = verts - center

    # 已经在blender中执行了平移操作，因此不需要再次平移
    verts_translated = verts

    # 初始化顶点颜色为白色
    verts_rgb = torch.ones_like(verts_translated)[None]  # (1, V, 3)
    # TexturesVertex使用verts_rgb创建顶点纹理，并将其移动到指定设备
    textures = TexturesVertex(verts_features=verts_rgb.to(device))

    # 创建3D网格对象
    obj_mesh = Meshes(
     verts=[verts_translated.to(device)],
     faces=[faces.to(device)],
     textures=textures
    )

    # bounding_boxes = obj_mesh.get_bounding_boxes()
    # print(bounding_boxes)

    # 返回3D网格对象
    return obj_mesh


def load_mask(mask_path):
    image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    image = (image > 128).astype(np.float32)
    return image


class NormalShader(ShaderBase):
    def forward(self, fragments, meshes, **kwargs):
        # Get per-face normals
        faces = meshes.faces_packed()
        verts = meshes.verts_packed()
        vertex_normals = meshes.verts_normals_packed()
        faces_normals = vertex_normals[faces]  # (F, 3, 3)

        # Interpolate normals for each pixel
        pix_to_face = fragments.pix_to_face  # (N, H, W, K)
        bary_coords = fragments.bary_coords  # (N, H, W, K, 3)
        N, H, W, K = pix_to_face.shape

        # Initialize pixel normals
        pixel_normals = torch.zeros((N, H, W, 3), device=device)

        for i in range(K):
            # Get the face index
            face_idx = pix_to_face[..., i]

            # Get valid mask
            valid_mask = face_idx >= 0

            # Get barycentric coordinates
            w = bary_coords[..., i, :]  # (N, H, W, 3)

            # Get normals for the corresponding faces
            face_normals = faces_normals[face_idx[valid_mask]]  # (P, 3, 3)

            # Interpolate normals
            normals = (face_normals * w[valid_mask].unsqueeze(-1)).sum(dim=1)

            # Assign to pixel normals
            pixel_normals[valid_mask] = normals

        # Normalize and map normals to [0, 1] for visualization
        colors = (pixel_normals + 1) / 2

        return colors, pixel_normals  # rgb？yes


def create_renderer(renderer_type, image_size, blend_params, cameras, device, light_position=None):
    if renderer_type == 'SoftSilhouetteShader':
        Silhouette_raster_settings = RasterizationSettings(
         image_size=image_size,  # 输出图像的大小(height,width)
         blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,  # 模糊半径，基于混合参数计算
         faces_per_pixel=100,  # 每个像素的面数，用于混合和抗锯齿
         bin_size=0
        )

        silhouette_renderer = MeshRenderer(
         rasterizer=MeshRasterizer(  # 将网格转换为屏幕上的像素
          cameras=cameras,  # 使用之前创建的透视相机
          raster_settings=Silhouette_raster_settings  # 使用配置好的栅格化设置
         ),
         shader=SoftSilhouetteShader(blend_params=blend_params)  # 使用软轮廓着色器，根据混合参数设置渲染图像
        )
        return silhouette_renderer

    elif renderer_type == 'HardPhongShader':
        if light_position is None:
            raise ValueError("light_position is required for HardPhongShader")
        else:
            lights = PointLights(device=device, location=(light_position,))

        Phong_raster_settings = RasterizationSettings(
         image_size=image_size,  # 图像大小
         blur_radius=0.0,  # 模糊半径为0
         faces_per_pixel=1,  # 每个像素仅渲染一个面
         bin_size=0
        )

        hard_phong_renderer = MeshRenderer(
         rasterizer=MeshRasterizer(
          cameras=cameras,
          raster_settings=Phong_raster_settings  # 使用新的栅格化设置
         ),
         shader=HardPhongShader(device=device, cameras=cameras, lights=lights)  # 使用硬Phong着色器，结合相机和光源进行渲染
        )
        return hard_phong_renderer
    elif renderer_type == 'NormalShader':
        normal_raster_settings = RasterizationSettings(
          image_size=image_size,  # 图像大小
          blur_radius=0.0,  # 模糊半径为0
          faces_per_pixel=1,  # 每个像素仅渲染一个面
          bin_size=0
         )

        normal_renderer = MeshRenderer(
         rasterizer=MeshRasterizer(
          cameras=cameras,
          raster_settings=normal_raster_settings  # 使用新的栅格化设置
         ),
         shader=NormalShader(device=device)  # 使用自定义法线着色器
        )

        return normal_renderer

# 目前还没有一个对应的转换函数，否则可以实现R=3约束
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

    # Combine the rotation matrices C2W
    R = torch.matmul(R_z, torch.matmul(R_y, R_x))

    # # If the inputs were scalars, remove the extra dimension
    # if R.shape[0] == 1:
    #     R = R.squeeze(0)

    return R


def extract_euler_angles(R):
    # 确保 R 是一个 3x3 矩阵
    assert R.shape == (3, 3)

    # 提取旋转矩阵的元素
    r11, r12, r13 = R[0, :]
    r21, r22, r23 = R[1, :]
    r31, r32, r33 = R[2, :]

    # # 计算 yaw, pitch, roll
    # yaw = np.arctan2(R[1, 0], R[0, 0])
    # pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
    # roll = np.arctan2(R[2, 1], R[2, 2])

    yaw = torch.asin(-r31)

    # Calculate pitch
    pitch = torch.atan2(r32, r33)

    # Calculate roll
    roll = torch.atan2(r21, r11)

    return pitch, yaw, roll


class Model(nn.Module):
    def __init__(self, meshes, img_renderer, normal_renderer, image_ref, image_ref_2,camera_location, camera_rotation,R_rel,T_rel):
        super().__init__()
        self.meshes = meshes
        self.device = meshes.device
        self.img_renderer = img_renderer
        self.normal_renderer = normal_renderer

        image_ref = torch.from_numpy(image_ref)
        self.register_buffer('image_ref', image_ref)

        # normal_ref = torch.from_numpy(normal_ref)
        # self.register_buffer('normal_ref', normal_ref)
        image_ref_2 = torch.from_numpy(image_ref_2)
        self.register_buffer('image_ref_2', image_ref_2)

        self.R_rel = torch.from_numpy(np.array(R_rel, dtype=np.float32)).to(meshes.device)
        self.T_rel = torch.from_numpy(np.array(T_rel, dtype=np.float32)).to(meshes.device)

        # 此处的R是1*3旋转角
        self.camera_rotation = nn.Parameter(
         torch.from_numpy(np.array(camera_rotation, dtype=np.float32)).to(meshes.device)
        )

        # 此处的T是1*3的平移向量W2C
        self.camera_position = nn.Parameter(
         torch.from_numpy(np.array(camera_location, dtype=np.float32)).to(meshes.device))

        # self.loss_eval_1 = nn.CrossEntropyLoss()
        self.loss_eval_1 = nn.L1Loss()

        # self.loss_eval_2 = nn.MSELoss()
        # self.loss_eval_2 = nn.L1Loss()

    def forward(self):
        # 此处的R和T都应该是W2C的，之前原始的look_at_rotation函数返回的也是R的转置，即W2C下的R -- 3D规定了必须使用W2C下的执行渲染
        # R = self.camera_rotation
        R = get_rotation_matrix(self.camera_rotation[0], self.camera_rotation[1], self.camera_rotation[2]) # w2c
        T = self.camera_position

        # R = R.unsqueeze(0)
        T = T.unsqueeze(0)

        R_2_W2C, T_2_W2C = cal_R2_RT(R, T, self.R_rel, self.T_rel)

        # Render the image(R and T are all w2c)
        image = self.img_renderer(meshes_world=self.meshes.clone(), R=R, T=T)
        # 此处不能执行0-1二值处理，导致梯度无法回传
        alpha_mask = image[0, ..., 3]

        image_2 = self.img_renderer(meshes_world=self.meshes.clone(), R=R_2_W2C, T=T_2_W2C)
        alpha_mask_2 = image_2[0, ..., 3]

        # _,pixel_normals = self.normal_renderer(meshes_world=self.meshes.clone(), R=R, T=T) # bgr [0,1] (1, H, W, 3)
        # pred_normals = pixel_normals[0]
        # pred_normals = torch.zeros_like(self.normal_ref)
        # color_normal = color_normal[0] # for visual

        img_loss_1 = self.loss_eval_1(alpha_mask, self.image_ref)
        img_loss_2 = self.loss_eval_1(alpha_mask_2, self.image_ref_2)
        # normal_loss = self.loss_eval_2(pred_normals, self.normal_ref)
        # loss = img_loss + normal_loss
        loss = img_loss_1+img_loss_2  # 对chicken只使用mask损失看看效果（因为Mask和物体完全一致）

        return loss, alpha_mask,alpha_mask_2

def main():
    """如何获取camera_location，camera_rotation，R_rel，T_rel

        执行 python render_colmap_mesh.py 可以得到如下输出(all is w2c)
        pytorch3d W2C R_0 is tensor([[[ 0.2654,  0.5792,  0.7707],
             [-0.4987, -0.6016,  0.6239],
             [ 0.8251, -0.5500,  0.1292]]])
        pytorch3d W2C T_0 is tensor([[-0.0654,  2.6518,  3.5998]])
        pytorch3d W2C R_5 is tensor([[[ 0.9441, -0.1215, -0.3064],
                 [ 0.2317, -0.4167,  0.8790],
                 [-0.2345, -0.9009, -0.3653]]])
        pytorch3d W2C T_5 is tensor([[0.8173, 2.4132, 3.2653]])
        R_rel tensor([[[-0.0559, -0.5889,  0.8062],
                 [ 0.4976,  0.6836,  0.5339],
                 [-0.8656,  0.4310,  0.2548]]], device='cuda:0')
        T_rel tensor([[-0.5269, -1.2890,  1.1484]], device='cuda:0')
        R2 tensor([[[[ 0.9441, -0.1215, -0.3064],
                  [ 0.2317, -0.4167,  0.8790],
                  [-0.2345, -0.9009, -0.3653]]]], device='cuda:0')
        T2 tensor([[[0.8173, 2.4132, 3.2653]]], device='cuda:0')

    其中的R_0，T_0对应camera_location，camera_rotation，R_rel和T_rel分别对应R_rel和T_rel

    """

    parser = argparse.ArgumentParser(description="Process some parameters for the rendering program.")
    # parser.add_argument('--obj_path', type=str, default='./data/blackdog_center.obj', help='Path to the .obj file')
    parser.add_argument('--obj_path', type=str, default='./data/dog_rotate_clean.obj', help='Path to the .obj file')
    # parser.add_argument('--mask_path', type=str, default='./data/mask_00.png', help='Path to the reference mask image')
    parser.add_argument('--mask_path', type=str, default='./data/mask_00.png', help='Path to the reference mask image')
    parser.add_argument('--mask_path_2', type=str, default='./data/mask_05.png', help='Path to the reference mask image')
    # parser.add_argument('--normal_path', type=str, default='./data/normal_00.png',help='Path to the reference normal image')
    # parser.add_argument('--normal_path', type=str, default='./data/normal_00.png', help='Path to the reference normal image')

    # 假设输入的R和T都是pytorch3d W2C下直接计算得到的3*3旋转矩阵和1*3平移向量
    parser.add_argument('--camera_location', type=parse_tuple, default=([-0.0654,  2.6518,  3.5998]), help='Initial camera location(w2c)')
    parser.add_argument('--camera_rotation', type=parse_nested_list, default=([[0.2654, 0.5792, 0.7707],
                                                                                         [-0.4987, -0.6016, 0.6239],
                                                                                         [0.8251, -0.5500, 0.1292]]), help='Initial camera rotation(w2c)')

    # 加入pytorch3d C2W下的相对R和相对T，基于COLMAP的尺度计算得到
    parser.add_argument('--R_rel', type=parse_nested_list, default=([[-0.0559, -0.5889,  0.8062],
                                                                                 [ 0.4976,  0.6836,  0.5339],
                                                                                 [-0.8656,  0.4310,  0.2548]]), help='Relative rotation matrix')
    parser.add_argument('--T_rel', type=parse_tuple, default=([-0.5269, -1.2890,  1.1484]), help='Relative translation vector')


    parser.add_argument('--loop_num', type=int, default=5000, help='Number of iterations')

    # 约束的时候如果使用原始分辨率大小需要的时间非常长
    parser.add_argument('--image_size', type=parse_tuple, default=(128, 153), help='Size of the output image')
    # parser.add_argument('--image_size', type=parse_tuple, default=(256, 306), help='Size of the output image')
    # parser.add_argument('--image_size', type=parse_tuple, default=(1024, 1224), help='Size of the output image')
    parser.add_argument('--final_image_size', type=parse_tuple, default=(1024, 1224),
                        help='Size of the output normal image')

    args = parser.parse_args()



    # 设置CUDA设备
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    obj_path = args.obj_path
    mask_path = args.mask_path
    mask_path_2 = args.mask_path_2
    # normal_path = args.normal_path

    camera_location = args.camera_location
    camera_rotation = args.camera_rotation
    camera_rotation_angle = extract_euler_angles(torch.tensor(camera_rotation, dtype=torch.float32))

    loop_num = args.loop_num
    image_size = args.image_size
    final_image_size = args.final_image_size
    R_rel = args.R_rel
    T_rel = args.T_rel

    obj_mesh = load_obj_mesh(obj_path, device=device)
    # cameras = FoVPerspectiveCameras(znear=0.1, zfar=10000, aspect_ratio=1, fov=30, degrees=True, device=device)
    cameras = FoVPerspectiveCameras(znear=0.1, zfar=10000, aspect_ratio=1, fov=29.16, degrees=True, device=device)
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)  # 默认渲染参数

    silhouette_renderer = create_renderer('SoftSilhouetteShader', image_size=image_size, blend_params=blend_params,
                                          cameras=cameras, device=device)
    normal_renderer = create_renderer('NormalShader', image_size=image_size, blend_params=blend_params, cameras=cameras,
                                      device=device)

    if mask_path and mask_path_2 is not None:
        image_ref_1 = load_mask(mask_path)
        # for accelate the optimization process
        image_ref_resize_1 = cv2.resize(image_ref_1, (image_size[1], image_size[0]))
        # 0-1二值化处理
        image_ref_resize_1 = (image_ref_resize_1 > 0).astype(np.float32)
        image_ref_2 = load_mask(mask_path_2)
        image_ref_resize_2 = cv2.resize(image_ref_2, (image_size[1], image_size[0]))
        image_ref_resize_2 = (image_ref_resize_2 > 0).astype(np.float32)

    # if normal_path is not None:
    #     # normal_ref = cv2.imread(normal_path, -1)[..., ::-1] # bgr -> rgb
    #     normal_ref = cv2.imread(normal_path, -1)[..., ::-1]  # bgr -> rgb
    #     if normal_ref.dtype == np.uint16:
    #         normal_ref = normal_ref.astype(np.float32) / 65535.0 * 2 - 1
    #     else:
    #         normal_ref = normal_ref.astype(np.float32) / 255.0 * 2 - 1
    #     # apply mask
    #     normal_ref = normal_ref * (image_ref[..., None] > 0)
    #     # resize to the same size
    #     normal_ref_resize = cv2.resize(normal_ref, (image_size[1], image_size[0]))
    #     # 归一化法线[-1,1]且长度为1
    #     epsilon = 1e-10
    #     normal_ref_resize = normal_ref_resize / (np.linalg.norm(normal_ref_resize, axis=2, keepdims=True) + epsilon)

    # model = Model(meshes=obj_mesh, img_renderer=silhouette_renderer, normal_renderer=normal_renderer,
    #               image_ref=image_ref_resize_1, image_ref_2=image_ref_resize_2, camera_location=camera_location,
    #               camera_rotation=camera_rotation,R_rel=R_rel,T_rel=T_rel).to(device)
    model = Model(meshes=obj_mesh, img_renderer=silhouette_renderer, normal_renderer=normal_renderer,
                  image_ref=image_ref_resize_1, image_ref_2=image_ref_resize_2, camera_location=camera_location,
                  camera_rotation=camera_rotation_angle,R_rel=R_rel,T_rel=T_rel).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.2)  # 学习率衰减

    _, image_init,image_init_2= model()
    # 展示图片
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(image_init.cpu().detach().numpy())
    plt.title('Initial image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(image_init_2.cpu().detach().numpy())
    plt.title('Initial image_2')
    plt.axis('off')
    plt.show()

    loop = tqdm(range(loop_num))  # 设置循环
    for i in loop:
        optimizer.zero_grad()  # 清零优化器中的梯度
        loss, mask_pred,mask_pred_2 = model()  # 获取当前相机位置下的损失
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数（即相机位置R和T）
        # 更新学习率
        scheduler.step()
        # 在进度条上显示当前损失值
        loop.set_description('Optimizing (loss %.4f)' % loss.data)

        if i % 200 == 0:
            # draw the alpha mask and self.image_ref_1
            plt.subplot(1, 4, 1)
            plt.imshow(mask_pred.cpu().detach().numpy())
            # plt.axis("off")
            plt.subplot(1, 4, 2)
            plt.imshow(model.image_ref.cpu().detach().numpy())
            plt.axis("off")
            plt.subplot(1, 4, 3)
            plt.imshow(mask_pred_2.cpu().detach().numpy())
            plt.axis("off")
            plt.subplot(1, 4, 4)
            plt.imshow((model.image_ref_2).cpu().detach().numpy())
            plt.axis("off")
            plt.title('loss: %.4f' % loss.data)
            # plt.savefig('camera_optimization_{:>3d}.png'.format(i))
            plt.show()
            # print(model.camera_rotation_angle, model.camera_position)

        # 如果损失函数的值小于阈值，则提前终止循环
        if loss.item() < 0.001:
            break


if __name__ == '__main__':
    main()



















