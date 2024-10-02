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
    # 不能在这里使用detach，否则无法反向传播 -- 一定记住torch中不要随意使用detach!!
    # R1 = R1.cpu().detach().numpy()
    # T1 = T1.cpu().detach().numpy()
    # R_rel = R_rel.cpu().detach().numpy()
    # T_rel = T_rel.cpu().detach().numpy()


    # R2_pre = R_rel @ R1[0]
    # T2_pre = R_rel @ T1[0] + T_rel
    # R2_pre = R2_pre[np.newaxis, ...]
    # T2_pre = T2_pre[np.newaxis, ...]
    # # 将numpy转换回tensor
    # R2_pre = torch.tensor(R2_pre, dtype=torch.float32, device=device)
    # T2_pre = torch.tensor(T2_pre, dtype=torch.float32, device=device)

    R2_pre = torch.matmul(R_rel, R1[0])
    T2_pre = torch.matmul(R_rel, T1[0]) + T_rel
    R2_pre = R2_pre.unsqueeze(0)  # (1, 3, 3)
    T2_pre = T2_pre.unsqueeze(0)  # (1, 3)

    return R2_pre, T2_pre  # (1,3,3) (1,3)


def parse_tuple(s):
    try:
        return tuple(map(int, s.strip("()").split(',')))
    except ValueError:
        raise argparse.ArgumentTypeError("Tuple must be in the form (x,y,z)")

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

        return colors, pixel_normals


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

# 转换函数以实现R=3约束
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
    def __init__(self, meshes, img_renderer, normal_renderer, image_ref_list,camera_location, camera_rotation,R_rel_list,T_rel_list):
        """
        :param meshes:
        :param img_renderer:
        :param normal_renderer:
        :param image_ref_list:   参考图像mask列表
        :param camera_location:  初始相机平移向量
        :param camera_rotation:  初始相机旋转矩阵
        :param R_rel_list:  相对旋转矩阵列表
        :param T_rel_list:  相对平移向量列表
        """

        super().__init__()
        self.meshes = meshes
        self.device = meshes.device
        self.img_renderer = img_renderer
        self.normal_renderer = normal_renderer
        self.R_rel_list = R_rel_list
        self.T_rel_list = T_rel_list

        for i in range(len(image_ref_list)):
            image_ref_list[i] = torch.tensor(image_ref_list[i], dtype=torch.float32, device=self.device)
            self.register_buffer(f"image_ref_{i}", image_ref_list[i])

        for j in range(len(R_rel_list)):
            self.R_rel_list[j] = torch.tensor(R_rel_list[j], dtype=torch.float32, device=self.device)

        for k in range(len(T_rel_list)):
            self.T_rel_list[k] = torch.tensor(T_rel_list[k], dtype=torch.float32, device=self.device)

        # 此处的R是1*3旋转角
        self.camera_rotation = nn.Parameter(torch.from_numpy(np.array(camera_rotation, dtype=np.float32)).to(meshes.device))

        # 此处的T是1*3的平移向量W2C
        self.camera_position = nn.Parameter(torch.from_numpy(np.array(camera_location, dtype=np.float32)).to(meshes.device))

        # # 设置网格缩放比例为可学习参数
        # self.scale = nn.Parameter(torch.tensor(initial_scale, dtype=torch.float32))

        self.loss_eval_image = nn.MSELoss()

    def forward(self):
        # origin camera
        R = get_rotation_matrix(self.camera_rotation[0], self.camera_rotation[1], self.camera_rotation[2])
        T = self.camera_position
        R_row = R.permute(0, 2, 1)
        T = T.unsqueeze(0)

        # relative camera
        R_W2C_row_list = []
        T_W2C_list = []
        for i in range(len(self.R_rel_list)):
            R_W2C, T_W2C = cal_R2_RT(R, T, self.R_rel_list[i], self.T_rel_list[i])
            R_W2C_row = R_W2C.permute(0, 2, 1)
            R_W2C_row_list.append(R_W2C_row)
            T_W2C_list.append(T_W2C)

        # # scale the mesh
        # verts = self.meshes.verts_packed()
        # scaled_verts = verts * self.scale
        # scaled_mesh = Meshes(verts=[scaled_verts], faces=self.meshes.faces_list())
        #
        # # print
        # print("scaled factor", self.scale.item())
        # print("camera rotation", self.camera_rotation)
        # print("camera position", self.camera_position)

        # render the image for the origin camera and relative cameras
        image_1 = self.img_renderer(meshes_world=self.meshes.clone(), R=R_row, T=T)
        alpha_1_mask = image_1[0, ..., 3]

        alpha_mask_list = []
        alpha_mask_list.append(alpha_1_mask)

        for i in range(len(R_W2C_row_list)):
            rel_image = self.img_renderer(meshes_world=self.meshes.clone(), R=R_W2C_row_list[i], T=T_W2C_list[i])
            alpha_mask = rel_image[0, ..., 3]
            alpha_mask_list.append(alpha_mask)

        # calculate the loss
        image_loss = 0
        # loss_list = []
        for i in range(len(alpha_mask_list)):

            loss_i = self.loss_eval_image(alpha_mask_list[i], getattr(self, f"image_ref_{i}"))
            image_loss += loss_i
            # loss_list.append(loss_i)

        # print("loss is %f, %f, %f" % (loss_list[0].item(), loss_list[1].item(), loss_list[2].item()))

        # image_loss = self.loss_eval_image(alpha_mask_list[1], getattr(self, f"image_ref_{1}"))

        # [print(loss_list[i].item()) for i in range(len(loss_list))]

        return image_loss, alpha_mask_list


def main():
    parser = argparse.ArgumentParser(description="Process some parameters for the rendering program.")
    # parser.add_argument('--obj_path', type=str, default='./data/dog-delaunay_clean.obj', help='Path to the .obj file')
    # parser.add_argument('--obj_path', type=str, default='./data/dog_rotate_clean.obj', help='Path to the .obj file')
    parser.add_argument('--obj_path', type=str, default='./data/dog_rotate_2.obj', help='Path to the .obj file')
    # parser.add_argument('--obj_path', type=str, default='./data/blackdog_center.obj', help='Path to the .obj file')

    parser.add_argument('--mask_paths', type=str, nargs='+', default=['./data/mask_00.png', './data/mask_05.png','./data/mask_10.png',
                                                                      './data/mask_15.png','./data/mask_20.png'], help='Paths to the reference mask images')


    parser.add_argument('--camera_rotation', type=parse_nested_list, default=([[0.2654, -0.4987, 0.8251],
                                                                               [0.5792, -0.6016, -0.5500],
                                                                               [0.7707, 0.6239, 0.1292]]),help='Initial camera rotation(w2c)')
    parser.add_argument('--camera_location', type=parse_tuple, default=[-0.0654,  2.6518,  3.5998], help='Initial camera location')



    parser.add_argument('--R_rel', type=parse_nested_list,nargs='+', default=([[[-0.0584,  0.5365,  0.8419],
                                                                                     [-0.5677,  0.6758, -0.4700],
                                                                                     [-0.8211, -0.5054,  0.2651]],
                                                                               [[-0.9746, 0.0975, 0.2015],
                                                                                [-0.1475, 0.3971, -0.9058],
                                                                                [-0.1683, -0.9126, -0.3727]],
                                                                               [[-0.5112, -0.4939, -0.7034],
                                                                                [0.4541, 0.5397, -0.7089],
                                                                                [0.7297, -0.6818, -0.0516]],
                                                                               [[0.5729, -0.4576, -0.6800],
                                                                                [0.4492, 0.8692, -0.2065],
                                                                                [0.6856, -0.1872, 0.7035]]
                                                                               ]), help='Relative rotation matrices')
    parser.add_argument('--T_rel', type=parse_nested_list,nargs='+', default=([[-3.6398,  2.2759,  3.5975],
                                                                               [-0.5309, 4.2041, 6.4695],
                                                                               [4.0681, 3.2073, 4.8001],
                                                                               [2.9877, 0.8685, 1.3180]
                                                                               ]), help='Relative translation vectors')

    parser.add_argument('--loop_num', type=int, default=5000, help='Number of iterations')
    parser.add_argument('--image_size', type=parse_tuple, default=(128, 153), help='Size of the output image')
    parser.add_argument('--final_image_size', type=parse_tuple, default=(1024, 1224),help='Size of the output normal image')
    # parser.add_argument('--initial_scale', type=float, default=1/70, help='Initial scale of the mesh')

    args = parser.parse_args()

    obj_path = args.obj_path
    mask_paths_list = args.mask_paths

    camera_location = args.camera_location
    camera_rotation = args.camera_rotation
    camera_rotation_angle = extract_euler_angles(torch.tensor(camera_rotation, dtype=torch.float32))


    loop_num = args.loop_num
    image_size = args.image_size
    final_image_size = args.final_image_size

    R_rel_list = args.R_rel
    T_rel_list = args.T_rel

    # initial_scale = args.initial_scale


############################################################################################################
    obj_mesh = load_obj_mesh(obj_path, device=device)
    cameras = PerspectiveCameras(focal_length=((291.76901860550583, 291.76901860550583),),principal_point=((76.6875, 64.0625),), in_ndc=False, image_size=(image_size,),device=device)
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)  # 默认渲染参数
    silhouette_renderer = create_renderer('SoftSilhouetteShader', image_size=image_size, blend_params=blend_params,cameras=cameras, device=device)
    normal_renderer = create_renderer('NormalShader', image_size=image_size, blend_params=blend_params, cameras=cameras,device=device)

    if len(mask_paths_list) == 0:
        raise ValueError("The mask_paths_list is empty.")
    else:
        mask_list = []
        for mask_path in mask_paths_list:
            mask = load_mask(mask_path)
            mask_resized = cv2.resize(mask, (image_size[1], image_size[0]))
            mask_resized = (mask_resized>0).astype(np.float32)
            mask_list.append(mask_resized)

############################################################################################################
    model = Model(obj_mesh, silhouette_renderer, normal_renderer, mask_list, camera_location, camera_rotation_angle, R_rel_list, T_rel_list)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.2)

    if not os.path.exists('vis'):
        os.makedirs('vis')

    loop = tqdm(range(loop_num))
    for i in loop:
        optimizer.zero_grad()
        loss,mask_pred_list = model()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        loop.set_description(f"Loss: {loss.item()}")

        if i % 200 == 0:
            subplot_n = len(mask_pred_list)*2
            for j in range(len(mask_pred_list)):
                plt.subplot(1, subplot_n, 1+j*2)
                plt.imshow(mask_pred_list[j].cpu().detach().numpy())
                plt.axis('off')
                plt.subplot(1, subplot_n, 2+j*2)
                plt.imshow(getattr(model, f"image_ref_{j}").cpu().detach().numpy())
                plt.axis('off')
            plt.title('loss: %.4f' % loss.data)
            plt.tight_layout()
            plt.savefig("vis/%d.png" % i)
            plt.show()


        if loss.item() < 0.001:
            break

    R = get_rotation_matrix(model.camera_rotation[0], model.camera_rotation[1], model.camera_rotation[2])
    T = model.camera_position
    R_row = R.permute(0, 2, 1)
    T = T.unsqueeze(0)
    final_normal_renderer = create_renderer('NormalShader', image_size=final_image_size, blend_params=blend_params,cameras=cameras, device=device)

    normal_final, _ = final_normal_renderer(meshes_world=model.meshes.clone(), R=R_row, T=T)
    normal_final = normal_final.detach().squeeze().cpu().numpy()
    normal_final = (normal_final * 255).astype(np.uint8)
    normal_final = normal_final[..., ::-1]
    cv2.imwrite("final_normal.png", normal_final)

if __name__ == '__main__':
    main()




