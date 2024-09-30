# 9/29 16:42 在这个版本上进行调整，接下来增加法线约束
# 9/29 20:54 本质上法线并不属于GT，因此约束没什么意义

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

def parse_tuple(s):
    try:
        return tuple(map(int, s.strip("()").split(',')))
    except ValueError:
        raise argparse.ArgumentTypeError("Tuple must be in the form (x,y,z)")

# 自定义解析函数
def parse_nested_list(string):
    # 使用 ast.literal_eval 安全地解析字符串为列表
    return ast.literal_eval(string)

def load_obj_mesh(obj_path,device):
    verts, faces_idx, _ = load_obj(obj_path,device=device)
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

        return colors,pixel_normals # rgb？yes


def create_renderer(renderer_type,image_size,blend_params,cameras,device,light_position=None):
    if renderer_type == 'SoftSilhouetteShader':
        Silhouette_raster_settings = RasterizationSettings(
            image_size=image_size,                                      # 输出图像的大小(height,width)
            blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,    # 模糊半径，基于混合参数计算
            faces_per_pixel=100,                                        # 每个像素的面数，用于混合和抗锯齿
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
            bin_size= 0
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
            bin_size= 0
        )

        normal_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=normal_raster_settings  # 使用新的栅格化设置
            ),
            shader=NormalShader(device=device)  # 使用自定义法线着色器
        )

        return normal_renderer


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

    # If the inputs were scalars, remove the extra dimension
    if R.shape[0] == 1:
        R = R.squeeze(0)

    return R



class Model(nn.Module):
    def __init__(self, meshes, img_renderer,normal_renderer, image_ref,normal_ref,camera_location,camera_rotation):
        super().__init__()
        self.meshes = meshes
        self.device = meshes.device
        self.img_renderer = img_renderer
        self.normal_renderer = normal_renderer

        image_ref = torch.from_numpy(image_ref)
        self.register_buffer('image_ref', image_ref)

        normal_ref = torch.from_numpy(normal_ref)
        self.register_buffer('normal_ref', normal_ref)


        self.camera_rotation_angle = nn.Parameter(
            torch.from_numpy(np.array(camera_rotation, dtype=np.float32)).to(meshes.device)
        )

        self.camera_position = nn.Parameter(
            torch.from_numpy(np.array(camera_location, dtype=np.float32)).to(meshes.device))

        # self.loss_eval_1 = nn.CrossEntropyLoss()
        self.loss_eval_1 = nn.L1Loss()

        self.loss_eval_2 = nn.MSELoss()
        # self.loss_eval_2 = nn.L1Loss()


    def forward(self):
        R = get_rotation_matrix(self.camera_rotation_angle[0],self.camera_rotation_angle[1],self.camera_rotation_angle[2]).unsqueeze(0).to(self.device)
        T = -torch.bmm(R.transpose(1, 2), self.camera_position[None, :, None])[:, :, 0]  # (1, 3)

        image = self.img_renderer(meshes_world=self.meshes.clone(), R=R, T=T)
        # 此处不能执行0-1二值处理，导致梯度无法回传
        alpha_mask = image[0, ..., 3]


        # _,pixel_normals = self.normal_renderer(meshes_world=self.meshes.clone(), R=R, T=T) # bgr [0,1] (1, H, W, 3)
        # pred_normals = pixel_normals[0]
        pred_normals = torch.zeros_like(self.normal_ref)
        # color_normal = color_normal[0] # for visual

        img_loss = self.loss_eval_1(alpha_mask, self.image_ref)
        # normal_loss = self.loss_eval_2(pred_normals, self.normal_ref)
        # loss = img_loss + normal_loss
        loss = img_loss # 对chicken只使用mask损失看看效果（因为Mask和物体完全一致）

        return loss, alpha_mask,pred_normals

def main():
    parser = argparse.ArgumentParser(description="Process some parameters for the rendering program.")
    # parser.add_argument('--obj_path', type=str, default='./data/pineapple_center.obj', help='Path to the .obj file')
    # parser.add_argument('--obj_path', type=str, default='./data/rooster_center.obj', help='Path to the .obj file')
    parser.add_argument('--obj_path', type=str, default='./data/rooster_rate.obj', help='Path to the .obj file')
    parser.add_argument('--mask_path', type=str, default='./data/mask.png',help='Path to the reference mask image')
    parser.add_argument('--normal_path', type=str, default='./data/normal.png', help='Path to the reference normal image')

    parser.add_argument('--camera_location', type=parse_tuple, default=(0,0,200), help='Initial camera location')
    parser.add_argument('--camera_rotation', type=parse_nested_list, default=[0, np.pi, 0], help='Initial camera rotation')

    parser.add_argument('--loop_num', type=int, default=5000, help='Number of iterations')

    # 约束的时候如果使用原始分辨率大小需要的时间非常长
    parser.add_argument('--image_size', type=parse_tuple, default=(128, 153), help='Size of the output image')
    # parser.add_argument('--image_size', type=parse_tuple, default=(256, 306), help='Size of the output image')
    # parser.add_argument('--image_size', type=parse_tuple, default=(1024, 1224), help='Size of the output image')
    parser.add_argument('--final_image_size', type=parse_tuple, default=(1024, 1224), help='Size of the output normal image')

    args = parser.parse_args()

    # 设置CUDA设备
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    obj_path = args.obj_path
    mask_path = args.mask_path
    normal_path = args.normal_path

    camera_location = args.camera_location
    camera_rotation = args.camera_rotation
    loop_num = args.loop_num
    image_size = args.image_size
    final_image_size = args.final_image_size

    obj_mesh = load_obj_mesh(obj_path,device=device)
    # cameras = FoVPerspectiveCameras(znear=0.1, zfar=10000, aspect_ratio=1, fov=30, degrees=True, device=device)
    cameras = FoVPerspectiveCameras(znear=0.1, zfar=10000, aspect_ratio=1, fov=29.16, degrees=True, device=device)
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)  # 默认渲染参数

    silhouette_renderer = create_renderer('SoftSilhouetteShader',image_size=image_size,blend_params=blend_params,cameras=cameras,device=device)
    normal_renderer = create_renderer('NormalShader',image_size=image_size,blend_params=blend_params,cameras=cameras,device=device)

    if mask_path is not None:
        image_ref = load_mask(mask_path)
        # for accelate the optimization process
        image_ref_resize = cv2.resize(image_ref, (image_size[1], image_size[0]))
        # 0-1二值化处理
        image_ref_resize = (image_ref_resize > 0).astype(np.float32)
    if normal_path is not None:
        # normal_ref = cv2.imread(normal_path, -1)[..., ::-1] # bgr -> rgb
        normal_ref = cv2.imread(normal_path, -1)[..., ::-1] # bgr -> rgb
        if normal_ref.dtype == np.uint16:
            normal_ref = normal_ref.astype(np.float32) / 65535.0 * 2 - 1
        else:
            normal_ref = normal_ref.astype(np.float32) / 255.0 * 2 - 1
        #apply mask
        normal_ref = normal_ref * (image_ref[..., None] > 0)
        # resize to the same size
        normal_ref_resize = cv2.resize(normal_ref, (image_size[1], image_size[0]))
        # 归一化法线[-1,1]且长度为1
        epsilon = 1e-10
        normal_ref_resize = normal_ref_resize / (np.linalg.norm(normal_ref_resize, axis=2, keepdims=True) + epsilon)


    model = Model(meshes=obj_mesh, img_renderer=silhouette_renderer,normal_renderer=normal_renderer,image_ref=image_ref_resize,normal_ref=normal_ref_resize,camera_location=camera_location,camera_rotation=camera_rotation).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)  # 学习率衰减

    _, image_init,normal_init = model()



    loop = tqdm(range(loop_num))  # 设置循环
    for i in loop:
        optimizer.zero_grad()  # 清零优化器中的梯度
        loss, mask_pred,normal_pred = model()  # 获取当前相机位置下的损失
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
            plt.imshow(normal_pred.cpu().detach().numpy())
            plt.axis("off")
            plt.subplot(1, 4, 4)
            plt.imshow((model.normal_ref).cpu().detach().numpy())
            plt.axis("off")
            plt.title('loss: %.4f' % loss.data)
            # plt.savefig('camera_optimization_{:>3d}.png'.format(i))
            plt.show()
            # print(model.camera_rotation_angle, model.camera_position)


        # 如果损失函数的值小于阈值，则提前终止循环
        if loss.item() < 0.001:
            break

    # 提取最终优化过后的相机参数
    R = get_rotation_matrix(model.camera_rotation_angle[0], model.camera_rotation_angle[1],
                            model.camera_rotation_angle[2]).unsqueeze(0).to(model.device)
    T = -torch.bmm(R.transpose(1, 2), model.camera_position[None, :, None])[:, :, 0]  # (1, 3)
    print("对齐旋转矩阵：" + str(R))
    print("对齐平移向量：" + str(T))

    # 保存最终的渲染结果（渲染法线图）,以原始分辨率输出
    final_normal_renderer = create_renderer('NormalShader',image_size=final_image_size,blend_params=blend_params,cameras=cameras,device=device)
    normal_final,_ = final_normal_renderer(meshes_world=model.meshes.clone(), R=R, T=T)
    normal_final = normal_final.detach().squeeze().cpu().numpy()
    normal_final = (normal_final * 255).astype(np.uint8)
    # 切换通道顺序 brg -> rgb(本质上是因为cv2.imwrite默认bgr通道写入)
    normal_final = normal_final[..., ::-1]
    cv2.imwrite("final_normal.png", normal_final)


if __name__ == '__main__':
    main()


