import os
import torch
import numpy as np
from tqdm import tqdm
import imageio
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
# from skimage import img_as_ubyte
from PIL import Image
import argparse
import cv2
import ast
import pytorch3d

# load_obj从文件中加载3D对象（.obj格式）
from pytorch3d.io import load_obj

# Meshes用于存储和操作3D网格数据结构
from pytorch3d.structures import Meshes

# Rotate进行3D旋转变换，Translate进行3D平移变换
from pytorch3d.transforms import Rotate, Translate

from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
    PerspectiveCameras,FoVOrthographicCameras
)

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
    # 归一化mask（大于128的地方均设置为1 -- 降采样的原因导致这个mask可能不是完全二值化）
    image = (image > 128).astype(np.float32)
    return image


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




class Model(nn.Module):
    def __init__(self, meshes, renderer, image_ref_1,camera_location,camera_rotation):
        super().__init__()
        self.meshes = meshes             # 存储3D网格对象
        self.device = meshes.device
        self.renderer = renderer         # 存储渲染器对象


        image_ref_1 = torch.from_numpy(image_ref_1)
        self.register_buffer('image_ref_1', image_ref_1)

        self.camera_rotation = nn.Parameter(
            torch.from_numpy(np.array(camera_rotation, dtype=np.float32)).to(meshes.device)
        )

        self.camera_position = nn.Parameter(
            torch.from_numpy(np.array(camera_location, dtype=np.float32)).to(meshes.device))

    def forward(self):
        # R = look_at_rotation(self.camera_position[None, :], device=self.device)  # (1, 3, 3)
        R = self.camera_rotation[None, :, :]
        T = -torch.bmm(R.transpose(1, 2), self.camera_position[None, :, None])[:, :, 0]  # (1, 3)


        image_1 = self.renderer(meshes_world=self.meshes.clone(), R=R, T=T)
        alpha_mask = image_1[..., 3]  # 这个alpha通道并不一定是0-1二值化


        loss_1 = torch.sum((alpha_mask - self.image_ref_1) ** 2)

        loss = loss_1

        return loss, image_1

def main():
    parser = argparse.ArgumentParser(description="Process some parameters for the rendering program.")
    # parser.add_argument('--obj_path', type=str, default='./data/rooster_center.obj', help='Path to the .obj file')
    parser.add_argument('--obj_path', type=str, default='./data/pineapple_center.obj', help='Path to the .obj file')
    parser.add_argument('--start_end_output', type=str, default='./visualization.png',
                        help='Output path for start and end comparison image')
    parser.add_argument('--filename_output', type=str, default='./camera_optimization_demo.gif',
                        help='Output path for the rendered GIF')
    # parser.add_argument('--mask_path_1', type=str, default='./data/mask_downsampled.png', help='Path to the first mask file')
    parser.add_argument('--mask_path_1', type=str, default='./data/mask.png',help='Path to the first mask file')
    # parser.add_argument('--mask_path_1', type=str, default='./data/rooster_mask.png', help='Path to the first mask file')
    parser.add_argument('--camera_location', type=parse_tuple, default=(0,0,200), help='Initial camera location')
    parser.add_argument('--camera_rotation', type=parse_nested_list, default=[[-1, 0, 0],[0,1,0],[0,0,-1]], help='Initial camera rotation')
    parser.add_argument('--loop_num', type=int, default=50, help='Number of iterations')  # 这个值尽量设置的大一点，模型迭代不动之后表示收敛
    parser.add_argument('--image_size', type=parse_tuple, default=(1024, 1224), help='Size of the output image')

    args = parser.parse_args()

    # 设置CUDA设备
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    obj_path = args.obj_path
    start_end_output = args.start_end_output
    filename_output = args.filename_output
    mask_path_1 = args.mask_path_1

    camera_location = args.camera_location
    camera_rotation = args.camera_rotation
    loop_num = args.loop_num
    image_size = args.image_size

    # 加载物体的3D网格数据(很明显这里并没有对obj进行缩放，这里的bounding box和blender中完全一样)
    obj_mesh = load_obj_mesh(obj_path,device=device)
    bounding_boxes = obj_mesh.get_bounding_boxes()
    # print(bounding_boxes)

    scale_x = abs(bounding_boxes[0][0][1] - bounding_boxes[0][0][0])
    scale_y = abs(bounding_boxes[0][1][1] - bounding_boxes[0][1][0])
    scale_z = abs(bounding_boxes[0][2][1] - bounding_boxes[0][2][0])
    # print(scale_x, scale_y, scale_z)



    light_position = (0, scale_z.item(), scale_z.item())             # 通过物体尺寸自动设置点光源位置
    cameras = FoVPerspectiveCameras(znear=0.1,zfar=10000,aspect_ratio=1,fov=30,degrees=True,device=device)      # 使用透视相机（遵循blender中的设定）
    # cameras = PerspectiveCameras(focal_length=(2354), principal_point=(612, 512),device=device,in_ndc=False,image_size=image_size)
    # cameras = FoVOrthographicCameras(device=device)

    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)  # 默认渲染参数

    silhouette_renderer = create_renderer('SoftSilhouetteShader',image_size=image_size,blend_params=blend_params,cameras=cameras,device=device)
    hard_phong_renderer = create_renderer('HardPhongShader',image_size=image_size,blend_params=blend_params,cameras=cameras,device=device,light_position=light_position)

    if mask_path_1 is not None:
        image_ref_1 = load_mask(mask_path_1)

    # 关掉writer，这对于目前的任务来说，可视化没什么意义
    # writer = imageio.get_writer(filename_output, mode='I', duration=1)
    model = Model(meshes=obj_mesh, renderer=silhouette_renderer, image_ref_1=image_ref_1,camera_location=camera_location,camera_rotation=camera_rotation).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    _, image_init = model()
    # plt.figure(figsize=(10, 10))
    # plt.subplot(1, 2, 1)
    # plt.imshow(image_init.detach().squeeze().cpu().numpy()[..., 3], cmap='gray')
    # plt.grid(False)
    # plt.title("Starting position")
    # plt.subplot(1, 2, 2)
    # plt.imshow(model.image_ref_1.cpu().numpy().squeeze(), cmap='gray')
    # plt.grid(False)
    # plt.title("Reference position")
    # plt.savefig(start_end_output)
    # 处理第一张图像（Starting position）
    # 从 tensor 转为 numpy 数组，并取第4通道
    image_init_np = image_init.detach().squeeze().cpu().numpy()[..., 3]
    # 确保图像是灰度图像（cv2 expects uint8 for grayscale images）
    image_init_np = (image_init_np * 255).astype(np.uint8)
    # 使用 cv2 保存第一张图像
    cv2.imwrite("starting_position.png", image_init_np)
    # 处理第二张图像（Reference position）
    image_ref_np = model.image_ref_1.cpu().numpy().squeeze()
    # 同样转换为 uint8 类型的灰度图像
    image_ref_np = (image_ref_np * 255).astype(np.uint8)
    # 使用 cv2 保存第二张图像
    cv2.imwrite("reference_position.png", image_ref_np)

    loop = tqdm(range(loop_num))  # 设置循环
    for i in loop:
        optimizer.zero_grad()  # 清零优化器中的梯度
        loss, _ = model()  # 获取当前相机位置下的损失
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数（即相机位置R和T）
        # 在进度条上显示当前损失值
        loop.set_description('Optimizing (loss %.4f)' % loss.data)
        # 如果损失函数的值小于200，则提前终止循环
        if loss.item() < 200:
            break

        # # 每10次迭代保存一次渲染结果
        # if i % 10 == 0:
        #     # 计算相机的旋转矩阵
        #     # R = look_at_rotation(model.camera_position[None, :], device=model.device)
        #     R = model.camera_rotation[None, :, :]
        #     # 计算相机的平移向量
        #     T = -torch.bmm(R.transpose(1, 2), model.camera_position[None, :, None])[:, :, 0]  # (1, 3)
        #     # 使用Phong渲染器渲染当前相机位置下的图像
        #     image = hard_phong_renderer(meshes_world=model.meshes.clone(), R=R, T=T)
        #     # image = silhouette_renderer(meshes_world=model.meshes.clone(), R=R, T=T)
        #     # 提取并转换图像数据
        #     image = image[0, ..., :3].detach().squeeze().cpu().numpy()
        #     # 将图像转换为8位无符号整数格式
        #     image = img_as_ubyte(image)
        #     # 将图像添加到GIF写入器
        #     writer.append_data(image)
        #
        #     plt.figure()
        #     plt.imshow(image[..., :3])
        #     plt.title("iter: %d, loss: %0.2f" % (i, loss.data))
        #     plt.axis("off")

    # writer.close()

    # 提取相机优化之后的参数
    # R = look_at_rotation(model.camera_position[None, :], device=model.device)  # (1, 3, 3)
    R = model.camera_rotation[None, :, :]
    T = -torch.bmm(R.transpose(1, 2), model.camera_position[None, :, None])[:, :, 0]  # (1, 3)

    print("对齐旋转矩阵：" + str(R))
    print("对齐平移向量：" + str(T))

    # 保存最终的渲染结果
    image_final = silhouette_renderer(meshes_world=model.meshes.clone(), R=R, T=T)
    image_final = image_final.detach().squeeze().cpu().numpy()[..., 3]
    image_final = (image_final * 255).astype(np.uint8)
    cv2.imwrite("final_position.png", image_final)


if __name__ == '__main__':
    main()


