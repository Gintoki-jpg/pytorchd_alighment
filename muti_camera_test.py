import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
import cv2
import ast
import re
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
    image = (image > 128).astype(np.float32)  # 二值化
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
            # blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,  # 模糊半径，基于混合参数计算
            # blur_radius=np.log(1. / 1e-3 - 1.) * blend_params.sigma,
            blur_radius=0.0001,  # 模糊半径
            faces_per_pixel=100,
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
            # blur_radius=0.0,  # 模糊半径为0
            blur_radius=0.0001,
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

# 转换函数以实现R自由度为3的约束
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

    return R


def extract_euler_angles(R):
    # 确保 R 是一个 3x3 矩阵
    assert R.shape == (3, 3)

    # 提取旋转矩阵的元素
    r11, r12, r13 = R[0, :]
    r21, r22, r23 = R[1, :]
    r31, r32, r33 = R[2, :]

    yaw = torch.asin(-r31)

    # Calculate pitch
    pitch = torch.atan2(r32, r33)

    # Calculate roll
    roll = torch.atan2(r21, r11)

    return pitch, yaw, roll

def extrac_R_T_from_txt(txt_path):
    with open(txt_path, 'r') as file:
        content = file.read()
    pattern = r'(\w+)\n([\[\]\-\d\.,\s\(\)]+)'  # 匹配变量名和数组值
    matches = re.findall(pattern, content)
    # 创建一个字典来存储所有变量
    variables = {}
    # 遍历所有匹配的变量名和值
    for name, value_str in matches:
        # 处理成 numpy 数组或普通 Python 列表
        try:
            value = eval(value_str)  # 使用 eval 直接转换字符串为数组或列表
        except:
            value = value_str.strip()  # 保留原始字符串
        variables[name] = np.array(value) if isinstance(value, list) else value
    # 提取各个变量
    R_col_0 = variables.get('R_col_0')
    T_col_0 = variables.get('T_col_0')
    fcl_screen_0 = variables.get('fcl_screen_0')
    prp_screen_0 = variables.get('prp_screen_0')
    return R_col_0, T_col_0, fcl_screen_0, prp_screen_0

def extract_rel_R_T_from_txt(txt_path):
    with open(txt_path, 'r') as file:
        content = file.read()
    pattern_R = r'R_rel_\d+\s(\[\[.*?\]\])'
    pattern_T = r'T_rel_\d+\s(\[.*?\])'
    matches_R = re.findall(pattern_R, content,re.DOTALL)
    matches_T = re.findall(pattern_T, content)

    R_rel_list = [np.array(eval(r)) for r in matches_R]
    T_rel_list = [np.array(eval(t)) for t in matches_T]
    return R_rel_list, T_rel_list

class Model(nn.Module):
    def __init__(self, meshes, img_renderer, normal_renderer, image_ref_list,camera_location, camera_rotation,R_rel_list,T_rel_list,initial_scale):
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
        self.camera_rotation = nn.Parameter(torch.from_numpy(np.array(camera_rotation, dtype=np.float32)).to(self.device))

        # 此处的T是1*3的平移向量W2C
        self.camera_position = nn.Parameter(torch.from_numpy(np.array(camera_location, dtype=np.float32)).to(self.device))

        # 设置网格缩放比例为可学习参数
        self.scale = nn.Parameter(torch.tensor(initial_scale, dtype=torch.float32, device=self.device))


        # self.loss_eval_image = nn.MSELoss()
        self.loss_eval_image = nn.BCELoss()

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


        # scale the mesh
        verts = self.meshes.verts_packed()
        scaled_verts = verts * self.scale
        scaled_mesh = Meshes(verts=[scaled_verts], faces=self.meshes.faces_list())

        # render normal for compare
        normal_image,_ = self.normal_renderer(meshes_world=scaled_mesh, R=R_row, T=T)

        # render the image for the origin camera and relative cameras
        image_1 = self.img_renderer(meshes_world=scaled_mesh, R=R_row, T=T)
        alpha_1_mask = image_1[0, ..., 3]
        # 此处无法执行二值化操作，会导致梯度消失
        alpha_mask_list = []
        alpha_mask_list.append(alpha_1_mask)

        for i in range(len(R_W2C_row_list)):
            rel_image = self.img_renderer(meshes_world=scaled_mesh, R=R_W2C_row_list[i], T=T_W2C_list[i])
            alpha_mask = rel_image[0, ..., 3]
            alpha_mask_list.append(alpha_mask)

        # calculate the loss
        image_loss = 0
        for i in range(len(alpha_mask_list)):
            loss_i = self.loss_eval_image(alpha_mask_list[i], getattr(self, f"image_ref_{i}"))
            image_loss += loss_i

        return image_loss, alpha_mask_list,normal_image

def save_normal(render_normal,R, save_path):
    # 世界法线图转换为相机法线图
    R_w2c = R[0]
    render_normal = render_normal[0]
    H, W, _ = render_normal.shape
    render_normal_flat = render_normal.view(-1, 3)
    rotated_normal_flat = torch.matmul(render_normal_flat, R_w2c.T)
    rotated_normal = rotated_normal_flat.view(H, W, 3)
    rotated_normal[:, :, 2] = -rotated_normal[:, :, 2]

    rotated_normal = (rotated_normal + 1) / 2 * 255
    rotated_normal_np = rotated_normal.detach().cpu().numpy()
    # cv2.imwrite(save_path, rotated_normal_np[..., ::-1])
    # cv2.imwrite(save_path, rotated_normal_np)
    # 交换通道顺序
    cv2.imwrite(save_path, rotated_normal_np[:, :, [2, 1, 0]])

def main():
    parser = argparse.ArgumentParser(description="Process some parameters for the rendering program.")


    parser.add_argument('--obj_path', type=str, default='./data/face/face_rotate.obj', help='Path to the .obj file')
    parser.add_argument('--camera_rotation_location_txt', type=str, default='./data/face/R_T_col_0.txt', help='Path to the origin camera rotation and location txt file')
    parser.add_argument('--relative_rotation_location_txt', type=str, default='./data/face/R_T_rel.txt', help='Path to the relative rotation and location txt file')
    parser.add_argument('--relative_cameras',type = parse_nested_list,nargs='+',default=[5,10,15,20],help='The camera number for the pickle file')
    parser.add_argument('--loop_num', type=int, default=6000, help='Number of iterations')
    parser.add_argument('--image_size', type=parse_tuple, default=(128, 153), help='Size of the train image')
    parser.add_argument('--final_image_size', type=parse_tuple, default=(1024, 1224),help='Size of the output image')
    parser.add_argument('--initial_scale', type=float, default=1.0, help='Initial scale of the mesh')
    parser.add_argument('--save_path', type=str, default='./normal_results', help='Path to save the output normal image')

    parser.add_argument('--mask_paths', type=str, nargs='+', default=['./data/face/mask_00.png', './data/face/mask_05.png',
                                                                       './data/face/mask_10.png',
                                                                      './data/face/mask_15.png',
                                                                      './data/face/mask_20.png'], help='Paths to the reference mask images')

    args = parser.parse_args()

    obj_path = args.obj_path
    mask_paths_list = args.mask_paths
    loop_num = args.loop_num
    train_image_size = args.image_size
    final_image_size = args.final_image_size
    relative_cameras = args.relative_cameras
    initial_scale = args.initial_scale
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    camera_rotaion_location_path = args.camera_rotation_location_txt
    relative_rotation_location_path = args.relative_rotation_location_txt

    camera_rotation, camera_location, fcl_screen_0, prp_screen_0 = extrac_R_T_from_txt(camera_rotaion_location_path)
    camera_rotation_angle = extract_euler_angles(torch.tensor(camera_rotation, dtype=torch.float32))

    R_rel_list_all, T_rel_list_all = extract_rel_R_T_from_txt(relative_rotation_location_path)
    R_rel_list = [R_rel_list_all[i-1] for i in relative_cameras]
    T_rel_list = [T_rel_list_all[i-1] for i in relative_cameras]


############################################################################################################
    obj_mesh = load_obj_mesh(obj_path, device=device)
    train_cameras = PerspectiveCameras(focal_length=fcl_screen_0,principal_point=prp_screen_0, in_ndc=False, image_size=(train_image_size,),device=device)
    final_cameras = PerspectiveCameras(focal_length=((fcl_screen_0[0][0] * 8, fcl_screen_0[0][1] * 8),),principal_point=((prp_screen_0[0][0]*8,prp_screen_0[0][1]*8),), in_ndc=False, image_size=(final_image_size,),device=device)


    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)  # 默认渲染参数
    silhouette_renderer = create_renderer('SoftSilhouetteShader', image_size=train_image_size, blend_params=blend_params,cameras=train_cameras, device=device)
    normal_renderer = create_renderer('NormalShader', image_size=train_image_size, blend_params=blend_params, cameras=train_cameras,device=device)

    if len(mask_paths_list) == 0:
        raise ValueError("The mask_paths_list is empty.")
    else:
        mask_list = []
        for mask_path in mask_paths_list:
            mask = load_mask(mask_path)
            mask_resized = cv2.resize(mask, (train_image_size[1], train_image_size[0]))
            mask_list.append(mask_resized)

############################################################################################################
    model = Model(obj_mesh, silhouette_renderer, normal_renderer, mask_list, camera_location, camera_rotation_angle, R_rel_list, T_rel_list,initial_scale)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    numOfStage  = 3
    milestones = np.linspace(0, 5000, numOfStage + 2)[1:-1].astype('int')
    milestones = milestones.tolist()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    if not os.path.exists('vis'):
        os.makedirs('vis')

    # 冻结 scale 参数的初始设置
    model.scale.requires_grad = False

    loop = tqdm(range(loop_num))
    for i in loop:
        # if i > 2000:
            # model.scale.requires_grad = True


        optimizer.zero_grad()
        loss,mask_pred_list,pred_normal_1 = model()
        loss.backward()
        optimizer.step()
        scheduler.step()

        loop.set_description(f"Loss: {loss.item()}")


        if i % 200 == 0:

            plt.subplot(1, 3, 1)
            plt.imshow(mask_pred_list[0].cpu().detach().numpy())
            plt.axis('off')
            plt.subplot(1, 3, 2)
            plt.imshow(getattr(model, f"image_ref_{0}").cpu().detach().numpy())
            plt.axis('off')
            plt.subplot(1, 3, 3)
            plt.imshow(pred_normal_1[0].cpu().detach().numpy())
            plt.axis('off')


            plt.title('loss: %.4f' % loss.data)
            plt.tight_layout()
            plt.show()



        if loss.item() < 0.001:
            print("The loss is less than 0.001, the training is finished.")
            break

############################################################################################################
    # 将R和T保存在txt文档中
    txt_path = os.path.join(save_path, 'R_T_final.txt')

    R = get_rotation_matrix(model.camera_rotation[0], model.camera_rotation[1], model.camera_rotation[2])
    T = model.camera_position
    # 计入文档
    with open(txt_path, 'w') as file:
        file.write(f"R_col_0\n{R.cpu().detach().numpy()}\n")
        file.write(f"T_col_0\n{T.cpu().detach().numpy()}\n")

    R_row = R.permute(0, 2, 1) # W2C下的旋转矩阵，行主序
    T = T.unsqueeze(0)

    # scale the mesh
    scale = model.scale.item()
    print('scale:', scale)
    verts = model.meshes.verts_packed()
    scaled_verts = verts * scale
    scaled_mesh = Meshes(verts=[scaled_verts], faces=model.meshes.faces_list())

    final_normal_renderer = create_renderer('NormalShader', image_size=final_image_size, blend_params=blend_params,cameras=final_cameras, device=device)
    _,render_normal = final_normal_renderer(meshes_world=scaled_mesh, R=R_row, T=T)
    save_path_0 = os.path.join(save_path, "normal_00.png")
    save_normal(render_normal, R, save_path_0)


    for i in range(len(R_rel_list_all)):
        R_rel = torch.tensor(R_rel_list_all[i], dtype=torch.float32, device=device)
        T_rel = torch.tensor(T_rel_list_all[i], dtype=torch.float32, device=device)
        R_W2C, T_W2C = cal_R2_RT(R, T, R_rel, T_rel)
        # 计入文档
        with open(txt_path, 'a') as file:
            file.write(f"R_col_{i+1}\n{R_W2C.cpu().detach().numpy()}\n")
            file.write(f"T_col_{i+1}\n{T_W2C.cpu().detach().numpy()}\n")
        R_W2C_row = R_W2C.permute(0, 2, 1) # pytorch3d中使用行主序为标准
        _,render_normal = final_normal_renderer(meshes_world=scaled_mesh, R=R_W2C_row, T=T_W2C)
        save_path_i = os.path.join(save_path, f"normal_{i+1:02d}.png")
        save_normal(render_normal, R_W2C, save_path_i)





if __name__ == '__main__':
    main()




