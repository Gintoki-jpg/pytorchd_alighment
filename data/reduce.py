import cv2
import numpy as np

# 打开当前目录中的mask.png图像
input_image = cv2.imread(r"C:\Users\SANFLAG\Desktop\mask_05.png", cv2.IMREAD_GRAYSCALE)

# 获取图像的原始尺寸
original_height, original_width = input_image.shape

# 降采样因子为4，因此新图像的尺寸为原来的1/4
new_width = original_width // 4
new_height = original_height // 4

# 确保图像只有0和255
_, binary_mask = cv2.threshold(input_image, 128, 255, cv2.THRESH_BINARY)

# 调整图像大小（降采样）
downsampled_image = cv2.resize(binary_mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

# 将降采样后的图像保存为新的文件
cv2.imwrite(r"C:\Users\SANFLAG\Desktop\mask05__downsampled.png", downsampled_image)

print(f"图像已成功降采样并保存为mask05__downsampled.png，新的尺寸为 {new_width}x{new_height} 像素。")

