import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimSun', 'Times New Roman']  # 中文宋体，英文Times New Roman
plt.rcParams['axes.unicode_minus'] = False

plt.rcParams['font.size'] = 14       # 全局字体大小
plt.rcParams['axes.titlesize'] = 16  # 标题字号
plt.rcParams['axes.labelsize'] = 14  # 坐标轴标签字号

# 设置图片文件夹路径
image_folder = 'illumination-GB/P10'  # 修改成你自己的路径

# 存储RGB颜色数据
colors = []

# 遍历文件夹内所有png图片
for filename in os.listdir(image_folder):
    if filename.lower().endswith('.png'):
        img_path = os.path.join(image_folder, filename)
        img = Image.open(img_path).convert('RGB')  # 确保转换成RGB格式

        # 提取像素数据
        img_data = np.array(img)

        # 将二维图片数据reshape成(N, 3)的RGB数组
        pixels = img_data.reshape(-1, 3)
        pixels = pixels[~np.all(pixels == [0, 0, 0], axis=1)]


        # 可以随机抽取一部分像素，避免内存占用过高
        if len(pixels) > 20000:
            idx = np.random.choice(len(pixels), 20000, replace=False)
            pixels = pixels[idx]

        colors.append(pixels)

# 将所有图片的颜色数据合并为一个大数组
colors = np.vstack(colors)

# 归一化颜色以便绘图显示
colors_norm = colors / 255.0

# 3D颜色散点图绘制
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    colors[:, 0], colors[:, 1], colors[:, 2],
    c=colors_norm,
    # s=500,  # 使用实际颜色
    marker='.', alpha=0.2, linewidth=0
)

ax.set_xlabel('红')
ax.set_ylabel('绿')
ax.set_zlabel('蓝')
ax.set_title('颜色分布散点图')

# 在绘图代码后添加布局调整
plt.tight_layout()  # 自动调整布局

# 修改保存参数
plt.savefig('3d_color_scatter.png', 
           dpi=300,
           bbox_inches='tight',  # 自动裁剪白边
           pad_inches=0.4)       # 保留少量内边距
