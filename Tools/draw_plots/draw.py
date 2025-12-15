import matplotlib.pyplot as plt
import numpy as np

# 这是一个用于画图的基础代码
# 我们使用 matplotlib 的 mplot3d 工具包

# 创建画布
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# ---------------------------------------------------------
# 生成一些虚拟数据用于展示 (你可以替换为你自己的数据)
# ---------------------------------------------------------


# 设置坐标轴标签
ax.set_xlabel('x=Social Needs')
ax.set_ylabel('y=Time')
ax.set_zlabel('z=Smile Intensity')

# ---------------------------------------------------------
# 关键设置：调整坐标轴比例
# ---------------------------------------------------------
# 这里设置 y 轴在视觉上的长度是 x 和 z 轴的 3 倍
# 格式为 (x_scale, y_scale, z_scale)
#z smile intensity
#x social needs
#y time

ax.set_zlim(10, 0)
ax.set_xlim(-1,1)
ax.set_ylim(0,60)
ax.set_box_aspect((1, 3, 1))

# 调整视角以便更好地观察长轴效果 (可选)
ax.view_init(elev=20, azim=135)

#绘制曲线
# ---------------------------------------------------------
# 在 X=0 平面绘制正弦曲线
# ---------------------------------------------------------
# 1. 生成 Y 轴数据 (Time: 0 到 60)
num_points = 100
y_curve = np.linspace(0, 60, num_points)

# 2. 生成 X 轴数据 (固定为 0)
x_curve = np.zeros(num_points)

# 3. 生成 Z 轴数据 (正弦波)
# 说明: sin 的值域是 [-1, 1]
# 我们乘以 4 再加 5，将值域映射到 [1, 9]，确保全程在 (0, 10) 之间
# y_curve / 5 用于调整波浪的频率，让它看起来更舒展
z_curve = 5 + 4 * np.sin(y_curve / 5)

# 4. 绘图
ax.plot(x_curve, y_curve, z_curve, color='blue', linewidth=2, label='Sine Curve')
ax.text(0, 30, 9, "Smile Intensity without social needs", color='blue', fontsize=12)
ax.text(0, 60, 9, "Base(y) = 5 + 4sin(y/5)", color='blue', fontsize=12)
# 展示图表

# ---------------------------------------------------------
# 绘制 x=0 的半透明平面
# ---------------------------------------------------------
# 1. 构造平面的网格点
# Y轴范围 0 到 60，Z轴范围 0 到 10
y_range = np.linspace(0, 60, 10)
z_range = np.linspace(0, 10, 10)
Y_plane, Z_plane = np.meshgrid(y_range, z_range)

# X轴固定为 0
X_plane = np.zeros_like(Y_plane)

# 2. 绘制平面
# alpha=0.3 表示 30% 不透明度 (即半透明)
# color='gray' 设置为灰色，避免干扰曲线颜色
ax.plot_surface(X_plane, Y_plane, Z_plane, alpha=0.1, color='gray')

#
# ---------------------------------------------------------
# 绘制平面: 基于中心脊椎曲线向两侧延伸
# 公式: Z = (5 + 4sin(y/5)) + 4x
# ---------------------------------------------------------

# 1. 生成网格数据
# x 从 -1 到 1, y 从 0 到 60
x_vals = np.linspace(-1, 1, 100)
y_vals = np.linspace(0, 60, 100)
X_surf, Y_surf = np.meshgrid(x_vals, y_vals)

# 2. 计算 Z 值
# 第一部分：基础的正弦波 (随 Y 变化) - 这是你的"脊椎"
base_curve = 5 + 4 * np.sin(Y_surf / 5)

# 第二部分：随 X 的线性变化 (倾斜)
# 这里的 4 是斜率，意味着 x 每增加 1，z 就增加 4
# 你可以修改这个数字来改变平面的倾斜程度
tilt = (-4) * X_surf

# 组合公式
Z_surf = base_curve + tilt

# 3. 裁剪数据 (Masking) - 强制限制 z 在 [0, 10] 之间
# 只有在这个范围内的点才会被绘制，形成切面效果
Z_surf[Z_surf < 0] = np.nan
Z_surf[Z_surf > 10] = np.nan

# 4. 绘制曲面
# alpha=0.6 半透明
# rstride 和 cstride 控制网格采样密度，设大一点可以减少线条密集度，设为1最细腻
ax.plot_surface(X_surf, Y_surf, Z_surf, cmap='viridis', alpha=0.6, rstride=2, cstride=2)
ax.text(0, 0, 0, "Smile Intensity Surface with social needs", color='black', fontsize=12)
ax.text(0, 0, -2, "Z(x, y) = [ 5 + 4sin(y/5) ] + 4x", color='black', fontsize=12)
plt.show()