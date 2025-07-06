import numpy as np
import matplotlib.pyplot as plt
import time

# 清除所有变量并初始化计时器
start_time = time.time()

# 定义原始正方形的坐标
x_1 = np.arange(-1, 1.1, 0.1)
x_2 = np.arange(-1, 1.1, 0.1)
y_1 = np.ones_like(x_1)
y_2 = -np.ones_like(x_2)
y_3 = np.arange(-1, 1.1, 0.1)
y_4 = np.arange(-1, 1.1, 0.1)
x_3 = np.ones_like(y_3)
x_4 = -np.ones_like(y_4)
x_5 = np.arange(-1, 1.1, 0.1)
y_5 = x_5

# 将所有的 x 和 y 坐标合并
x = np.concatenate([x_1, x_2, x_3, x_4, x_5])
y = np.concatenate([y_1, y_2, y_3, y_4, y_5])

# 绘制原始正方形
plt.figure(figsize=(8, 8))
plt.plot(x, y, 'b', linewidth=2)
plt.axis([-4, 4, -4, 4])
plt.axis('equal')
plt.grid(True)
plt.box(False)

# 定义变换矩阵 A
A = np.array([[1, 2], [0.5, 3]])

# 进行线性变换
T_A = A @ np.vstack((x, y))
x_A = T_A[0, :]
y_A = T_A[1, :]

# 绘制变换后的图形，并在图中显示原始图形以方便对比
plt.figure(figsize=(8, 8))
plt.plot(x_A, y_A, 'b', linewidth=2)
plt.plot(x, y, '.r', linewidth=0.7)  # 虚线画出原图形方便对比
plt.axis([-4, 4, -4, 4])
plt.axis('equal')
plt.grid(True)
plt.box(False)

# 计算并输出运行时间
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.6f} seconds")

# 显示图形
plt.show()