import numpy as np
import time

# 清除所有变量并初始化计时器
start_time = time.time()

# 定义矩阵 A 和 B，并确保它们是浮点数类型
A = np.array([[4, 5, 6], [5, 6, 7]], dtype=float)
B = np.array([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]], dtype=float)

# 计算矩阵乘法 C = A * B
C = np.dot(A, B)

# 输出结果
print("Matrix C:")
print(C)

# 计算并输出运行时间
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.6f} seconds")