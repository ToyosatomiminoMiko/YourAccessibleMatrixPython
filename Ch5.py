import numpy as np
import time

# 清除所有变量并初始化计时器
start_time = time.time()

# 定义矩阵 A，并确保它是浮点数类型
A = np.array([[1, 1, 2], [1, 2, 3], [2, 4, 5]], dtype=float)

# 计算矩阵 A 的逆矩阵
try:
    inv_A = np.linalg.inv(A)
    print("Matrix A:")
    print(A)
    print("\nInverse of Matrix A:")
    print(inv_A)
except np.linalg.LinAlgError as e:
    print(f"Error: {e}")

# 计算并输出运行时间
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nElapsed time: {elapsed_time:.6f} seconds")