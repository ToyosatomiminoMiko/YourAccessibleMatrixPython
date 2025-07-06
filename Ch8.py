import numpy as np
import time
from scipy.sparse import bmat

# 清除所有变量并初始化计时器
start_time = time.time()

# 生成一个 10x10 的随机整数矩阵 A
A = np.round(9 * np.random.rand(10, 10)).astype(int)

# 将矩阵 A 分割成多个子矩阵

# 定义行和列的分割点
row_indices = [0, 3, 6, 10]  # 对应于 [3, 3, 4]
col_indices = [0, 5, 10]     # 对应于 [5, 5]

# 使用列表推导式进行分割
B = [
    [A[row_indices[i]:row_indices[i+1], col_indices[j]:col_indices[j+1]]
     for j in range(len(col_indices)-1)]
    for i in range(len(row_indices)-1)
]

# 打印结果以验证
print("Matrix A:")
print(A)

print("\nSub-matrices B:")
for row in B:
    for sub_matrix in row:
        print(sub_matrix)
        print()

# 计算并输出运行时间
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.6f} seconds")