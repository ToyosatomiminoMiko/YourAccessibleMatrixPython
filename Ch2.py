import numpy as np

# 定义矩阵A，B，C和向量k，h
A = np.array([[1, 2], [-1, 3]])    # 矩阵A
B = np.array([[4, -2], [-6, 8]])   # 矩阵B
C = np.array([[0, 1, 2], [3, 7, 5]])  # 矩阵C
k = np.array([[2], [3]])           # 向量k
h = np.array([1, 3, -4])           # 向量h

# 运算L = A + B: 矩阵加法，对应位置元素相加
L = A + B

# 运算M = A - B: 矩阵减法，对应位置元素相减
M = A - B

# 运算N = 2 * A: 标量乘矩阵，每个元素都乘以2
N = 2 * A

# 运算p = A * k: 矩阵与列向量的乘积，按照矩阵乘法规则进行
p = A @ k

# 获取矩阵C的尺寸[m, n]：m为行数，n为列数
m, n = C.shape  # shape属性返回矩阵的维度（行数，列数）

# 输出结果
print("L:\n", L)
print("M:\n", M)
print("N:\n", N)
print("p:\n", p)
print("Size of C:", m, n)