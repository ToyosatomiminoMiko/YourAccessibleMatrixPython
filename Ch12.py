import numpy as np
from scipy.linalg import eig

# 定义矩阵 A, B, C
A = np.array([[2, -3, 2], [0, -1, 2], [1, -5, 5]])
B = np.array([[-1, 1, 0], [-4, 3, 0], [1, 0, 2]])
C = np.array([[1, -4, 8], [0, -4, 10], [0, -3, 7]])

# 计算矩阵 A 的特征值和特征向量
PA, UA = eig(A)
print("Matrix A:")
print("Eigenvalues:\n", PA)
print("Eigenvectors:\n", UA)

# 计算矩阵 B 的特征值和特征向量
PB, UB = eig(B)
print("\nMatrix B:")
print("Eigenvalues:\n", PB)
print("Eigenvectors:\n", UB)

# 计算矩阵 C 的特征值和特征向量
PC, UC = eig(C)
print("\nMatrix C:")
print("Eigenvalues:\n", PC)
print("Eigenvectors:\n", UC)