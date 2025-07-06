import numpy as np

# 定义矩阵 A 并计算其秩
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
rank_A = np.linalg.matrix_rank(A)
print(f"Matrix A:\n{A}")
print(f"Rank of A: {rank_A}")

# 定义矩阵 U 并计算其秩
U = np.array([[10, 0, 0, 0], [0, 25, 0, 0], [0, 0, 34, 0], [0, 0, 0, 1e-15]])
rank_U = np.linalg.matrix_rank(U, tol=1e-16)
print(f"\nMatrix U:\n{U}")
print(f"Rank of U with tolerance 1e-16: {rank_U}")