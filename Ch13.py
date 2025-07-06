import numpy as np
from scipy.linalg import eig
import time

def similar_diag(A):
    """
    对方阵 A 进行相似对角化。

    参数:
    A (np.ndarray): 输入的方阵。

    返回:
    tuple: 包含两个元素的元组 (P, U)。
           如果矩阵可以相似对角化，则 P 是变换矩阵，U 是对角化的结果；
           否则，P 和 U 都为空数组，并输出提示信息。
    """
    m, n = A.shape
    if m != n:
        raise ValueError("Input matrix must be square.")  # 输入必须是方阵
    else:
        eigenvalues, eigenvectors = eig(A)
        P = eigenvectors
        if np.linalg.matrix_rank(P) != n:  # 不可相似对角化
            P = np.array([])
            U = np.array([])
            print('Input matrix cannot be diagonalized through similar transformation.')
        else:
            # 将特征值构成对角矩阵
            U = np.diag(eigenvalues)
        return P, U

if __name__ == "__main__":
    # 清除所有变量并初始化计时器
    start_time = time.time()

    # 定义矩阵 A, B, C
    A = np.array([[2, -3, 2], [0, -1, 2], [1, -5, 5]])
    B = np.array([[-1, 1, 0], [-4, 3, 0], [1, 0, 2]])
    C = np.array([[1, -4, 8], [0, -4, 10], [0, -3, 7]])

    # 计算矩阵 A 的相似对角化
    PA, UA = similar_diag(A)
    if PA.size > 0 and UA.size > 0:
        print("Matrix A:")
        print("Transformation matrix P:\n", PA)
        print("Diagonalized matrix U:\n", UA)

    # 计算矩阵 B 的相似对角化
    PB, UB = similar_diag(B)
    if PB.size > 0 and UB.size > 0:
        print("\nMatrix B:")
        print("Transformation matrix P:\n", PB)
        print("Diagonalized matrix U:\n", UB)

    # 计算矩阵 C 的相似对角化
    PC, UC = similar_diag(C)
    if PC.size > 0 and UC.size > 0:
        print("\nMatrix C:")
        print("Transformation matrix P:\n", PC)
        print("Diagonalized matrix U:\n", UC)

    # 计算并输出运行时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nElapsed time: {elapsed_time:.6f} seconds")