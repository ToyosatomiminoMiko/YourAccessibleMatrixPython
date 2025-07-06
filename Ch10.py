import numpy as np
import time


def adjugate(A):
    """
    计算方阵 A 的伴随矩阵。

    参数:
    A (np.ndarray): 输入的方阵。

    返回:
    np.ndarray: 方阵 A 的伴随矩阵。
    """
    m, n = A.shape
    if m != n:
        raise ValueError("Input matrix must be square.")

    B = np.zeros_like(A, dtype=float)
    for i in range(n):
        for j in range(n):
            T = np.delete(np.delete(A, i, axis=0), j, axis=1)
            B[j, i] = ((-1) ** (i + j)) * det1(T)  # 确保 det1 已经定义

    return B


def det1(A):
    """
    计算方阵 A 的行列式。

    参数:
    A (np.ndarray): 输入的方阵。

    返回:
    float: 方阵 A 的行列式。
    """
    n, m = A.shape
    if n == m:
        if n == 1:
            return A[0, 0]
        elif n == 2:
            return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        else:
            d = 0
            for i in range(n):
                A2 = np.delete(np.delete(A, 0, axis=0), i, axis=1)
                d += A[0, i] * (-1) ** (1 + i) * det1(A2)
            return d
    else:
        raise ValueError("Input matrices must be square.")


if __name__ == "__main__":
    # 清除所有变量并初始化计时器
    start_time = time.time()

    # 定义矩阵 A, B, C
    A = np.array([[1, 1, 1, 1], [1, 1, 2, 3], [1, 3, 3, 4], [2, 3, 4, 5]], dtype=float)
    B = np.array([[1, 1, 1, 1], [1, 1, 2, 3], [1, 3, 3, 4], [1, 3, 4, 6]], dtype=float)
    C = np.array([[1, 1, 1, 1], [1, 1, 2, 3], [2, 2, 3, 4], [3, 3, 5, 7]], dtype=float)

    # 计算伴随矩阵
    A_adj = adjugate(A)
    B_adj = adjugate(B)
    C_adj = adjugate(C)

    # 打印伴随矩阵
    print("Adjugate of A:")
    print(A_adj)
    print("\nAdjugate of B:")
    print(B_adj)
    print("\nAdjugate of C:")
    print(C_adj)

    # 计算并输出运行时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nElapsed time: {elapsed_time:.6f} seconds")