import numpy as np
import time

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

    # 定义矩阵 A，并确保它是浮点数类型
    A = np.array([[1, 2, 5], [3, 2, 6], [9, 7, 4]], dtype=float)

    # 计算矩阵 A 的行列式
    det_A = det1(A)
    print(f"Determinant of A: {det_A}")

    # 计算并输出运行时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")