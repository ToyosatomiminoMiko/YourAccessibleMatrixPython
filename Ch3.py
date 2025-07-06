import numpy as np
import time


def solve_equation_system(A, b):
    """
    解线性方程组 Ax = b。

    参数:
    A (np.ndarray): 系数矩阵。
    b (np.ndarray): 结果向量。
    """
    # 确保矩阵和向量是浮点数类型
    A = A.astype(float)
    b = b.astype(float)

    m, n = A.shape  # 获取矩阵A的尺寸
    Ab = np.column_stack((A, b))  # 将A和b组合成增广矩阵[A|b]

    # 使用rref函数
    B, pivots = rref(Ab)

    rank_A = np.linalg.matrix_rank(A)
    rank_Ab = np.linalg.matrix_rank(Ab)

    if rank_A == n and rank_Ab == n:
        # 唯一解
        x = B[:, -1]
        print('Only one solution.')
        for i in range(m):
            print(f'{x[i]:.3f}')
    elif rank_Ab != rank_A:
        # 无解
        print('No solution.')
    else:
        # 无穷多解
        C = null_space_manual(A, pivots)  # 齐次方程组的基础解系
        x0 = B[:, -1]  # 非齐次方程组特解

        print('Infinitely many solutions.')
        print('Basic solution set - homo:')

        for j in range(C.shape[1]):
            print(f'c{j + 1} = ')
            for i in range(m):
                print(f'{C[i, j]:.3f}')

        if np.linalg.norm(b) != 0:
            print('One particular solution - non-homo:')
            print('x0 = ')
            for i in range(m):
                print(f'{x0[i]:.3f}')


def rref(matrix):
    """
    将矩阵化简为行最简形式（RREF）。

    参数:
    matrix (np.ndarray): 输入矩阵。

    返回:
    tuple: 化简后的矩阵和主变量序号。
    """
    # 确保矩阵是浮点数类型
    matrix = matrix.astype(float)

    m, n = matrix.shape
    A = matrix.copy()
    pivots = []

    for i in range(min(m, n)):
        # 找到当前列的最大值并交换行
        max_row = i + np.argmax(np.abs(A[i:, i]))
        if A[max_row, i] == 0:
            continue
        A[[i, max_row]] = A[[max_row, i]]
        pivots.append(i)

        # 消元操作
        for j in range(i + 1, m):
            factor = A[j, i] / A[i, i]
            A[j, :] -= factor * A[i, :]

    # 回代过程，将矩阵转换为简化阶梯形
    for i in reversed(range(len(pivots))):
        pivot_col = pivots[i]
        for j in range(i):
            factor = A[j, pivot_col] / A[i, pivot_col]
            A[j, :] -= factor * A[i, :]

    # 归一化主元为1
    for i in range(len(pivots)):
        pivot_col = pivots[i]
        A[i, :] /= A[i, pivot_col]

    return A, pivots


def null_space_manual(A, pivots):
    """
    计算矩阵的基础解系

    参数:
    A (np.ndarray): 输入矩阵。
    pivots (list): 主变量列索引。

    返回:
    np.ndarray: 零空间的基向量。
    """
    # 确保矩阵是浮点数类型
    A = A.astype(float)

    m, n = A.shape
    A_rref, _ = rref(A)

    free_vars = [i for i in range(n) if i not in pivots]

    # 初始化零空间基向量
    null_basis = []

    for var in free_vars:
        vec = np.zeros(n)
        vec[var] = 1

        for pivot in pivots[::-1]:
            vec[pivot] = -A_rref[pivot, var]

        null_basis.append(vec)

    return np.array(null_basis).T


if __name__ == "__main__":
    # 清除所有变量并初始化计时器
    np.set_printoptions(precision=3, suppress=True)

    start_time = time.time()

    # 定义系数矩阵A和常数项向量b，并确保它们是浮点数类型（使用时可以替换为其他方程组的情形）
    A = np.array([[1, 2, 3, 1], [1, -4, -1, -1], [2, 1, 4, 1], [1, -1, 1, 0]], dtype=float)
    b = np.array([3, 1, 5, 2], dtype=float)

    # 调用solve_equation_system函数
    solve_equation_system(A, b)

    # 计算并输出运行时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")