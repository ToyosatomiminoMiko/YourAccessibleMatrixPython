import numpy as np
import time

def linear_dependency(A, name):
    """
    判断向量组是否线性无关。

    参数:
    A (np.ndarray): 输入的向量组矩阵。
    name (str): 向量组的名称。

    返回:
    int: 如果向量组线性无关返回 1，否则返回 0。
    """
    _, n = A.shape
    if np.linalg.matrix_rank(A) == n:
        print(f'Vector Set {name} is linearly independent.')
        return 1
    else:
        print(f'Vector Set {name} is linearly dependent.')
        return 0

def equivalent(A, B, name_A, name_B):
    """
    判断两个向量组是否等价。

    参数:
    A (np.ndarray): 第一个向量组矩阵。
    B (np.ndarray): 第二个向量组矩阵。
    name_A (str): 第一个向量组的名称。
    name_B (str): 第二个向量组的名称。
    """
    C = np.hstack((A, B))
    if np.linalg.matrix_rank(A) == np.linalg.matrix_rank(B) and np.linalg.matrix_rank(A) == np.linalg.matrix_rank(C):
        print(f'Sets {name_A} and {name_B} are equivalent.')
    else:
        print(f'Sets {name_A} and {name_B} are not equivalent.')

def Gram_Schmidt(A):
    """
    使用施密特正交化将向量组转化为单位正交基向量组。

    参数:
    A (np.ndarray): 输入的向量组矩阵。

    返回:
    np.ndarray: 单位正交基向量组。
    """
    m, n = A.shape
    P = np.zeros((m, n))
    Q = np.zeros((m, n))
    P[:, 0] = A[:, 0]
    for i in range(1, n):
        p = np.zeros(m)
        for j in range(i):
            p += np.dot(A[:, i], P[:, j]) / np.dot(P[:, j], P[:, j]) * P[:, j]
        P[:, i] = A[:, i] - p
        Q[:, i] = P[:, i] / np.linalg.norm(P[:, i])
    Q[:, 0] = P[:, 0] / np.linalg.norm(P[:, 0])
    return Q

def transition_matrix(A, B):
    """
    求解从向量组 A 到向量组 B 的过渡矩阵。

    参数:
    A (np.ndarray): 第一个向量组矩阵。
    B (np.ndarray): 第二个向量组矩阵。

    返回:
    np.ndarray: 过渡矩阵。
    """
    augmented_matrix = np.hstack((A, B))
    rref_matrix = rref(augmented_matrix)[0]
    P = rref_matrix[:, A.shape[1]:]
    if A.shape[0] > A.shape[1]:
        P = P[:A.shape[1], :]
    return P

def rref(matrix):

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

if __name__ == "__main__":
    # 清除所有变量并初始化计时器
    start_time = time.time()

    # 输入向量组并横向并列为矩阵
    a1 = np.array([1, 1, 1]).reshape(-1, 1)
    a2 = np.array([1, 0, 1]).reshape(-1, 1)
    A = np.hstack((a1, a2))  # 向量组(I)

    b1 = np.array([2, 1, 2]).reshape(-1, 1)
    b2 = np.array([0, 1, 0]).reshape(-1, 1)
    B = np.hstack((b1, b2))  # 向量组(II)

    # 任务1: 判断它们是线性无关还是线性相关
    flag_independent1 = linear_dependency(A, '(I)')
    flag_independent2 = linear_dependency(B, '(II)')

    # 任务2: 判断它们是否等价
    equivalent(A, B, '(I)', '(II)')

    # 任务3: 使用施密特正交化对两者分别化为等价的单位正交基向量组
    QA = Gram_Schmidt(A)
    QB = Gram_Schmidt(B)
    print("Orthogonalized and normalized vectors of A:")
    print(QA)
    print("\nOrthogonalized and normalized vectors of B:")
    print(QB)

    # 任务4: 求从向量组(I)和(II)的过渡矩阵
    if flag_independent1 == 1 and flag_independent2 == 1:
        P = transition_matrix(A, B)
        print("\nTransition matrix from set (I) to set (II):")
        print(P)

    # 计算并输出运行时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nElapsed time: {elapsed_time:.6f} seconds")