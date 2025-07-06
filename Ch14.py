import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt
import time


def orthogonal_diag(A):
    """
    对实对称矩阵 A 进行相似对角化。

    参数:
    A (np.ndarray): 输入的实对称矩阵。

    返回:
    tuple: 包含两个元素的元组 (Q, U)。
           Q 是变换矩阵，U 是对角化的结果。
    """
    m, n = A.shape
    if m != n or not np.allclose(A, A.T):
        raise ValueError("Input matrix must be symmetric.")  # 输入必须是实对称矩阵
    else:
        eigenvalues, eigenvectors = eig(A)
        Q = eigenvectors
        U = np.diag(eigenvalues)
        return Q, U


def pn_definite(U):
    """
    求出对应的正、负惯性指数，并判定是正定还是负定二次型。

    参数:
    U (np.ndarray): 对角化的结果矩阵。
    """
    n = U.shape[0]
    positive = 0
    negative = 0
    for i in range(n):
        if U[i, i] > 0:
            positive += 1
        elif U[i, i] < 0:
            negative += 1
    print(f'The positive index of inertia is {positive}.')
    print(f'The negative index of inertia is {negative}.')
    if positive == n:
        print('f is positive definite.')
    elif negative == n:
        print('f is negative definite.')


def create_curve(A, U, xyinterval):
    """
    画出对应二元二次型对应的曲线。

    参数:
    A (np.ndarray): 输入的实对称矩阵。
    U (np.ndarray): 对角化的结果矩阵。
    xyinterval (list): 数值区间行向量 [xmin, xmax, ymin, ymax]。
    """
    a1, b1_half, c1 = A[0, 0], A[0, 1], A[1, 1]
    b1 = b1_half * 2  # 因为输入矩阵是对称矩阵，所以 b1 = A(1,2) + A(2,1)
    a2, c2 = U[0, 0], U[1, 1]

    def f1(x1, x2):
        return a1 * x1 ** 2 + b1 * x1 * x2 + c1 * x2 ** 2 - 1

    def f2(x1, x2):
        return a2 * x1 ** 2 + c2 * x2 ** 2 - 1

    xmin, xmax, ymin, ymax = xyinterval

    # 创建网格数据
    x = np.linspace(xmin, xmax, 400)
    y = np.linspace(ymin, ymax, 400)
    X, Y = np.meshgrid(x, y)

    # 计算函数值
    Z1 = f1(X, Y)
    Z2 = f2(X, Y)

    # 绘制原曲线
    plt.figure(figsize=(8, 8))
    plt.contour(X, Y, Z1, levels=[0], colors='blue', linestyles='dashed', linewidths=2)

    # 绘制标准形曲线
    plt.contour(X, Y, Z2, levels=[0], colors='red', linewidths=2)

    plt.axis(xyinterval)
    plt.axis('equal')
    plt.grid(True)
    plt.box(False)
    plt.show()


if __name__ == "__main__":
    # 清除所有变量并初始化计时器
    start_time = time.time()

    # 定义矩阵 A
    A = np.array([[1, 0.5], [0.5, 1]])

    # 计算矩阵 A 的相似对角化
    Q, U = orthogonal_diag(A)

    # 判定正、负惯性指数及正定性
    pn_definite(U)

    # 画出对应二元二次型的曲线
    xyinterval = [-3, 3, -3, 3]
    create_curve(A, U, xyinterval)

    # 计算并输出运行时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nElapsed time: {elapsed_time:.6f} seconds")