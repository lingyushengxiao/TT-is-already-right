import numpy as np
import torch
from cvxopt import matrix, solvers
from cvxpylayer import MyCPLayer
import cvxpy as cp
import time
from scipy.linalg import sqrtm
from matplotlib import pyplot as plt
def generate_data(n, m):
    """生成随机优化问题数据"""
    Q = np.random.rand(n, n)
    Q = Q.T @ Q + np.eye(n)  # 保证正定
    c = np.random.rand(n)
    A = np.random.rand(m, n) - 0.5
    b = np.random.rand(m)
    return Q, c, A, b

# QPTM 实现
def solve_qpth(Q, c, A, b):
    Q = matrix(Q)
    c = matrix(c)
    A = matrix(A)
    b = matrix(b)
    start_time = time.time()
    solution = solvers.qp(Q, c, A, b)
    runtime = time.time() - start_time
    return runtime

# CVXPYLayer 实现
def solve_cvxpylayer(Q, c, A, b):
    n = len(c)
    m = A.shape[0]
    
    # CVXPY parameters
    Qsqrt_param = cp.Parameter((n, n))
    c_param = cp.Parameter(n)
    A_param = cp.Parameter((m, n))
    b_param = cp.Parameter(m)

    # CVXPY problem definition
    x = cp.Variable(n)
    objective = cp.Minimize(0.5 * cp.sum_squares(Qsqrt_param @ x) + c_param.T @ x)
    constraints = [A_param @ x <= b_param]
    problem = cp.Problem(objective, constraints)

    # CVXPYLayer Reconstruction
    cvxpylayer = MyCPLayer(problem, parameters=[Qsqrt_param, c_param, A_param, b_param], variables=[x])

    # introduce tensor for parameters
    Q_torch = torch.tensor(sqrtm(Q), dtype=torch.float32)
    c_torch = torch.tensor(c, dtype=torch.float32)
    A_torch = torch.tensor(A, dtype=torch.float32)
    b_torch = torch.tensor(b, dtype=torch.float32)

    x_opt, = cvxpylayer(Q_torch, c_torch, A_torch, b_torch)
    # runtime test
    start_time = time.time()
    runtime = time.time() - start_time
    return runtime

# 测试不同问题
problems = [
    {"n": 2, "m": 3},
    {"n": 10, "m": 5},
    {"n": 100, "m": 200},
]

qptm_times=[]
cvxpylayer_times=[]
# 比较运行时间
n_values = range(10,201,10)
for i in range(10,201,10):
    n=i
    m=i
    Q, c, A, b = generate_data(n, m)

    qptm_time = solve_qpth(Q, c, A, b)
    cvxpylayer_time = solve_cvxpylayer(Q, c, A, b)
    qptm_times.append(qptm_time)
    cvxpylayer_times.append(cvxpylayer_time)
plt.figure(figsize=(10, 6))
plt.plot(n_values, qptm_times, label="QPTH Time", color="blue", linewidth=2)
plt.plot(n_values, cvxpylayer_times, label="CVXPYLayer Time", color="red", linestyle="--", linewidth=2)

# 图像设置
plt.title("Runtime Comparison: QPTH vs CVXPYLayer", fontsize=14)
plt.xlabel("Number of Variables (n)", fontsize=12)
plt.ylabel("Runtime (seconds)", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()

# 显示图像
plt.savefig('/home/xlyang/CVXPY_LAYER/fig_1.png')