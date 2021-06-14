import numpy as np
import random
from decimal import Decimal
from matplotlib import pyplot as plt


def calc_lambda(t, k, M, mu, A, w, cur_ans):
    ret = mu[k]
    for i in range(M):
        for j in range(len(cur_ans[i])):
            if t >= cur_ans[i][j] and w * (t - cur_ans[i][j]) < 200:
                ret += A[k][i] * np.exp(-w * (t - cur_ans[i][j]))
    return ret


def multi_hawkes(A, mu, w, M, N):
    t = 0
    cur_ans = []
    for i in range(M):
        cur_ans.append([])
    ans = []
    while len(ans) < N:
        lambda_list = [calc_lambda(t, i, M, mu, A, w, cur_ans) for i in range(M)]
        cur_lambda = sum(lambda_list)
        t += -np.log(random.uniform(0, 1)) / cur_lambda
        lambda_list = [calc_lambda(t, i, M, mu, A, w, cur_ans) for i in range(M)]
        D = random.uniform(0, 1)
        if D * cur_lambda <= sum(lambda_list):
            cur_lambda_sum = 0
            for i in range(M):
                if cur_lambda_sum + lambda_list[i] >= D * cur_lambda >= cur_lambda_sum:
                    cur_ans[i].append(t)
                    ans.append([i, t])
                    print(i, t)
                    break
                cur_lambda_sum += lambda_list[i]
    return cur_ans, ans


def integral_lambda(t, k, M, mu, A, w, cur_ans):
    ret = mu[k] * t
    for i in range(M):
        for j in range(len(cur_ans[i])):
            ret += A[k][i] / w * (1 - np.exp(-w * (t - cur_ans[i][j]))) * np.greater_equal(t, cur_ans[i][j])
    return ret

M = int(input())
A = []
for i in range(M):
    A.append(list(map(float, input().rstrip().split())))
w = float(input())
mu = []
for i in range(M):
    mu.append(float(input()))
N = int(input())

cur_ans, ans = multi_hawkes(A, mu, w, M, N)

for i in range(M):
    plt.figure(i)
    x = np.linspace(0, ans[29][1], 100000)
    y = []
    for xx in x:
        y.append(calc_lambda(xx, i, M, mu, A, w, cur_ans))
    plt.plot(x, y, label='intensity')

    x = []
    for j in range(30):
        x.append(ans[j][1])
    x = np.array(x)
    y = []
    for xx in x:
        y.append(calc_lambda(xx, i, M, mu, A, w, cur_ans))
    plt.step(x, y, linestyle='--', where="post", label='intensity maximum')
    y = [0.5 for _ in range(30)]
    plt.scatter(x, y, marker="^", label='event')
    plt.legend()
    plt.xlabel("Time")
    plt.show()

for i in range(M):
    plt.figure(i)
    plt.clf()
    y = integral_lambda(np.array(cur_ans[i]), i, M, mu, A, w, cur_ans)

    for j in range(len(y) - 1, 0, -1):
        y[j] = y[j] - y[j - 1]
    print(y)
    x = np.linspace(1, len(y), len(y))
    x = -np.log(1 - (x - 0.5) / len(y))
    y.sort()
    plt.scatter(x, y, linewidth=0.1)

    x = np.linspace(0, x[-1], 1000)
    y = 1 * x
    plt.plot(x, y)
    plt.xlabel('quantiles of exponential distribution')
    plt.ylabel('quantiles of input sample')
    plt.show()