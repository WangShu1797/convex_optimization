import matplotlib.pyplot as plt
import numpy as np

A = np.load('A.npy')
b = np.load('b.npy')
x = np.load('x.npy')

# See the report for the derivation process
def SubGradient(x):
    x_new = x.copy()
    for i, data in enumerate(x):
        if data == 0:
            x_new[i] = np.random.uniform(-1,1)
        else:
            x_new[i] = np.sign(x[i])
    return x_new

def SG(x):
    return 2 * np.dot(A.T, (np.dot(A, x) - b)) + P * SubGradient(x)

A_size = (50, 100)
B_size = 50
x_size = 100
alpha = 0.001
P = 0.001
alpha_k = alpha
i = 1

Xk_record = []
x_opt_dst_steps = []
x_dst_steps = []

Xk = np.zeros(x_size)

while True:
    Xk_record.append(Xk)
    Xk_new = Xk - alpha_k * SG(Xk)
    alpha_k = alpha / i
    i += 1
    if np.linalg.norm(Xk_new - Xk, ord=2) < 1e-5:
        break
    else:
        Xk = Xk_new.copy()

x_opt = Xk_record[-1]
x_real = x.copy()

x_opt_dst = [np.linalg.norm(Xk - x_opt) for Xk in Xk_record]
x_real_dst = [np.linalg.norm(Xk - x_real) for Xk in Xk_record]

plt.title("Distance")
plt.plot(x_opt_dst, label='X-opt-distance')
plt.plot(x_real_dst, label='X-real-distance')
plt.legend()
plt.show()
