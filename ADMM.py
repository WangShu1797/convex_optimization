import matplotlib.pyplot as plt
import numpy as np

A = np.load('A.npy')
b = np.load('b.npy')
x = np.load('x.npy')

A_size = (50, 100)
b_size = 50
x_size = 100

P = 0.01
#P = 0.001
#P = 0.0001
c = 0.005
Xk = np.zeros(x_size)
Yk = np.zeros(x_size)
Zk = np.zeros(x_size)

x_opt_dst = []
x_real_dst = []
Xk_record = []
# See the report for the derivation process
while True:
    Xk_record.append(Xk)
    
    Xk_new = np.dot(
        np.linalg.inv(np.dot(A.T, A) + c * np.eye(x_size, x_size)),
        c*Yk + Zk + np.dot(A.T, b)
    )
    Yk_new = np.zeros(x_size)
    # the same update method for y's 1st-order norm  
    for i in range(x_size):
        if Xk_new[i] - Zk[i] / c < - P / c:
            Yk_new[i] = Xk_new[i] - Zk[i] / c + P / c
        elif Xk_new[i] - Zk[i] / c > P / c:
            Yk_new[i] = Xk_new[i] - Zk[i] / c - P / c
        else:
            continue

    Zk_new = Zk + c * (Yk_new - Xk_new)

    if np.linalg.norm(Xk_new - Xk, ord=2) < 1e-5:
        break
    else:
        Xk = Xk_new.copy()
        Yk = Yk_new.copy()
        Zk = Zk_new.copy()

print(Xk)
print(x)

# plot the distance between each Xk and x_real/x_optimal
x_opt = Xk_record[-1]
x_real = x.copy()

x_opt_dst = [np.linalg.norm(Xk - x_opt) for Xk in Xk_record]
x_real_dst = [np.linalg.norm(Xk - x_real) for Xk in Xk_record]

plt.title("Distance")
plt.plot(x_opt_dst, label='X-opt-distance')
plt.plot(x_real_dst, label='X-real-distance')
plt.legend()
plt.show()
