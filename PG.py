import numpy as np

A = np.load('A.npy')
x = np.load('x.npy')
b = np.load('b.npy')
print(A.shape)
print(x.shape)
alpha = 0.001
P_half = 0.01
#P_half = 0.001
#P_half = 0.1
Xk = np.zeros(x_size)#initialize x_k as 0
zero = np.zeros(x_size)
Xk_record=[]
while True:
    Xk_record.append(Xk)
    # Update the target function in two parts
    # the first part is differentiable
    # the second part use adjacent point projection, more details about this are in my assignment report
    Xk_half = Xk - alpha * np.dot(A.T, np.dot(A, Xk) - b)
    #print(Xk_half.shape)
    Xk_new = zero.copy()
    for i in range(x_size):
        if Xk_half[i] < - alpha * P_half:
            Xk_new[i] = Xk_half[i] + alpha * P_half
        elif Xk_half[i] > alpha * P_half:
            Xk_new[i] = Xk_half[i] - alpha * P_half
        else:
          continue
    # when the x value does not update， stop
    if np.linalg.norm(Xk_new - Xk, ord=2) < 1e-5:
        break
    else:
        Xk = Xk_new.copy()

x_real = x
x_opt = Xk_record[-1]
x_opt_dis = [np.linalg.norm(i - x_opt, ord=2) for i in Xk_record]
x_real_dis = [np.linalg.norm(i - x_real, ord=2) for i in Xk_record]
plt.title("Distance")
plt.plot(x_opt_dis, label='x-opt-distance')
plt.plot(x_real_dis, label='x-real-distance')
plt.legend()
plt.show()

print(Xk_record)
#print(x_real_dis)
#print(Xk)
#print(x)