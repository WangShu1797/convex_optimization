import numpy as np
import random
A_size = (50, 100)#A 50*100 matrix
x_size = 100#x 100*1 vector
A = np.random.normal(0, 1, A_size)
x = np.zeros(x_size)
index = random.sample(list(range(x_size)), 5)#There are only five non-zero elements in x
for xi in index:
    x[xi] = np.random.randn()# a random number with normal distribution(0,1)
print(x)
e = np.random.normal(0, 0.1, 50)#e 50*1 noise
b = np.dot(A, x) + e#b 50*1 observation value
np.save("A.npy", A)
np.save("x.npy", x)
np.save("b.npy", b)