import torch
from time import perf_counter
import numpy as np

'''
t1 = perf_counter()
a = torch.randn(20000, 20000).cuda()
b = torch.randn(20000, 1).cuda()

t1 = perf_counter()
c = a @ b
t2 = perf_counter()
print(t2 - t1)

t1 = perf_counter()
c = torch.einsum("ij, jk -> ik", a, b)
t2 = perf_counter()
print(t2 - t1)

t1 = perf_counter()
c = torch.matmul(a, b)
t2 = perf_counter()
print(t2 - t1)

t1 = perf_counter()
c = a @ b
t2 = perf_counter()
print(t2 - t1)

t1 = perf_counter()
c = torch.einsum("ij, jk -> ik", a, b)
t2 = perf_counter()
print(t2 - t1)

t1 = perf_counter()
c = torch.matmul(a, b)
t2 = perf_counter()
print(t2 - t1)
'''


def hammingDistance(x, y):
    # x = int(x, 2)
    # y = int(y, 2)
    xor = x ^ y
    distance = 0
    while xor:
        distance = distance + 1
        xor = xor & (xor - 1)

    return distance


def get_hdis(n_qubits=7):
    n_qubits = 7
    h_distance = np.zeros((2**n_qubits, 2**n_qubits))
    for i in range(2**n_qubits):
        for j in range(2**n_qubits):
            h_distance[i, j] = (-2)**(-hammingDistance(i, j))

    np.save("measures/rm_temp/h_distance_"+str(n_qubits)+'.npy', h_distance)

    return h_distance


if __name__ == "__main__":
    get_hdis(n_qubits=7)
