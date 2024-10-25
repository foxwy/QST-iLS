# -*- coding: utf-8 -*-
# @Author: foxwy
# @Date:   2021-04-30 09:48:23
# @Last Modified by:   yong
# @Last Modified time: 2024-08-25 16:33:04
# @Function: Provide some of the most basic functions

import os
import numpy as np
import torch
from scipy.special import comb
from scipy.linalg import null_space
from time import perf_counter


def create_file_name(paths, names, root="results/"):
    file_path = root + "/".join(paths) + "/"
    if os.path.isdir(file_path):
        print('result dir exists, is: ' + file_path)
    else:
        os.makedirs(file_path)
        print('result dir not exists, has been created, is: ' + file_path)

    file_name = ""
    for name in names:
        file_name += (str(name) + "_")
    file_name = file_name[:-1]
    file_name += ".npy"

    return file_path + file_name


def save_file(file, paths, names, root="results/"):
    file_name = create_file_name(paths, names, root)
    np.save(file_name, file)


def create_unitary(v):
    """
    Construct a unitary matrix U given a unit column vector v

    Args:
        v (array): Pure state matrix (unit column vector).

    Returns:
        unitary matrix U = [v, Nullspace(v^\dagger)]
    """
    v /= np.linalg.norm(v)
    U = np.hstack((v, null_space(v.T.conj())))

    return U


def HammingDistance(x, y):
    x = int(x, 2)
    y = int(y, 2)
    xor = x ^ y
    distance = 0
    while xor:
        distance = distance + 1
        xor = xor & (xor - 1)

    return distance


def ten_to_k(num, k, N) -> list:
    """
    Convert decimal ``num`` to ``k`` decimal and complementary

    Args:
        num: Decimal numbers.
        k: k decimal.
        N: Total number of digits.

    Returns:
        Converted k decimal list.

    Examples::
        >>> ten_to_k(10, 2, 5)
        >>> [0, 1, 0, 1, 0]
        >>> ten_to_k(10, 4, 5)
        >>> [0, 0, 0, 2, 2]
    """
    transfer_num = []
    if num > k**N - 1:  # error num
        print('please input the right number!')
    else:
        while num != 0:
            num, a = divmod(num, k)
            transfer_num.append(a)
        transfer_num = transfer_num[::-1]
        if len(transfer_num) != N:
            transfer_num = [0] * (N - len(transfer_num)) + transfer_num

    return transfer_num


def median_of_mean(data_list, K):
    assert len(data_list) % K == 0

    groups = np.split(np.array(data_list), K)
    median_list = []
    for group_i in groups:
        mean_i = np.mean(group_i)
        median_list.append(mean_i)

    return np.median(median_list)


def mea_sqrt_W(state, j, k):
    """Pauli expection values for W state"""

    if sum(j) == 0:
        j1 = np.nonzero(k)[0]
        p_cal = (1 - 2 * (state[j1].conj().T @ state[j1])[0, 0])
    elif sum(j) == 2:
        j1 = np.nonzero(j)[0]
        me = np.array(j) @ np.array(k)
        if me % 2 == 0:
            p_cal = (2 * (state[j1[0]].conj() * state[j1[1]]).real)[0]
        else:
            if k[j1][0] == 0 and k[j1][1] == 1:
                p_cal = (2 * (state[j1[0]].conj() * state[j1[1]]).imag)[0]
            else:
                p_cal = (-2 * (state[j1[0]].conj() * state[j1[1]]).imag)[0]
    else:
        p_cal = 0

    return p_cal.real  # / np.sqrt(2**self.n_qubits)


def mea_sqrt_GHZ(state, n_qubits, j, k):
    """Pauli expection values for GHZ state"""

    if sum(j) == 0:
        p_cal = state[0, 0] * state[0, 0].conj() + (-1)**sum(k) * \
            state[-1, 0] * state[-1, 0].conj()
    elif sum(j) == n_qubits:
        p_cal = state[0, 0] * state[-1, 0].conj() * (1j)**sum(k) + \
            state[0, 0].conj() * state[-1, 0] * (-1j)**sum(k)
    else:
        p_cal = 0

    return p_cal.real  # / np.sqrt(2**self.n_qubits)


def get_qw(n_qubits):
    qw = np.zeros(n_qubits + 1)
    for i in range(n_qubits + 1):
        qw[i] = comb(n_qubits, i) * (n_qubits - 2 * i)**2 / \
            (n_qubits * 2**n_qubits)

    return qw / sum(qw)


def sample_W(n_qubits):
    """samping Pauli operators for W state, refer to [Direct Fidelity Estimation from Few Pauli Measurements]"""

    x = np.random.rand()
    if x < 1 / n_qubits:
        j = np.zeros(n_qubits, dtype=int)

        qw = get_qw(n_qubits)
        w = np.random.choice(n_qubits + 1, replace=True, p=qw)
        if w == 0:
            k = np.zeros(n_qubits, dtype=int)
        else:
            idx = np.random.choice(n_qubits, size=(w), replace=False)
            k = np.zeros(n_qubits, dtype=int)
            k[idx] = 1
    else:
        idx = np.random.choice(n_qubits, size=(2), replace=False)
        idx.sort()
        j = np.zeros(n_qubits, dtype=int)
        j[idx] = 1

        k1 = np.random.choice([0, 1], size=(n_qubits - 1), replace=True)
        k = np.zeros(n_qubits, dtype=int)
        k[idx[0]] = k1[0]
        if k1[0] == 0:
            k[idx[1]] = 0
        else:
            k[idx[1]] = 1

        if n_qubits > 2:
            k[:idx[0]] = k1[1:(idx[0]+1)]
            k[(idx[1]+1):] = k1[idx[1]:]

        # k = np.random.choice([0, 1], size=(n_qubits), replace=True)

    return j.astype(int), k.astype(int)


def sample_GHZ(n_qubits):
    """samping Pauli operators for GHZ state, refer to [Direct Fidelity Estimation from Few Pauli Measurements]"""

    x = np.random.rand()
    if x < 0.5:
        j = np.zeros(n_qubits)

        k_n = np.random.choice(np.arange(n_qubits//2 + 1)
                               * 2, size=1, replace=True)[0]
        idx = np.random.choice(n_qubits, size=(k_n), replace=False)
        k = np.zeros(n_qubits)
        k[idx] = 1
    else:
        j = np.ones(n_qubits)

        k_n = np.random.choice(np.arange(n_qubits//2 + 1)
                               * 2, size=1, replace=True)[0]
        idx = np.random.choice(n_qubits, size=(k_n), replace=False)
        k = np.zeros(n_qubits)
        k[idx] = 1

    return j.astype(int),  k.astype(int)


def data_handle(rho, sigma):
    r = rho.shape[1]
    p_f = 0
    if r == 1:
        rho /= torch.norm(rho)
        rho_t = rho @ rho.conj().T
        p_f = 1
    else:
        rho_t = rho / torch.trace(rho)

    s = sigma.shape[1]
    if s == 1:
        sigma /= torch.norm(sigma)
        sigma_t = sigma @ sigma.conj().T
        p_f = 1
    else:
        sigma_t = sigma / torch.trace(sigma)

    return rho_t, sigma_t, p_f


def cal_HS(rho, sigma):
    if isinstance(rho, np.ndarray):
        rho = torch.tensor(rho).to(torch.complex64)
        sigma = torch.tensor(sigma).to(torch.complex64)

    rho_t, sigma_t, _ = data_handle(rho, sigma)
    hs = 0.5 * torch.trace((rho_t - sigma_t) @ (rho_t - sigma_t)).real

    return hs.item()


def cal_IF(rho, sigma):
    if isinstance(rho, np.ndarray):
        rho = torch.tensor(rho).to(torch.complex64)
        sigma = torch.tensor(sigma).to(torch.complex64)
        
    rho_t, sigma_t, p_f = data_handle(rho, sigma)

    if p_f == 1:
        IF = 1 - torch.trace(rho_t @ sigma_t)
    else:
        rho_tmp = torch.matmul(rho_t, sigma_t)
        eigenvalues = torch.linalg.eigvals(rho_tmp)  # low

        sqrt_eigvals = torch.sqrt(torch.abs(eigenvalues))
        # trace(sqrtm(sqrtm(self.rho_star).dot(rho).dot(sqrtm(self.rho_star))))**2
        IF = 1 - torch.sum(sqrt_eigvals)**2

    return abs(IF.real.item())


def mean_remove_min_max(data_list):

    if len(data_list) == 0:
        return 0

    if len(data_list) > 2:
        data_list.remove(min(data_list))
        data_list.remove(max(data_list))
        average_data = float(sum(data_list)) / len(data_list)
        return average_data

    elif len(data_list) <= 2:
        average_data = float(sum(data_list)) / len(data_list)
        return average_data


def shuffle_forward(rho, dims):
    """
    To transpose the density matrix, see
    the matlab version in paper ``Superfast maximum likelihood reconstruction for quantum tomography``,
    this is the [numpy] version we implemented.
    """
    N = len(dims)
    rho = rho.T
    rho = rho.reshape(np.concatenate((dims, dims), 0))
    ordering = np.reshape(np.arange(2*N).reshape(2, -1).T, -1)
    rho = np.transpose(rho, ordering)
    return rho


def qmt(X, operators, allow_negative=False):
    """
    Simplifying the computational complexity of mixed state measurements using the product structure of POVM, see 
    the matlab version in paper ``Superfast maximum likelihood reconstruction for quantum tomography``,
    this is the [numpy] version we implemented.

    Args:
        X (array): Density matrix.
        operators (list, [array]): The set of single-qubit measurement, such as [M1, M2, ....], the size of M is (k, 2, 2).
        allow_negative (bool): Flag for whether to set the calculated value negative to zero.
    """
    if not isinstance(operators, list):
        operators = [operators]

    N = len(operators)  # qubits number
    Ks = np.zeros(N, dtype=int)
    Ds = np.zeros(N, dtype=int)
    for i in range(N):
        dims = operators[i].shape
        Ks[i] = dims[0]
        Ds[i] = dims[1]
        operators[i] = operators[i].reshape((Ks[i], Ds[i]**2))

    X = shuffle_forward(X, Ds[::-1])
    for i in range(N - 1, -1, -1):
        P = operators[i]
        X = X.reshape(-1, Ds[i]**2)
        X = np.matmul(P, X.T)

    P_all = np.real(X.reshape(-1))
    if not allow_negative:
        P_all = np.maximum(P_all, 0)
        P_all /= np.sum(P_all)

    return P_all


def shuffle_adjoint(R, dims):
    """
    To transpose the density matrix, see
    the matlab version in paper ``Superfast maximum likelihood reconstruction for quantum tomography``,
    this is the [numpy] version we implemented.
    """
    N = len(dims)
    R = R.reshape(np.concatenate((dims, dims), 0))
    ordering = np.arange(2 * N).reshape(-1, 2).T.reshape(-1)
    R = np.transpose(R, ordering)
    R = R.reshape(np.prod(dims), np.prod(dims))

    return R


def qmt_matrix(coeffs, operators):
    """
    Simplifying the computational complexity of mixed state measurement operator mixing using the product structure of POVM, see 
    the matlab version in paper ``Superfast maximum likelihood reconstruction for quantum tomography``,
    this is the [numpy] version we implemented.

    Args:
        coeffs (array): Density matrix.
        operators (list, [array]): The set of single-qubit measurement, such as [M1, M2, ....], the size of M is (k, 2, 2).
        allow_negative (bool): Flag for whether to set the calculated value negative to zero.

    Examples::
        >>> M = numpy.array([a, b, c, d])
        >>> qmt_matrix([1, 2, 3, 4], [M])
        >>> 1 * a + 2 * b + 3 * c + 4 * d
        a, b, c, d is a matrix.
    """
    if not isinstance(operators, list):
        operators = [operators]

    X = np.array(coeffs)
    N = len(operators)  # qubits number
    Ks = np.zeros(N, dtype=int)
    Ds = np.zeros(N, dtype=int)
    for i in range(N):
        dims = operators[i].shape
        Ks[i] = dims[0]
        Ds[i] = dims[1]
        operators[i] = operators[i].reshape((Ks[i], Ds[i]**2))

    for i in range(N - 1, -1, -1):
        P = operators[i]
        X = X.reshape(-1, Ks[i])
        X = X.dot(P)
        X = X.T

    X = shuffle_adjoint(X, Ds[::-1])
    X = 0.5 * (X + X.T.conj())

    return X


def fidelity_pure(X, operators):
    """
    Simplifying the computational complexity of pure state measurements using 
    the product structure of POVM, this is the [numpy] version we implemented.

    Args:
        X (array): Pure state matrix (unit column vector).
        operators (list, [array]): The set of single-qubit measurement, such as [M1, M2, ....], the size of M is (2, 2).
    """
    if not isinstance(operators, list):
        operators = [operators]

    X = np.array(X)
    N = len(operators)  # qubits number
    Ks = np.zeros(N, dtype=int)
    Ds = np.zeros(N, dtype=int)
    for i in range(N):
        dims = operators[i].shape
        Ks[i] = 1
        Ds[i] = dims[1]
        if i < N - 1:
            operators[i] = operators[i].reshape((Ks[i], Ds[i]**2))

    X = X.reshape(-1, 2).T
    X_T = X.T.conjugate()
    X_k = X_T.dot(operators[-1]).dot(X)

    X_k = shuffle_forward_pure(X_k, Ds[:-1])
    for i in range(N - 2, -1, -1):
        P = operators[i]
        X_k = X_k.reshape(-1, Ds[i]**2).T
        X_k = P.dot(X_k)

    P_all = X_k

    return P_all


def shuffle_forward_pure(rho, dims):
    """
    To transpose the density matrix, see
    the matlab version in paper ``Superfast maximum likelihood reconstruction for quantum tomography``,
    this is the [numpy] version we implemented.
    """
    N = len(dims)
    rho = rho.reshape(np.concatenate((dims, dims), 0))
    ordering = np.reshape(np.arange(2*N).reshape(2, -1).T, -1)
    rho = np.transpose(rho, ordering)

    return rho


def shuffle_forward_torch(rho, dims):
    """
    To transpose the density matrix, see
    the matlab version in paper ``Superfast maximum likelihood reconstruction for quantum tomography``,
    this is the [torch] version we implemented.
    """
    N = len(dims)
    rho = rho.T
    rho = rho.reshape(tuple(torch.cat([dims, dims], 0)))
    ordering = torch.reshape(torch.arange(2*N).reshape(2, -1).T, (1, -1))[0]
    rho = rho.permute(tuple(ordering))
    return rho


def qmt_torch(X, operators, allow_negative=False):
    """
    Simplifying the computational complexity of mixed state measurements using the product structure of POVM, see 
    the matlab version in paper ``Superfast maximum likelihood reconstruction for quantum tomography``,
    this is the [torch] version we implemented.

    Args:
        X (tensor): Density matrix.
        operators (list, [tensor]): The set of single-qubit measurement, such as [M1, M2, ....], the size of M is (k, 2, 2).
        allow_negative (bool): Flag for whether to set the calculated value negative to zero.
    """
    if not isinstance(operators, list):
        operators = [operators]

    N = len(operators)  # qubits number
    Ks = torch.zeros(N, dtype=torch.int)
    Ds = torch.zeros(N, dtype=torch.int)
    for i in range(N):
        dims = operators[i].shape
        Ks[i] = dims[0]
        Ds[i] = dims[1]
        operators[i] = operators[i].reshape((Ks[i], Ds[i]**2))

    X = shuffle_forward_torch(X, Ds)
    if N > 12:  # torch does not support more dimensional operations
        X = X.cpu()
    X = X.reshape(-1, Ds[-1]**2)
    if N > 12:
        X = X.to(operators[0].device)

    for i in range(N - 1, -1, -1):
        P = operators[i]
        X = torch.matmul(P, X.T)

        if i > 0:
            X = X.reshape(-1, Ds[i]**2)

    P_all = torch.real(X.reshape(-1))
    if not allow_negative:
        P_all = torch.maximum(P_all, torch.tensor(0))
        P_all /= torch.sum(P_all)
    return P_all


def shuffle_adjoint_torch(R, dims):
    """
    To transpose the density matrix, see
    the matlab version in paper ``Superfast maximum likelihood reconstruction for quantum tomography``,
    this is the [torch] version we implemented.
    """
    N = len(dims)
    R = R.reshape(tuple(torch.cat([dims, dims], 0)))
    ordering = torch.arange(2 * N).reshape(-1, 2).T.reshape(-1)
    R = R.permute(tuple(ordering))
    R = R.reshape(torch.prod(dims), torch.prod(dims))

    return R


def qmt_matrix_torch(X, operators):
    """
    Simplifying the computational complexity of mixed state measurement operator mixing using the product structure of POVM, see 
    the matlab version in paper ``Superfast maximum likelihood reconstruction for quantum tomography``,
    this is the [torch] version we implemented.

    Args:
        coeffs (tensor): Density matrix.
        operators (list, [tensor]): The set of single-qubit measurement, such as [M1, M2, ....], the size of M is (k, 2, 2).
        allow_negative (bool): Flag for whether to set the calculated value negative to zero.

    Examples::
        >>> M = torch.tensor([a, b, c, d])
        >>> qmt_matrix(torch.tensor([1, 2, 3, 4]), [M])
        >>> 1 * a + 2 * b + 3 * c + 4 * d
        a, b, c, d is a matrix.
    """
    if not isinstance(operators, list):
        operators = [operators]

    N = len(operators)  # qubits number
    Ks = torch.zeros(N, dtype=torch.int)
    Ds = torch.zeros(N, dtype=torch.int)
    for i in range(N):
        dims = operators[i].shape
        Ks[i] = dims[0]
        Ds[i] = dims[1]
        operators[i] = operators[i].reshape((Ks[i], Ds[i]**2))

    for i in range(N - 1, -1, -1):
        P = operators[i]
        X = X.reshape(-1, Ks[i])
        X = torch.matmul(X, P)
        X = X.T

    X = shuffle_adjoint_torch(X, Ds.flip(dims=[0]))
    X = 0.5 * (X + X.T.conj())
    return X


def proj_to_sum_one(x, fl):
    """
    project x in to a unit vector according to fl
    """
    if fl == 1:
        x = x / torch.sum(x)

    elif fl == 2:
        x = x + (1 - torch.sum(x)) / len(x)

    return x


def proj_trans_S(rho):
    """
    Transformation of Hermitian matrix to nearest density matrix, F projection state-mapping method, 
    see paper ``Efficient method for computing the maximum-likelihood quantum state from 
    measurements with additive gaussian noise``,
    this is [torch] version we implemented.
    """
    eigenvalues, eigenvecs = torch.linalg.eigh(
        rho)  # eigenvalues[i], eigenvecs[:, i]
    device = rho.device

    eigenvalues = proj_to_sum_one(eigenvalues, 2)  # unit sum

    u, _ = torch.sort(eigenvalues)
    csu = torch.cumsum(u, 0)
    csu0 = torch.zeros_like(csu).to(device)
    csu0[1:] = csu[:-1]
    t = csu0 / torch.arange(len(u), 0, -1).to(device)
    idx = torch.nonzero(u + t > 0)[0, 0]
    eigenvalues = torch.maximum(eigenvalues + t[idx], torch.tensor(0))

    A = eigenvecs * abs(eigenvalues)
    rho = torch.matmul(A, eigenvecs.T.conj())

    A = eigenvecs * abs(eigenvalues)**0.5
    rho_sqrt = torch.matmul(A, eigenvecs.T.conj())

    return rho, rho_sqrt

def proj_trans_P(rho):
    eigenvalues, eigenvecs = torch.linalg.eigh(
        rho)  # eigenvalues[i], eigenvecs[:, i]
    device = rho.device

    eigenvalues[eigenvalues < 0] = 0

    A = eigenvecs * eigenvalues
    rho = torch.matmul(A, eigenvecs.T.conj())

    A = eigenvecs * eigenvalues**0.5
    rho_sqrt = torch.matmul(A, eigenvecs.T.conj())

    return rho, rho_sqrt


def proj_trans_pure(rho):
    lamda = torch.linalg.eigh(rho)
    index = torch.argmax(abs(lamda[0]))
    A = lamda[1][:, index]
    A /= torch.norm(A)
    A = A.reshape(-1, 1)

    return A @ A.conj().T


# --------------------main--------------------
if __name__ == '__main__':
    '''u1 = np.array([[1, 2], [4, 3j]])
    u2 = np.array([[3j, 2], [1, 4]])
    u3 = np.array([[4, 4j], [3, 1]])
    u4 = np.array([[4, 6j], [7, 1]])
    u5 = np.array([[4, 6j], [7, 2]])

    state = np.random.rand(2**8, 1) + 1j * np.random.rand(2**8, 1)
    t1 = time.perf_counter()
    ideal = state.conj().T @ np.kron(u1, np.kron(u2, np.kron(u3, np.kron(u4, np.kron(u5, np.kron(u5, np.kron(u5, u5))))))) @ state
    t2 = time.perf_counter()
    real = fidelity_pure(state, [u1, u2, u3, u4, u5, u5, u5, u5])
    print(time.perf_counter() - t2, t2 - t1)
    print(ideal, real)'''

    '''
    u1 = np.array([[1, 2], [4, 3j]])
    u2 = np.array([[3j, 2], [1, 4]])

    t1 = time.perf_counter()
    print(np.kron(u1, np.kron(u2, u2)))
    t2 = time.perf_counter()
    u1_t = u1.reshape(-1)
    u2_t = u2.reshape(-1)
    print(np.kron(u1_t, np.kron(u2_t, u2_t)))
    print(time.perf_counter() - t2, t2 - t1)
    '''

    '''
    v = np.array([1j-2, 4-2j, 3+9j, 4j+2]).reshape(-1, 1)
    v /= np.linalg.norm(v)
    print(v)
    print(create_unitary(v)[:, 0])'''

    t1 = perf_counter()
    n_qubits = 14
    pauli_basis = torch.tensor([[[1, 0], [0, 0]],
                                [[1, 0], [0, 0]],
                                [[1, 0], [0, 0]],
                                [[0.5, 0.5j], [-0.5j, 0.5]]]).to(torch.complex64).cuda()

    A = torch.randn((2**n_qubits, 2**n_qubits)).to(torch.complex64).cuda()
    rho = A @ A.conj().T
    PA = qmt_torch(rho, [pauli_basis] * n_qubits, allow_negative=False)
    print("time", perf_counter() - t1, len(PA))
