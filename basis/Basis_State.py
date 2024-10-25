# -*- coding: utf-8 -*-
# @Author: foxwy
# @Date:   2021-01-19 15:38:06
# @Last Modified by:   yong
# @Last Modified time: 2024-08-25 19:57:04
# @Function: Quantum state and quantum measurment

import numpy as np
from numpy.linalg import qr


class State():
    """
    Some basic quantum pure states, 
    including Pauli matrices, |0>, |1>, GHZ-class states, W-class states.

    Examples::
        >>> st = State()
        >>> GHZ_state = st.Get_GHZ(2)
        >>> [[0.70710678+0.j        ]
            [0.        +0.70710678j]]
    """

    def __init__(self):
        # Pauli matrices
        self.I = np.array([[1, 0], [0, 1]])
        self.X = np.array([[0, 1], [1, 0]])
        self.s1 = self.X
        self.Y = np.array([[0, -1j], [1j, 0]])
        self.s2 = self.Y
        self.Z = np.array([[1, 0], [0, -1]])
        self.s3 = self.Z

        # state
        self.state0 = np.matrix([[1], [0]])
        self.state1 = np.matrix([[0], [1]])
        self.state01 = 1 / np.sqrt(2) * (self.state0 + self.state1)

    def Get_GHZ_s(self, N, w_s=None):
        """
        GHZ-class states:
            |GHZ> = 1 / sqrt(2) * (|0...0> + alpha * |1...1>), 
            alpha from {1, -1, j, -j}
        """
        if w_s is None:
            choice = [1.0]  #, -1.0, 1.0j, -1.0j]
            w_s = np.random.choice(choice, 1, replace=True)[0]

        GHZ_state = 1 / np.sqrt(2) * np.array([1.0, w_s])
        GHZ_state = GHZ_state.reshape(-1, 1)

        return GHZ_state

    def Get_state_from_array(self, array):
        """Calculate the corresponding pure state according to the given array"""
        st = {0: self.state0, 1: self.state1}
        State = st[array[0]]
        for i in array[1:]:
            State = np.kron(State, st[i])
        return State

    def Get_W(self, N):
        I_array = np.identity(N)
        W_state = 0
        for row in I_array:
            W_state += self.Get_state_from_array(row)
        W_state = 1 / np.sqrt(N) * W_state

        return W_state

    def Get_W_s(self, N, w_s=None):
        """
        W-class states:
            |W> = 1 / sqrt(N) * (alpha_1 * |10...0> + ... + alpha_i * |00..1.0> + alpha_N * |00...1>), 
            alpha_i from {1, -1, j, -j}
        """
        if w_s is None:
            choice = [1.0]  #, -1.0, 1.0j, -1.0j]
            w_s = np.random.choice(choice, N, replace=True)

        w_s /= np.linalg.norm(w_s)
        w_s = w_s.reshape(-1, 1)

        return w_s

    def Get_Haar_random(self, N):
        d = 2**N
        A, B = np.random.normal(size=(d, d)), np.random.normal(size=(d, d))
        Z = A + 1j * B

        Q, R = qr(Z)
        diag = np.diagonal(R)
        Lambda = np.diag(np.sign(diag))
        
        rho = np.zeros((d, 1))
        rho[0] = 1.0

        return np.dot(Q, Lambda) @ rho


class Mea_basis(State):
    """
    Defining Quantum Measurement, include POVM and Pauli measurement.

    Examples::
        >>> Me = Mea_basis(basis='Tetra4')
        >>> M = Me.M
        >>> [[[ 0.39433756+0.j          0.14433756-0.14433756j]
              [ 0.14433756+0.14433756j  0.10566244+0.j        ]]

             [[ 0.39433756+0.j         -0.14433756+0.14433756j]
              [-0.14433756-0.14433756j  0.10566244+0.j        ]]

             [[ 0.10566244+0.j         -0.14433756-0.14433756j]
              [-0.14433756+0.14433756j  0.39433756+0.j        ]]

             [[ 0.10566244+0.j          0.14433756+0.14433756j]
              [ 0.14433756-0.14433756j  0.39433756+0.j        ]]]
    """
    def __init__(self, basis='Tetra'):
        """
        Selection of different measurement bases.

        Args:
            basis: ['Tetra'], ['Tetra4'], ['6Pauli'], ['4Pauli'], ['Pauli'], 
                   ['Pauli_rebit'], [Pauli_6'], ['Trine'], ['Psi2'], ['Pauli_normal'].

        Variables: 
            self.K: The number of POVM elements.
            slef.M: POVM or Pauli operators.
        """
        super().__init__()
        self.basis = basis
        self.Get_basis()

    def Get_basis(self):
        """POVM and Pauli operators"""
        if self.basis == 'Tetra':
            self.K = 4

            self.M = np.zeros((self.K, 2, 2), dtype=np.complex64)

            v1 = np.array([0, 0, 1.0])
            self.M[0, :, :] = 1.0 / 4.0 * \
                (self.I + v1[0] * self.s1 + v1[1] * self.s2 + v1[2] * self.s3);

            v2 = np.array([2.0 * np.sqrt(2.0) / 3.0, 0.0, -1.0 / 3.0])
            self.M[1, :, :] = 1.0 / 4.0 * \
                (self.I + v2[0] * self.s1 + v2[1] * self.s2 + v2[2] * self.s3);

            v3 = np.array(
                [-np.sqrt(2.0) / 3.0, np.sqrt(2.0 / 3.0), -1.0 / 3.0])
            self.M[2, :, :] = 1.0 / 4.0 * \
                (self.I + v3[0] * self.s1 + v3[1] * self.s2 + v3[2] * self.s3);

            v4 = np.array(
                [-np.sqrt(2.0) / 3.0, -np.sqrt(2.0 / 3.0), -1.0 / 3.0])
            self.M[3, :, :] = 1.0 / 4.0 * \
                (self.I + v4[0] * self.s1 + v4[1] * self.s2 + v4[2] * self.s3);

        elif self.basis == 'Tetra4':
            self.K = 4

            self.M = np.zeros((self.K, 2, 2), dtype=np.complex64)

            self.M[0, :, :] = 0.25 * (self.I + (self.X + self.Y + self.Z) / np.sqrt(3))
            self.M[1, :, :] = 0.25 * (self.I + (-self.X - self.Y + self.Z) / np.sqrt(3))
            self.M[2, :, :] = 0.25 * (self.I + (-self.X + self.Y - self.Z) / np.sqrt(3))
            self.M[3, :, :] = 0.25 * (self.I + (self.X - self.Y - self.Z) / np.sqrt(3))

        elif self.basis == '6Pauli':
            self.K = 6

            self.M = np.zeros((self.K, 2, 2), dtype=np.complex64)

            self.M[0, :, :] = (self.I + self.X) / 6
            self.M[1, :, :] = (self.I - self.X) / 6
            self.M[2, :, :] = (self.I + self.Y) / 6
            self.M[3, :, :] = (self.I - self.Y) / 6
            self.M[4, :, :] = (self.I + self.Z) / 6
            self.M[5, :, :] = (self.I - self.Z) / 6

        elif self.basis == '4Pauli':  # different from paper
            self.K = 4

            self.M = np.zeros((self.K, 2, 2), dtype=np.complex64)

            self.M[0, :, :] = 1.0 / 3.0 * np.array([[1, 0], [0, 0]])
            self.M[1, :, :] = 1.0 / 6.0 * np.array([[1, 1], [1, 1]])
            self.M[2, :, :] = 1.0 / 6.0 * np.array([[1, -1j], [1j, 1]])
            self.M[3, :, :] = 1.0 / 3.0 * (np.array([[0, 0], [0, 1]]) +
                                           0.5 * np.array([[1, -1], [-1, 1]])
                                           + 0.5 * np.array([[1, 1j], [-1j, 1]]))

        elif self.basis == 'Pauli':
            self.K = 6
            Ps = np.array([1. / 3., 1. / 3., 1. / 3.,
                           1. / 3., 1. / 3., 1. / 3.])
            self.M = np.zeros((self.K, 2, 2), dtype=np.complex64)
            theta = np.pi / 2.0
            self.M[0, :, :] = Ps[0] * self.pXp(theta, 0.0)
            self.M[1, :, :] = Ps[1] * self.mXm(theta, 0.0)
            self.M[2, :, :] = Ps[2] * self.pXp(theta, np.pi / 2.0)
            self.M[3, :, :] = Ps[3] * self.mXm(theta, np.pi / 2.0)
            self.M[4, :, :] = Ps[4] * self.pXp(0.0, 0.0)
            self.M[5, :, :] = Ps[5] * self.mXm(0, 0.0)

        elif self.basis == 'Pauli_rebit':  # X
            self.K = 4
            Ps = np.array([1. / 2., 1. / 2., 1. / 2., 1. / 2.])
            self.M = np.zeros((self.K, 2, 2), dtype=np.complex64)
            theta = np.pi / 2.0
            self.M[0, :, :] = Ps[0] * self.pXp(theta, 0.0)
            self.M[1, :, :] = Ps[1] * self.mXm(theta, 0.0)
            self.M[2, :, :] = Ps[2] * self.pXp(0.0, 0.0)
            self.M[3, :, :] = Ps[3] * self.mXm(0, 0.0)
            self.M = self.M.real

        elif self.basis == 'Pauli_6':
            self.K = 6
            Ps = np.array([1. / 3., 1. / 6., 1. / 2.])
            self.M = np.zeros((self.K, 2, 2), dtype=np.complex64)
            self.M[0, :, :] = Ps[0] * np.array([[1, 0], [0, 0]])
            self.M[1, :, :] = Ps[0] * np.array([[0, 0], [0, 1]])
            self.M[2, :, :] = Ps[1] / 2 * np.array([[1, 1], [1, 1]])
            self.M[3, :, :] = Ps[1] / 2 * np.array([[1, -1], [-1, 1]])
            self.M[4, :, :] = Ps[2] / 2 * np.array([[1, -1j], [1j, 1]])
            self.M[5, :, :] = Ps[2] / 2 * np.array([[1, 1j], [-1j, 1]])

        elif self.basis == 'Trine':
            self.K = 3
            self.M = np.zeros((self.K, 2, 2), dtype=np.complex64)
            phi0 = 0.0
            for k in range(self.K):
                phi = phi0 + (k) * 2 * np.pi / 3.0
                self.M[k, :, :] = 0.5 * (self.I + np.cos(phi)
                                         * self.Z + np.sin(phi) * self.X) * 2 / 3.0
            self.M = self.M.real

        elif self.basis == 'Psi2':
            self.K = 2
            self.M = np.zeros((self.K, 2, 2), dtype=np.complex64)
            self.M[0, 0, 0] = 1
            self.M[1, 1, 1] = 1

        elif self.basis == 'Pauli_normal':  # projective measurement, not POVM
            self.K = 4
            self.M = np.zeros((self.K, 2, 2), dtype=np.complex64)
            self.M[0, :, :] = 1 / np.sqrt(2) * np.array([[1.0, 0.0], [0.0, 1.0]])  # I
            self.M[1, :, :] = 1 / np.sqrt(2) * np.array([[1.0, 0.0], [0.0, -1.0]])  # Z
            self.M[2, :, :] = 1 / np.sqrt(2) * np.array([[0.0, 1.0], [1.0, 0.0]])  # X
            self.M[3, :, :] = 1 / np.sqrt(2) * np.array([[0.0, -1j], [1j, 0.0]])  # Y

        else:
            print(self.basis, 'does not exist!')

    @staticmethod
    def pXp(theta, phi):
        return np.array([[np.cos(theta / 2.0)**2, np.cos(theta / 2.0) * np.sin(theta / 2.0) * np.exp(-1j * phi)],
                         [np.cos(theta / 2.0) * np.sin(theta / 2.0) * np.exp(1j * phi), np.sin(theta / 2.0)**2]])

    @staticmethod
    def mXm(theta, phi):
        return np.array([[np.sin(theta / 2.0)**2, -np.cos(theta / 2.0) * np.sin(theta / 2.0) * np.exp(-1j * phi)],
                         [-np.cos(theta / 2.0) * np.sin(theta / 2.0) * np.exp(1j * phi), np.cos(theta / 2.0)**2]])


if __name__ == "__main__":
    GHZ = State().Get_GHZ(2)
    print(GHZ)