# -*- coding: utf-8 -*-
# @Author: yong
# @Date:   2023-07-29 14:16:20
# @Last Modified by:   yong
# @Last Modified time: 2024-03-26 07:21:24
# Experiment Appendix

from basis.Basis_State import Mea_basis
from basis.Basic_Function import qmt, qmt_matrix, cal_HS, cal_IF, ten_to_k
import numpy as np
from scipy.special import comb
from tqdm import tqdm
from time import perf_counter

import sys
sys.path.append('..')


class FEQST():

    def __init__(self, args):
        """
        args: experimental setups
        """
        super().__init__()

        self.na_state = args.na_state
        self.n_qubits = args.n_qubits
        self.n_tols = args.n_tols
        self.n_epochs = args.n_epochs
        self.n_shots = args.n_shots
        self.n_meas = args.n_meas

        self.method = args.method
        self.dtype = args.dtype

        self.povm = args.POVM
        self.K = args.K
        self.M = Mea_basis(self.povm).M

        # optimizer parameters
        self.lr = args.lr

    def guide(self, circ, rho_star, rho_init, nonzero_idx):
        """main execute function of QST"""

        # temp variables
        self.epoch = 0
        time_all = 0
        self.circ = circ
        self.rho_star = rho_star
        self.nonzero_idx = nonzero_idx
        self.dim = len(rho_star)

        # observations
        if self.method in ["iLS", "fiLS", "DFE"]:
            self.f_star, self.f_idxs = self.get_observations()
            print('measurement number:', len(self.f_idxs))
            self.f_zeroidxs = list(
                set(np.arange(self.K**self.n_qubits)) - set(self.f_idxs))

        elif self.method == "MLE":
            self.n_shots = round(self.n_tols / self.n_meas)
            self.paulis, self.m_results = self.circ.circuit_simulator(
                meastype='mea_mle', shots=self.n_shots, pauli_num=self.n_meas, pos=self.na_state
            )

        # init
        A = rho_init.copy()
        U = A.copy()

        save_data = {"epoch": [], "time": [], "HS": [], "iFq": [], 'iFqe': 0}
        all_method = {"iLS": self.LS_GD, "fiLS": self.DFE_GD,
                      "DFE": self.DFE_GD, "MLE": self.MLE_GD}
        GD_method = all_method[self.method]

        # learning QST
        if self.method in ["iLS", "MLE"]:
            pbar = tqdm(range(self.n_epochs))

            for _ in pbar:
                self.epoch += 1
                time_b = perf_counter()

                A, U = GD_method(A, U)

                time_all += (perf_counter() - time_b)

                if self.epoch % 10 == 0 or self.epoch == 1:  # save data
                    rho = A @ A.conj().T
                    rho /= np.trace(rho)

                    HS = cal_HS(self.rho_star, rho)
                    IF = cal_IF(self.rho_star, rho)

                    if np.isnan(HS) or np.isnan(IF):
                        break

                    save_data["epoch"].append(self.epoch)
                    save_data["time"].append(time_all)
                    save_data["HS"].append(HS)
                    save_data["iFq"].append(IF)
                    save_data["iFqe"] = IF

                    pbar.set_description(
                        "epoch {:d} | time {:.3f} | HS {:.8f} | IF {:.8f}".
                        format(self.epoch, time_all, HS, IF)
                    )

        else:
            time_b = perf_counter()
            A = GD_method()
            time_all += (perf_counter() - time_b)

            rho = A @ A.conj().T
            rho /= np.trace(rho)

            HS = cal_HS(self.rho_star, rho)
            IF = cal_IF(self.rho_star, rho)

            save_data["epoch"].append(1)
            save_data["time"].append(time_all)
            save_data["HS"].append(HS)
            save_data["iFq"].append(IF)
            save_data["iFqe"] = IF

            print("epoch {:d} | time {:.3f} | HS {:.8f} | IF {:.8f}".format(
                self.epoch, time_all, HS, IF))

        return save_data

    def get_observations(self):
        # sample
        choice_idxs = np.random.choice(np.arange(
            3**self.n_qubits), size=(3**self.n_qubits), replace=False)  # add random

        j_all = []
        k_all = []
        for idx in choice_idxs:
            j = []
            k = []
            pauli = ten_to_k(idx, 3, self.n_qubits)
            for n in range(self.n_qubits):
                if pauli[n] == 0:  # I
                    j.append(0)
                    k.append(1)
                elif pauli[n] == 1:  # X
                    j.append(1)
                    k.append(0)
                else:  # Y
                    j.append(1)
                    k.append(1)

            if self.sample_jud(j, k):
                j_all.append(j)
                k_all.append(k)

            if len(j_all) > self.n_meas - 1:
                break

        j_all, k_all = np.array(j_all), np.array(k_all)

        # extend to Pauli operators contains "I"
        idxs = []
        target_obs = []
        target_locs = []
        jk_all = {-1}
        for i in range(len(j_all)):

            j, k = j_all[i], k_all[i]

            pos_num = self.n_qubits
            for p in range(2**pos_num - 1):
                j_t = j.copy()
                k_t = k.copy()
                t1 = ten_to_k(p, 2, pos_num)
                for m in range(pos_num):
                    if t1[m] == 0:  # X, Y, Z
                        j_t[m] = j[m]
                        k_t[m] = k[m]
                    else:  # I
                        j_t[m] = 0
                        k_t[m] = 0

                jk_s = np.concatenate(
                    [j_t, k_t]) @ 2**np.arange(2 * self.n_qubits)[::-1]
                if jk_s not in jk_all:
                    jk_all.add(jk_s)

                    data = []
                    for n in range(self.n_qubits):
                        data.append(j_t[n] * 2 + k_t[n])
                    data = np.array(data[::-1])
                    # index of Pauli string, from 0 to 4^n - 1
                    idx = data @ self.K**np.arange(self.n_qubits)[::-1]
                    idxs.append(idx)

                    # measure
                    # derandomized shadow
                    ob = []
                    loc = []
                    for m in range(self.n_qubits):
                        if j_t[m] == 1 and k_t[m] == 0:
                            ob.append(1)
                            loc.append(m)
                        elif j_t[m] == 1 and k_t[m] == 1:
                            ob.append(2)
                            loc.append(m)
                        elif j_t[m] == 0 and k_t[m] == 1:
                            ob.append(0)
                            loc.append(m)

                    target_obs.append(ob)
                    target_locs.append(loc)

        all_observables = (target_obs, target_locs)

        # derandomized shadows
        p_rho_star = self.circ.circuit_simulator(
            meastype='mea_pauli_expect',
            shots=self.n_shots,
            all_observables=all_observables,
            n_tols=self.n_tols,
            derandomized=True
        )

        # identity matrix
        p_rho_star = list(p_rho_star)
        if 0 not in idxs:
            idxs.append(0)
            p_rho_star.append(1)

        p_rho_star = np.array(p_rho_star)
        idxs = np.array(idxs)
        p_rho_star_all = np.zeros((self.K**self.n_qubits))
        p_rho_star_all[idxs] = p_rho_star
        p_rho_star_all[0] = 1

        return p_rho_star_all / np.sqrt(2**self.n_qubits), idxs

    def momentum(self, U, G, A, beta=0):
        U = beta * U + G
        A -= self.lr * U

        return A, U

    def LS_GD(self, A, U):
        rho_res = np.zeros((2**self.n_qubits, 1)).astype(self.dtype)
        rho_res[self.nonzero_idx] = A

        PA = qmt(rho_res @ rho_res.conj().T,
                 [self.M] * self.n_qubits, allow_negative=True)
        e = PA - self.f_star

        if self.method in ("LS", "iLS"):  # use measured data
            if len(self.f_zeroidxs) > 0:
                e[self.f_zeroidxs] = 0

        R = qmt_matrix(e, [self.M] * self.n_qubits)
        G = ((R[self.nonzero_idx].T)[self.nonzero_idx].T) @ A

        A, U = self.momentum(U, G, A, beta=0.9)

        return A, U

    def MLE_GD(self, A, U):
        pauli_basis = np.array([[[1, 0], [0, 0]],
                                [[0, 0], [0, 1]],
                                [[0.5, 0.5], [0.5, 0.5]],
                                [[0.5, -0.5], [-0.5, 0.5]],
                                [[0.5, -0.5j], [0.5j, 0.5]],
                                [[0.5, 0.5j], [-0.5j, 0.5]]]).astype(complex)

        rho_res = np.zeros((2**self.n_qubits, 1)).astype(self.dtype)
        rho_res[self.nonzero_idx] = A

        PA = qmt(rho_res @ rho_res.conj().T,
                 [pauli_basis]*self.n_qubits, allow_negative=False)
        e = -self.m_results / (PA + 1e-12) / len(self.m_results)
        R = qmt_matrix(e, [pauli_basis]*self.n_qubits)

        G = 2 * ((R[self.nonzero_idx].T)[self.nonzero_idx].T) @ A

        A, U = self.momentum(U, G, A, beta=0.9)

        return A, U

    def DFE_GD(self):
        e = self.f_star
        if len(self.f_zeroidxs) > 0:
            e[self.f_zeroidxs] = 0
        R = qmt_matrix(e, [self.M] * self.n_qubits)
        R = ((R[self.nonzero_idx].T)[self.nonzero_idx].T)

        lamda = np.linalg.eigh(R)
        index = np.argmax(abs(lamda[0]))
        A = lamda[1][:, index]
        A /= np.linalg.norm(A)

        return A.reshape(-1, 1)

    def mea_sqrt_W(self, state, j, k):
        """Pauli expection values"""

        if sum(j) == 0:
            j1 = np.nonzero(k)[0]
            p_cal = (1 - 2 * (state[j1].conj().T @ state[j1])[0, 0])
        elif sum(j) == 2:
            j1 = np.nonzero(j)[0]
            me = np.array(j) @ np.array(k)
            if me % 2 == 0:
                p_cal = (2 * (state[j1[0]].conj() * state[j1[1]]).real)[0]
            else:
                # p_cal = (2 * (state[j1[0]].conj() * state[j1[1]]).imag)[0]
                if k[j1][0] == 0 and k[j1][1] == 1:
                    p_cal = (2 * (state[j1[0]].conj() * state[j1[1]]).imag)[0]
                else:
                    p_cal = (-2 * (state[j1[0]].conj() * state[j1[1]]).imag)[0]
        else:
            p_cal = 0

        return p_cal.real  # / np.sqrt(2**self.n_qubits)

    def mea_sqrt_GHZ(self, state, j, k):
        """Pauli expection values"""

        if sum(j) == 0:
            p_cal = state[0, 0] * state[0, 0].conj() + (-1)**sum(k) * \
                state[-1, 0] * state[-1, 0].conj()
        elif sum(j) == self.n_qubits:
            p_cal = state[0, 0] * state[-1, 0].conj() * (1j)**sum(k) + \
                state[0, 0].conj() * state[-1, 0] * (-1j)**sum(k)
        else:
            p_cal = 0

        return p_cal.real  # / np.sqrt(2**self.n_qubits)

    def mea_sqrt(self, state, j, k):
        if self.na_state == "GHZ":
            p = self.mea_sqrt_GHZ(state, j, k)
        elif self.na_state == "W":
            p = self.mea_sqrt_W(state, j, k)

        return p

    @staticmethod
    def sample_W_jud(j, k):
        if sum(j) == 0 or sum(j) == 2:
            return True
        else:
            return False

    def sample_GHZ_jud(self, j, k):
        if sum(j) == 0 or sum(j) == self.n_qubits:
            return True
        else:
            return False

    def sample_jud(self, j, k):
        if self.na_state == "GHZ":
            return self.sample_GHZ_jud(j, k)
        elif self.na_state == "W":
            return self.sample_W_jud(j, k)
