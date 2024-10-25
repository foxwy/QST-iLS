# -*- coding: utf-8 -*-
# @Author: yong
# @Date:   2023-07-29 14:16:20
# @Last Modified by:   yong
# @Last Modified time: 2024-03-31 16:22:55

from basis.Basic_Function import cal_HS, cal_IF, ten_to_k
import numpy as np
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

        # optimizer parameters
        self.a = args.a
        self.b = args.b
        self.metric_k = 1

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
        self.get_observations()

        # init
        A = rho_init.copy()
        U = A.copy()

        save_data = {"epoch": [], "time": [], "HS": [], "iFq": [], 'iFqe': 0}

        # learning QST
        pbar = tqdm(range(self.n_epochs))

        for i in pbar:
            self.epoch += 1
            time_b = perf_counter()

            A, U = self.SPSA(A, U)

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

        return save_data

    def get_observations(self):
        j_all = np.zeros((self.n_meas, self.n_qubits))
        k_all = np.zeros((self.n_meas, self.n_qubits))
        jk_all = np.concatenate((j_all, k_all), axis=1)
        target_obs = []
        target_locs = []

        i = 0
        while i < self.n_meas:
            j, k = self.sample()
            jk = np.concatenate((j, k))
            if sum(np.all(jk_all == jk, axis=1)) == 0:
                j_all[i] = j
                k_all[i] = k
                jk_all = np.concatenate((j_all, k_all), axis=1)
                i += 1

                ob = []
                loc = []
                for m in range(self.n_qubits):
                    if j[m] == 1 and k[m] == 0:
                        ob.append(1)
                        loc.append(m)
                    elif j[m] == 1 and k[m] == 1:
                        ob.append(2)
                        loc.append(m)
                    elif j[m] == 0 and k[m] == 1:
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
            n_tols=self.n_tols
        )

        # identity matrix
        self.j_all = j_all
        self.k_all = k_all
        self.p_rho_star_all = np.array(p_rho_star)

    def SPSA(self, rho, U):
        """basic SPSA algorithm"""

        alpha, beta = self.params()
        rho_delta = self.rho_delta()

        rho_plus = rho + beta * rho_delta
        m_plus = self.metric(rho_plus)
        rho_sub = rho - beta * rho_delta
        m_sub = self.metric(rho_sub)

        G = (m_plus - m_sub) / (2 * beta * rho_delta.conj())

        '''
        U_new = rho - alpha * G
        rho = U_new + 0.99 * (U_new - U)
        U = U_new'''

        # G += 0 / (self.epoch // 100 + 1) * rho
        U = 0.99 * U + G
        G += 0.99 * U

        rho -= alpha * G

        rho /= np.linalg.norm(rho)

        return rho, U

    def params(self):
        """the parameters of SPSA"""

        A = 1000  # 0
        s = 1  # 0.602
        r = 1 / 6  # 0.101
        alpha = self.a / (self.epoch + A)**s  # 1
        beta = self.b / self.epoch**r  # 1/6

        return alpha, beta

    def rho_delta(self):
        """random direction of SPSA"""

        choice = [1.0, -1.0, 1.0j, -1.0j]
        data = np.random.choice(choice, self.dim, replace=True)
        data_nonzero = data.astype(self.dtype).reshape(-1, 1)

        '''
        data_r = 2 * (np.random.rand(self.dim, 1) > 0.5).astype(int) - 1
        data_i = 1j * (2 * (np.random.rand(self.dim, 1) > 0.5).astype(int) - 1)
        data_nonzero = (data_r + data_i).astype(self.dtype)'''

        return data_nonzero

    def metric(self, rho):
        rho /= np.linalg.norm(rho)

        if self.method == "iLS" or self.method == "DFE":
            m_flag = 0  # random (1) or sequence (0)
            Fq = 0
            Fq_n = 0
            n_sample_one = 5
            if n_sample_one >= len(self.j_all):
                for idx, j in enumerate(self.j_all):
                    k, p_rho_star = self.k_all[idx], self.p_rho_star_all[idx]
                    p_rho = self.mea_sqrt(rho, j, k)
                    if self.method == "iLS":
                        Fq += 1 - (p_rho - p_rho_star)**2
                    else:
                        Fq += (p_rho / (p_rho_star + 1e-20))
                    Fq_n += 1
                    # p_rho_star_star = self.mea_sqrt(self.rho_star, j, k)
                    # print('---:', p_rho_star, p_rho_star_star)
            else:
                if m_flag == 1:
                    idxs = np.random.choice(
                        np.arange(len(self.j_all)), n_sample_one, replace=False)
                else:
                    if self.metric_k == 1:
                        self.idxs = np.arange(len(self.j_all))
                        self.idxs_k = 1

                    if self.idxs_k * n_sample_one > len(self.j_all):
                        np.random.shuffle(self.idxs)
                        self.idxs_k = 1

                    idxs = self.idxs[(self.idxs_k - 1) *
                                     n_sample_one: self.idxs_k * n_sample_one]
                    if self.metric_k % 2 == 0:
                        self.idxs_k += 1

                for idx in idxs:
                    j, k, p_rho_star = self.j_all[idx], self.k_all[idx], self.p_rho_star_all[idx]
                    p_rho = self.mea_sqrt(rho, j, k)
                    if self.method == "iLS":
                        Fq += 1 - (p_rho - p_rho_star)**2
                    else:
                        Fq += (p_rho / (p_rho_star + 1e-20))
                    # p_rho_star_star = self.mea_sqrt(self.rho_star, j, k)
                    # print('-:', p_rho_star_star, p_rho_star)
                    Fq_n += 1

            Fq /= Fq_n
            iFq = 1 - Fq

        self.metric_k += 1
        if self.method == "DFE":
            iFq = min(max(iFq, 0), 1)  # right range

        return iFq

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

    def sample_W(self):
        """samping Pauli operators"""

        x = np.random.rand()
        if x < 0.5:
            j = np.zeros(self.n_qubits, dtype=int)
        else:
            idx = np.random.choice(self.n_qubits, size=(2), replace=False)
            idx.sort()
            j = np.zeros(self.n_qubits, dtype=int)
            j[idx] = 1

        k = np.random.choice([0, 1], size=(self.n_qubits), replace=True)

        return j.astype(int), k.astype(int)

    def sample_GHZ(self):
        """samping Pauli operators"""

        x = np.random.rand()
        if x < 0.5:
            j = np.zeros(self.n_qubits)
        else:
            j = np.ones(self.n_qubits)

        k = np.random.choice([0, 1], size=(self.n_qubits), replace=True)

        return j.astype(int),  k.astype(int)

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

    def sample(self):
        if self.na_state == "GHZ":
            j, k = self.sample_GHZ()
        elif self.na_state == "W":
            j, k = self.sample_W()

        return j, k
