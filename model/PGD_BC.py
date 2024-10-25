# -*- coding: utf-8 -*-
# @Author: yong
# @Date:   2023-07-29 14:16:20
# @Last Modified by:   yong
# @Last Modified time: 2024-09-06 21:33:24
# Experiment B and C

from basis.Basic_Function import (qmt,
                                  qmt_matrix,
                                  cal_HS,
                                  cal_IF,
                                  ten_to_k,
                                  qmt_torch,
                                  qmt_matrix_torch,
                                  proj_trans_S,
                                  proj_trans_pure,
                                  proj_trans_P,
                                  create_file_name,
                                  save_file)
from basis.Basis_State import Mea_basis
import os
import numpy as np
import torch
from tqdm import tqdm
from time import perf_counter
from copy import deepcopy

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
        self.rank = args.rank
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
        self.lamda = args.lamda
        self.p = args.p
        self.p_esti = args.p_esti
        self.purity_target = args.purity_target
        self.meada_used = args.meada_used

        # other parameter
        self.device = torch.device("cuda")

    def guide(self, circ, rho_star):
        # temp variables
        self.circ = circ
        self.rho_star = rho_star
        self.dim = len(rho_star)

        self.rho_star = torch.from_numpy(self.rho_star).to(
            self.device).to(torch.complex64)

        # observations
        if self.method in ["LS", "iLS", "fLS", "fiLS", "DFE", "CSE"]:
            self.n_shots = 1

            if self.meada_used:  # use history measure data
                file_name_s = create_file_name(['data', self.na_state], [
                                               'f-star', self.n_qubits, self.n_tols, self.n_meas, self.p], root="measures/")
                file_name_i = create_file_name(['data', self.na_state], [
                                               'f-idxs', self.n_qubits, self.n_tols, self.n_meas, self.p], root="measures/")
                pauli_s = create_file_name(['data', self.na_state], [
                                           'paulis', self.n_qubits, self.n_tols, self.n_meas, self.p], root="measures/")
                if os.path.exists(file_name_s) and os.path.exists(file_name_i):
                    self.f_star = np.load(file_name_s)
                    self.f_idxs = np.load(file_name_i)
                    self.paulis = np.load(pauli_s)
                else:
                    self.f_star, self.f_idxs = self.get_observations()
                    np.save(file_name_s, self.f_star)
                    np.save(file_name_i, self.f_idxs)
                    np.save(pauli_s, self.paulis)
            else:
                self.f_star, self.f_idxs = self.get_observations()

            if self.method in ["iLS", "fiLS"]:
                if self.p_esti:
                    self.f_zeroidxs_ori = list(
                        set(np.arange(self.K**self.n_qubits)) - set(self.f_idxs))

                    purity_data = {}
                    purity_data["ideal"] = (
                        1 + (self.dim - 1) * self.p**2) / self.dim

                    purity_dic = (self.f_star @ self.f_star).real.item()
                    purity_dic = min(purity_dic, 1)
                    purity_data["purity_dic"] = purity_dic

                    # purity estimation
                    self.n_shots = round(self.n_tols / self.n_meas)
                    prr, pr = self.circ.circuit_simulator(
                        meastype='mea_purity_esti', shots=self.n_shots, pauli_num=self.n_meas, paulis=self.paulis
                    )
                    purity_data["RM"] = prr
                    purity_data["CS"] = pr

                    print(purity_data)
                else:
                    self.purity_target = self.purity_target

            if self.method in ["CSE"] and self.p_esti:
                self.f_zeroidxs_ori = list(
                    set(np.arange(self.K**self.n_qubits)) - set(self.f_idxs))

                purity_data = {}
                purity_data["ideal"] = (
                    1 + (self.dim - 1) * self.p**2) / self.dim

            # to GPU
            self.M = torch.from_numpy(self.M).to(
                self.device).to(torch.complex64)
            self.f_star = torch.from_numpy(self.f_star).to(
                self.device).to(torch.complex64)
            self.f_zeroidxs = list(
                set(np.arange(self.K**self.n_qubits)) - set(self.f_idxs))
            self.f_zeroidxs = torch.tensor(
                self.f_zeroidxs).to(self.device).to(torch.long)

            rho = self.rho_init()
            print('init purity', round(torch.trace(rho @ rho).real.item(), 3))

        elif self.method == "MLE":
            self.n_shots = round(self.n_tols / self.n_meas)
            self.paulis, self.m_results = self.circ.circuit_simulator(
                meastype='mea_mle', shots=self.n_shots, pauli_num=self.n_meas
            )
            self.m_results = torch.from_numpy(self.m_results).to(
                self.device).to(torch.complex64)

        if self.method in ("LS", "fLS", "MLE") or (self.method in ("iLS", "fiLS") and self.rank == 1):
            self.lamda_f = 0
        else:
            self.lamda_f = 1

        # cross validation
        if self.method in ["iLS", "fiLS", "CSE"] and self.p_esti:
            # 5-fold to detemine purity of target
            K_f = 1
            f_size = len(self.f_idxs)
            f_size_kf = f_size // K_f

            # CV
            # np.linspace(round(torch.trace(rho @ rho).real.item(), 3), 1, 21)
            prs = [1]
            cv_errs = []
            losss = []
            IFs = []
            HSs = []
            for purity_target in prs:
                self.purity_target = purity_target

                # cv
                cv_error = []
                for _ in range(1):
                    # np.random.shuffle(self.f_idxs)

                    for kf in range(K_f):
                        f_idxs_kf = self.f_idxs[kf *
                                                f_size_kf: min((kf + 1) * f_size_kf, f_size)]

                        self.f_zeroidxs = list(
                            set(self.f_zeroidxs_ori) | set(f_idxs_kf))
                        self.f_zeroidxs = torch.tensor(
                            self.f_zeroidxs).to(self.device).to(torch.long)
                        f_testidxs = deepcopy(f_idxs_kf)
                        f_testidxs = torch.tensor(f_testidxs).to(
                            self.device).to(torch.long)

                        # training
                        _, rho = self.learning()

                        # predict
                        PA = qmt_torch(
                            rho, [self.M] * self.n_qubits, allow_negative=True)
                        e = (PA - self.f_star)[f_testidxs]
                        cv_error.append((e @ e).real.item())

                cv_errs.append(np.sum(cv_error))

                # loss function
                self.f_zeroidxs = self.f_zeroidxs_ori
                self.f_zeroidxs = torch.tensor(
                    self.f_zeroidxs).to(self.device).to(torch.long)

                _, rho = self.learning()

                PA = qmt_torch(rho, [self.M] *
                               self.n_qubits, allow_negative=True)
                e = PA - self.f_star
                if len(self.f_zeroidxs) > 0:  # importance
                    e.index_fill_(0, self.f_zeroidxs, 0)

                if self.method in ["iLS", "fiLS"]:
                    loss = [(e @ e).real.item(),
                            (-torch.trace(rho @ rho)).real.item()]
                else:
                    loss = [(e @ e).real.item(), torch.trace(rho).real.item()]
                losss.append(loss)

                # infidelity
                IF = cal_IF(self.rho_star, rho)
                IFs.append(IF)

                hs = cal_HS(self.rho_star, rho)
                HSs.append(hs)

            # estimate
            purity_data["lamda"] = self.lamda
            purity_data["CV"] = prs[cv_errs.index(min(cv_errs))]

            purity_data["CV_errs"] = cv_errs
            purity_data["Loss"] = losss
            purity_data["IF"] = IFs
            purity_data["HS"] = HSs
            purity_data["purity"] = torch.trace(rho @ rho).real.item()

            print(purity_data)

            return purity_data

        else:
            save_data, _ = self.learning()

            return save_data

    def learning(self):
        self.epoch = 0
        time_all = 0

        # init
        rho = self.rho_init()
        U = torch.zeros_like(rho)

        IF = cal_IF(self.rho_star, rho)
        pr = torch.trace(rho @ rho).real.item()
        print("rho init: IF {:.5f}, purity {:.5f}".format(IF, pr))

        save_data = {"epoch": [], "time": [], "HS": [],
                     "iFq": [], 'iFqe': 0, "purity": []}
        all_method = {"LS": self.LS_GD, "iLS": self.LS_GD,
                      "fLS": self.DFE_GD, "fiLS": self.LS_GD,
                      "MLE": self.MLE_GD, "CSE": self.CSE_GD}
        if self.method == "fiLS" and self.rank == 1:  # direct estimate
            all_method["fiLS"] = self.DFE_GD
        GD_method = all_method[self.method]

        # learning QST
        if self.method in ["LS", "iLS", "MLE", "CSE"] or (self.method == "fiLS" and self.rank > 1):
            pbar = tqdm(range(self.n_epochs))

            for _ in pbar:
                self.epoch += 1
                time_b = perf_counter()

                rho, U = GD_method(rho, U)

                time_all += (perf_counter() - time_b)

                if (self.epoch % 2 == 0 or self.epoch == 1):  # save data
                    if self.rank > 1:
                        rho_t = rho / torch.trace(rho)
                    else:
                        rho_t = proj_trans_pure(rho)
                    HS = cal_HS(self.rho_star, rho_t)
                    IF = cal_IF(self.rho_star, rho_t)
                    pr = torch.trace(rho_t @ rho_t).real.item()

                    save_data["epoch"].append(self.epoch)
                    save_data["time"].append(time_all)
                    save_data["HS"].append(HS)
                    save_data["iFq"].append(IF)
                    save_data["iFqe"] = IF
                    save_data["purity"].append(pr)

                    pbar.set_description(
                        "epoch {:d} | time {:.3f} | HS {:.5f} | IF {:.5f} | pur et {:.5f} | pur tar {:.5f} | lamda {:.4f}".
                        format(self.epoch, time_all, HS, IF, pr,
                               self.purity_target, self.lamda)
                    )

                    # Early stopping of iterations
                    if self.epoch > 200:
                        # ifidelity essentially unchanged
                        if np.std(save_data["iFq"][-10:]) < 1 / self.n_tols:
                            pass
                            # break
        else:
            time_b = perf_counter()
            rho = GD_method()
            time_all += (perf_counter() - time_b)

            rho /= torch.trace(rho)
            HS = cal_HS(self.rho_star, rho)
            IF = cal_IF(self.rho_star, rho)
            pr = torch.trace(rho @ rho).real.item()

            save_data["epoch"].append(1)
            save_data["time"].append(time_all)
            save_data["HS"].append(HS)
            save_data["iFq"].append(IF)
            save_data["purity"].append(pr)

            print(
                "epoch {:d} | time {:.4f} | Fq {:.8f}".
                format(self.epoch, time_all, 1 - IF)
            )

        rho /= torch.trace(rho).real
        # rho = rho.cpu().numpy()
        '''
        np.save('estimated_rho/'+str(self.n_qubits)+'_'+self.na_state+'_'+str(self.p) +
                '_'+str(self.n_meas)+'_'+self.method+'_rho.npy', rho)
        np.save('estimated_rho/'+str(self.n_qubits)+'_'+self.na_state+'_'+str(self.p) +
                '_'+str(self.n_meas)+'_'+self.method+'_save_data.npy', save_data)'''

        return save_data, rho

    def get_observations(self):
        # sample
        choice_idxs = np.random.choice(
            # add random
            np.arange(3**self.n_qubits), size=(self.n_meas), replace=False)
        self.paulis = [ten_to_k(pauli_i, 3, self.n_qubits)
                       for pauli_i in choice_idxs]

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

            j_all.append(j)
            k_all.append(k)

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

    def rho_init(self):
        # LS method initialization
        # T = qmt_matrix_torch(self.f_star, [self.M] * self.n_qubits)
        # rho, rho_sqrt = proj_trans_S(T)
        A = np.random.normal(size=(self.dim, self.rank)).astype(self.dtype)
        A = torch.from_numpy(A).to(self.device).to(torch.complex64)
        rho = A @ A.conj().T
        rho /= torch.trace(rho)

        return rho

    def momentum(self, U, G, rho, beta=0):
        U = beta * U + G
        rho -= self.lr * U

        if self.rank > 1:
            if self.method != "CSE":
                rho, _ = proj_trans_S(rho)
            else:
                rho, _ = proj_trans_P(rho)
        else:
            rho = proj_trans_pure(rho)

        rho = 0.5 * (rho + rho.conj().T)
        # print(torch.trace(rho))

        return rho, U

    def LS_GD(self, rho, U):
        # cal the error
        PA = qmt_torch(rho, [self.M] * self.n_qubits, allow_negative=True)
        e = PA - self.f_star

        if self.method in ("LS", "iLS"):  # use measured data
            # remove the unmeausred data
            if len(self.f_zeroidxs) > 0:  # importance
                e.index_fill_(0, self.f_zeroidxs, 0)

        T = qmt_matrix_torch(e, [self.M] * self.n_qubits)

        if self.lamda_f == 1:
            G = T - self.lamda * rho
        else:
            G = T

        rho, U = self.momentum(U, G, rho, beta=0.0)

        return rho, U

    def CSE_GD(self, rho, U):
        PA = qmt_torch(rho, [self.M] * self.n_qubits, allow_negative=True)
        e = PA - self.f_star

        if len(self.f_zeroidxs) > 0:  # importance
            e.index_fill_(0, self.f_zeroidxs, 0)

        T = qmt_matrix_torch(e, [self.M] * self.n_qubits)

        G = T + self.purity_target * torch.eye(self.dim).to(self.device)

        rho, U = self.momentum(U, G, rho, beta=0)

        return rho, U

    def MLE_GD(self, rho, U):
        pauli_basis = torch.tensor([[[1, 0], [0, 0]],
                                    [[0, 0], [0, 1]],
                                    [[0.5, 0.5], [0.5, 0.5]],
                                    [[0.5, -0.5], [-0.5, 0.5]],
                                    [[0.5, -0.5j], [0.5j, 0.5]],
                                    [[0.5, 0.5j], [-0.5j, 0.5]]]).to(torch.complex64).to(self.device)

        PA = qmt_torch(rho, [pauli_basis]*self.n_qubits, allow_negative=False)
        e = -self.m_results / len(self.m_results) / (PA + 1e-12)
        R = qmt_matrix_torch(e, [pauli_basis]*self.n_qubits)

        G = R

        rho, U = self.momentum(U, G, rho, beta=0)

        return rho, U

    def DFE_GD(self):
        e = self.f_star
        if len(self.f_zeroidxs) > 0:
            e[self.f_zeroidxs] = 0
        T = qmt_matrix_torch(e, [self.M] * self.n_qubits)

        if self.rank > 1:
            rho, _ = proj_trans_S(T)
        else:
            rho = proj_trans_pure(T)

        return rho
