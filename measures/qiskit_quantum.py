# -*- coding: utf-8 -*-
# @Author: foxwy
# @Date:   2021-04-27 13:41:55
# @Last Modified by:   yong
# @Last Modified time: 2024-08-25 20:15:57

# system module
import os
from measures.data_acquisition_shadow import derandomized_classical_shadow
from basis.Basic_Function import qmt_matrix, ten_to_k
from measures.Q_hdis import get_hdis
import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt
from copy import deepcopy
import warnings

# Qiskit module
from qiskit import QuantumRegister, QuantumCircuit, transpile
import qiskit.quantum_info as qi
from qiskit_aer import AerSimulator, Aer

# Qiskit Aer noise module
from qiskit_aer.noise import (NoiseModel,
                              QuantumError,
                              ReadoutError,
                              depolarizing_error,
                              pauli_error,
                              thermal_relaxation_error)


import sys
sys.path.append('..')

# User module


warnings.filterwarnings("ignore", category=DeprecationWarning)


def Plot_counts(counts):
    """
    plot counts based on measurement outcomes of qiskit simulator
    """
    plt.figure()
    qubits = sorted(counts)
    values = []
    for qubit in qubits:
        values.append(counts[qubit])
    probabilities = [i / sum(values) for i in values]
    print(probabilities)
    plt.bar(qubits, probabilities)
    for a, b in zip(qubits, probabilities):
        plt.text(a, round(b, 3), round(b, 3), ha='center', va='bottom')
    plt.ylabel('probabilities')
    plt.xlabel('qubits')
    plt.grid(axis='y', linestyle='-.')
    plt.tight_layout()
    plt.show()


class Circuit_meas():
    """
    Implementation of all qiskit simulator parts as a source of
    experimental data for fidelity estimation.
    """

    def __init__(self, circtype='GHZ', n_qubits=2, backend="aer", p=1):
        """
        Args:
            circtype (str): Type of state of the experiment, choices =
                ["GHZ", "W", "Random_Haar"].
            n_qubits (int): The number of qubits.
            backend (str): Qiskit experiment backend, choices =
                ['aer', 'qasm', 'FakePerth', 'MPS', 'stabilizer', 'IBMQ'].
            p (float): Level of werget state
        """
        assert circtype in ("GHZ", "W", "Random_Haar"), print(
            'please input right state type')
        assert backend in ('aer', 'qasm', 'MPS', 'stabilizer', 'IBMQ'), print(
            'please input right backend')

        self.circtype = circtype  # the type of state
        self.n_qubits = n_qubits  # the number of qubit
        self.backend = self.get_backend(backend)
        self.p = p
        self.get_init_circuit()

    def get_backend(self, backend):
        """The backend of quantum machine"""

        if backend == 'aer':
            p01 = 0.01
            p10 = 0.01

            # QuantumError objects
            readout_err = ReadoutError([[1 - p10, p10], [p01, 1 - p01]])

            # Add errors to noise model
            noise_model = NoiseModel()
            noise_model.add_all_qubit_readout_error(readout_err)

            backend = AerSimulator(noise_model=noise_model)

            # backend = Aer.get_backend('aer_simulator')  # aer_simulator
        elif backend == 'qasm':
            backend = Aer.get_backend('qasm_simulator')
        elif backend == 'MPS':
            backend = Aer.get_backend('aer_simulator_matrix_product_state')
        elif backend == 'stabilizer':
            backend = AerSimulator(method='extended_stabilizer')
        else:
            print('please input right backend')

        return backend

    def get_init_circuit(self, circtype=None):
        """prepare desired quantum state"""

        if circtype is not None:
            self.circtype = circtype

        qr = QuantumRegister(self.n_qubits)
        qc = QuantumCircuit(qr, name='st_circ')

        # obtain the circuit according to the type of state
        if self.circtype == "GHZ":  # GHZ-class state
            choice = [1.0]  # , -1.0, 1.0j, -1.0j]
            w_s = np.random.choice(choice, 1, replace=True)[0]
            self.w_s = w_s

            if w_s == 1:  # GHZ
                qc.h(0)
                for i in range(1, self.n_qubits):
                    qc.cx(0, i)
            elif w_s == -1:
                qc.x(0)
                qc.h(0)
                for i in range(1, self.n_qubits):
                    qc.cx(0, i)
            elif w_s == 1.0j:
                qc.h(0)
                qc.s(0)
                for i in range(1, self.n_qubits):
                    qc.cx(0, i)
            elif w_s == -1.0j:
                qc.x(0)
                qc.h(0)
                qc.s(0)
                for i in range(1, self.n_qubits):
                    qc.cx(0, i)

        elif self.circtype == "W":  # W-class state
            theta = np.arccos(1 / np.sqrt(self.n_qubits))
            qc.ry(2 * theta, 0)
            for i in range(self.n_qubits - 2):
                theta = np.arccos(1 / np.sqrt(self.n_qubits - i - 1))
                qc.cry(2 * theta, i, i + 1)
            for i in range(self.n_qubits - 1):
                qc.cx(self.n_qubits - 2 - i, self.n_qubits - 1 - i)
            qc.x(0)

            # [1, -1, 1j, -1j]
            choice = [1.0]  # , -1.0, 1.0j, -1.0j]
            w_s = np.random.choice(choice, self.n_qubits, replace=True)
            self.w_s = w_s
            for i in range(self.n_qubits):
                if w_s[i] == -1:
                    qc.z(i)
                elif w_s[i] == 1j:
                    qc.s(i)
                elif w_s[i] == -1j:
                    qc.sdg(i)

        elif self.circtype == "Random_Haar":  # random state based on Haar measure
            U_gate = qi.random_unitary(2**self.n_qubits)
            qc.append(U_gate, range(self.n_qubits))

        else:
            print('please input right state type')

        self.qr = qr
        self.qc = qc
        self.inst = self.qc.to_instruction()

        # Get ideal output state
        if self.circtype == "Random_Haar":
            self.target_state = qi.Statevector(self.qc)

    def get_mea_circuit_basis(self, meas_element):
        """
        quantum measurement based on different basis: Z:0, X:1, Y:2.

        Args:
            meas_element (list): Pauli measurement in each qubit, based on Qiskit, from bottom to top.
                example-["X", "Z", "Y"] for 3-qubit.

        Returns:
            Quantum circuit of measurement.
        """

        mea_qr = QuantumRegister(self.n_qubits, 'q')
        mea_qc = QuantumCircuit(mea_qr)

        for idx, m in enumerate(meas_element):
            if m in (1, "X"):  # X
                mea_qc.h(self.n_qubits - 1 - idx)
            elif m in (2, "Y"):  # Y
                mea_qc.sdg(self.n_qubits - 1 - idx)
                mea_qc.h(self.n_qubits - 1 - idx)

        return mea_qc

    def circuit_basis(self, meas_element):
        """
        Quantum circuit combines state circuit and measurement circuit.

        Args:
            meas_element (list): Pauli measurement in each qubit, based on Qiskit, from bottom to top.
                example-["X", "Z", "Y"] for 3-qubit.

        Returns:
            Final Quantum circuit.
        """

        mea_qc = self.get_mea_circuit_basis(meas_element)
        circ_mea = self.qc.compose(mea_qc, range(self.n_qubits), inplace=False)
        circ_mea.measure_all()

        return circ_mea

    def circuit_pauli(self):
        """All quantum circuits of 3^n Pauli measurements"""

        circs = []
        for mea_idx in range(3**self.n_qubits):
            meas_element = ten_to_k(mea_idx, 3, self.n_qubits)
            circ_mea = self.circuit_basis(meas_element)
            circs.append(circ_mea)

        return circs

    def circuit_dfe(self, j, k):
        """quantum circuit for direct fidelity estimation"""

        mea_qr = QuantumRegister(self.n_qubits, 'q')
        mea_qc = QuantumCircuit(mea_qr)

        for idx, m in enumerate(j):
            if j[idx] == 1 and k[idx] == 0:  # X
                mea_qc.h(idx)
            elif j[idx] == 1 and k[idx] == 1:  # Y
                mea_qc.sdg(idx)
                mea_qc.h(idx)

        circ_mea = self.qc.compose(mea_qc, range(self.n_qubits), inplace=False)
        circ_mea.measure_all()

        return circ_mea

    def get_measure(self, circ, shots):
        # Transpile the ideal circuit to a circuit that can be directly executed by the backend
        transpiled_circuit = transpile(circ, self.backend)
        # transpiled_circuit.draw('mpl')

        # counts
        result = self.backend.run(transpiled_circuit, shots=shots).result()
        counts = result.get_counts()
        counts = self.full_counts_to_depolar(counts)

        return counts

    # add to deploar noise
    def full_counts_to_depolar(self, counts):
        counts_all = {}
        shots = sum(counts.values())
        for i in range(2**self.n_qubits):
            i2 = ten_to_k(i, 2, self.n_qubits)
            key = "".join([str(j) for j in i2])
            if key in counts:
                counts_all[key] = self.trans_to_depolar(counts[key], shots)
            else:
                counts_all[key] = self.trans_to_depolar(0, shots)

        return counts_all

    def trans_to_depolar(self, count, shots):
        return self.p * count + (1 - self.p) / 2**self.n_qubits * shots

    # calculate expectation of Pauli measure
    def cal_exp(self, shots, counts, j, k):
        """
        estimate Pauli expection value based on measurment outcomes.

        Args:
            shots (int): The number of shots.
            counts (turple): Quantum measurement outcomes.
            j, k (list): The measurement of direct fidelity estimation.

        Returns:
            Pauli (j, k) expection value.
        """

        beta = 0
        num_shots = shots
        num_items = len(counts)
        frequencies = {
            n: (v + beta) / (num_shots + num_items * beta)
            for n, v in counts.items()
        }
        parity_frequencies = [(-1)**self.parity(n, j, k)
                              * v for n, v in frequencies.items()]
        exp_p = sum(parity_frequencies)

        return exp_p  # / np.sqrt(2**self.n_qubits)

    def parity(self, key, j, k):
        indices = [self.n_qubits - 1 - i for i,
                   symbol in enumerate(j) if symbol == 0 and k[i] == 0]
        digit_list = list(key)
        for i in indices:
            digit_list[i] = '0'
        effective_key = ''.join(digit_list)

        return effective_key.count('1')

    # shadow tomo
    def circuit_cliff_shadow(self, num_shadow, shots):
        """Clifford shadow from https://github.com/ryanlevy/shadow-tutorial/tree/main"""

        cliffords = [
            qi.random_clifford(self.n_qubits) for _ in range(num_shadow)
        ]

        shadows = []
        for cliff in cliffords:
            # simulate
            circuit = cliff.to_circuit()
            qc_c = self.qc.compose(circuit)
            qc_c.measure_all()
            f_counts = self.get_measure(qc_c, shots)

            shadows_shots = []
            mat = cliff.adjoint().to_matrix()  # Operator(circuit).to_matrix()
            for bit, count in f_counts.items():
                Ub = mat[:, int(bit, 2)]  # this is Udag|b>

                if self.circtype == "GHZ":
                    nonzero_idx = 2**np.array([0, self.n_qubits]) - 1
                elif self.circtype == "W":
                    nonzero_idx = 2**np.arange(self.n_qubits)

                ub_w = Ub[nonzero_idx].reshape(-1, 1)

                shadows_shots.append(
                    self.Minv(ub_w @ ub_w.conj().T, self.n_qubits) *
                    self.trans_to_depolar(count, shots)
                )

            shadows.append(np.sum(shadows_shots, axis=0) / shots)

        rho_shadow = np.sum(shadows, axis=0) / num_shadow

        return rho_shadow, shadows

    def get_shadows(self, pauli_string, counts, bf=1):
        counts_sum = sum(counts.values())
        # single-shot
        if bf == 0:
            shadows_shots = []
            for bit, count in counts.items():
                mat = 1.0
                # U_all = []
                for i, bi in enumerate(bit[::-1]):
                    b = self.rotGate(pauli_string[i])[int(bi), :]
                    mat = np.kron(self.Minv(np.outer(b.conj(), b), 1), mat)
                shadows_shots.append(mat * count)

            shadow = np.sum(shadows_shots, axis=0) / counts_sum

        else:  # more eff
            M = []
            for n in range(self.n_qubits):
                M_n = np.zeros((2, 2, 2), dtype=np.complex64)
                for i in range(2):
                    b = self.rotGate(pauli_string[n])[i, :]
                    mat = self.Minv(np.outer(b.conj(), b), 1)
                    M_n[i, :, :] = mat
                M.append(M_n)

            shadow = qmt_matrix(
                np.array(list(counts.values())), M) / counts_sum

        return shadow

    def circuit_pauli_shadow(self, num_shadow, shots, paulis=None, pos=None):
        """Pauli shadow from https://github.com/ryanlevy/shadow-tutorial/tree/main"""

        if paulis is None:
            if pos is None:
                # paulis = np.random.randint(0, 3, size=(pauli_num, self.n_qubits))
                paulis = [ten_to_k(pauli_i, 3, self.n_qubits) for pauli_i in np.random.choice(
                    np.arange(3**self.n_qubits), size=(num_shadow), replace=False)]
            else:
                choice_idxs = np.random.choice(np.arange(
                    # add random
                    3**self.n_qubits), size=(3**self.n_qubits), replace=False)

                paulis = []
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

                    if self.sample_jud(pos, j, k):
                        paulis.append(pauli)

                    if len(paulis) > num_shadow - 1:
                        break

            paulis = np.array(paulis).reshape(-1, self.n_qubits)

        shadows = []
        for pauli_string in paulis:
            qc_c = self.circuit_basis(pauli_string[::-1])
            f_counts = self.get_measure(qc_c, shots)

            # single-shot
            shadow = self.get_shadows(pauli_string, f_counts)
            shadows.append(shadow)

        rho_shadow = np.sum(shadows, axis=0) / num_shadow

        return rho_shadow, shadows  # @ rho_shadow @ rho_shadow

    def circuit_purity_estimation(self, num_pauli, shots, paulis=None):
        if paulis is None:
            # paulis = np.random.randint(0, 3, size=(num_pauli, self.n_qubits))
            paulis = [ten_to_k(pauli_i, 3, self.n_qubits) for pauli_i in np.random.choice(
                np.arange(3**self.n_qubits), size=(num_pauli), replace=False)]
            paulis = np.array(paulis).reshape(-1, self.n_qubits)
        else:
            paulis = np.array(paulis).reshape(-1, self.n_qubits)

        prr = 0
        shadows = []

        file_name = "measures/rm_temp/h_distance_"+str(self.n_qubits)+'.npy'
        if os.path.exists(file_name):
            h_distance_matrix = np.load(file_name)
        else:
            h_distance_matrix = get_hdis(self.n_qubits)

        for idx, pauli_string in enumerate(paulis):
            qc_c = self.circuit_basis(pauli_string[::-1])
            f_counts = self.get_measure(qc_c, shots)

            # classical shadows
            shadow = self.get_shadows(pauli_string, f_counts)
            shadows.append(shadow)

            # purity of RM
            val = np.array(list(f_counts.values())).reshape(-1, 1)
            idxs = [int(int(bit, 2)) for bit in f_counts.keys()]
            h_dis = h_distance_matrix[idxs].T[idxs].T

            val_sum = sum(val)
            prr += ((val.T @ h_dis @ val - val_sum) /
                    (val_sum**2 - val_sum)).item()

            '''
            for bit1, count1 in f_counts.items():
                for bit2, count2 in f_counts.items():
                    if bit1 == bit2:
                        prr += (count1 * (count2 - 1))
                    else:
                        hamming_distance = sum(
                            [i != j for i, j in zip(bit1, bit2)])
                        prr += (-2)**(-hamming_distance) * count1 * count2'''

            if idx % 10 == 0:
                print("purity esti", idx, prr * 2**self.n_qubits / (idx + 1))

        # rm
        prr = prr * 2**self.n_qubits / num_pauli
        # prr = prr * 2**self.n_qubits / num_pauli / shots / (shots - 1)
        print("rm purity:", prr.real)

        # cs
        pr = 0
        for i in range(num_pauli):
            pr += np.trace(shadows[i] @ shadows[i]).real

        pr_t = 0
        for i in range(num_pauli - 1):
            for j in range(i + 1, num_pauli):
                pr_t += np.trace(shadows[i] @ shadows[j]).real

        pr += pr_t * 2
        pr /= (num_pauli**2)
        print("cs purity:", pr.real)

        return max(min(prr.real, 1), 0), max(min(pr, 1), 0)

    def circuit_mle_sample(self, pauli_num, shots, paulis=None, pos=None):
        if paulis is None:
            if pos is None:
                # paulis = np.random.randint(0, 3, size=(pauli_num, self.n_qubits))
                paulis = [ten_to_k(pauli_i, 3, self.n_qubits) for pauli_i in np.random.choice(
                    np.arange(3**self.n_qubits), size=(pauli_num), replace=False)]
            else:
                choice_idxs = np.random.choice(np.arange(
                    # add random
                    3**self.n_qubits), size=(3**self.n_qubits), replace=False)

                paulis = []
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

                    if self.sample_jud(pos, j, k):
                        paulis.append(pauli)

                    if len(paulis) > pauli_num - 1:
                        break

            paulis = np.array(paulis).reshape(-1, self.n_qubits)

        '''
        m_results = []
        for pauli_string in paulis:
            qc_c = self.circuit_basis(pauli_string[::-1])
            f_counts = self.get_measure(qc_c, shots)

            m_result = np.zeros(2**self.n_qubits)
            counts_sum = sum(f_counts.values())
            for k in f_counts:
                idx = np.array([int(i) for i in k[::-1]]) @ 2**np.arange(self.n_qubits)[::-1]
                m_result[idx] = f_counts[k] / counts_sum

            m_results.append(m_result)'''

        # |0>: 0, |1>: 1, |+>: 2, |->: 3, |+i>: 4, |-i>: 5
        basis_tras = {0: 0, 1: 2, 2: 4}
        m_results = np.zeros(6**self.n_qubits)
        for pauli_string in paulis:
            qc_c = self.circuit_basis(pauli_string[::-1])
            f_counts = self.get_measure(qc_c, shots)

            counts_sum = sum(f_counts.values())
            for key in f_counts:
                m = [int(k) + basis_tras[pauli_string[idx]]
                     for idx, k in enumerate(list(key[::-1]))]
                idx = np.array(m) @ 6**np.arange(self.n_qubits)
                m_results[idx] = f_counts[key] / counts_sum

        return paulis, m_results

    def circuit_pauli_expect_shadow(self, all_observables, shots, n_tols, derandomized=True):
        """
        compute Pauli expection value based on derandomized shadow.
        1. we fix the case where derandomized shadows may lead to too few Pauli measurements, 
           resulting in an expectation of 0 for measured observations;
        2. we develop multi-shot derandomized shadows to improve the efficiency of expectation estimation.

        """

        # 1 derandomization protol
        print('\n-----begin shadow sampling-----')
        t1 = perf_counter()
        mea_map = {"1": -1, "0": 1}

        target_obs, target_locs = all_observables
        if derandomized:
            measurement_procedure = derandomized_classical_shadow(
                target_obs, target_locs, shots, self.n_qubits
            )
        else:
            measurement_procedure = np.random.choice(
                [0, 1, 2], size=(shots, self.n_qubits), replace=True)

        # fix bug of derandomized
        '''
        measurement_procedure = np.array(measurement_procedure)
        measurement = deepcopy(measurement_procedure)
        for idx, single_obs in enumerate(target_obs):
            indices = np.all(measurement[:, target_locs[idx]] == np.array(single_obs), axis=1)
            if sum(indices) == 0:
                single_mea = np.zeros(self.n_qubits)
                single_mea[target_locs[idx]] = np.array(single_obs)
                measurement_procedure = np.r_[measurement_procedure, [single_mea]]'''

        # np.save('measurement_procedure_W_'+str(self.n_qubits)+'.npy', measurement_procedure)

        N_Mb = len(measurement_procedure)
        print('shadow pauli measurement num:', len(measurement_procedure))

        if N_Mb > n_tols:
            measurement_procedure = measurement_procedure[:n_tols]
            N_Mb = len(measurement_procedure)

        # 2 measurement in quantum machine or simulator
        full_measurement = []
        outcomes = []
        outcounts = []
        for measurement in measurement_procedure:
            qc_c = self.circuit_basis(measurement[::-1])
            f_counts = self.get_measure(qc_c, round(
                n_tols / len(measurement_procedure)))

            for bit, count in f_counts.items():  # single shots
                single_outcome = [mea_map[bit[::-1][n]]
                                  for n in range(self.n_qubits)]
                full_measurement.append(measurement)
                outcomes.append(single_outcome)
                outcounts.append(count)

        Ntol = sum(outcounts)
        t2 = perf_counter()
        print('shadow measurement num:', Ntol)

        # 3 Pauli expection estimation
        full_measurement = np.array(full_measurement)
        outcomes = np.array(outcomes)
        outcounts = np.array(outcounts)

        # pauli expection based on classical shadows
        ob_values = []
        for idx, single_obs in enumerate(target_obs):
            indices = np.all(
                full_measurement[:, target_locs[idx]] == np.array(single_obs), axis=1)

            if sum(indices) > 0:
                product = np.prod(
                    outcomes[indices][:, target_locs[idx]], axis=1)
                # / np.sqrt(2**self.n_qubits)
                ob = (product @ outcounts[indices]) / sum(outcounts[indices])
                ob_values.append(ob)
            else:
                ob_values.append(0)
        print(
            "-----shadow sampling time: {:.5f} | {:.5f} s-----\n".format(perf_counter() - t2, t2 - t1))

        return ob_values

    @staticmethod
    def rotGate(g):
        '''produces gate U such that U|psi> is in Pauli basis g'''
        if g == 1:  # X
            return 1 / np.sqrt(2) * np.array([[1., 1.], [1., -1.]])
        elif g == 2:  # Y
            return 1 / np.sqrt(2) * np.array([[1., -1.0j], [1., 1.j]])
        elif g == 0:  # Z
            return np.eye(2)
        else:
            raise NotImplementedError(f"Unknown gate {g}")

    @staticmethod
    def rotGate2(g, bi):
        d_dtype = np.complex128
        if g == 1:
            if bi == 0:
                return np.array([[0.5, 1.5], [1.5, 0.5]], dtype=d_dtype)
            else:
                return np.array([[0.5, -1.5], [-1.5, 0.5]], dtype=d_dtype)
        elif g == 2:
            if bi == 0:
                return np.array([[0.5, 1.5j], [-1.5j, 0.5]], dtype=d_dtype)
            else:
                return np.array([[0.5, -1.5j], [1.5j, 0.5]], dtype=d_dtype)
        elif g == 0:
            if bi == 0:
                return np.array([[2., 0.], [0., -1.]], dtype=d_dtype)
            else:
                return np.array([[-1., 0.], [0., 2.]], dtype=d_dtype)
        else:
            raise NotImplementedError(f"Unknown gate {g}")

    def Minv(self, X, N):
        '''inverse shadow channel'''
        return (2**N + 1.) * X - np.eye(len(X))

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

    def sample_jud(self, pos, j, k):
        if pos == "GHZ":
            return self.sample_GHZ_jud(j, k)
        elif pos == "W":
            return self.sample_W_jud(j, k)

    def circuit_simulator(
        self,
        meastype='pauli',
        meas_element=None,
        shots=2048,
        mea_state=None,
        j=None,
        k=None,
        num_shadow=None,
        pauli_num=None,
        all_observables=None,
        n_tols=None,
        derandomized=True,
        paulis=None,
        pos=None
    ):
        """Run the quantum circuit on a Qiskit simulator backend"""

        # circuit
        circ = 0
        if meastype == 'basis':
            if meas_element is None:
                print('please input meas_element')
            else:
                circ = self.circuit_basis(meas_element)
        elif meastype == 'pauli':
            circ = self.circuit_pauli()
        elif meastype == 'mea_povm':
            circ = self.circuit_povm(mea_state)
        elif meastype == 'mea_state':
            circ = self.circuit_state(mea_state)
        elif meastype == 'mea_dfe':
            circ = self.circuit_dfe(j, k)
        elif meastype == 'mea_cliff_shadow':
            rho_shadow, shadows = self.circuit_cliff_shadow(num_shadow, shots)
            return rho_shadow, shadows
        elif meastype == 'mea_pauli_shadow':
            rho_shadow, shadows = self.circuit_pauli_shadow(
                num_shadow, shots, paulis, pos)
            return rho_shadow, shadows
        elif meastype == 'mea_purity_esti':
            prr, pr = self.circuit_purity_estimation(pauli_num, shots, paulis)
            return prr, pr
        elif meastype == 'mea_pauli_expect':
            ob_values = self.circuit_pauli_expect_shadow(
                all_observables, shots, n_tols, derandomized)
            return ob_values
        elif meastype == "mea_mle":
            paulis, m_results = self.circuit_mle_sample(
                pauli_num, shots, paulis, pos)
            return paulis, m_results

        counts = self.get_measure(circ, shots)

        return counts


if __name__ == '__main__':
    n_qubits = 3

    t1 = perf_counter()
    circ = Circuit_meas(circtype="GHZ", n_qubits=n_qubits)
    print(circ.target_state, circ.w_s)

    t2 = perf_counter()
    print("time:", t2 - t1)
