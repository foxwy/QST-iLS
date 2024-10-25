# -*- coding: utf-8 -*-
# @Author: yong
# @Date:   2023-07-29 14:16:20
# @Last Modified by:   yong
# @Last Modified time: 2024-09-11 00:05:50

"""
main file for experiments B and C
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from measures.qiskit_quantum import Circuit_meas
from model.PGD_BC import FEQST
from basis.Basis_State import State
from basis.Basic_Function import save_file


def basic_exp(args):
    """basic experiment based different setups (args)"""

    print('\nargs:', args)
    circ = Circuit_meas(args.na_state, args.n_qubits, p=args.p)

    # prepare state
    if args.na_state == "GHZ":
        rho = State().Get_GHZ_s(args.n_qubits, circ.w_s)
        rho = np.array(rho).astype(args.dtype).reshape(-1, 1)
        nonzero_idx = 2**np.array([0, args.n_qubits]) - 1

        rho_star = np.zeros((2**args.n_qubits, 1)).astype(args.dtype)
        rho_star[nonzero_idx] = rho

    elif args.na_state == "W":
        rho = State().Get_W_s(args.n_qubits, circ.w_s)
        rho = np.array(rho).astype(args.dtype).reshape(-1, 1)
        nonzero_idx = 2**np.arange(args.n_qubits)

        rho_star = np.zeros((2**args.n_qubits, 1)).astype(args.dtype)
        rho_star[nonzero_idx] = rho

    rho_star = args.p * rho_star @ rho_star.T.conj() + (1 - args.p) / \
        2**args.n_qubits * np.eye(2**args.n_qubits)

    # learn state from measurement
    qst = FEQST(args)
    results = qst.guide(circ, rho_star)

    # plt.plot(np.log10(results["epoch"]), np.log10(np.array(results["iFq"])))
    # plt.show()

    return results


# Appendix A, purity estimation experiment
def exp_purity_esti(args):  # no prior
    args.n_qubits = 7
    args.n_epochs = int(300)
    args.n_shots = 1
    args.rank = int(2**args.n_qubits)
    args.method = "iLS"
    args.lr = 0.5
    args.p_esti = True
    args.meada_used = True

    dim = 2**args.n_qubits

    for na_state in ["W"]:
        args.na_state = na_state

        for n_meas in [200]:
            args.n_meas = n_meas

            for n_tols in [7]:
                args.n_tols = 10**n_tols

                prs = [0.4]
                for pr in [prs[0]]:
                    args.p = np.sqrt((pr - 1 / dim) / (1 - 1 / dim))
                    args.purity_target = pr

                    save_data = {}
                    for lamda in np.insert(10**np.linspace(-3, 0, 21), 0, 0):
                        args.lamda = lamda
                        print(n_meas, n_tols, pr, lamda)
                        results = basic_exp(args)
                        save_data[lamda] = results

                    save_file(
                        save_data, ["exp_purity_esti", args.na_state],
                        [args.method, args.n_qubits, args.n_meas, n_tols, pr]
                    )


# Experiment B, IF with werner state weight, full estimate state, full target state
def exp_full(args):
    args.n_qubits = 7
    args.n_epochs = int(1e3)
    args.n_shots = 1
    args.rank = int(2**args.n_qubits)
    args.lamda = 10
    args.p_esti = False
    args.meada_used = False

    dim = 2**args.n_qubits
    lrs = {"LS": 0.5, "fLS": 0.5, "iLS": 0.5,
           "fiLS": 0.5, "MLE": 0.01, "CSE": 0.5}

    p_target = {"GHZ": [0.03109016, 0.33879062, 0.62913846,
                        1.], "W": [0.03109016, 0.32615999, 0.67366785, 0.98916525]}
    n_exp = 10

    for na_state in ["GHZ", "W"]:
        args.na_state = na_state

        for n_meas in [100]:
            args.n_meas = n_meas

            for n_tols in [5, 6]:
                args.n_tols = 10**n_tols

                for method in ["LS", "fLS", "iLS", "fiLS", "CSE", "MLE"]:
                    args.method = method
                    args.lr = lrs[method]

                    prs = np.linspace(1 / dim, 1, 4)[::-1]
                    for idx, pr in enumerate(prs):
                        args.p = np.sqrt((pr - 1 / dim) / (1 - 1 / dim))

                        if args.method == 'CSE':
                            args.purity_target = 0.1
                        elif args.method in ['iLS', 'fiLS']:
                            args.purity_target = p_target[na_state][idx]

                        save_data = {}
                        for idx in range(n_exp):
                            print(n_meas, n_tols, pr, idx)
                            results = basic_exp(args)
                            save_data[idx] = results

                        save_file(
                            save_data, ["exp_full", args.na_state],
                            [args.method, args.n_qubits, args.n_meas, n_tols, pr]
                        )


# Experiment B, IF with number of samples, full estimate state, full target state
def exp_1(args):
    args.n_qubits = 7
    args.n_epochs = int(1e3)
    args.n_shots = 1
    args.rank = int(2**7)
    args.lamda = 10
    args.p_esti = False
    args.p = 1
    args.purity_target = 0.1
    args.meada_used = False

    lrs = {"LS": 0.5, "fLS": 0.5, "iLS": 0.5,
           "fiLS": 0.5, "MLE": 0.01, "CSE": 0.5}
    n_exp = 10

    for na_state in ["GHZ", "W"]:
        args.na_state = na_state

        for n_meas in [100]:
            args.n_meas = n_meas

            for method in ["CSE"]:  # "LS", "fLS", "iLS", "fiLS", "CSE", "MLE"
                args.method = method
                args.lr = lrs[method]

                for n_tols in [4, 5, 6]:
                    args.n_tols = 10**n_tols

                    save_data = {}
                    for idx in range(n_exp):
                        print(idx)
                        results = basic_exp(args)
                        save_data[idx] = results

                    save_file(
                        save_data, ["exp1-t", args.na_state],
                        [args.method, args.n_qubits,
                            args.n_meas, args.rank, args.n_tols]
                    )


# Experiment C, IF with number of samples, full estimate state, pure target state
def exp_1p(args):
    args.n_qubits = 7
    args.n_epochs = int(1e3)
    args.n_shots = 1
    args.rank = int(1)
    args.lamda = 1
    args.p_esti = False
    args.p = 1
    args.purity_target = 0.1
    args.meada_used = False

    lrs = {"LS": 0.5, "fLS": 0.5, "iLS": 0.5,
           "fiLS": 0.5, "MLE": 0.01, "CSE": 0.5}
    n_exp = 10

    for na_state in ["GHZ", "W"]:
        args.na_state = na_state

        for n_meas in [100]:
            args.n_meas = n_meas

            for method in ["CSE"]:
                args.method = method
                args.lr = lrs[method]

                for n_tols in [4, 5, 6]:
                    args.n_tols = 10**n_tols

                    save_data = {}
                    for idx in range(n_exp):
                        print(idx)
                        results = basic_exp(args)
                        save_data[idx] = results

                    save_file(
                        save_data, ["exp1-p", args.na_state],
                        [args.method, args.n_qubits,
                            args.n_meas, args.rank, args.n_tols]
                    )


# Experiment B, IF with number of operators, full estimate state, full target state
def exp_2(args):
    args.n_qubits = 7
    args.n_epochs = int(1e3)
    args.n_shots = 1
    args.rank = int(2**7)
    args.lamda = 10
    args.p_esti = False
    args.p = 1
    args.purity_target = 0.1
    args.meada_used = False

    lrs = {"LS": 0.5, "fLS": 0.5, "iLS": 0.5,
           "fiLS": 0.5, "MLE": 0.01, "CSE": 0.5}
    n_exp = 10

    for na_state in ["GHZ", "W"]:
        args.na_state = na_state

        for n_meas in [2, 20, 50, 70, 100][::-1]:
            args.n_meas = n_meas

            for method in ["CSE"]:
                args.method = method
                args.lr = lrs[method]
                args.n_tols = int(n_meas * 1e4)

                save_data = {}
                for idx in range(n_exp):
                    print(idx)
                    results = basic_exp(args)
                    save_data[idx] = results

                save_file(
                    save_data, ["exp2-t", args.na_state],
                    [args.method, args.n_qubits,
                        args.n_meas, args.rank, args.n_tols]
                )


# Experiment C, IF with number of operators, full estimate state, pure target state
def exp_2p(args):
    args.n_qubits = 7
    args.n_epochs = int(1e3)
    args.n_shots = 1
    args.rank = 1
    args.lamda = 10
    args.p_esti = False
    args.p = 1
    args.purity_target = 0.1
    args.meada_used = False

    lrs = {"LS": 0.5, "fLS": 0.5, "iLS": 0.5,
           "fiLS": 0.5, "MLE": 0.01, "CSE": 0.5}
    n_exp = 10

    for na_state in ["GHZ", "W"]:
        args.na_state = na_state

        for n_meas in [2, 20, 50, 70][::-1]:
            args.n_meas = n_meas

            for method in ["CSE"]:
                args.method = method
                args.lr = lrs[method]
                args.n_tols = int(n_meas * 1e4)

                save_data = {}
                for idx in range(n_exp):
                    print(idx)
                    results = basic_exp(args)
                    save_data[idx] = results

                save_file(
                    save_data, ["exp2-p", args.na_state],
                    [args.method, args.n_qubits,
                        args.n_meas, args.rank, args.n_tols]
                )


if __name__ == "__main__":
    # ----------parameters----------
    print('-'*20 + 'set parser' + '-'*20)
    parser = argparse.ArgumentParser()

    # parameters
    parser.add_argument("--POVM", type=str,
                        default="Pauli_normal", help="type of POVM")
    parser.add_argument("--K", type=int, default=4,
                        help='number of operators in single-qubit POVM')

    parser.add_argument("--na_state", type=str, default="GHZ", choices=["GHZ", "W"],
                        help="name of state")
    parser.add_argument("--n_qubits", type=int, default=7,
                        help="number of qubits")
    parser.add_argument("--rank", type=int, default=int(2**7),
                        help="rank of estimated state")
    parser.add_argument("--n_tols", type=int, default=int(1e6),
                        help="number of total samples")
    parser.add_argument("--n_epochs", type=int,
                        default=int(1e3), help="number of epochs")
    parser.add_argument("--n_shots", type=int, default=1,
                        help="number of shots of each Pauli operator")
    parser.add_argument("--n_meas", type=int, default=100,
                        help="number of Pauli operator")

    # method
    parser.add_argument("--method", type=str, default="CSE", choices=[
                        "LS", "iLS", "fLS", "fiLS", "DFE", "MLE", "CSE"], help="method of QST")

    parser.add_argument("--lr", type=float, default=0.5,
                        help="learning rate")
    parser.add_argument("--lamda", type=float,
                        default=10, help="parameter of iLS and fiLS")
    parser.add_argument("--p", type=float, default=1,
                        help="p of werner state (p * \rho + (1-p)/d * I)")
    parser.add_argument("--purity_target", type=float,
                        default=0.1, help="purity estimation of target state and CSE parameter")
    parser.add_argument("--p_esti", type=bool, default=False,
                        help="purity estimation or not")
    parser.add_argument("--meada_used", type=bool, default=False,
                        help="use history measure data or not")

    args = parser.parse_args()
    args.dtype = np.complex64

    results = basic_exp(args)
    # exp_full(args)
    # exp_1(args)
    # exp_2(args)
    # exp_1p(args)
    # exp_2p(args)
