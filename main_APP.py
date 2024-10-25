# -*- coding: utf-8 -*-
# @Author: yong
# @Date:   2023-07-29 14:16:20
# @Last Modified by:   yong
# @Last Modified time: 2024-03-28 13:19:58

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from measures.qiskit_quantum import Circuit_meas
from model.GD_APP import FEQST
from basis.Basis_State import State
from basis.Basic_Function import save_file


def basic_exp(args):
    """basic experiment based different setups (args)"""

    print('\nargs:', args)
    circ = Circuit_meas(args.na_state, args.n_qubits)

    # prepare state
    if args.na_state == "GHZ":
        rho_star = State().Get_GHZ_s(args.n_qubits, circ.w_s)
        rho_star = np.array(rho_star).astype(args.dtype).reshape(-1, 1)
        nonzero_idx = 2**np.array([0, args.n_qubits]) - 1
        rho_init = np.zeros((2, 1)).astype(args.dtype)
        rho_init[0] = 1

    elif args.na_state == "W":
        rho_star = State().Get_W_s(args.n_qubits, circ.w_s)
        rho_star = np.array(rho_star).astype(args.dtype).reshape(-1, 1)
        nonzero_idx = 2**np.arange(args.n_qubits)
        rho_init = np.zeros((args.n_qubits, 1)).astype(args.dtype)
        rho_init[0] = 1

    # learn state from measurement
    qst = FEQST(args)
    results = qst.guide(circ, rho_star, rho_init, nonzero_idx)

    return results


def exp_3(args):  # pos prior
    args.n_qubits = 7
    args.n_epochs = int(1e3)
    args.n_shots = 1

    lrs = {"iLS": 0.1, "DFE": 0.002, "MLE": 0.001}
    n_exp = 10

    for na_state in ["GHZ", "W"]:
        args.na_state = na_state

        for n_meas in [100]:
            args.n_meas = n_meas

            for method in ["iLS", "DFE", "MLE"]:
                args.method = method
                args.lr = lrs[method]

                for n_tols in [4, 5, 6]:
                    args.n_tols = 10**n_tols

                    if method in ["MLE"]:
                        args.n_shots = round(args.n_tols / args.n_meas)
                    else:
                        args.n_shots = 1

                    save_data = {}
                    for idx in range(n_exp):
                        print(idx)
                        results = basic_exp(args)
                        save_data[idx] = results

                    save_file(
                        save_data, ["exp3-t", args.na_state],
                        [args.method, args.n_qubits, n_meas, args.n_tols]
                    )


def exp_4(args):  # pos prior
    args.n_qubits = 7
    args.n_epochs = int(1e3)
    args.n_shots = 1

    lrs = {"iLS": 0.1, "DFE": 0.002, "MLE": 0.001}
    n_exp = 10

    for na_state in ["GHZ", "W"]:

        args.na_state = na_state

        for n_meas in [2, 20, 50, 70, 100]:
            args.n_meas = n_meas

            for method in ["DFE"]:
                args.method = method
                args.lr = lrs[method]
                args.n_tols = int(n_meas * 1e4)

                if method in ["MLE"]:
                    args.n_shots = round(args.n_tols / args.n_meas)
                    args.n_epochs = int(1e3)
                else:
                    args.n_shots = 1
                    args.n_epochs = int(1e3)

                save_data = {}
                for idx in range(n_exp):
                    print(idx)
                    results = basic_exp(args)
                    save_data[idx] = results

                save_file(
                    save_data, ["exp4-t", args.na_state],
                    [args.method, args.n_qubits, n_meas, args.n_tols]
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

    parser.add_argument("--na_state", type=str, default="GHZ",
                        choices=["GHZ", "W", "Random_Haar"], help="name of state")
    parser.add_argument("--n_qubits", type=int, default=7,
                        help="number of qubits")
    parser.add_argument("--n_tols", type=int, default=int(1e6),
                        help="number of total measurement samples")
    parser.add_argument("--n_epochs", type=int,
                        default=int(1e3), help="number of epochs of SPSA")
    parser.add_argument("--n_shots", type=int, default=1,
                        help="number of shots of each Pauli operator")
    parser.add_argument("--n_meas", type=int, default=100,
                        help="number of Pauli operator")

    # method
    parser.add_argument("--method", type=str, default="iLS",
                        choices=["iLS", "DFE", "MLE"], help="method of QST")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")

    args = parser.parse_args()
    args.dtype = np.complex64

    results = basic_exp(args)
    # exp_3(args)
    # exp_4(args)
