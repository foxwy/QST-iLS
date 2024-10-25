# -*- coding: utf-8 -*-
# @Author: yong
# @Date:   2023-07-29 14:16:20
# @Last Modified by:   yong
# @Last Modified time: 2024-04-01 09:55:48

import argparse
import numpy as np
import matplotlib.pyplot as plt

from measures.qiskit_quantum import Circuit_meas
from model.GD_E import FEQST
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

    '''
    plt.plot(results["epoch"], results["iFq"])
    plt.yscale('log')
    plt.xscale('log')
    plt.show()'''

    return results


def exp_6(args):  # DFE_P_S
    args.na_state = "W"
    args.n_epochs = int(1e5)
    args.n_shots = 1
    args.n_meas = int(1e3)
    args.n_sample_one = 5

    args.method = "DFE"
    a = {10: 0.01, 15: 0.01, 20: 0.04, 25: 0.05}
    args.b = 0.1
    n_exp = 5

    for n_qubits in [25]:
        args.n_qubits = n_qubits
        args.a = a[n_qubits]

        for n_tols in [4]:
            args.n_tols = 10**n_tols

            save_data = {}
            for idx in range(n_exp):
                print(idx)
                results = basic_exp(args)
                save_data[idx] = results

            save_file(
                save_data, ["exp6-t", args.na_state], [
                    args.method, args.n_qubits, args.n_meas, n_tols, args.n_sample_one
                ]
            )


def exp_6_GHZ(args):  # DFE_P_S
    args.na_state = "GHZ"
    args.n_epochs = int(1e5)
    args.n_shots = 1
    args.n_meas = int(1e3)
    args.n_sample_one = 5

    args.method = "DFE"
    a = {10: 0.1, 15: 0.1, 20: 0.1, 25: 0.1}
    args.b = 0.1
    n_exp = 5

    for n_qubits in [25]:
        args.n_qubits = n_qubits
        args.a = a[n_qubits]

        for n_tols in [4, 5, 6]:
            args.n_tols = 10**n_tols

            save_data = {}
            for idx in range(n_exp):
                print(idx)
                results = basic_exp(args)
                save_data[idx] = results

            save_file(
                save_data, ["exp6-t", args.na_state], [
                    args.method, args.n_qubits, args.n_meas, n_tols, args.n_sample_one
                ]
            )


def exp_6_ils(args):  # DFE_P_S
    args.na_state = "W"
    args.n_epochs = 2 * int(1e5)
    args.n_shots = 1
    args.n_meas = int(1e3)
    args.n_sample_one = 5

    args.method = "iLS"
    a = {10: 0.2, 15: 1, 20: 10, 25: 5}
    args.b = 0.1
    n_exp = 5

    for n_qubits in [10, 15, 20, 25]:
        args.n_qubits = n_qubits
        args.a = a[n_qubits]

        for n_tols in [4, 5, 6]:
            args.n_tols = 10**n_tols

            save_data = {}
            for idx in range(n_exp):
                print(idx)
                results = basic_exp(args)
                save_data[idx] = results

            save_file(
                save_data, ["exp6-t", args.na_state], [
                    args.method, args.n_qubits, args.n_meas, n_tols, args.n_sample_one
                ]
            )


def exp_6_GHZ_ils(args):  # DFE_P_S
    args.na_state = "GHZ"
    args.n_epochs = 2 * int(1e5)
    args.n_shots = 1
    args.n_meas = int(1e3)
    args.n_sample_one = 5

    args.method = "iLS"
    a = {10: 1, 15: 2, 20: 5, 25: 15}
    args.b = 0.1
    n_exp = 5

    for n_qubits in [10, 15, 20, 25]:
        args.n_qubits = n_qubits
        args.a = a[n_qubits]

        for n_tols in [4, 5, 6]:
            args.n_tols = 10**n_tols

            save_data = {}
            for idx in range(n_exp):
                print(idx)
                results = basic_exp(args)
                save_data[idx] = results

            save_file(
                save_data, ["exp6-t", args.na_state], [
                    args.method, args.n_qubits, args.n_meas, n_tols, args.n_sample_one
                ]
            )


if __name__ == "__main__":
    # ----------parameters----------
    print('-'*20 + 'set parser' + '-'*20)
    parser = argparse.ArgumentParser()

    # parameters
    parser.add_argument("--na_state", type=str, default='GHZ',
                        choices=["GHZ", "W", "Random_Haar"], help="name of state")
    parser.add_argument("--n_qubits", type=int, default=10,
                        help="number of qubits")
    parser.add_argument("--n_tols", type=int, default=int(1e6),
                        help="number of total measurement samples")
    parser.add_argument("--n_epochs", type=int,
                        default=int(1e5), help="number of epochs of SPSA")
    parser.add_argument("--n_shots", type=int, default=1,
                        help="number of shots of each Pauli operator")
    parser.add_argument("--n_meas", type=int, default=1000,
                        help="number of Pauli operator")

    # method
    parser.add_argument("--method", type=str, default="iLS",
                        choices=["iLS", "DFE", "Shadow"], help="method of QST")

    parser.add_argument("--a", type=float, default=0.01,
                        help="parameter of SPSA")  # 2
    parser.add_argument("--b", type=float, default=0.1,
                        help="parameter of SPSA")  # 0.1

    args = parser.parse_args()
    args.dtype = np.complex64

    results = basic_exp(args)
    # exp_6(args)
    # exp_6_GHZ(args)
    # exp_6_ils(args)
    # exp_6_GHZ_ils(args)
