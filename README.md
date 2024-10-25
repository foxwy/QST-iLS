# **Quantum state tomography based on infidelity estimation**

The official Pytorch implementation of the paper named `Quantum state tomography based on infidelity estimation`.

### **Abstract**

Quantum state tomography is a fundamental technique in quantum information science for characterizing and benchmarking quantum systems based on measurement statistics. In this work, we introduce an infidelity-based least-squares estimator that integrates the state purity information, resulting in a orders of magnitude higher tomography accuracy compared to previous methods. This estimator is further enhanced by incorporating randomized technique from direct fidelity estimation, making it applicable to large-scale quantum systems. We validate the proposed estimators through extensive experiments conducted on the IBM Qiskit simulator. Our results demonstrate that the estimator achieves an $\mathcal{O}(1/N)$ infidelity scaling with Pauli sample size $N$ for (nearly) pure states. Further, it enables high-precision pure-state tomography for systems of up to 25-qubit states, given some priors about the state. The proposed method offers a novel perspective on the union of advanced tomography techniques and state property estimation.

## Getting started

This code was tested on the computer with a single Intel(R) Core(TM) i7-12700KF CPU @ 3.60GHz with 64GB RAM and a single NVIDIA GeForce RTX 3090 Ti GPU with 24.0GB RAM, and requires:

- Python 3.9
- conda3
- matplotlib==3.8.4
- numpy==1.24.3
- qiskit==0.43.1
- qiskit_aer==0.12.0
- qiskit_ibmq_provider==0.20.2
- qiskit_ignis==0.7.1
- qiskit_terra==0.24.1
- SciencePlots==2.1.1
- scipy==1.14.1
- seaborn==0.13.2
- torch==2.4.1
- tqdm==4.66.5

## Runs exp from B to E and Appendix

### 1. Run exp B and C (`main_BC`)

```python
parser = argparse.ArgumentParser()

# parameters
parser.add_argument("--POVM", type=str, default="Pauli_normal", help="type of POVM")
parser.add_argument("--K", type=int, default=4, help='number of operators in single-qubit POVM')

parser.add_argument("--na_state", type=str, default="GHZ", choices=["GHZ", "W"], help="name of state")
parser.add_argument("--n_qubits", type=int, default=7, help="number of qubits")
parser.add_argument("--rank", type=int, default=int(2**7), help="rank of estimated state")
parser.add_argument("--n_tols", type=int, default=int(1e6), help="number of total samples")
parser.add_argument("--n_epochs", type=int, default=int(1e3), help="number of epochs")
parser.add_argument("--n_shots", type=int, default=1, help="number of shots of each Pauli operator")
parser.add_argument("--n_meas", type=int, default=100, help="number of Pauli operator")

# method
parser.add_argument("--method", type=str, default="CSE", choices=["LS", "iLS", "fLS", "fiLS", "DFE", "MLE", "CSE"], help="method of QST")

parser.add_argument("--lr", type=float, default=0.5, help="learning rate")
parser.add_argument("--lamda", type=float, default=10, help="parameter of iLS and fiLS")
parser.add_argument("--p", type=float, default=1, help="p of werner state (p * \rho + (1-p)/d * I)")
parser.add_argument("--purity_target", type=float, default=0.1, help="purity estimation of target state and CSE parameter")
parser.add_argument("--p_esti", type=bool, default=False, help="purity estimation or not")
parser.add_argument("--meada_used", type=bool, default=False, help="use history measure data or not")

args = parser.parse_args()
args.dtype = np.complex64

results = basic_exp(args)

```

### 2. Run exp Appendix (`main_APP`)

```python
parser = argparse.ArgumentParser()

# parameters
parser.add_argument("--POVM", type=str, default="Pauli_normal", help="type of POVM")
parser.add_argument("--K", type=int, default=4, help='number of operators in single-qubit POVM')

parser.add_argument("--na_state", type=str, default="GHZ", choices=["GHZ", "W", "Random_Haar"], help="name of state")
parser.add_argument("--n_qubits", type=int, default=7, help="number of qubits")
parser.add_argument("--n_tols", type=int, default=int(1e6), help="number of total measurement samples")
parser.add_argument("--n_epochs", type=int, default=int(1e3), help="number of epochs of SPSA")
parser.add_argument("--n_shots", type=int, default=1, help="number of shots of each Pauli operator")
parser.add_argument("--n_meas", type=int, default=100, help="number of Pauli operator")

# method
parser.add_argument("--method", type=str, default="iLS", choices=["iLS", "DFE", "MLE"], help="method of QST")
parser.add_argument("--lr", type=float, default=0.1, help="learning rate")

args = parser.parse_args()
args.dtype = np.complex64

results = basic_exp(args)
```

### 3. Run exp D (`main_D`)

```python
parser = argparse.ArgumentParser()

# parameters
parser.add_argument("--na_state", type=str, default="GHZ", choices=["GHZ", "W", "Random_Haar"], help="name of state")
parser.add_argument("--n_qubits", type=int, default=15, help="number of qubits")
parser.add_argument("--n_tols", type=int, default=int(1e5), help="number of total measurement samples")
parser.add_argument("--n_epochs", type=int, default=int(2 * 1e5), help="number of epochs of SPSA")
parser.add_argument("--n_shots", type=int, default=1, help="number of shots of each Pauli operator")
parser.add_argument("--n_meas", type=int, default=1000, help="number of Pauli operator")

# method
parser.add_argument("--method", type=str, default="iLS", choices=["iLS", "DFE"], help="method of QST")

parser.add_argument("--a", type=float, default=1, help="parameter of SPSA")  # 2
parser.add_argument("--b", type=float, default=0.1, help="parameter of SPSA")  # 0.1

args = parser.parse_args()
args.dtype = np.complex64

results = basic_exp(args)
```

### 4. Run exp E (`main_E`)

```python
parser = argparse.ArgumentParser()

# parameters
parser.add_argument("--na_state", type=str, default='GHZ', choices=["GHZ", "W", "Random_Haar"], help="name of state")
parser.add_argument("--n_qubits", type=int, default=10, help="number of qubits")
parser.add_argument("--n_tols", type=int, default=int(1e6), help="number of total measurement samples")
parser.add_argument("--n_epochs", type=int, default=int(1e5), help="number of epochs of SPSA")
parser.add_argument("--n_shots", type=int, default=1, help="number of shots of each Pauli operator")
parser.add_argument("--n_meas", type=int, default=1000, help="number of Pauli operator")

# method
parser.add_argument("--method", type=str, default="iLS", choices=["iLS", "DFE", "Shadow"], help="method of QST")

parser.add_argument("--a", type=float, default=0.01, help="parameter of SPSA")  # 2
parser.add_argument("--b", type=float, default=0.1, help="parameter of SPSA")  # 0.1

args = parser.parse_args()
args.dtype = np.complex64

results = basic_exp(args)
```

## **License**

This code is distributed under an [Mozilla Public License Version 2.0](LICENSE).
