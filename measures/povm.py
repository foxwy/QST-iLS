# -*- coding: utf-8 -*-
# @Author: yong
# @Date:   2023-07-30 19:53:29
# @Last Modified by:   WY
# @Last Modified time: 2023-10-12 15:21:32
# @source: https://github.com/petr-ivashkov/qiskit-community-tutorials/blob/master/terra/qis_adv/two_approaches_to_implement_povms.ipynb

import numpy as np
from numpy.linalg import svd
import scipy
from scipy.linalg import sqrtm
from cmath import pi, sqrt, exp
from qiskit.visualization import plot_histogram

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, Aer, transpile
from qiskit.extensions import UnitaryGate


def theoretical_probs(povm, state, binary_tree=False):
    """ Precalculates the expected probabilities for a pure state
        for the given POVM.
    """
    rho = state.T.conj() @ state  # density matrix of the state
    probs = [np.trace(povm[i] @ rho).round(2).real for i in range(len(povm))]

    l = len(probs)
    n = int(np.ceil(np.log2(l)))
    prob_counts = {}
    for i in range(l):
        if binary_tree:
            key = np.binary_repr(i, n)[::-1]
        else:
            key = np.binary_repr(i, n)
        value = probs[i]
        prob_counts[key] = value

    return prob_counts


def get_measurement_op(povm, start, end):
    """
    Returns a cumulative measurement operator by grouping together
    POVM elements from povm[start] to povm[end-1]
    """
    return np.sum(povm[start:end], axis=0).round(5)


def get_diagonalization(povm, start, end):
    """
    Returns: 
        Kraus operator <M>, diagonal <D> and the modal matrix <V> for a given 
        measurement operator by diagonalizing the measurement operator in the form:

        M = V@D@Vh such that M@M = E
    """
    E = get_measurement_op(povm, start, end)
    d2, V = np.linalg.eig(E)
    D = np.real(np.sqrt(np.diag(d2)))
    M = V @ D @ np.linalg.inv(V)
    return M, D, V


def get_next_level_binary_kraus_ops(povm, start, end):
    """
    Computes two next level binary Kraus operators 
    Args:
        povm: numpy array of POVM elements
        start/end: indices which define the cumulative POVM element
    Returns: 
        Two binary Kraus operators b0 and b1 which take from a higher to lower branch

    * <M> is the Kraus operator corresponding to the current level in binary tree
    * <M0> (<M1>) is the Kraus operators corresponding to the left (right) branch
    * <Q> asserts the completeness condition: b0@b0.T.conj() +  b1@b1.T.conj() = I
    * <M_psinv> is the Moore-Penrose pseudo-inverse of <M>
    """
    mid = int(start + (end - start) / 2)
    # computing <M>
    M, D, V = get_diagonalization(povm, start, end)
    # computing the null space of <M>
    P = np.sign(D.round(5))
    Pc = np.eye(len(M)) - P
    Q = V @ Pc @ V.T.conj()
    # computing <M_psinv>
    # D_inv = np.zeros_like(D)
    # for i in range(len(M)):
    #    if D[i,i].round(5) == 0: continue
    #    else: D_inv[i,i] = np.real(1/D[i,i])
    D_inv = np.linalg.pinv(D)
    M_psinv = V @ D_inv @ V.T.conj()
    # computing <M0> and <M1>
    M0, _, _ = get_diagonalization(povm, start, mid)
    M1, _, _ = get_diagonalization(povm, mid, end)
    # computing <b0> and <b1>
    b0 = M0 @ M_psinv + Q / np.sqrt(2)
    b1 = M1 @ M_psinv + Q / np.sqrt(2)
    return b0, b1


def closest_unitary(A):
    """ Calculates the unitary matrix U that is closest with respect to the
        operator norm distance to the general matrix A.
    """
    V, _, Wh = scipy.linalg.svd(A)
    U = np.matrix(V.dot(Wh))
    return U


def closest_unitary_schur(A):
    T, Z = scipy.linalg.schur(A, output='complex')
    return Z @ np.diag(np.diag(T) / abs(np.diag(T))) @ Z.T.conj()


def extend_to_unitary(b0, b1):
    """ Creates a coupling unitary between the system and ancilla.

        The condition for unitary is: <0|U|0> = b0 and <1|U|0> = b1,
        whereby the ancilla is projected onto states |0> and |1>.

        A two-column matrix A, with its upper left block given by b0, 
        and bottom left block by b1, is extended to unitary U by appending  
        a basis of the null space of A.
    """
    A = np.concatenate((b0, b1))
    u, _, _ = scipy.linalg.svd(A)
    y = u[:, len(A[0]):]
    U = np.hstack((A, y))
    # verify U is close to unitary
    assert np.allclose(U.T.conj() @ U, np.eye(len(A)), atol=1e-03), "Failed to construct U"
    return closest_unitary(U)


# Class definitions
class POVM:
    """Base class that holds an arbitrary POVM <povm> as a list of <N> POVM elements.
    """

    def __init__(self, povm):
        """
        Constructor asserts that the given POVM is valid.
        """
        self.povm = povm
        self.N = len(povm)
        self.depth = int(np.ceil(np.log2(self.N)))  # required depth of the binary tree
        self.povm_dim = len(povm[0])  # dimension of the POVM operators
        self.n_qubits = int(np.log2(self.povm_dim))  # number of system qubits
        assert self.is_valid()

    def is_valid(self):
        """Verifies the hermiticity, positivity of the POVM and that
        the POVM resolves the identity.
        Returns:
            True: if all conditions are satisfied
        Raises:
            Assertion Error: if one of conditions is not satisfied
        """
        for E in self.povm:
            assert np.allclose(E.conj().T, E), "Some POVM elements are not hermitian"
            assert np.all(np.linalg.eigvals(E).round(3) >= 0), "Some POVM elements are not positive semi-definite"
        assert np.allclose(sum(self.povm), np.eye(self.povm_dim)), "POVM does not resolve the identity"
        return True


class BinaryTreePOVM(POVM):
    """Class which implements the binary tree approach as described in https://arxiv.org/abs/0712.2665 
    to contruct a POVM measurement tree. 
    """

    def __init__(self, povm, inst, dual=False):
        """Creates a binary stree structure with BinaryMeasurementNode objects
        stored in <nodes> dictionary. The keys in the dictionary are the bitstrings 
        corresponding to the states of the classical register at the point when the
        corresponding node has been "reached".

        Args:
            povm: list of POVM elements
        """
        super().__init__(povm)
        # pad with zero operators if necessary
        while np.log2(self.N) - self.depth != 0:
            self.povm.append(np.zeros_like(self.povm[0]))
            self.N += 1

        self.nodes = {}
        self.dual = dual

        self.create_binary_tree(key="0", start=0, end=self.N)
        self.qc = self.construct_measurement_circuit(inst)

    def create_binary_tree(self, key, start, end):
        """Recursive method to build the measurement tree.
        Terminates when the fine-grain level corresponding to the single POVM
        elements is reached.

        <start> and <end> are the first and (last-1) indices of POVM elements
        which were grouped together to obtain a cumulative coarse-grain operator. 
        The range [start, end) corresponds to the possible outcomes which "sit" in 
        the branches below.
        """
        if start >= (end - 1):
            return
        new_node = BinaryMeasurementNode(self.povm, key=key, start=start, end=end)
        self.nodes[key] = new_node
        mid = int(start + (end - start) / 2)
        self.create_binary_tree(new_node.left, start=start, end=mid)
        self.create_binary_tree(new_node.right, start=mid, end=end)

    def construct_measurement_circuit(self, inst):
        """Contructs a quantum circuit <qc> for a given POVM by sequentially appending
        coupling unitaries <U> and measurements conditioned on the state of the
        classical register <cr>. The method uses BFS traversal of the precomputed 
        binary measurement tree, i.e. the measurement nodes are visited in level-order.

        * Traversal terminates when the fine-grain level was reached.
        * Ancilla qubit is reset before each level.
        * The root node has the key "0".

        * The <if_test> instruction is applied to the entire classical register <cr>,
          whereby the value is the key of the corresponding node - padded with zeros 
          from right to the length of the <cr> register - and interpreted as an integer.

          Example:
            At the first level the two nodes have keys:
                left = "00" and right = "01"
            If the <cr> is 3 bits long, then the left/right unitary is applied if 
            the state of <cr> is int("000",2) = 0 / int("010",2) = 2
        """
        qr = QuantumRegister(self.n_qubits + 1)
        cr = ClassicalRegister(self.depth)
        qc = QuantumCircuit(qr, cr, name="measurement-circuit")

        qc.append(inst, range(self.n_qubits))
        root = self.nodes["0"]
        U_gate = UnitaryGate(root.U, label=root.key)
        qc.append(U_gate, range(self.n_qubits + 1))
        if self.dual:  # works for depth = 2
            qc.x(self.n_qubits)
        qc.measure(self.n_qubits, cr[0])
        if self.depth == 1:
            return qc
        qc.x(self.n_qubits).c_if(cr[0], 1)

        current_level = [self.nodes["00"], self.nodes["10"]]

        for i in range(1, self.depth):
            next_level = []
            for node in current_level:
                U_gate = UnitaryGate(node.U, label=node.key)
                cr_state = int(node.key[:-1], 2)
                with qc.if_test((cr, cr_state)):
                    qc.append(U_gate, range(self.n_qubits + 1))
                if node.left in self.nodes:
                    next_level.append(self.nodes[node.left])
                if node.right in self.nodes:
                    next_level.append(self.nodes[node.right])
                current_level = next_level
            # dual condition must be checked here for larger systems
            qc.measure(self.n_qubits, cr[i])
            if i == self.depth - 1:
                continue
            qc.x(self.n_qubits).c_if(cr[i], 1)  # instead of resetting apply conditional X gate

        return qc


class BinaryMeasurementNode(POVM):
    """A BinaryMeasurementNode object is a node in the BinaryTreePOVM.
    It contains:
        1. Its <key> in the <nodes> dictionary.
        2. <start> and <end>: the first and (last-1) indices of the accumulated 
        POVM elements, corresponding this node.
        3. Coupling unitary <U>.
        4. Keys <left> and <right> of the two children nodes.
        5. Attributes of the POVM class: <M>, <N>, <M_dim>.
        6. Its level <level> in the binary tree, where level of the root node is 0.
    """

    def __init__(self, povm, key, start, end):
        super().__init__(povm)
        self.key = key
        self.level = len(self.key) - 1
        self.start = start
        self.end = end
        self.left = "0" + self.key
        self.right = "1" + self.key
        b0, b1 = get_next_level_binary_kraus_ops(self.povm, self.start, self.end)
        self.U = extend_to_unitary(b0, b1)

    def __str__(self):
        line1 = 'Node with the key {} at level {}\n'.format(self.key, self.level)
        line2 = 'Cumulative operator = [{},{})'.format(self.start, self.end)
        line3 = 'left = {}, right = {}\n'.format(self.left, self.right)
        line4 = 'U = \n{}\n'.format(self.U)
        return line1 + line2 + line3 + line4


def check_for_rank_one(povm):
    """
    function to check if a povm is a rank-1 povm
    """
    rank_one = True
    for p in povm:
        if np.linalg.matrix_rank(p) != 1:
            rank_one = False
            return rank_one
        else:
            continue
    return rank_one


def compute_rank_one_unitary(povm, atol=1e-13, rtol=0):
    """
    This function computes the unitary that rotates the system to the Hilbert space of the ancilla
    Input:  POVM ---> a list of the elements of POVM
    Output: Unitary matrix
    """

    # check if povm is a rank-1 povm:
    assert check_for_rank_one(povm), "This is not a rank-1 povm"
    new_povm = []
    for p in povm:
        if np.log2(len(povm)) % 2 == 0:  # still under investigation
            w, v = np.linalg.eig(p)
        else:
            w, v = np.linalg.eigh(p)  # note the that the eigenvenvector is computer for hermitian eigh
        for eigenvalue, engenvector in zip(w, v):
            if np.isclose(np.abs(eigenvalue), 0):
                continue
            else:
                new_p = np.sqrt(eigenvalue) * engenvector
                new_povm.append(new_p)
    v = np.vstack(new_povm)  # arrange the povm elements to form a matrix of dimension: Row X Col*len(povm)
    v = np.atleast_2d(v.T)  # convert to 2d matrix

    u, s, vh = svd(v)    # apply svd
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:]         # missing rows of v

    # add the missing rows of v to v
    V = np.vstack((v, ns))

    # make the unitary a square matrix of dimension N=2^n where n = int(np.ceil(np.log(V.shape[0])))
    n = int(np.ceil(np.log2(V.shape[0])))
    N = 2**n      # dimension of system and ancilla Hilber space
    r, c = V.shape

    U = np.eye(N, dtype=complex)  # initialize Unitary matrix to the identity. Ensure it is complex
    U[:r, :c] = V[:r, :c]  # assign all the elements of V to the corresponding elements of U

    U = U.conj().T  # Transpose the unitary so that the rows are the povm

    # check for unitarity of U
    assert np.allclose(U.T.conj() @ U, np.eye(N), atol=1e-13), "Failed to construct U"

    return U


# Using the original unitary generator
def compute_full_rank_unitary(povm, atol=1e-13, rtol=0):
    """
    This function computes the unitary that rotates the system to the Hilbert space of the ancilla
    Input:  POVM ---> a list of the elements of POVM
    Output: Unitary matrix
    """

    # Here square root of the POVM elements were used as a replacement for the vector that form the povm
    povm = [sqrtm(M)for M in povm]

    v = np.hstack(povm)  # arrange the povm elements to form a matrix of dimension: Row X Col*len(povm)
    v = np.atleast_2d(v)  # convert to 2d matrix
    u, s, vh = svd(v)    # apply svd
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:]         # missing rows of v

    # add the missing rows of v to v
    V = np.vstack((v, ns))

    # make the unitary a square matrix of dimension N=2^n where n = int(np.ceil(np.log(V.shape[0])))
    n = int(np.ceil(np.log2(V.shape[0])))
    N = 2**n      # dimension of system and ancilla Hilber space
    r, c = V.shape

    U = np.eye(N, dtype=complex)  # initialize Unitary matrix to the identity. Ensure it is complex
    U[:r, :c] = V[:r, :c]  # assign all the elements of V to the corresponding elements of U

    U = U.conj().T  # Transpose the unitary so that the rows are the povm

    # check for unitarity of U
    assert np.allclose(U.T.conj() @ U, np.eye(N), atol=1e-07), "Failed to construct U"

    return U


def rank_one_circuit(povm, inst, U, num_system_qubit):
    N = U.shape[0]  # Dimension of the unitary to be applied to system and ancilla
    num_ancilla_qubit = int(np.ceil(np.log2(N))) - num_system_qubit  # total number of qubits for system

    all_num = num_ancilla_qubit + num_system_qubit
    qr = QuantumRegister(all_num)
    qc = QuantumCircuit(qr, name='circuit')
    qc.append(inst, range(num_system_qubit))

    # reset ancilla to zero
    qc.reset(range(num_system_qubit, all_num))

    # append the unitary gate
    U_gate = UnitaryGate(U, label='U')  # unitary gate to be applied between system and ancilla
    qc.append(U_gate, range(all_num))

    # measure only the ancilliary qubits
    qc.measure_all()

    return qc


def full_rank_circuit(povm, inst, U, num_system_qubit):
    N = U.shape[0]  # Dimension of the unitary to be applied to system and ancilla
    num_ancilla_qubit = int(np.ceil(np.log2(N))) - num_system_qubit  # total number of qubits for system

    all_num = num_ancilla_qubit + num_system_qubit
    classical_reg = ClassicalRegister(num_ancilla_qubit, name='measure')  # classical register
    qr = QuantumRegister(all_num)
    qc = QuantumCircuit(qr, classical_reg, name='circuit')
    qc.append(inst, range(num_system_qubit))

    # reset ancilla to zero
    qc.reset(range(num_system_qubit, all_num))

    # append the unitary gate
    U_gate = UnitaryGate(U, label='U')  # unitary gate to be applied between system and ancilla
    qc.append(U_gate, range(all_num))

    # measure only the ancilliary qubits
    qc.measure(range(num_system_qubit, all_num), classical_reg)

    return qc


def construct_quantum_circuit(povm, inst, num_system_qubit):

    # compute unitary matrix
    if check_for_rank_one(povm):
        U = compute_rank_one_unitary(povm)
        qc = rank_one_circuit(povm, inst, U, num_system_qubit)
    else:
        U = compute_full_rank_unitary(povm)
        qc = full_rank_circuit(povm, inst, U, num_system_qubit)

    return qc


def draw_circuit(qc, idle_wires=True):
    """
    This functions draws the naimark extension quantum circuit
    """

    return qc.draw(output='mpl', idle_wires=idle_wires)


def naimark_plot(povm, state, counts, names, save=False, file_name="povm_output.pdf"):
    theory_count = theoretical_probs(povm, state)
    count_list = [theory_count]
    # legend_list = ["Theoretical result"]
    legend_list = ['Theory (t)']

    for count, name in zip(counts, names):
        count_list.append(count)
        legend_list.append(name)

    fig = plot_histogram(count_list, legend=legend_list, bar_labels=True)
    ax = fig.axes[0]
    title = f"Tetrad POVM with Naimark's extension approach"
    ax.set_title(title, fontsize=12)
    ax.set_ylabel('Probabilities', fontsize=12)
    ax.set_xlabel('POVM elements', fontsize=12)

    x_lables = []
    for i in range(len(povm)):
        x_lables.append("M" + str(i))
    ax.set_xticks(range(len(povm)))
    ax.set_xticklabels(x_lables)

    ax.legend(fontsize=12)
    fig.tight_layout()

    if save:
        fig.savefig(file_name)

    return fig


if __name__ == "__main__":
    # Define the set of vectors on the Bloch sphere
    psi_0 = np.array([[1 / sqrt(2)], [0]])
    psi_1 = np.array([[1], [sqrt(2) * exp(2j * pi / 3)]]) / sqrt(6)
    psi_2 = np.array([[1], [sqrt(2) * exp(4j * pi / 3)]]) / sqrt(6)
    psi_3 = np.array([[1], [sqrt(2)]]) / sqrt(6)

    # POVM elements are given as M_i = |psi_i><psi_i|
    M0 = psi_0 @ psi_0.conj().T
    M1 = psi_1 @ psi_1.conj().T
    M2 = psi_2 @ psi_2.conj().T
    M3 = psi_3 @ psi_3.conj().T

    # POVM is a list of its elements
    povm = [np.kron(M0, M0), np.eye(4) - np.kron(M0, M0)]

    # Initial (pure) state of the system is |0>
    state = np.array([[1, 0, 0, 0]])
    state = state / np.linalg.norm(state)

    #btp = BinaryTreePOVM(povm)
    #qc = btp.qc
    qc = construct_quantum_circuit(povm, state)
    backend = Aer.get_backend('aer_simulator_matrix_product_state')
    transpiled_circuit = transpile(qc, backend)

    # counts
    result = backend.run(transpiled_circuit, shots=1024).result()
    counts = result.get_counts()
    print(counts)

    naimark_plot(povm, state, [counts], ['simu'], save=False)
    # plt.show()
