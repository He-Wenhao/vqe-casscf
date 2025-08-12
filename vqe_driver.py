import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
from openfermion import InteractionOperator, get_fermion_operator
from openfermion.transforms import jordan_wigner
from openfermion.linalg import get_sparse_operator
from itertools import product
from pennylane.qchem import hf_state, excitations
from pyscf import gto, scf, mcscf
from pyscf.fci.direct_spin1 import FCI as DirectSpinFCI
import matplotlib.pyplot as plt
from pathlib import Path
import scipy
from pyscf.fci import cistring
from pyscf import fci
from math import comb

# Core VQE implementation
def pennylane_vqe(H_const, h1, g2, nele, init_params=None):
    # Integrals
    def add_spin_1bd(m):
        dim = m.shape[0]
        res = np.zeros((2*dim,2*dim))
        for p, q in product(range(dim), repeat=2):
            res[2*p, 2*q] = m[p,q]
            res[2*p+1, 2*q+1] = m[p,q]
        return res

    def add_spin_2bd(g):
        dim = g.shape[0]
        res = np.zeros((2*dim,)*4)
        for p, q, r, s in product(range(dim), repeat=4):
            val = g[p,q,r,s]
            if abs(val) < 1e-12:
                continue
            for σ1, σ2 in product(range(2), repeat=2):
                res[2*p+σ1, 2*q+σ2, 2*r+σ2, 2*s+σ1] = val
        return res / 2.0

    int1_spin = add_spin_1bd(h1)
    g_phys = g2.transpose((0, 2, 3, 1))
    int2_spin = add_spin_2bd(g_phys)

    # fermionic to qubit Hamiltonian
    intop = InteractionOperator(H_const, int1_spin, int2_spin)
    ferm_ham = get_fermion_operator(intop)
    qub_ham = jordan_wigner(ferm_ham)
    n_qubits = max(i for term in qub_ham.terms for i, _ in term) + 1
    Hmat = get_sparse_operator(qub_ham, n_qubits).toarray()
    H = qml.Hermitian(Hmat, wires=range(n_qubits))

    # UCCSD ansatz
    singles, doubles = excitations(nele, n_qubits)
    hf_bits = hf_state(nele, n_qubits)
    n_sing = len(singles)
    n_doub = len(doubles)

    def ansatz(params):
        qml.BasisState(hf_bits, wires=range(n_qubits))
        for θ, ex in zip(params[:n_sing], singles):
            qml.SingleExcitation(θ, wires=ex)
        for θ, ex in zip(params[n_sing:], doubles):
            qml.DoubleExcitation(θ, wires=ex)

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(params):
        ansatz(params)
        return qml.expval(H)

    cost = lambda p: pnp.real(circuit(p))
    n_params = n_sing + n_doub

    # initialise parameters
    if init_params is None:
        #np.random.seed(0)
        #params = pnp.random.normal(0, 0.01, n_params, requires_grad=True)
        params = pnp.zeros(n_params, requires_grad=True)
    else:
        params = pnp.array(init_params, requires_grad=True)

    # Run VQE
    opt = qml.AdamOptimizer(0.05)
    energy = None
    for i in range(200):
        params, energy = opt.step_and_cost(cost, params)
        if i % 50 == 0:
            print(f"  PL-VQE iter {i:3d} → E = {energy:.8f} Ha")

    energy = pnp.real(circuit(params))
    ################################
    ## implementation of ci_vec
    ################################
    #raise NotImplementedError("ci_vec not defined.")
    @qml.qnode(dev, interface="autograd")
    def _state_qnode(params):
        ansatz(params)
        return qml.state()
    
    psi = _state_qnode(params)
    n_spatial = n_qubits // 2
    
    ci_vec = qubit_state_to_ci(psi, n_orb=n_spatial, nelec=nele)
    assert abs(ci_vec.imag).max() < 1e-10, "CI vector has non-zero imaginary part."
    ci_vec = ci_vec.real
    print('data type of ci_vec:', ci_vec.dtype)
    e_cas_from_ci = fci.direct_spin1.energy(h1, g2, ci_vec, n_spatial, (nele//2, nele//2))
    e_cas_from_ci += H_const
    print(f"  FCI energy from CI vector: {e_cas_from_ci:.8f} Ha")
    print(f"  VQE energy: {energy:.8f} Ha")
    #print("Hconst:", H_const)
    #assert abs(e_cas_from_ci - energy) < 1e-6, "FCI energy from CI vector does not match VQE energy."
    print(f"energy difference: {abs(e_cas_from_ci - energy):.8f} Ha")


    return float(energy), params, ci_vec


# PennyLane‐based FCI solver
class PennyLaneSolver:
    def __init__(self, mf):
        self._mf = mf
        self.params = None
        
        # target the singlet ground‐state
        self.spin = 0
        self.nroots = 1
        self.root = 0
        
        # classical FCI solver for RDM
        self.classical = DirectSpinFCI()
        self.classical.spin = self.spin
        self.classical.nroots = self.nroots
        self.classical.root = self.root

    def kernel(self, h1e, g2e, norb, nelec, ci0=None, verbose=0, max_memory=None, ecore=None, **kwargs):
        # run PennyLane VQE for energy & params
        H_const = self._mf.energy_nuc()
        nele = nelec[0] + nelec[1]
        E, p, ci_vec = pennylane_vqe(H_const, h1e, g2e, nele, init_params=self.params)
        self.params = p

        # classical FCI for CI vector & RDM
        #e2, ci_vec = self.classical.kernel(h1e, g2e, norb, nelec, ci0=ci0, verbose=verbose, max_memory=max_memory, ecore=ecore, **kwargs)
        return E, ci_vec

    def make_rdm12(self, ci_vec, ncas, nelecas):
        # 1 and 2-RDM from the classical CI vector.
        return self.classical.make_rdm12(ci_vec, ncas, nelecas)
