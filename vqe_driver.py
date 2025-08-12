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

from utils import *

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

        return E, ci_vec

    def make_rdm12(self, ci_vec, ncas, nelecas):
        # 1 and 2-RDM from the classical CI vector.
        return self.classical.make_rdm12(ci_vec, ncas, nelecas)

# Helper for VQE optimization
def run_vqe_cas(mol, mo_guess, ncas, nelecas, solver='VQE', mode='SCF'):
    mf_init = scf.RHF(mol); mf_init.kernel()
    if mode == 'SCF':
        mc = mcscf.CASSCF(mf_init, ncas=ncas, nelecas=nelecas)
    else:
        mode == "CI"
        mc = mcscf.CASCI(mf_init, ncas=ncas, nelecas=nelecas)
        
    mc.mo_coeff = mo_guess
    if solver == 'VQE':
        mc.fcisolver = PennyLaneSolver(mf_init)
    else:
        assert solver == 'FCI'

    e_tot, e_cas, ci_vec, mo_coeff, mo_energy = mc.kernel()
    if solver == 'FCI':
        mc.fcisolver.params = None
    return e_tot, mc.fcisolver.params, mo_coeff, ci_vec, mc

def casscf_overlap(mc1, mc2):
    """
    Compute the overlap between two CASSCF wavefunctions.
    
    Args:
        mc1: PySCF CASSCF object for wavefunction 1
        mc2: PySCF CASSCF object for wavefunction 2
        
    Returns:
        overlap: <Psi1 | Psi2>
        infidelity: 1 - |<Psi1 | Psi2>|^2
    """
    mol = mc1.mol
    assert mc1.ncas == mc2.ncas, "Active space size must match"
    assert mc1.nelecas == mc2.nelecas, "Active electron count must match"
    assert mol == mc2.mol, "Both wavefunctions must share the same molecule (AO basis)"
    
    # Extract CI and MO coefficients
    ci1, ci2 = mc1.ci, mc2.ci
    mo1, mo2 = mc1.mo_coeff, mc2.mo_coeff
    ncas, nelecas = mc1.ncas, mc1.nelecas

    # AO overlap matrix (shared AO basis)
    S_ao = mol.intor('int1e_ovlp')
    
    # MO overlap in active space
    mo1_act = mo1[:, mc1.ncore:mc1.ncore+ncas]
    mo2_act = mo2[:, mc2.ncore:mc2.ncore+ncas]
    S_mo = mo1_act.T @ S_ao @ mo2_act
    
    # CI overlap
    overlap = fci.addons.overlap(ci1, ci2, ncas, nelecas, S_mo)
    infidelity = 1 - abs(overlap)**2
    
    return infidelity

def perm_orca2pyscf(**kargs):
    mol = gto.M(**kargs)
    mol.verbose = 0
    def direct_sum(A, B):
        if type(A) == str:
            return B
        if type(B) == str:
            return A
        result = np.zeros((A.shape[0] + B.shape[0], A.shape[1] + B.shape[1]))
        result[:A.shape[0], :A.shape[1]] = A
        result[A.shape[0]:, A.shape[1]:] = B
        return result
    
    perm_block = {
        's':np.array([[1]]),
        'p':np.array([
                    [0,1,0],
                    [0,0,1],
                    [1,0,0]]),
        'd':np.array([
                    [0,0,0,0,1],
                    [0,0,1,0,0],
                    [1,0,0,0,0],
                    [0,1,0,0,0],
                    [0,0,0,1,0]]),
        }
    ind = 0
    perm_mat = 'None'
    while ind < len(mol.ao_labels()):
        l_val = mol.ao_labels()[ind][5]
        if l_val == 's':
            ind += 1
        elif l_val == 'p':
            ind += 3
        elif l_val == 'd':
            ind += 5
        else:
            raise TypeError('wrong l value')
        perm_mat = direct_sum(perm_mat,perm_block[l_val])
    return perm_mat

def ucc_ansatz_eval(mol, mo_guess, ncas, nelecas, label="", solver='VQE', vqe_params=None):
    if solver.upper() != 'VQE':
        raise ValueError("Only VQE solver is supported in this clean version")
    
    if vqe_params is None:
        raise ValueError("vqe_params must be provided for VQE")
    
    # Get active space integrals
    h1e, g2e = get_active_space_integrals(mol, mo_guess, ncas, nelecas)
    H_const = mol.energy_nuc()
    
    # Handle nelecas
    if isinstance(nelecas, (tuple, list)):
        n_electrons = sum(nelecas)
    else:
        n_electrons = nelecas
    
    # Single-shot evaluation
    
    # Add spin
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

    int1_spin = add_spin_1bd(h1e)
    g_phys = g2e.transpose((0, 2, 3, 1))
    int2_spin = add_spin_2bd(g_phys)

    # Build Hamiltonian
    intop = InteractionOperator(H_const, int1_spin, int2_spin)
    ferm_ham = get_fermion_operator(intop)
    qub_ham = jordan_wigner(ferm_ham)
    n_qubits = max(i for term in qub_ham.terms for i, _ in term) + 1
    Hmat = get_sparse_operator(qub_ham, n_qubits).toarray()
    H = qml.Hermitian(Hmat, wires=range(n_qubits))

    # Build ansatz
    singles, doubles = excitations(n_electrons, n_qubits)
    hf_bits = hf_state(n_electrons, n_qubits)
    n_sing = len(singles)

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

    # Single evaluation
    params = pnp.array(vqe_params, requires_grad=False)
    energy = float(pnp.real(circuit(params)))
    
    print(f"Single-shot VQE ({label}) energy: {energy:.8f} Ha")
    return None, 0, mo_guess, energy
