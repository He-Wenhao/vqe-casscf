import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
from openfermion import InteractionOperator, get_fermion_operator
from openfermion.transforms import jordan_wigner
from openfermion import get_sparse_operator
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
import json
from numpy.linalg import eigh, inv

from utils import *
from vqe_driver import *

BOND_LENGTH = 0.4 # Just change right here. Nothing else needs to be touched.

def pennylane_vqe_casci(H_const, h1, g2, nele, init_params=None, track_history=True):
    # CASCI may provide g2 in reduced form, so handle different g2 formats
    if hasattr(g2, 'shape'):
        if g2.ndim == 4:
            # Standard 4D format
            g_phys = g2.transpose((0, 2, 3, 1))
        elif g2.ndim == 2:
            # Restore to 4D
            from pyscf.ao2mo import restore
            g2_4d = restore(1, g2, h1.shape[0])
            g_phys = g2_4d.transpose((0, 2, 3, 1))
        else:
            raise ValueError(f"Unexpected g2 dimensions: {g2.ndim}")
    else:
        g_phys = g2

    int1_spin = add_spin_1bd(h1)
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
    
    n_spatial = n_qubits // 2

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

    @qml.qnode(dev, interface="autograd")
    def _state_qnode(params):
        ansatz(params)
        return qml.state()

    cost = lambda p: pnp.real(circuit(p))
    n_params = n_sing + n_doub

    # Initialize parameters
    if init_params is None:
        params = pnp.zeros(n_params, requires_grad=True)
    else:
        params = pnp.array(init_params, requires_grad=True)

    # Run VQE
    opt = qml.AdamOptimizer(0.05)
    energy = None
    energy_history = []
    ci_vec_history = []
    
    for i in range(200):
        params, energy = opt.step_and_cost(cost, params)
        energy_history.append(energy)
        
        if track_history and i % 10 == 0:
            psi_temp = _state_qnode(params)
            ci_temp = qubit_state_to_ci(psi_temp, n_orb=n_spatial, nelec=nele).real
            ci_vec_history.append(ci_temp)
            
        if i % 50 == 0:
            print(f"  PL-VQE iter {i:3d} → E = {energy:.8f} Ha")

    energy = float(pnp.real(circuit(params)))
    
    # Get final CI vector
    psi = _state_qnode(params)
    ci_vec = qubit_state_to_ci(psi, n_orb=n_spatial, nelec=nele)
    assert abs(ci_vec.imag).max() < 1e-10, "CI vector has non-zero imaginary part."
    ci_vec = ci_vec.real
    
    e_cas_from_ci = fci.direct_spin1.energy(h1, g2, ci_vec, n_spatial, (nele//2, nele//2))
    e_cas_from_ci += H_const
    print(f"  FCI energy from CI vector: {e_cas_from_ci:.8f} Ha")
    print(f"  VQE energy: {energy:.8f} Ha")

    return float(energy), params, ci_vec, energy_history, ci_vec_history

class PennyLaneSolverCASCI:
    def __init__(self, mf):
        self._mf = mf
        self.params = None
        self.energy_history = []
        self.ci_vec_history = []
        
        self.spin = 0
        self.nroots = 1
        self.root = 0
        
        self.classical = DirectSpinFCI()
        self.classical.spin = self.spin
        self.classical.nroots = self.nroots
        self.classical.root = self.root

    def kernel(self, h1e, g2e, norb, nelec, ci0=None, verbose=0, max_memory=None, ecore=None, **kwargs):
        H_const = ecore if ecore is not None else self._mf.energy_nuc()
        nele = nelec[0] + nelec[1] if isinstance(nelec, (tuple, list)) else nelec
        E, p, ci_vec, e_hist, ci_hist = pennylane_vqe_casci(
            H_const, h1e, g2e, nele, init_params=self.params
        )
        self.params = p
        self.energy_history = e_hist
        self.ci_vec_history = ci_hist

        return E, ci_vec

    def make_rdm1(self, ci_vec, ncas, nelecas):
        return self.classical.make_rdm1(ci_vec, ncas, nelecas)
    
    def make_rdm12(self, ci_vec, ncas, nelecas):
        return self.classical.make_rdm12(ci_vec, ncas, nelecas)


def run_vqe_casci(mol, mo_guess, ncas, nelecas, solver='VQE'):
    mf_init = scf.RHF(mol)
    mf_init.kernel()
    
    mc = mcscf.CASCI(mf_init, ncas=ncas, nelecas=nelecas)
    mc.mo_coeff = mo_guess
    
    mc.canonicalization = False
    
    if solver == 'VQE':
        mc.fcisolver = PennyLaneSolverCASCI(mf_init)
    else:
        assert solver == 'FCI'
    
    e_tot, e_cas, ci_vec, mo_coeff, mo_energy = mc.kernel()
    
    
    return e_tot, mc.fcisolver.params if hasattr(mc.fcisolver, 'params') else None, mo_coeff, ci_vec, mc


def get_bond_length_index(bond_length, inference_data):
    """
    Map bond length to index in inference.json based on positions.
    
    The inference.json file contains data for 23 different H4 configurations.
    We need to find which index corresponds to our bond length.
    """
    # Get positions from inference data
    positions = inference_data['pos']
    
    # Calculate bond lengths for each configuration
    # Bond length is the distance between consecutive H atoms
    target_found = False
    for idx, pos_set in enumerate(positions):
        # Calculate first bond length (between H0 and H1)
        first_bond = pos_set[1][0] - pos_set[0][0]
        
        # Check if this matches our target bond length (with small tolerance)
        if abs(first_bond - bond_length) < 1e-6:
            return idx
    
    # If exact match not found, find closest
    min_diff = float('inf')
    best_idx = 0
    for idx, pos_set in enumerate(positions):
        first_bond = pos_set[1][0] - pos_set[0][0]
        diff = abs(first_bond - bond_length)
        if diff < min_diff:
            min_diff = diff
            best_idx = idx
    
    print(f"Warning: Exact bond length {bond_length} not found in inference.json. Using closest match at index {best_idx}")
    return best_idx


def analyze_vqe_casci_vs_fci_casscf(bond_length):
    d = bond_length
    pos = [[0,0,0], [d,0,0], [2*d,0,0], [3*d,0,0]]
    elements = ["H", "H", "H", "H"]
    atom = [[el, tuple(c)] for el, c in zip(elements, pos)]
    mol = gto.M(atom=atom, basis='cc-pVDZ', spin=0, charge=0, verbose=0)
    ncas, nelecas = 4, 4
    
    # Load NN projection from inference.json
    inference_path = Path("data_H4/inference.json")
    if inference_path.exists():
        with open(inference_path) as f:
            inference_data = json.load(f)
        
        # Get the appropriate projection matrix for this bond length
        bond_idx = get_bond_length_index(d, inference_data)
        proj = np.array(inference_data["proj"][bond_idx])
        
        print(f"Using projection matrix at index {bond_idx} for bond length {d}")
        
        # Apply permutation
        S = mol.intor("int1e_ovlp")
        sqrtS = scipy.linalg.sqrtm(S).real
        perm = perm_orca2pyscf(atom=atom, basis="cc-pVDZ")
        proj = perm @ proj @ perm.T
        
        # Get NN orbitals
        from numpy.linalg import eigh, inv
        eigvals, eigvecs = eigh(proj)
        idx = np.argsort(eigvals)[::-1]
        sqrtS_inv = inv(sqrtS)
        nn_orbitals = sqrtS_inv @ eigvecs[:, idx]
    else:
        print(f"Warning: inference.json not found, using HF orbitals for NN")
        mf_temp = scf.RHF(mol)
        mf_temp.kernel()
        nn_orbitals = mf_temp.mo_coeff
    
    # Get HF orbitals
    mf_hf = scf.RHF(mol)
    mf_hf.kernel()
    hf_orbitals = mf_hf.mo_coeff
    
    print("="*60)
    print("Step 1: Running FCI-CASSCF as Reference")
    print("="*60)
    
    # Run FCI-CASSCF (optimizes both orbitals and CI)
    mc_fci_casscf = mcscf.CASSCF(mf_hf, ncas=ncas, nelecas=nelecas)
    mc_fci_casscf.mo_coeff = hf_orbitals
    e_fci_casscf = mc_fci_casscf.kernel()[0]
    print(f"FCI-CASSCF reference energy: {e_fci_casscf:.8f} Ha")
    
    print("\n" + "="*60)
    print("Step 2: Running HF-initialized VQE-CASCI")
    print("="*60)
    
    E_hf_casci, params_hf, hf_mo_casci, hf_ci_casci, mc_hf_casci = run_vqe_casci(
        mol, hf_orbitals, ncas, nelecas, solver='VQE'
    )
    print(f"HF-VQE-CASCI final energy: {E_hf_casci:.8f} Ha")
    
    print("\n" + "="*60)
    print("Step 3: Running NN-initialized VQE-CASCI")
    print("="*60)
    
    E_nn_casci, params_nn, nn_mo_casci, nn_ci_casci, mc_nn_casci = run_vqe_casci(
        mol, nn_orbitals, ncas, nelecas, solver='VQE'
    )
    print(f"NN-VQE-CASCI final energy: {E_nn_casci:.8f} Ha")
    
    # Track infidelities during optimization
    print("\n" + "="*60)
    print("Step 4: Computing Infidelities vs FCI-CASSCF")
    print("="*60)
    
    hf_ci_history = mc_hf_casci.fcisolver.ci_vec_history
    nn_ci_history = mc_nn_casci.fcisolver.ci_vec_history
    hf_energy_history = mc_hf_casci.fcisolver.energy_history
    nn_energy_history = mc_nn_casci.fcisolver.energy_history
    
    # Create temporary CASCI objects for overlap calculation
    def create_temp_mc(base_mc, ci_vec):
        temp_mc = type('TempMC', (), {})()
        temp_mc.ci = ci_vec
        temp_mc.mo_coeff = base_mc.mo_coeff
        temp_mc.ncas = base_mc.ncas
        temp_mc.nelecas = base_mc.nelecas
        temp_mc.ncore = base_mc.ncore
        temp_mc.mol = base_mc.mol
        return temp_mc
    
    infidelities_hf = []
    infidelities_nn = []
    steps = []
    
    n_steps = min(len(hf_ci_history), len(nn_ci_history))
    
    for i in range(n_steps):
        hf_temp = create_temp_mc(mc_hf_casci, hf_ci_history[i])
        nn_temp = create_temp_mc(mc_nn_casci, nn_ci_history[i])
        
        infid_hf = casscf_overlap(hf_temp, mc_fci_casscf)
        infid_nn = casscf_overlap(nn_temp, mc_fci_casscf)
        
        infidelities_hf.append(infid_hf)
        infidelities_nn.append(infid_nn)
        steps.append(i * 10)  # History saved every 10 steps
    
    # Final infidelities
    final_infid_hf = casscf_overlap(mc_hf_casci, mc_fci_casscf)
    final_infid_nn = casscf_overlap(mc_nn_casci, mc_fci_casscf)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    
    plt.semilogy(steps, infidelities_hf, 'ro-', label='HF-VQE-CASCI vs FCI-CASSCF', alpha=0.7, linewidth=2, markersize=6)
    plt.semilogy(steps, infidelities_nn, 'bo-', label='NN-VQE-CASCI vs FCI-CASSCF', alpha=0.7, linewidth=2, markersize=6)
    
    plt.xlabel('VQE Optimization Step', fontsize=14)
    plt.ylabel('Infidelity (1 - |⟨ψ|ψ_ref⟩|²)', fontsize=14)
    plt.title('VQE-CASCI Wavefunction Infidelity vs FCI-CASSCF Reference', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig('vqe_casci_infidelity_vs_fci_casscf.png', dpi=150)
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"FCI-CASSCF reference energy:  {e_fci_casscf:.8f} Ha")
    print(f"HF-VQE-CASCI final energy:     {E_hf_casci:.8f} Ha")
    print(f"NN-VQE-CASCI final energy:     {E_nn_casci:.8f} Ha")
    print(f"\nEnergy differences from FCI-CASSCF:")
    print(f"HF-VQE-CASCI: {(E_hf_casci - e_fci_casscf)*1000:.3f} mHa")
    print(f"NN-VQE-CASCI: {(E_nn_casci - e_fci_casscf)*1000:.3f} mHa")
    print(f"\nFinal infidelities vs FCI-CASSCF:")
    print(f"HF-VQE-CASCI: {final_infid_hf:.6e}")
    print(f"NN-VQE-CASCI: {final_infid_nn:.6e}")
    
    if final_infid_hf > 0:
        print(f"Infidelity ratio (NN/HF): {final_infid_nn/final_infid_hf:.2f}")
    
    # Save results
    results = {
        "description": "VQE-CASCI infidelity vs FCI-CASSCF reference",
        "reference_energy": float(e_fci_casscf),
        "hf_vqe_casci_energy": float(E_hf_casci),
        "nn_vqe_casci_energy": float(E_nn_casci),
        "energy_diff_mHa": {
            "hf": float((E_hf_casci - e_fci_casscf)*1000),
            "nn": float((E_nn_casci - e_fci_casscf)*1000)
        },
        "final_infidelities": {
            "hf": float(final_infid_hf),
            "nn": float(final_infid_nn)

            
        },
        "infidelity_history": {
            "steps": steps,
            "hf": [float(x) for x in infidelities_hf],
            "nn": [float(x) for x in infidelities_nn]
        },
        "energy_history": {
            "all_steps": list(range(len(hf_energy_history))),
            "hf": [float(x) for x in hf_energy_history],
            "nn": [float(x) for x in nn_energy_history]
        }
    }
    
    with open(f'energy_and_infidelity({d} bond).json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved")
    
    return results
    
def main():
    d_1 = BOND_LENGTH
    pos = [[0,0,0], [d_1,0,0], [2*d_1,0,0], [3*d_1,0,0]]
    elements = ["H", "H", "H", "H"]
    atom = [[el, tuple(c)] for el, c in zip(elements, pos)]
        
    # Load projection from inference.json
    inference_path = Path("data_H4/inference.json")
    if inference_path.exists():
        with open(inference_path) as f:
            inference_data = json.load(f)
        
        # Get the appropriate projection matrix for this bond length
        bond_idx = get_bond_length_index(d_1, inference_data)
        proj = np.array(inference_data["proj"][bond_idx])
    else:
        print("ERROR: inference.json not found!")
        return None
        
    # Overlap and permutation
    S = gto.M(atom=atom, basis="cc-pVDZ").intor("int1e_ovlp")
    sqrtS = scipy.linalg.sqrtm(S).real
    perm = perm_orca2pyscf(atom=atom, basis="cc-pVDZ")
    proj = perm @ proj @ perm.T
        
    # Build molecule
    mol = gto.M(atom=atom, basis='cc-pVDZ', spin=0, charge=0, verbose=0)
    ncas, nelecas = 4, 4
        
    # NN initial guess from projection
    eigvals, eigvecs = eigh(proj)
    idx = np.argsort(eigvals)[::-1]
    sqrtS_inv = inv(sqrtS)
    sorted_eigvecs = sqrtS_inv @ eigvecs[:, idx]
        
    # run classical vqe
        
    # HF initial guess
    mf_hf = scf.RHF(mol)
    mf_hf.kernel()
    fci_E_hf, fci_params_hf, fci_hf_orbitals, fci_hf_ci, mc_hf_ci = run_vqe_cas(mol,  mf_hf.mo_coeff, ncas, nelecas,solver='FCI')
    print(f"HF-initialized CI-CAS energy: {fci_E_hf:.8f} Ha")
        
    # run classical vqe
    fci_E_nn, fci_params_nn, fci_nn_orbitals, fci_nn_ci, mc_nn_ci = run_vqe_cas(mol,  sorted_eigvecs, ncas, nelecas,solver='FCI')
    print(f"NN-initialized CI-CAS energy: {fci_E_nn:.8f} Ha")

    results = analyze_vqe_casci_vs_fci_casscf(d_1)
    return results

if __name__ == "__main__":
    results = main()
