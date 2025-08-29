import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
from numpy.linalg import norm, eigh, inv
from openfermion import InteractionOperator, get_fermion_operator
from openfermion.transforms import jordan_wigner
from openfermion import get_sparse_operator
from pennylane.qchem import hf_state, excitations
from pyscf import gto, scf, mcscf, ao2mo, fci
from pyscf.ao2mo import restore
from pyscf.fci import direct_spin1, cistring
from pyscf.fci.direct_spin1 import FCI as DirectSpinFCI
from pathlib import Path
import scipy
from math import comb
import json
import os

from utils import *
from vqe_driver import *

BOND_LENGTH = 0.7  # Change this as needed

def ensure_dir(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)

def pennylane_vqe_casci(H_const, h1, g2, nele, init_params=None, track_history=True):
    # CASCI may provide g2 in reduced form, so handle different g2 formats
    if hasattr(g2, 'shape'):
        if g2.ndim == 4:
            g_phys = g2.transpose((0, 2, 3, 1))
        elif g2.ndim == 2:
            g2_4d = restore(1, g2, h1.shape[0])
            g_phys = g2_4d.transpose((0, 2, 3, 1))
        else:
            raise ValueError(f"Unexpected g2 dimensions: {g2.ndim}")
    else:
        g_phys = g2

    int1_spin = add_spin_1bd(h1)
    int2_spin = add_spin_2bd(g_phys)

    # Fermionic to qubit Hamiltonian
    intop = InteractionOperator(H_const, int1_spin, int2_spin)
    ferm_ham = get_fermion_operator(intop)
    qub_ham = jordan_wigner(ferm_ham)
    n_qubits = max(i for term in qub_ham.terms for i, _ in term) + 1
    Hmat = get_sparse_operator(qub_ham, n_qubits).toarray()
    assert norm(Hmat - Hmat.conj().T) < 1e-5, "Hmat not Hermitian"
    Hmat = (Hmat + Hmat.conj().T) / 2
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
    params_history = []
    
    for i in range(200):
        params, energy = opt.step_and_cost(cost, params)
        energy_history.append(float(energy))
        params_history.append(params.copy())
        
        if track_history and i % 10 == 0:
            psi_temp = _state_qnode(params)
            ci_temp = qubit_state_to_ci(psi_temp, n_orb=n_spatial, nelec=nele).real
            ci_vec_history.append(ci_temp.tolist())
            
        if i % 50 == 0:
            print(f"  PL-VQE iter {i:3d} → E = {energy:.8f} Ha")

    energy = float(pnp.real(circuit(params)))
    
    # Get final CI vector
    psi = _state_qnode(params)
    ci_vec = qubit_state_to_ci(psi, n_orb=n_spatial, nelec=nele)
    assert abs(ci_vec.imag).max() < 1e-10, "CI vector has non-zero imaginary part."
    ci_vec = ci_vec.real
    
    e_cas_from_ci = direct_spin1.energy(h1, g2, ci_vec, n_spatial, (nele//2, nele//2))
    e_cas_from_ci += H_const
    print(f"  FCI energy from CI vector: {e_cas_from_ci:.8f} Ha")
    print(f"  VQE energy: {energy:.8f} Ha")

    return float(energy), params.tolist(), ci_vec.tolist(), energy_history, ci_vec_history

class PennyLaneSolverCASCI:
    def __init__(self, mf):
        self._mf = mf
        self.params = None
        self.energy_history = []
        self.ci_vec_history = []
        
        self.spin = 0
        wef = self.spin  # keep attribute access compatible if used elsewhere
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
        
        return E, np.array(ci_vec)

    def make_rdm1(self, ci_vec, ncas, nelecas):
        return self.classical.make_rdm1(ci_vec, ncas, nelecas)
    
    def make_rdm12(self, ci_vec, ncas, nelecas):
        return self.classical.make_rdm12(ci_vec, ncas, nelecas)

def run_vqe_casci(mol, mo_guess, ncas, nelecas, solver='VQE'):
    mf_init = scf.RHF(mol); mf_init.kernel()
    mc = mcscf.CASCI(mf_init, ncas=ncas, nelecas=nelecas)
    mc.mo_coeff = mo_guess
    mc.canonicalization = False
    if solver == 'VQE':
        mc.fcisolver = PennyLaneSolverCASCI(mf_init)
    else:
        assert solver == 'FCI'

    # Robust to PySCF version differences in return shape
    res = mc.kernel()
    if isinstance(res, tuple):
        if len(res) == 2:
            e_tot, ci_vec = res
            mo_coeff = mc.mo_coeff
        elif len(res) == 3:
            e_tot, e_cas, ci_vec = res
            mo_coeff = mc.mo_coeff
        elif len(res) >= 5:
            e_tot, e_cas, ci_vec, mo_coeff, mo_energy = res[:5]
        else:
            e_tot = res[0]
            ci_vec = res[-1] if res[-1] is not None and not np.isscalar(res[-1]) else res[1]
            mo_coeff = mc.mo_coeff
    else:
        e_tot = float(res)
        ci_vec = mc.ci
        mo_coeff = mc.mo_coeff

    params = mc.fcisolver.params if hasattr(mc.fcisolver, 'params') else None
    return e_tot, params, mo_coeff, ci_vec, mc

def get_bond_length_index(bond_length, inference_data):
    positions = inference_data['pos']
    for idx, pos_set in enumerate(positions):
        first_bond = pos_set[1][0] - pos_set[0][0]
        if abs(first_bond - bond_length) < 1e-6:
            return idx
    # closest match
    min_diff, best_idx = float('inf'), 0
    for idx, pos_set in enumerate(positions):
        first_bond = pos_set[1][0] - pos_set[0][0]
        diff = abs(first_bond - bond_length)
        if diff < min_diff:
            min_diff, best_idx = diff, idx
    print(f"Warning: Using closest bond length match at index {best_idx}")
    return best_idx

# === Full-FCI utilities (for fidelity against full Hilbert space cc-pVDZ FCI) ===

def full_fci_in_basis(mol, mo_full):
    """Full-space FCI in provided MO basis. Returns TOTAL energy (incl. E_nuc)."""
    mf = scf.RHF(mol); mf.kernel()
    h1_ao = mf.get_hcore()
    h1_mo = mo_full.T @ h1_ao @ mo_full
    eri_mo = ao2mo.restore(1, ao2mo.kernel(mol, mo_full), mo_full.shape[1])
    norb = mo_full.shape[1]
    nelec = (mol.nelectron//2, mol.nelectron//2)
    solver = direct_spin1.FCI()
    e_elec, ci_full = solver.kernel(h1_mo, eri_mo, norb, nelec)
    e_tot = e_elec + mol.energy_nuc()
    return e_tot, ci_full, norb, nelec

def embed_cas_ci_into_full(ci_cas, norb_full, nelec, act_idx):
    """
    Embed a CAS CI matrix (alpha x beta) into the full-basis CI table by
    placing amplitudes at determinants whose occupations lie entirely in act_idx.
    """
    na, nb = nelec
    ci_cas = np.asarray(ci_cas)
    # determinant lists
    stra_full = cistring.make_strings(range(norb_full), na)
    strb_full = cistring.make_strings(range(norb_full), nb)
    ncas = len(act_idx)
    stra_cas = cistring.make_strings(range(ncas), na)
    strb_cas = cistring.make_strings(range(ncas), nb)

    posA = {int(s): i for i, s in enumerate(stra_full)}
    posB = {int(s): i for i, s in enumerate(strb_full)}

    def expand_bits(s_cas_bits, act_index):
        s_full = 0
        for j, orb in enumerate(act_index):
            if (s_cas_bits >> j) & 1:
                s_full |= (1 << orb)
        return s_full

    ci_full_embed = np.zeros((len(stra_full), len(strb_full)))
    for ia_cas, sA_cas in enumerate(stra_cas):
        sA_full = expand_bits(int(sA_cas), act_idx)
        ia_full = posA[sA_full]
        for ib_cas, sB_cas in enumerate(strb_cas):
            sB_full = expand_bits(int(sB_cas), act_idx)
            ib_full = posB[sB_full]
            ci_full_embed[ia_full, ib_full] = ci_cas[ia_cas, ib_cas]
    return ci_full_embed

def cas_fullspace_fidelity(ci_cas, ci_full, norb_full, nelec, act_idx):
    """
    True full-space fidelity: |<phi_full | Psi_full>|^2
    where phi_full is the CAS state embedded into the full determinant basis.
    """
    phi_full = embed_cas_ci_into_full(ci_cas, norb_full, nelec, act_idx)
    # Ensure normalization
    n_phi = np.linalg.norm(phi_full.ravel())
    if n_phi == 0:
        return 0.0
    phi_full = phi_full / n_phi
    n_psi = np.linalg.norm(ci_full.ravel())
    if n_psi == 0:
        return 0.0
    psi_full = ci_full / n_psi
    ov = np.vdot(phi_full.ravel(), psi_full.ravel())
    return float(np.abs(ov)**2)

def run_experiment(bond_length):
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
        bond_idx = get_bond_length_index(d, inference_data)
        proj = np.array(inference_data["proj"][bond_idx])
        print(f"Using projection matrix at index {bond_idx} for bond length {d}")

        S = mol.intor("int1e_ovlp")
        sqrtS = scipy.linalg.sqrtm(S).real
        perm = perm_orca2pyscf(atom=atom, basis="cc-pVDZ")
        proj = perm @ proj @ perm.T

        eigvals, eigvecs = eigh(proj)
        idx = np.argsort(eigvals)[::-1]
        sqrtS_inv = inv(sqrtS)
        nn_orbitals = sqrtS_inv @ eigvecs[:, idx]
    else:
        print(f"ERROR: inference.json not found!")
        return None
    
    # Get HF orbitals
    mf_hf = scf.RHF(mol); mf_hf.kernel()
    hf_orbitals = mf_hf.mo_coeff
    
    print("="*60)
    print("Step 1: Running FCI-CASSCF as Reference")
    print("="*60)
    mc_fci_casscf = mcscf.CASSCF(mf_hf, ncas=ncas, nelecas=nelecas)
    mc_fci_casscf.mo_coeff = hf_orbitals
    e_fci_casscf = mc_fci_casscf.kernel()[0]
    fci_casscf_ci = mc_fci_casscf.ci
    fci_casscf_mo = mc_fci_casscf.mo_coeff
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
    
    def create_temp_mc(base_mc, ci_vec):
        temp_mc = type('TempMC', (), {})()
        temp_mc.ci = np.array(ci_vec) if isinstance(ci_vec, list) else ci_vec
        temp_mc.mo_coeff = base_mc.mo_coeff
        temp_mc.ncas = base_mc.ncas
        temp_mc.nelecas = base_mc.nelecas
        temp_mc.ncore = base_mc.ncore
        temp_mc.mol = base_mc.mol
        return temp_mc
    
    # Histories
    hf_ci_history = mc_hf_casci.fcisolver.ci_vec_history
    nn_ci_history = mc_nn_casci.fcisolver.ci_vec_history
    
    infidelities_hf = []
    infidelities_nn = []
    infidelities_hf_full = []
    infidelities_nn_full = []
    steps = []
    
    n_steps = min(len(hf_ci_history), len(nn_ci_history))

    # Full Hilbert Space FCI references (total energies). EMBEDDING is used for fidelity.
    e_full_hf, ci_full_hf, norb_hf, nelec_hf = full_fci_in_basis(mol, hf_mo_casci)
    e_full_nn, ci_full_nn, norb_nn, nelec_nn = full_fci_in_basis(mol, nn_mo_casci)

    act_idx_hf = list(range(mc_hf_casci.ncore, mc_hf_casci.ncore + mc_hf_casci.ncas))
    act_idx_nn = list(range(mc_nn_casci.ncore, mc_nn_casci.ncore + mc_nn_casci.ncas))

    for i in range(n_steps):
        hf_temp = create_temp_mc(mc_hf_casci, hf_ci_history[i])
        nn_temp = create_temp_mc(mc_nn_casci, nn_ci_history[i])

        # Infidelity vs FCI-CASSCF reference
        infid_hf = casscf_overlap(hf_temp, mc_fci_casscf)
        infid_nn = casscf_overlap(nn_temp, mc_fci_casscf)

        # Full Hilbert space fidelity
        F_hf_global_i = cas_fullspace_fidelity(hf_ci_history[i], ci_full_hf, norb_hf, nelec_hf, act_idx_hf)
        F_nn_global_i = cas_fullspace_fidelity(nn_ci_history[i], ci_full_nn, norb_nn, nelec_nn, act_idx_nn)

        infidelities_hf.append(float(infid_hf))
        infidelities_nn.append(float(infid_nn))
        infidelities_hf_full.append(float(1.0 - F_hf_global_i))
        infidelities_nn_full.append(float(1.0 - F_nn_global_i))
        steps.append(i * 10)
    
    # Final infidelities vs Full Hilbert Space FCI
    final_infid_hf_full = float(1.0 - cas_fullspace_fidelity(hf_ci_casci, ci_full_hf, norb_hf, nelec_hf, act_idx_hf))
    final_infid_nn_full = float(1.0 - cas_fullspace_fidelity(nn_ci_casci, ci_full_nn, norb_nn, nelec_nn, act_idx_nn))

    # Final infidelities vs FCI-CASSCF reference
    final_infid_hf = casscf_overlap(mc_hf_casci, mc_fci_casscf)
    final_infid_nn = casscf_overlap(mc_nn_casci, mc_fci_casscf)

    E_casscf_nn, params_casscf_nn, mo_casscf_nn, ci_casscf_nn, mc_casscf_nn = run_cas(
        mol, nn_orbitals, ncas, nelecas, solver='VQE'
    )
    e_full_cass_nn, ci_full_cass_nn, norb_cass_nn, nelec_cass_nn = full_fci_in_basis(mol, mo_casscf_nn)
    act_idx_cass_nn = list(range(mc_casscf_nn.ncore, mc_casscf_nn.ncore + mc_casscf_nn.ncas))
    F_casscf_nn_global = cas_fullspace_fidelity(ci_casscf_nn, ci_full_cass_nn, norb_cass_nn, nelec_cass_nn, act_idx_cass_nn)
    final_infid_casscf_nn_full = float(1.0 - F_casscf_nn_global)
    casscf_grey_line = [final_infid_casscf_nn_full] * len(steps)
    
    results = {
        "bond_length": float(d),
        "description": "VQE-CASCI infidelity vs FCI-CASSCF reference",
        "reference_energy": float(e_fci_casscf),
        "reference_ci": fci_casscf_ci.tolist() if hasattr(fci_casscf_ci, 'tolist') else fci_casscf_ci,
        "reference_mo": fci_casscf_mo.tolist() if hasattr(fci_casscf_mo, 'tolist') else fci_casscf_mo,
        "hf_vqe_casci_energy": float(E_hf_casci),
        "nn_vqe_casci_energy": float(E_nn_casci),
        "hf_params": params_hf,
        "nn_params": params_nn,
        "energy_diff_mHa": {
            "hf": float((E_hf_casci - e_fci_casscf)*1000),
            "nn": float((E_nn_casci - e_fci_casscf)*1000)
        },
        "final_infidelities": {
            "hf": float(final_infid_hf),
            "nn": float(final_infid_nn)
        },
        "final_infidelities_full_fci": {
            "hf": float(final_infid_hf_full),
            "nn": float(final_infid_nn_full)
        },
        "reference_full_fci_energy_hf_basis": float(e_full_hf),
        "reference_full_fci_energy_nn_basis": float(e_full_nn),
        "infidelity_history": {
            "steps": steps,
            "hf": infidelities_hf,
            "nn": infidelities_nn
        },
        "infidelity_history_full_fci": {
            "steps": steps,
            "hf": infidelities_hf_full,
            "nn": infidelities_nn_full,
            "casscf_nn": casscf_grey_line
        },
        "energy_history": {
            "all_steps": list(range(len(mc_hf_casci.fcisolver.energy_history))),
            "hf": mc_hf_casci.fcisolver.energy_history,
            "nn": mc_nn_casci.fcisolver.energy_history
        },
        "full_fci_overlap_mode": "embedded",
        "vqe_casscf_nn": {
            "energy": float(E_casscf_nn),
            "final_infidelity_full_fci": final_infid_casscf_nn_full
        }
    }
    return results

def main():
    bond_length = BOND_LENGTH
    results_dir = "results/fig3/Data2"
    ensure_dir(results_dir)
    print(f"\n{'='*60}")
    print(f"Running VQE-CASCI experiment for bond length {bond_length}")
    print(f"{'='*60}\n")
    results = run_experiment(bond_length)
    if results:
        filename = f"{results_dir}/vqe_casci_results_bond({bond_length}).json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n{'='*60}")
        print(f"Results saved to {filename}")
        print(f"{'='*60}")
        print("\nSUMMARY:")
        print(f"Reference energy: {results['reference_energy']:.8f} Ha")
        print(f"HF-VQE energy:    {results['hf_vqe_casci_energy']:.8f} Ha")
        print(f"NN-VQE energy:    {results['nn_vqe_casci_energy']:.8f} Ha")
        print(f"Final infidelity ratio (NN/HF): {results['final_infidelities']['nn']/results['final_infidelities']['hf']:.2f}")
    return results

if __name__ == "__main__":
    results = main()
