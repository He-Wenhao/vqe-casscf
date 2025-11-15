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
from pyscf.mcscf import avas
from pathlib import Path
import scipy
import json

from utils import *
from vqe_driver import *

BOND_LENGTH = 0.7  # Change as needed

# --- Utility functions ---
def ensure_dir(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)

def pennylane_vqe_casci(H_const, h1, g2, nele, init_params=None, track_history=True):
    # Handle 2-body integrals
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

    # Fermionic -> Qubit Hamiltonian
    intop = InteractionOperator(H_const, int1_spin, int2_spin)
    ferm_ham = get_fermion_operator(intop)
    qub_ham = jordan_wigner(ferm_ham)
    n_qubits = max(i for term in qub_ham.terms for i, _ in term) + 1
    Hmat = get_sparse_operator(qub_ham, n_qubits).toarray()
    Hmat = (Hmat + Hmat.conj().T)/2
    H = qml.Hermitian(Hmat, wires=range(n_qubits))

    # UCCSD Ansatz
    singles, doubles = excitations(nele, n_qubits)
    hf_bits = hf_state(nele, n_qubits)
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

    @qml.qnode(dev, interface="autograd")
    def _state_qnode(params):
        ansatz(params)
        return qml.state()

    # Cost function
    cost = lambda p: pnp.real(circuit(p))
    n_params = n_sing + len(doubles)
    params = pnp.zeros(n_params, requires_grad=True) if init_params is None else pnp.array(init_params, requires_grad=True)

    # Run VQE
    opt = qml.AdamOptimizer(0.05)
    energy_history, ci_vec_history, params_history = [], [], []
    for i in range(200):
        params, energy = opt.step_and_cost(cost, params)
        energy_history.append(float(energy))
        params_history.append(params.copy())
        if track_history and i % 10 == 0:
            psi_temp = _state_qnode(params)
            ci_temp = qubit_state_to_ci(psi_temp, n_orb=n_qubits//2, nelec=nele).real
            ci_vec_history.append(ci_temp.tolist())
        if i % 50 == 0:
            print(f"  PL-VQE iter {i:3d} → E = {energy:.8f} Ha")

    # Final CI vector & energy
    psi = _state_qnode(params)
    ci_vec = qubit_state_to_ci(psi, n_orb=n_qubits//2, nelec=nele).real
    e_cas_from_ci = direct_spin1.energy(h1, g2, ci_vec, n_qubits//2, (nele//2, nele//2)) + H_const
    energy = float(pnp.real(circuit(params)))
    print(f"  FCI energy from CI vector: {e_cas_from_ci:.8f} Ha")
    print(f"  VQE energy: {energy:.8f} Ha")

    return energy, params.tolist(), ci_vec.tolist(), energy_history, ci_vec_history

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
# --- PennyLane CASCI solver ---
class PennyLaneSolverCASCI:
    def __init__(self, mf):
        self._mf = mf
        self.params = None
        self.energy_history = []
        self.ci_vec_history = []
        self.classical = DirectSpinFCI()
        self.spin = 0
        self.nroots = 1
        self.root = 0

    def kernel(self, h1e, g2e, norb, nelec, ci0=None, verbose=0, **kwargs):
        H_const = self._mf.energy_nuc()
        nele = sum(nelec) if isinstance(nelec, (tuple, list)) else nelec
        E, p, ci_vec, e_hist, ci_hist = pennylane_vqe_casci(H_const, h1e, g2e, nele, init_params=self.params)
        self.params = p
        self.energy_history = e_hist
        self.ci_vec_history = ci_hist
        return E, np.array(ci_vec)

# --- AVAS wrapper ---
def run_avas_h1s(mol, mf_hf, threshold=0.95):
    ao_labels = ["H 1s"]
    avas_obj = avas.AVAS(mf_hf, aolabels=ao_labels,threshold=threshold)
    ncas_avas, nelecas_avas, mo_avas = avas_obj.kernel()
    print(f"AVAS selected {ncas_avas} orbitals with {nelecas_avas} electrons.")
    return mo_avas, 4, 4


# --- Main experiment ---
def run_experiment(bond_length):
    # H4 positions
    pos = [[0,0,0], [bond_length,0,0], [2*bond_length,0,0], [3*bond_length,0,0]]
    elements = ["H"]*4
    atom = [[el, tuple(c)] for el, c in zip(elements, pos)]

    

    # Molecule & RHF
    mol = gto.M(atom=atom, basis='cc-pVDZ', spin=0, charge=0, verbose=0)
    mf_hf = scf.RHF(mol).run()
    hf_orbitals = mf_hf.mo_coeff

    # --- AVAS ---
    mo_avas, ncas_avas, nelecas_avas = run_avas_h1s(mol, mf_hf, threshold=0.95)

    # --- HF-initialized VQE-CASCI ---
    ncas, nelecas = 4, 4
    E_hf_casci, params_hf, hf_mo_casci, hf_ci_casci, mc_hf_casci = run_vqe_casci(mol, hf_orbitals, ncas, nelecas, solver='VQE')


    inference_path = Path("data_H4/inference.json")
    if inference_path.exists():
        with open(inference_path) as f:
            inference_data = json.load(f)
        bond_idx = get_bond_length_index(bond_length, inference_data)
        proj = np.array(inference_data["proj"][bond_idx])
        print(f"Using projection matrix at index {bond_idx} for bond length {bond_length}")

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
    
    E_nn_casci, params_nn, nn_mo_casci, nn_ci_casci, mc_nn_casci = run_vqe_casci(
        mol, nn_orbitals, ncas, nelecas, solver='VQE'
    )

    # --- AVAS-initialized VQE-CASCI ---
    E_avas_casci, params_avas, avas_mo_casci, avas_ci_casci, mc_avas_casci = run_vqe_casci(
        mol, mo_avas, ncas_avas, nelecas_avas, solver='VQE'
    )

    # --- Save results ---
    results_dir = "results/fig3/Data3"
    ensure_dir(results_dir)

    results = {
        "bond_length": float(bond_length),
        "hf_vqe_casci_energy": float(E_hf_casci),
        "nn_vqe_casci_energy": float(E_nn_casci),
        "avas_vqe_casci_energy": float(E_avas_casci),
        "hf_params": params_hf,
        "nn_params": params_nn,
        "avas_params": params_avas,
        "energy_history": {
            "hf":   mc_hf_casci.fcisolver.energy_history,
            "nn":   mc_nn_casci.fcisolver.energy_history,
            "avas": mc_avas_casci.fcisolver.energy_history,
        },
        "ci_vec_history": {
            "hf":   mc_hf_casci.fcisolver.ci_vec_history,
            "nn":   mc_nn_casci.fcisolver.ci_vec_history,
            "avas": mc_avas_casci.fcisolver.ci_vec_history,
        },
    }


    filename = f"{results_dir}/vqe_casci_results_bond({bond_length:.1f}).json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filename}")

    print("\nSUMMARY:")
    print(f"HF-VQE energy:    {E_hf_casci:.8f} Ha")
    print(f"NN-VQE energy:    {E_nn_casci:.8f} Ha")
    print(f"AVAS-VQE energy:  {E_avas_casci:.8f} Ha")

    return results

# --- Main ---
def main(bond):
    bond_length = bond
    results = run_experiment(bond_length)
    return results

if __name__ == "__main__":
    bonds = np.arange(.3, 2.31, 0.1)
    for i in bonds:
        main(i)
