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
import scipy.linalg as sla
from pyscf.fci import cistring
from pyscf import fci
from math import comb
import json
from numpy.linalg import eigh, inv
import matplotlib as mpl

from utils import *
from vqe_driver import *
from checker import *

BOND_LENGTH = 0.5 # Just change bond length right here. Nothing else needs to be touched
FILE = "data_H4/inference.json"


def load_projection_data(filename=FILE, atom=None, bond_length=BOND_LENGTH):
    # Load from inference.json instead of individual files
    inference_path = Path(filename)
    with open(inference_path, "r") as f:
        inference_data = json.load(f)
    
    # Find the right index for this bond length
    positions = inference_data['pos']
    bond_idx = None
    
    for idx, pos_set in enumerate(positions):
        first_bond = pos_set[1][0] - pos_set[0][0]
        if abs(first_bond - bond_length) < 1e-6:
            bond_idx = idx
            break
    
    if bond_idx is None:
        # Find closest match
        min_diff = float('inf')
        for idx, pos_set in enumerate(positions):
            first_bond = pos_set[1][0] - pos_set[0][0]
            diff = abs(first_bond - bond_length)
            if diff < min_diff:
                min_diff = diff
                bond_idx = idx
        print(f"Warning: Using closest bond length match at index {bond_idx}")
    
    # Get the projection matrix for this bond length
    proj = np.array(inference_data["proj"][bond_idx])

    # Overlap and permutation (rest stays the same)
    S = gto.M(atom=atom, basis="cc-pVDZ").intor("int1e_ovlp")
    sqrtS = scipy.linalg.sqrtm(S).real
    perm = perm_orca2pyscf(atom=atom, basis="cc-pVDZ")
    proj = perm @ proj @ perm.T

    # NN initial guess from projection
    eigvals, eigvecs = eigh(proj)
    idx = np.argsort(eigvals)[::-1]
    sqrtS_inv = inv(sqrtS)
    sorted_eigvecs = sqrtS_inv @ eigvecs[:, idx]

    return sorted_eigvecs



def run_fci_calculations(mol, hf_mo_coeff, nn_mo_coeff, ncas=4, nelecas=4):
    print("=== Running FCI Calculations ===")
    
    # FCI with HF initial guess
    fci_E_hf, fci_params_hf, fci_hf_orbitals, fci_hf_ci, mc_hf_ci = run_vqe_cas(
        mol, hf_mo_coeff, ncas, nelecas, solver='FCI'
    )
    print(f"HF-initialized FCI-CAS energy: {fci_E_hf:.8f} Ha")
    
    # FCI with NN initial guess
    fci_E_nn, fci_params_nn, fci_nn_orbitals, fci_nn_ci, mc_nn_ci = run_vqe_cas(
        mol, nn_mo_coeff, ncas, nelecas, solver='FCI'
    )
    print(f"NN-initialized FCI-CAS energy: {fci_E_nn:.8f} Ha")

    return (fci_E_hf, fci_params_hf, fci_hf_orbitals, fci_hf_ci, mc_hf_ci,
            fci_E_nn, fci_params_nn, fci_nn_orbitals, fci_nn_ci, mc_nn_ci)


def run_vqe_casscf_calculations(mol, hf_mo_coeff, nn_mo_coeff, ncas=4, nelecas=4):
    print("\n=== Running VQE-CASSCF Calculations ===")
    
    # VQE with NN initial guess
    E_nn, params_nn, nn_orbitals, nn_ci, mc_nn = run_vqe_cas(
        mol, nn_mo_coeff, ncas, nelecas
    )
    print(f"NN-initialized VQE-CAS energy: {E_nn:.8f} Ha")

    # VQE with HF initial guess
    E_hf, params_hf, hf_orbitals, hf_ci, mc_hf = run_vqe_cas(
        mol, hf_mo_coeff, ncas, nelecas
    )
    print(f"HF-initialized VQE-CAS energy: {E_hf:.8f} Ha")

    return (E_hf, params_hf, hf_orbitals, hf_ci, mc_hf, E_nn, params_nn, nn_orbitals, nn_ci, mc_nn)

# Main scanning function
def extended_vqe_optimization_scan(mol, mo1, mo2, params1, params2, ncas, nelecas, t_min=-0.3, t_max=1.3, n_points=21):
    S = mol.intor("int1e_ovlp")
    ts = np.linspace(t_min, t_max, n_points)
    
    energies_opt = []
    params_opt = []
    converged = []
    
    print(f"\n=== Extended VQE Optimization Scan [{t_min}, {t_max}] ===")
    print("Optimizing VQE at each interpolation point...")
    print("\nt       Energy (Ha)    Status")
    print("-" * 40)
    
    for i, t in enumerate(ts):
        mo_t = interpolate_mo_cayley(mo1, mo2, t, S)

        params_init = (1 - t) * params1 + t * params2
        
        print(f"\nOptimizing at t={t:.4f}...")
        
        _, _, _, e_tot = ucc_ansatz_eval(
            mol, mo_t, ncas, nelecas,
            solver='VQE', vqe_params=params_init
        )
        energies_opt.append(e_tot)
        converged.append(True)
    
    return ts, np.array(energies_opt), params_opt, converged


def visualize_extended_scan(ts, energies, converged, E_hf_orig, E_nn_orig):
    energies = np.array(energies)
    converged = np.array(converged)

    # Derive t_min and t_max from the data itself
    t_min = float(ts.min())
    t_max = float(ts.max())

    # Key indices
    idx_min = np.argmin(energies)
    idx_max = np.argmax(energies)

    print("\n=== Extended Scan Analysis ===")
    print(f"Global minimum: {energies[idx_min]:.8f} Ha at t={ts[idx_min]:.2f}")
    print(f"Global maximum: {energies[idx_max]:.8f} Ha at t={ts[idx_max]:.2f}")
    barrier = (energies[idx_max] - energies[idx_min]) * 627.509
    print(f"Barrier height: {barrier:.2f} kcal/mol")
    
    print("Warning: set E_ref manually")
    E_ref = -2.0140511195198982
    rel = (energies - E_ref)

    mpl.rcParams['pdf.fonttype'] = 42

    # plt.figure(figsize=(10, 8))
    plt.figure()
    plt.plot(ts[converged], rel[converged], 'go-')
    plt.axhline(0, color='black', linestyle='-', alpha=0.3)
    plt.axvspan(t_min, 0, color='red', alpha=0.1)
    plt.axvspan(0, 1, color='green', alpha=0.1)
    plt.axvspan(1, t_max, color='blue', alpha=0.1)
    plt.tick_params(axis='both', direction='in',which='major', length=6, width=1, labelsize=14, top=True, right=True)
    plt.yscale('log')
    plt.xlim(-0.2, 1.2)
    plt.ylim(1e-3, 1e1)
    plt.axhline((E_hf_orig - E_ref), color='red', linestyle=':', label='Original HF')
    plt.axhline((E_nn_orig - E_ref) , color='blue', linestyle=':', label='Original NN')

    plt.xlabel('Interpolation Parameter t')
    plt.ylabel('Relative Energy (Ha)')
    plt.title('Energy Relative to t=0 (HF)')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('figs/double_well.pdf', metadata={"TextAsShapes": False})
    
    return idx_min, idx_max

def main():
    mol, atom = setup_molecule(d=BOND_LENGTH)
    ncas, nelecas = 4, 4

    nn_mo_coeff = load_projection_data(FILE, atom, bond_length=BOND_LENGTH)

    mf_hf = scf.RHF(mol)
    mf_hf.kernel()
    hf_mo_coeff = mf_hf.mo_coeff

    # Run FCI calculations
    (fci_E_hf, fci_params_hf, fci_hf_orbitals, fci_hf_ci, mc_hf_ci, fci_E_nn, fci_params_nn, fci_nn_orbitals, fci_nn_ci, mc_nn_ci) = run_fci_calculations(
        mol, hf_mo_coeff, nn_mo_coeff, ncas, nelecas)
    
    # Run VQE calculations  
    (E_hf, params_hf, hf_orbitals, hf_ci, mc_hf, E_nn, params_nn, nn_orbitals, nn_ci, mc_nn) = run_vqe_casscf_calculations(
        mol, hf_mo_coeff, nn_mo_coeff, ncas, nelecas)

    # Sanity checks
    analyze_rdms(mc_hf_ci, fci_hf_ci, mc_nn_ci, fci_nn_ci, ncas, nelecas)
    infidelities_and_energies(mc_hf, mc_nn, mc_hf_ci, mc_nn_ci, mol)
    orbital_difference(hf_orbitals, nn_orbitals, fci_hf_orbitals, fci_nn_orbitals)

    # Run extended VQE optimization scan
    ts_ext, E_opt_ext, params_opt_ext, converged_ext = extended_vqe_optimization_scan(
        mol, hf_orbitals, nn_orbitals, params_hf, params_nn,
        ncas, nelecas, t_min=-0.2, t_max=1.2, n_points=29
    )

    # Visualize and analyze results
    idx_min, idx_max = visualize_extended_scan(ts_ext, E_opt_ext, converged_ext, E_hf, E_nn)

    # Save results for further analysis
    results = {
        't': ts_ext.tolist(),
        'energy': E_opt_ext.tolist(),
        'converged': converged_ext,
        'params': params_opt_ext,
        'E_hf_original': float(E_hf),
        'E_nn_original': float(E_nn)
    }

    return results


if __name__ == "__main__":
    results = main()
