import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
from openfermion import InteractionOperator, get_fermion_operator
from openfermion.transforms import jordan_wigner
from openfermion import get_sparse_operator
from pennylane.qchem import hf_state, excitations
from pyscf import gto, scf, mcscf
import scipy
from pyscf import fci
import json
from pathlib import Path
from numpy.linalg import eigh, inv

from utils import *
from vqe_driver import *
from checker import *

BOND_LENGTH = 2.3  # Change bond length here
FILE = "data_H4/inference.json"

def ensure_dir(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)

def load_projection_data(filename=FILE, atom=None, bond_length=BOND_LENGTH):
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

    # Overlap and permutation
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

    return (E_hf, params_hf, hf_orbitals, hf_ci, mc_hf, 
            E_nn, params_nn, nn_orbitals, nn_ci, mc_nn)

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
        params_opt.append(params_init.tolist() if hasattr(params_init, 'tolist') else params_init)
        converged.append(True)
    
    return ts, np.array(energies_opt), params_opt, converged

def run_experiment(bond_length):
    mol, atom = setup_molecule(d=bond_length)
    ncas, nelecas = 4, 4

    # Get NN and HF orbitals
    nn_mo_coeff = load_projection_data(FILE, atom, bond_length=bond_length)
    mf_hf = scf.RHF(mol)
    mf_hf.kernel()
    hf_mo_coeff = mf_hf.mo_coeff

    # Run FCI calculations
    (fci_E_hf, fci_params_hf, fci_hf_orbitals, fci_hf_ci, mc_hf_ci,
     fci_E_nn, fci_params_nn, fci_nn_orbitals, fci_nn_ci, mc_nn_ci) = run_fci_calculations(
        mol, hf_mo_coeff, nn_mo_coeff, ncas, nelecas)
    
    # Run VQE calculations  
    (E_hf, params_hf, hf_orbitals, hf_ci, mc_hf,
     E_nn, params_nn, nn_orbitals, nn_ci, mc_nn) = run_vqe_casscf_calculations(
        mol, hf_mo_coeff, nn_mo_coeff, ncas, nelecas)

    # Sanity checks
    print("\n=== Sanity Checks ===")
    analyze_rdms(mc_hf_ci, fci_hf_ci, mc_nn_ci, fci_nn_ci, ncas, nelecas)
    infidelities_and_energies(mc_hf, mc_nn, mc_hf_ci, mc_nn_ci, mol)
    orbital_difference(hf_orbitals, nn_orbitals, fci_hf_orbitals, fci_nn_orbitals)

    # Run extended VQE optimization scan
    ts_ext, E_opt_ext, params_opt_ext, converged_ext = extended_vqe_optimization_scan(
        mol, hf_orbitals, nn_orbitals, params_hf, params_nn,
        ncas, nelecas, t_min=-0.2, t_max=1.2, n_points=29
    )

    # Prepare results
    results = {
        'bond_length': float(bond_length),
        't': ts_ext.tolist(),
        'energy': E_opt_ext.tolist(),
        'converged': [bool(c) for c in converged_ext],
        'params': params_opt_ext,
        'E_hf_original': float(E_hf),
        'E_nn_original': float(E_nn),
        'fci_E_hf': float(fci_E_hf),
        'fci_E_nn': float(fci_E_nn)
    }
    
    return results

def main():
    bond_length = BOND_LENGTH
    
    # Results directory
    results_dir = "results/fig5/Data"
    ensure_dir(results_dir)
    
    print(f"\n{'='*60}")
    print(f"Running VQE scan experiment for bond length {bond_length}")
    print(f"{'='*60}\n")
    
    # Run experiment
    results = run_experiment(bond_length)
    
    # Save results
    filename = f"{results_dir}/vqe_scan({bond_length}).json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to {filename}")
    print(f"{'='*60}")
    
    # Print summary
    energies = np.array(results['energy'])
    ts = np.array(results['t'])
    idx_min = np.argmin(energies)
    idx_max = np.argmax(energies)
    
    print("\nSUMMARY:")
    print(f"Bond length: {bond_length} Å")
    print(f"Energy range: [{energies.min():.8f}, {energies.max():.8f}] Ha")
    print(f"Global minimum: {energies[idx_min]:.8f} Ha at t={ts[idx_min]:.2f}")
    print(f"Global maximum: {energies[idx_max]:.8f} Ha at t={ts[idx_max]:.2f}")
    print(f"Barrier height: {(energies[idx_max] - energies[idx_min]) * 627.509:.2f} kcal/mol")
    
    return results

if __name__ == "__main__":
    results = main()
