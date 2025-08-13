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
import json
from numpy.linalg import eigh, inv

from utils import *
from vqe_driver import *


def analyze_vqe_casci_vs_fci_casscf(bond_length):
    d = bond_length
    pos = [[0,0,0], [d,0,0], [2*d,0,0], [3*d,0,0]]
    elements = ["H", "H", "H", "H"]
    atom = [[el, tuple(c)] for el, c in zip(elements, pos)]
    mol = gto.M(atom=atom, basis='cc-pVDZ', spin=0, charge=0, verbose=0)
    ncas, nelecas = 4, 4
    
    # Load NN projection
    obs_path = Path(f"obs/{int(d*10-3)}.json")
    if obs_path.exists():
        with open(obs_path) as f:
            obs_data = json.load(f)
        proj = np.array(obs_data["proj"])
        
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
        print(f"Warning: {d}.json not found, using HF orbitals for NN")
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
    results = analyze_vqe_casci_vs_fci_casscf(0.7)
    return results

if __name__ == "__main__":
    results = main()
