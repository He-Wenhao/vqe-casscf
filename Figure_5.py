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
import matplotlib as mpl

from utils import *
from vqe_driver import *
from downfolding_methods_pytorch import perm_orca2pyscf


d = 2.3
pos = [[0,0,0], [d,0,0], [2*d,0,0], [3*d,0,0]]
elements = ["H", "H", "H", "H"]
atom = [[el, tuple(c)] for el, c in zip(elements, pos)]

# Load projection from obs/20.json
obs_path = Path("obs/20.json")
obs_data = np.array(scipy.json.load(open(obs_path))) if False else None
# actually use JSON
obs_data = json.load(open(obs_path))
proj = np.array(obs_data["proj"])

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

"""========FCI=========="""
# HF initial guess
mf_hf = scf.RHF(mol)
mf_hf.kernel()
fci_E_hf, fci_params_hf, fci_hf_orbitals, fci_hf_ci, mc_hf_ci = run_vqe_cas(mol,  mf_hf.mo_coeff, ncas, nelecas,solver='FCI')
print(f"HF-initialized CI-CAS energy: {fci_E_hf:.8f} Ha")

# run classical vqe
fci_E_nn, fci_params_nn, fci_nn_orbitals, fci_nn_ci, mc_nn_ci = run_vqe_cas(mol,  sorted_eigvecs, ncas, nelecas,solver='FCI')
print(f"HF-initialized CI-CAS energy: {fci_E_nn:.8f} Ha")

np.linalg.norm(fci_hf_orbitals[:,:4] @ fci_hf_orbitals[:,:4].T - fci_nn_orbitals[:,:4] @ fci_nn_orbitals[:,:4].T)

"""========VQE-CASSCF=========="""
# Run for NN guess
E_nn, params_nn,nn_orbitals, nn_ci, mc_nn  = run_vqe_cas(mol, sorted_eigvecs, ncas, nelecas)
print(f"NN-initialized VQE-CAS energy: {E_nn:.8f} Ha")

# HF initial guess
mf_hf = scf.RHF(mol)
mf_hf.kernel()
E_hf, params_hf, hf_orbitals, hf_ci, mc_hf = run_vqe_cas(mol,  mf_hf.mo_coeff, ncas, nelecas)
print(f"HF-initialized VQE-CAS energy: {E_hf:.8f} Ha")

np.linalg.norm(hf_orbitals[:,:4] @ hf_orbitals[:,:4].T - nn_orbitals[:,:4] @ nn_orbitals[:,:4].T)

"""==============================""""
# Compute 1-RDM in the active space
rdm1 = mc_hf_ci.fcisolver.make_rdm1(fci_hf_ci, mc_hf_ci.ncas, mc_hf_ci.nelecas)

# Verify trace equals electron number (active electrons)
trace_rdm1 = rdm1.trace()
eigvals = np.linalg.eigvalsh(rdm1)[::-1]  # Hermitian, so eigvalsh is faster
print("Eigenvalues of the active-space 1-RDM:")
print(eigvals)

# Compute 1-RDM in the active space
rdm1 = mc_nn_ci.fcisolver.make_rdm1(fci_nn_ci, mc_nn_ci.ncas, mc_nn_ci.nelecas)

# Verify trace equals electron number (active electrons)
trace_rdm1 = rdm1.trace()
eigvals = np.linalg.eigvalsh(rdm1)  # Hermitian, so eigvalsh is faster
print("Eigenvalues of the active-space 1-RDM:")
print(eigvals)

print((nn_ci).shape)
print("Infid(nn-vqe,hf-vqe):", casscf_overlap(mc_hf, mc_nn))
print("Infid(nn-vqe,fci):", casscf_overlap(mc_nn, mc_hf_ci))
print("Infid(hf-vqe,fci):", casscf_overlap(mc_hf, mc_hf_ci))
print("Infid(fci-1,fci-2):", casscf_overlap(mc_nn_ci, mc_hf_ci))

dm1_ao_nn, dm1_ao_hf, dm2_ao_nn, dm2_ao_hf = compare_rdms_ao(mc_nn, mc_hf)
e_nn = energy_from_ao_rdm(mol, dm1_ao_nn, dm2_ao_nn)
e_hf = energy_from_ao_rdm(mol, dm1_ao_hf, dm2_ao_hf)

print(f"Energy from NN-init RDMs: {e_nn:.12f} Ha")
print(f"Energy from HF-init RDMs: {e_hf:.12f} Ha")
print(f"ΔE = {e_nn - e_hf:.6e} Ha")

"""========Figure 4 Scanning Function========"""
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
        # Interpolate orbitals
        mo_t = interpolate_mo_cayley(mo1, mo2, t, S)
        #mo_t = mo1

        params_init = (1 - t) * params1 + t * params2
        #params_init = params1
        # Initial guess for parameters
        #if t < 0:
        #    params_init = (1 - 2*abs(t)) * params1 + 2*abs(t) * np.zeros_like(params1)
        #elif t > 1:
        #    params_init = (2 - t) * params2 + (t - 1) * np.zeros_like(params2)
        #else:
        #    params_init = (1 - t) * params1 + t * params2
        
        print(f"\nOptimizing at t={t:.4f}...")
        
        #try:
            # Full VQE optimization
        #e_tot, params_final, _, _, _ = run_vqe_cas(mol, mo_t, ncas, nelecas,mode='CI')
        #    energies_opt.append(e_tot)
        #    params_opt.append(params_final)
        #    converged.append(True)
        #    status = "Converged"
        #except Exception as e:
        #    print(f"  Warning: Optimization failed at t={t:.2f}: {str(e)}")
            # Use single-shot energy with interpolated params as fallback
        _, _, _, e_tot = run_casscf_with_guess(
            mol, mo_t, ncas, nelecas,
            solver='VQE', vqe_params=params_init
        )
        energies_opt.append(e_tot)
        #    params_opt.append(params_init)
        converged.append(True)
        #    status = "Failed"
        
        #print(f"{t:5.2f}   {energies_opt[-1]:12.8f}   {status}")
    
    return ts, np.array(energies_opt), params_opt, converged


# Run scan from -0.2 to 1.2
ts_ext, E_opt_ext, params_opt_ext, converged_ext = extended_vqe_optimization_scan(
    mol, hf_orbitals, nn_orbitals, params_hf, params_nn,
    ncas, nelecas, t_min=-0.2, t_max=1.2, n_points=29
)

"""=============Save Data & Plot Sample Figure============="""
mpl.rcParams['pdf.fonttype'] = 42

# Visualization function
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
    '''
    # ==========================
    # Figure 1: Energy Surface
    # ==========================
    plt.figure(figsize=(10, 5))
    plt.plot(ts[converged], energies[converged], 'go-', label='Optimized VQE')
    if np.any(~converged):
        plt.plot(ts[~converged], energies[~converged], 'rx', label='Failed optimization')

    # Highlight regions
    plt.axvspan(t_min, 0, color='blue', alpha=0.1, label='HF extrapolation')
    plt.axvspan(0, 1, color='green', alpha=0.1, label='Interpolation region')
    plt.axvspan(1, t_max, color='red', alpha=0.1, label='NN extrapolation')

    # Reference lines
    plt.axhline(E_hf_orig, color='blue', linestyle=':', label='Original HF')
    plt.axhline(E_nn_orig, color='red', linestyle=':', label='Original NN')
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(1, color='gray', linestyle='--', alpha=0.5)

    plt.ylabel('Energy (Ha)')
    plt.xlabel('Interpolation Parameter t')
    plt.title('Extended VQE Energy Surface')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    '''
    # ==========================
    # Figure 2: Relative Energy (kcal/mol)
    # ==========================
    print("Warning: set E_ref manually")
    E_ref = -2.0140511195198982
    rel = (energies - E_ref) 

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
    #plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    #plt.show()
    plt.savefig('figs/double_well.pdf', metadata={"TextAsShapes": False})
    
    return idx_min, idx_max

# Visualize results
idx_min, idx_max = visualize_extended_scan(ts_ext, E_opt_ext, converged_ext, E_hf, E_nn)

# Save results for further analysis
results = {
    't': ts_ext,
    'energy': E_opt_ext,
    'converged': converged_ext,
    'params': params_opt_ext,
    'E_hf_original': E_hf,
    'E_nn_original': E_nn
}
