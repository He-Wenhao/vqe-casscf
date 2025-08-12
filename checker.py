# sanity check to make sure the code is working (most are already integrated into each of the Figure.py files)
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

def test_vqe_ci(d=2.3, ncas=4, nelecas=4):
    print("\n=== Test VQE CI ===")
    
    mol, atom = setup_molecule(d)
    
    mf_hf = scf.RHF(mol)
    mf_hf.kernel()
    
    E_hf, params_hf, hf_orbitals, hf_ci, mc_hf = run_vqe_cas(mol, mf_hf.mo_coeff, ncas, nelecas)
    print(f"HF-initialized VQE-CAS energy: {E_hf:.8f} Ha")
    
    # Extract RDMs and calculate energy
    dm1_ao, dm2_ao = extract_rdms_ao(mc_hf)
    e_rdm = energy_from_ao_rdm(mol, dm1_ao, dm2_ao)
    print(f"Energy from RDMs: {e_rdm:.8f} Ha")
    
    energy_diff = abs(e_rdm - E_hf)
    print(f"Energy difference: {energy_diff:.2e} Ha")
    assert energy_diff < 1e-5, f"Energy mismatch: {e_rdm:.12f} vs {E_hf:.12f}"


def analyze_rdms(mc_hf_ci, fci_hf_ci, mc_nn_ci, fci_nn_ci, ncas=4, nelecas=4):
    print("\n=== RDM Analysis ===")
    
    # HF-initialized FCI RDM
    rdm1_hf = mc_hf_ci.fcisolver.make_rdm1(fci_hf_ci, ncas, nelecas)
    eigvals_hf = np.linalg.eigvalsh(rdm1_hf)[::-1]
    print("HF-FCI 1-RDM eigenvalues:", eigvals_hf)
    
    # NN-initialized FCI RDM  
    rdm1_nn = mc_nn_ci.fcisolver.make_rdm1(fci_nn_ci, ncas, nelecas)
    eigvals_nn = np.linalg.eigvalsh(rdm1_nn)[::-1]
    print("NN-FCI 1-RDM eigenvalues:", eigvals_nn)


def orbital_difference(hf_orbitals, nn_orbitals, fci_hf_orbitals, fci_nn_orbitals):
    print("\n=== Orbital Difference ===")
    
    # VQE orbital differences
    vqe_diff = np.linalg.norm(hf_orbitals[:,:4] @ hf_orbitals[:,:4].T - nn_orbitals[:,:4] @ nn_orbitals[:,:4].T)
    print(f"VQE orbital difference norm: {vqe_diff:.6f}")
    
    # FCI orbital differences
    fci_diff = np.linalg.norm(fci_hf_orbitals[:,:4] @ fci_hf_orbitals[:,:4].T - fci_nn_orbitals[:,:4] @ fci_nn_orbitals[:,:4].T)
    print(f"FCI orbital difference norm: {fci_diff:.6f}")


def infidelities_and_energies(mc_hf, mc_nn, mc_hf_ci, mc_nn_ci, mol):
    print("\n=== Infidelities and Energy Analysis ===")
    
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
