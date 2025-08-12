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

def parity_from_occ(occ):
    """Return permutation parity (+1 or -1) from occupation string when sorting by spin label."""
    # Step 1: Convert to spin labels
    labels = [1 if i % 2 == 0 else 2 for i, o in enumerate(occ) if o == 1]
    
    # Step 2: Compute parity when sorting
    target = sorted(labels)
    perm = []
    used = set()
    for t in target:
        for i, o in enumerate(labels):
            if o == t and i not in used:
                perm.append(i)
                used.add(i)
                break
    parity = 1
    for i in range(len(perm)):
        for j in range(i + 1, len(perm)):
            if perm[i] > perm[j]:
                parity *= -1
    return parity

def build_qubit_state_from_ci(ci, n_orb, nelec):
    n_alpha = nelec // 2
    n_beta = nelec // 2
    assert n_alpha == n_beta, "This function assumes equal number of alpha and beta electrons"
    dim_alpha = int(comb(n_orb, n_alpha))
    dim_beta = int(comb(n_orb, n_beta))
    assert ci.shape == (dim_alpha, dim_beta)

    # Generate all occupation strings
    #alpha_strs = list(combinations(range(n_orb), n_alpha))
    #beta_strs = list(combinations(range(n_orb), n_beta))
    result = []
    for n in cistring.make_strings(range(n_orb), n_alpha):
        # Convert from binary string to integer, then to bit positions
        nb = int(bin(n),2)
        ones_pos = tuple(i for i in range(nb.bit_length()) if (n >> i) & 1)
        result.append(ones_pos)

    alpha_strs = result
    beta_strs = result
    
    n_spin_orb = 2 * n_orb
    dim_qubit = 2 ** n_spin_orb
    wf_qubit = np.zeros(dim_qubit, dtype=ci.dtype)

    for i, a_str in enumerate(alpha_strs):
        for j, b_str in enumerate(beta_strs):
            occ = [0] * n_spin_orb
            for p in a_str:
                occ[2*p] = 1       # spin-up (alpha)
            for p in b_str:
                occ[2*p+1] = 1     # spin-down (beta)
            #occ = occ[::-1]
            s = parity_from_occ(occ)
            idx = int("".join(str(b) for b in occ), 2)
            wf_qubit[idx] = s * ci[i, j]
    return wf_qubit

def qubit_state_to_ci(wf_qubit, n_orb, nelec):
    n_alpha = nelec // 2
    n_beta = nelec // 2
    assert n_alpha == n_beta, "This function assumes equal number of alpha and beta electrons"
    dim_alpha = int(comb(n_orb, n_alpha))
    dim_beta = int(comb(n_orb, n_beta))
    
    ci = np.zeros((dim_alpha, dim_beta), dtype=wf_qubit.dtype)

    # Generate all occupation strings
    #alpha_strs = list(combinations(range(n_orb), n_alpha))
    #beta_strs = list(combinations(range(n_orb), n_beta))
    result = []
    for n in cistring.make_strings(range(n_orb), n_alpha):
        # Convert from binary string to integer, then to bit positions
        nb = int(bin(n),2)
        ones_pos = tuple(i for i in range(nb.bit_length()) if (n >> i) & 1)
        result.append(ones_pos)

    alpha_strs = result
    beta_strs = result
    
    n_spin_orb = 2 * n_orb
    
    for i, a_str in enumerate(alpha_strs):
        for j, b_str in enumerate(beta_strs):
            occ = [0] * n_spin_orb
            for p in a_str:
                occ[2*p] = 1
            for p in b_str:
                occ[2*p+1] = 1
            s = parity_from_occ(occ)
            idx = int("".join(str(b) for b in occ), 2)
            ci[i, j] = s * wf_qubit[idx]

    return ci

def extract_rdms_ao(mc):
    """
    Extract the 1-RDM and 2-RDM in AO basis from a CASSCF object.

    Args:
        mc: PySCF CASSCF object

    Returns:
        dm1_ao: 1-RDM in AO basis
        dm2_ao: 2-RDM in AO basis (chemist's notation: (pq|rs))
    """
    ci = mc.ci
    ncas = mc.ncas
    nelecas = mc.nelecas
    mo = mc.mo_coeff
    mo_act = mo[:, mc.ncore:mc.ncore+ncas]

    # 1- and 2-RDMs in the active space
    dm1_cas, dm2_cas = fci.direct_spin1.make_rdm12(ci, ncas, nelecas)

    # Transform 1-RDM to AO basis
    dm1_ao = mo_act @ dm1_cas @ mo_act.T

    # Transform 2-RDM to AO basis
    # (pq|rs) = ∑_{ijkl} C_{pi} C_{qj} C_{rk} C_{sl} * dm2_ijkl
    C = mo_act
    dm2_ao = np.einsum('pi,qj,rk,sl,ijkl->pqrs',
                       C, C, C, C, dm2_cas, optimize=True)

    return dm1_ao, dm2_ao

def compare_rdms_ao(mc1, mc2):
    """
    Compare 1-RDM and 2-RDM in AO basis between two CASSCF wavefunctions.

    Args:
        mc1, mc2: PySCF CASSCF objects

    Returns:
        dm1_ao_1, dm1_ao_2, dm2_ao_1, dm2_ao_2
    """
    dm1_ao_1, dm2_ao_1 = extract_rdms_ao(mc1)
    dm1_ao_2, dm2_ao_2 = extract_rdms_ao(mc2)

    # Frobenius norm of difference
    diff_dm1 = np.linalg.norm(dm1_ao_1 - dm1_ao_2)
    diff_dm2 = np.linalg.norm(dm2_ao_1 - dm2_ao_2)

    print("1-RDM AO difference (Frobenius norm):", diff_dm1)
    print("2-RDM AO difference (Frobenius norm):", diff_dm2)

    return dm1_ao_1, dm1_ao_2, dm2_ao_1, dm2_ao_2


def energy_from_ao_rdm(mol, dm1_ao, dm2_ao):
    """
    Compute total electronic energy from 1-RDM and 2-RDM in AO basis.

    Args:
        mol: PySCF Mole object
        dm1_ao: 1-RDM in AO basis (shape [n_ao, n_ao])
        dm2_ao: 2-RDM in AO basis (shape [n_ao, n_ao, n_ao, n_ao])

    Returns:
        total_energy: float, total electronic energy in Hartree
    """
    # One-electron AO integrals (T + V)
    h1_ao = mol.intor('int1e_kin') + mol.intor('int1e_nuc')

    # Two-electron AO integrals (physicist's notation)
    eri_ao = mol.intor('int2e')  # (μν|λσ)

    # Energy contributions
    e1 = np.einsum('pq,qp->', h1_ao, dm1_ao)  # Tr[h * D]
    e2 = 0.5 * np.einsum('pqrs,pqrs->', eri_ao, dm2_ao)

    return e1 + e2 + mol.energy_nuc()

# Extract active space integrals
def get_active_space_integrals(mol, mo_coeff, ncas, nelecas):
    # Get AO integrals
    h_ao = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    eri_ao = mol.intor('int2e')
    
    # Extract active space MOs (assuming they are the first ncas orbitals)
    mo_cas = mo_coeff[:, :ncas]
    
    # Transform to active space
    h1e = mo_cas.T @ h_ao @ mo_cas
    
    # Two-electron integrals in chemist notation (11|22)
    g2e = np.einsum('pqrs,pi,qj,rk,sl->ijkl', eri_ao, mo_cas, mo_cas, mo_cas, mo_cas)
    
    return h1e, g2e


# Orthogonalize MO coefficients with respect to overlap matrix S
def orthogonalize_mo_gram_schmidt(mo, S):
    n_aos, n_mos = mo.shape
    mo_orth = mo.copy()
    
    for i in range(n_mos):
        # Normalize
        norm = np.sqrt(mo_orth[:, i].T @ S @ mo_orth[:, i])
        mo_orth[:, i] /= norm
        
        # Orthogonalize against previous vectors
        for j in range(i+1, n_mos):
            overlap = mo_orth[:, i].T @ S @ mo_orth[:, j]
            mo_orth[:, j] -= overlap * mo_orth[:, i]
    
    return mo_orth


# Orbital interpolation using Cayley transform
def interpolate_mo_cayley(mo1, mo2, t, S=None):
    if t == 0.0:
        return mo1.copy()
    elif t == 1.0:
        return mo2.copy()
    
    # Compute rotation matrix
    if S is not None:
        R = mo1.T @ S @ mo2
    else:
        R = mo1.T @ mo2
    
    I = np.eye(R.shape[0])
    

    # Fallback to eigenvalue interpolation
    eigvals, eigvecs = np.linalg.eig(R)
    angles = np.angle(eigvals)
    angles_t = t * angles
    R_t = np.real(eigvecs @ np.diag(np.exp(1j * angles_t)) @ eigvecs.conj().T)
    
    # Apply rotation
    mo_t = mo1 @ R_t
    
    # Ensure orthonormality
    if S is not None:
        mo_t = orthogonalize_mo_gram_schmidt(mo_t, S)
    else:
        Q, _ = np.linalg.qr(mo_t)
        mo_t = Q
    
    return mo_t
