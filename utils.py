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
