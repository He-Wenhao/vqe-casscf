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
from downfolding_methods_pytorch import perm_orca2pyscf

### test vqe ci
# HF initial guess
d = 0.7
pos = [[0,0,0], [d,0,0], [2*d,0,0], [3*d,0,0]]
elements = ["H", "H", "H", "H"]
atom = [[el, tuple(c)] for el, c in zip(elements, pos)]
mol = gto.M(atom=atom, basis='cc-pVDZ', spin=0, charge=0, verbose=0)
ncas = nelecas = 4

mf_hf = scf.RHF(mol)
mf_hf.kernel()
E_hf, params_hf, hf_orbitals, hf_ci, mc_hf = run_vqe_cas(mol,  mf_hf.mo_coeff, ncas, nelecas)
print(f"HF-initialized VQE-CAS energy: {E_hf:.8f} Ha")

dm1_ao, dm2_ao = extract_rdms_ao(mc_hf)
e_rdm = energy_from_ao_rdm(mol, dm1_ao, dm2_ao)
print("e_rdm",e_rdm)
assert abs(e_rdm - E_hf) < 1e-5, f"Energy mismatch: {e_rdm:.12f} vs {E_hf:.12f}"
