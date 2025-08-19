import json
import numpy as np
import scipy
from pyscf import gto, scf, mcscf
import sys
import time
import os
from vqe_driver import PennyLaneSolver, perm_orca2pyscf

FILE = "data_H4/inference.json"

# Load data once globally
with open(FILE, "r") as file:
    data = json.load(file)
pos_l = data["pos"]
elements_l = data["elements"]
proj_l = data["proj"]
name_l = data["name"]


def run_casscf_with_guess(mol, mo_guess, ncas, nelecas, label=""):
    print(f"\nRunning CASSCF with {label} initial guess...")

    mf_init = scf.RHF(mol)
    mf_init.verbose = 0
    mf_init.kernel()

    mc = mcscf.CASSCF(mf_init, ncas=ncas, nelecas=nelecas)
    mc.fcisolver = PennyLaneSolver(mf_init)
    mc.mo_coeff = mo_guess
    mc.verbose = 0               # suppress PySCF banner so we control output

    # Capture counts from PySCF callback
    counts = {"macro": 0, "jk": 0, "micro": 0}
    def _cb(envs):
        if "imacro" in envs:
            counts["macro"] = max(counts["macro"], int(envs["imacro"]))
        if "njk" in envs:
            counts["jk"] = max(counts["jk"], int(envs["njk"]))
        if "imicro" in envs:
            counts["micro"] = max(counts["micro"], int(envs["imicro"]))
    mc.callback = _cb

    t0 = time.time()
    e_tot, *_ = mc.kernel()
    t1 = time.time()

    print(f"CASSCF converged in {counts['macro']} macro ({counts['jk']} JK {counts['micro']} micro)")
    print(f"CASSCF energy = {e_tot:.14f}")

    print(f"CASSCF ({label} guess) took {t1 - t0:.2f} seconds")
    return mc


def process_index(ind):
    pos = pos_l[ind]
    elements = elements_l[ind]
    proj = np.array(proj_l[ind])
    name = name_l[ind]

    atom = [[ele, tuple(coord)] for ele, coord in zip(elements, pos)]

    log_path = f"results/fig4/cas_init_result/log_{name[:-5]}.txt"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    sys.stdout = open(log_path, "w")

    print(name)
    print(atom)

    # Overlap and projection matrix processing
    S = gto.M(atom=atom, basis="cc-pVDZ").intor("int1e_ovlp")
    sqrtS = scipy.linalg.sqrtm(S).real
    perm = perm_orca2pyscf(atom=atom, basis="cc-pVDZ")
    proj = perm @ proj @ perm.T
    eigvals, eigvecs = np.linalg.eig(proj)
    idx = np.argsort(eigvals)[::-1]
    sorted_eigvecs = eigvecs[:, idx]
    sorted_eigvecs = np.linalg.inv(sqrtS) @ sorted_eigvecs
    rand_orbitals = sorted_eigvecs

    # Build molecule
    mol = gto.M(atom=atom, basis='ccpVDZ', spin=0, charge=0, verbose=4, output=log_path)

    # Run HF
    mf_hf = scf.RHF(mol)
    mf_hf.kernel()
    hf_orbitals = mf_hf.mo_coeff

    # Run CASSCFs
    run_casscf_with_guess(mol, rand_orbitals, ncas=4, nelecas=4, label="NN")
    run_casscf_with_guess(mol, hf_orbitals, ncas=4, nelecas=4, label="HF")

    sys.stdout.close()
    sys.stdout = sys.__stdout__

if __name__ == "__main__":
    for i in range(0,23):
        process_index(i)

    print("Files saved to results/fig4/cas_init_result/")
