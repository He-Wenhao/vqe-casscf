import json
import numpy as np
import scipy
from pyscf import gto, scf, mcscf
from pyscf.mcscf import avas
from vqe_driver import perm_orca2pyscf
import sys
import time
import os

FILE = "data_H4/inference.json"

# Load data once globally
with open(FILE, "r") as file:
    data = json.load(file)
pos_l = data["pos"]
elements_l = data["elements"]
proj_l = data["proj"]
name_l = data["name"]

def avas_mo_guess_H4(mol, threshold=0.95):
    mf = scf.RHF(mol).run()
    cas, elecas, orbital = avas.avas(mf, ["H 1s"], threshold=threshold)

    print(f"[AVAS] returning mo_guess with shape {orbital.shape}, ncas={cas}, nelecas={elecas}")
    return orbital, cas, elecas


def run_casscf_with_guess(mol, mo_guess, ncas, nelecas, label=""):
    print(f"\nRunning CASSCF with {label} initial guess...")
    mf_init = scf.RHF(mol)
    mf_init.kernel()

    # Classical CASSCF (no VQE)
    mc = mcscf.CASSCF(mf_init, ncas=ncas, nelecas=nelecas)
    mc.mo_coeff = mo_guess
    mc.verbose = 4

    start = time.time()
    mc.kernel()
    end = time.time()

    print(f"CASSCF ({label} guess) took {end - start:.2f} seconds")
    return mc

def process_index(ind):
    pos = pos_l[ind]
    elements = elements_l[ind]
    proj = np.array(proj_l[ind])
    name = name_l[ind]

    atom = [[ele, tuple(coord)] for ele, coord in zip(elements, pos)]

    log_path = f"results/fig6/cas_init_result/log_{name[:-5]}.txt"
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

    try:
        mo_avas, ncas_avas, nelecas_avas = avas_mo_guess_H4(mol, threshold=0.95)
        run_casscf_with_guess(mol, mo_avas, ncas_avas,nelecas_avas, label="AVAS(H 1s, 0.95)")
        print(f"[AVAS] (suggested) ncas={ncas_avas}, nelecas={nelecas_avas}")
    except Exception as e:
        print(f"[AVAS] Failed to build guess: {e}")

    run_casscf_with_guess(mol, rand_orbitals, ncas=4, nelecas=4, label="NN")


    sys.stdout.close()
    sys.stdout = sys.__stdout__

if __name__ == "__main__":
    for i in range(0,23):
        process_index(i)

    print("Files saved to results/fig6/cas_init_result/")
