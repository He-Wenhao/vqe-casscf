import json
import numpy as np
import scipy
from pyscf import gto, scf, mcscf
from pyscf.mcscf import avas
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

def avas_mo_guess_H4(mol, threshold=0.95):
    mf = scf.RHF(mol).run()
    cas, elecas, orbital = avas.avas(mf, ["H 1s"], threshold=threshold)

    print(f"[AVAS] returning mo_guess with shape {orbital.shape}, ncas={cas}, nelecas={elecas}")
    return orbital, cas, elecas

def run_casscf_with_guess(mol, mo_guess, ncas, nelecas, label=""):
    print(f"\nRunning CASSCF with {label} initial guess...")
    mf_init = scf.RHF(mol); mf_init.kernel()
    mc = mcscf.CASSCF(mf_init, ncas=ncas, nelecas=nelecas)
    mc.fcisolver = PennyLaneSolver(mf_init)
    mc.mo_coeff = mo_guess
    mc.verbose = 4  # suppress PySCF banner so we control output
    
    # Count VQE calls directly
    vqe_call_count = {"count": 0}
    original_kernel = mc.fcisolver.kernel
    
    def wrapped_kernel(*args, **kwargs):
        vqe_call_count["count"] += 1
        return original_kernel(*args, **kwargs)
    
    mc.fcisolver.kernel = wrapped_kernel
    
    # Track counts and print
    counts = {"macro": 0, "jk_total": 0, "jk_current_macro_max": 0, "last_macro": 0}
    def _cb(envs):
        if "imacro" in envs:
            current_macro = int(envs["imacro"])
            
            if current_macro > counts["last_macro"] and counts["last_macro"] > 0:
                counts["jk_total"] += counts["jk_current_macro_max"]
                counts["jk_current_macro_max"] = 0
            
            counts["macro"] = current_macro
            counts["last_macro"] = current_macro
            
        if "njk" in envs:
            # Track the maximum njk within this macro iteration
            counts["jk_current_macro_max"] = max(counts["jk_current_macro_max"], int(envs["njk"]))
            
        if "e_tot" in envs:
            print(f"macro iter {counts['macro']:2d}  CASSCF E = {envs['e_tot']:.14f}")
    
    mc.callback = _cb
    t0 = time.time()
    e_tot, *_ = mc.kernel()
    t1 = time.time()
    
    # Add the final macro iteration's JK count
    if counts["macro"] > 0:
        counts["jk_total"] += counts["jk_current_macro_max"]
    
    # Calculate micro so that macro + micro = VQE calls
    total_vqe = vqe_call_count["count"]
    macro = counts["macro"]
    micro = total_vqe - macro
    
    print(f"CASSCF converged in {macro} macro ({counts['jk_total']} JK {micro} micro)")
    print(f"CASSCF energy = {e_tot:.14f}")
    print(f"CASSCF ({label} guess) took {t1 - t0:.2f} seconds")
    print(f"VQE calls: {total_vqe} = {macro} macro + {micro} micro")
    return mc


def process_index(ind):
    pos = pos_l[ind]
    elements = elements_l[ind]
    proj = np.array(proj_l[ind])
    name = name_l[ind]

    atom = [[ele, tuple(coord)] for ele, coord in zip(elements, pos)]
    print(ind,name)
    #return 0
    log_path = f"results/fig4/cas_init_result_avas/log_{name[:-5]}.txt"
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
    mol = gto.M(atom=atom, basis='cc-pVDZ', spin=0, charge=0, verbose=4, output=log_path)

    # Run CASSCFs
    run_casscf_with_guess(mol, rand_orbitals, ncas=4, nelecas=4, label="NN")
    try:
        mo_avas, ncas_avas, nelecas_avas = avas_mo_guess_H4(mol, threshold=0.95)
        run_casscf_with_guess(mol, mo_avas, ncas=4, nelecas=4, label="AVAS(H 1s, 0.95)")
        print(f"[AVAS] (suggested) ncas={ncas_avas}, nelecas={nelecas_avas}")
    except Exception as e:
        print(f"[AVAS] Failed to build guess: {e}")

    sys.stdout.close()
    sys.stdout = sys.__stdout__

if __name__ == "__main__":
    for i in range(0,23):
        process_index(i)

    print("Files saved to results/fig4/cas_init_result_avas/")
