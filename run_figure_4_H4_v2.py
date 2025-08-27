import json
import numpy as np
import scipy
from pyscf import gto, scf, mcscf
import sys
import time
import os
from vqe_driver import PennyLaneSolver, perm_orca2pyscf

FILE = "data_H4/inference.json"
log_path_beginning = f"results/fig4/cas_init_result_exp(2-step)/"
# log_path_beginning = f"results/fig4/cas_init_result(1-step)/"

# Load data once globally
with open(FILE, "r") as file:
    data = json.load(file)
pos_l = data["pos"]
elements_l = data["elements"]
proj_l = data["proj"]
name_l = data["name"]


def run_casscf_with_guess(mol, mo_guess, ncas, nelecas, label="", *, two_step=False, max_cycle_micro=0):
    print(f"\nRunning CASSCF with {label} initial guess...  [two_step={two_step}, max_cycle_micro={max_cycle_micro}]")
    mf_init = scf.RHF(mol); mf_init.kernel()

    if two_step:
        try:
            from pyscf.mcscf import mc2step as _mc2
            if hasattr(_mc2, "CASSCF"):
                mc = _mc2.CASSCF(mf_init, ncas=ncas, nelecas=nelecas)
            elif hasattr(_mc2, "casscf"):
                mc = _mc2.casscf(mf_init, ncas, nelecas)
            else:
                mc = mcscf.CASSCF(mf_init, ncas=ncas, nelecas=nelecas)

            print("uses mc2step pre-CI:", getattr(mc, 'ci_response_space', None))
            print("ah_start_cycle:", getattr(mc, 'ah_start_cycle', None))
        except:
            print("Two-step is not working")
    else:
        mc = mcscf.CASSCF(mf_init, ncas=ncas, nelecas=nelecas)

    mc.fcisolver = PennyLaneSolver(mf_init)
    mc.mo_coeff = mo_guess
    mc.verbose = 0
    
    if not two_step:
        mc.max_cycle_micro = max_cycle_micro

    # Count VQE calls directly
    solver_calls = {"total": 0}
    original_kernel = mc.fcisolver.kernel

    def wrapped_kernel(*args, **kwargs):
        solver_calls["total"] += 1
        return original_kernel(*args, **kwargs)

    mc.fcisolver.kernel = wrapped_kernel

    # Tracking
    counts = {
        "macro": 0,
        "jk_total": 0,
        "jk_current_macro_max": 0,
        "last_macro": 0,
    }

    def _cb(envs):
        imacro = int(envs.get("imacro", counts["macro"]))
        if imacro > counts["last_macro"]:
            if counts["last_macro"] > 0:
                counts["jk_total"] += counts["jk_current_macro_max"]
                counts["jk_current_macro_max"] = 0
            counts["last_macro"] = imacro
        counts["macro"] = imacro

        if "njk" in envs:
            counts["jk_current_macro_max"] = max(counts["jk_current_macro_max"], int(envs["njk"]))

        if "e_tot" in envs:
            print(f"macro iter {imacro:2d}  CASSCF E = {envs['e_tot']:.14f}")

    mc.callback = _cb

    t0 = time.time()
    e_tot, *_ = mc.kernel()
    t1 = time.time()

    # Add the final macro iteration's JK count
    if counts["macro"] > 0:
        counts["jk_total"] += counts["jk_current_macro_max"]

    total_vqe = solver_calls["total"]
    macro = counts["macro"]

    if not two_step:
        micro_report = macro
    else:
        micro_report = total_vqe - macro

    print(f"CASSCF converged in {macro} macro ({counts['jk_total']} JK {micro_report} micro)")
    print(f"CASSCF energy = {e_tot:.14f}")
    print(f"CASSCF ({label} guess) took {t1 - t0:.2f} seconds")
    print(f"VQE calls: {total_vqe}")

    return mc


def process_index(ind, *, two_step=False, max_cycle_micro=0):
    pos = pos_l[ind]
    elements = elements_l[ind]
    proj = np.array(proj_l[ind])
    name = name_l[ind]

    atom = [[ele, tuple(coord)] for ele, coord in zip(elements, pos)]

    log_path = f"{log_path_beginning}log_{name[:-5]}.txt"
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
    nn_orbitals = sorted_eigvecs

    # Build molecule
    mol = gto.M(atom=atom, basis='cc-pVDZ', spin=0, charge=0, verbose=4, output=log_path)

    # Run HF
    mf_hf = scf.RHF(mol)
    mf_hf.kernel()
    hf_orbitals = mf_hf.mo_coeff

    # Run CASSCFs (both with the same settings)
    run_casscf_with_guess(mol, nn_orbitals, ncas=4, nelecas=4, label="NN", two_step=two_step, max_cycle_micro=max_cycle_micro)
    run_casscf_with_guess(mol, hf_orbitals, ncas=4, nelecas=4, label="HF", two_step=two_step, max_cycle_micro=max_cycle_micro)

    sys.stdout.close()
    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    if log_path_beginning == "results/fig4/cas_init_result(1-step)/":
        for i in range(0, 23):
            process_index(i, two_step=False, max_cycle_micro=0)
        
        print("Files saved to results/fig4/cas_init_result(1-step)/")

    if log_path_beginning == "results/fig4/cas_init_result(2-step)/":
        for i in range(0, 23):
            process_index(i, two_step=True)

        print("Files saved to results/fig4/cas_init_result(2-step)/")
