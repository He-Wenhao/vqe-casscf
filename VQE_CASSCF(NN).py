import json
from pathlib import Path
import numpy as np
from numpy.linalg import eigh, inv
from pyscf import scf, gto
import scipy
from utils import setup_molecule
from vqe_driver import run_cas, perm_orca2pyscf

START = 0.3
STOP = 2.3
STEP = 0.1
NCAS, NELECAS = 4, 4
OUT_DIR = Path("results/VQE-CASSCF")
OUT_FILE = OUT_DIR / "vqe_casscf_nn_energies.json"
INFER_FILE = Path("data_H4/inference.json")

def bond_grid(start=START, stop=STOP, step=STEP):
    n = int(round((stop - start) / step)) + 1
    return [round(start + i*step, 1) for i in range(n)]

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def _get_bond_length_index(bond_length, inference_data):
    # exact match
    for idx, pos_set in enumerate(inference_data["pos"]):
        first_bond = pos_set[1][0] - pos_set[0][0]
        if abs(first_bond - bond_length) < 1e-6:
            return idx
            
    # closest match if fails (this should never happen)
    print(f"Warning: Using closest bond length match.")
    diffs = [abs(ps[1][0] - ps[0][0] - bond_length) for ps in inference_data["pos"]]
    return int(np.argmin(diffs))

def _load_nn_mos(atom, bond_length):
    with open(INFER_FILE, "r") as f:
        data = json.load(f)
    idx = _get_bond_length_index(bond_length, data)
    proj = np.array(data["proj"][idx])

    # Build AO overlap in same geometry/basis and apply ORCA->PySCF permutation
    S = gto.M(atom=atom, basis="cc-pVDZ").intor("int1e_ovlp")
    sqrtS = scipy.linalg.sqrtm(S).real
    perm = perm_orca2pyscf(atom=atom, basis="cc-pVDZ")
    proj = perm @ proj @ perm.T

    # NN orbitals = S^{-1/2} * eigenvectors of 'proj' sorted by descending eigenvalue
    evals, evecs = eigh(proj)
    order = np.argsort(evals)[::-1]
    sqrtS_inv = inv(sqrtS)
    nn_mos = sqrtS_inv @ evecs[:, order]
    return nn_mos

def run_vqe_casscf_nn_for_bond(bond_length: float):
    mol, atom = setup_molecule(d=bond_length)
    nn_mo = _load_nn_mos(atom, bond_length)
    e_tot, params, mo_coeff, ci_vec, mc = run_cas(mol, nn_mo, NCAS, NELECAS, solver='VQE')
    return float(e_tot)

def sweep_and_save():
    ensure_dir(OUT_DIR)
    bonds = bond_grid()

    results = {
        "system": "H4 linear chain",
        "basis": "cc-pVDZ",
        "method": "VQE-CASSCF (NN initialization)",
        "ncas": NCAS,
        "nelecas": NELECAS,
        "bond_lengths_A": [],
        "energies_Ha": [],
    }

    for d in bonds:
        try:
            e = run_vqe_casscf_nn_for_bond(d)
            print(f"Bond {d:>3.1f} Å → E = {e:.8f} Ha")
        except Exception as exc:
            print(f"Bond {d:>3.1f} Å → FAILED: {exc}")
            e = float("nan")

        results["bond_lengths_A"].append(d)
        results["energies_Ha"].append(e)

    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_FILE}")

def main():
    sweep_and_save()

if __name__ == "__main__":
    main()
