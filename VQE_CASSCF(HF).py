import json
import os
from pathlib import Path
import numpy as np
from pyscf import scf
from utils import setup_molecule
from vqe_driver import run_cas


START = 0.3
STOP = 2.3
STEP = 0.1
NCAS, NELECAS = 4, 4
OUT_DIR = Path("results/VQE-CASSCF")
OUT_FILE = OUT_DIR / "vqe_casscf_hf_energies.json"

def bond_grid(start=START, stop=STOP, step=STEP):
    n = int(round((stop - start) / step)) + 1
    return [round(start + i*step, 1) for i in range(n)]

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def run_vqe_casscf_hf_for_bond(bond_length: float):
    mol, _ = setup_molecule(d=bond_length)

    mf = scf.RHF(mol); mf.kernel()
    hf_mo = mf.mo_coeff

    e_tot, params, mo_coeff, ci_vec, mc = run_cas(mol, hf_mo, NCAS, NELECAS, solver='VQE')
    return float(e_tot)

def sweep_and_save():
    ensure_dir(OUT_DIR)
    bonds = bond_grid()

    results = {
        "system": "H4 linear chain",
        "basis": "cc-pVDZ",
        "method": "VQE-CASSCF (HF initialization)",
        "ncas": NCAS,
        "nelecas": NELECAS,
        "bond_lengths_A": [],
        "energies_Ha": [],
    }

    for d in bonds:
        try:
            e = run_vqe_casscf_hf_for_bond(d)
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
