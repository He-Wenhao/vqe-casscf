import json
import numpy as np
import scipy
from pyscf import gto, scf, mcscf
import sys
import time
import os
import re
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
from vqe_driver import PennyLaneSolver, perm_orca2pyscf, ucc_ansatz_eval

def load_inference_data(filename="inference.json"):
    with open(filename, "r") as file:
        data = json.load(file)
    pos_l = data["pos"]
    elements_l = data["elements"]
    proj_l = data["proj"]
    name_l = data["name"]

    return pos_l, elements_l, proj_l, name_l

def process_index(ind, pos_l, elements_l, proj_l, name_l):
    pos = pos_l[ind]
    elements = elements_l[ind]
    proj = np.array(proj_l[ind])
    name = name_l[ind]

    atom = [[ele, tuple(coord)] for ele, coord in zip(elements, pos)]

    log_path = f"cas_init_result/log_{name[:-5]}.txt"
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

    # Run HF
    mf_hf = scf.RHF(mol)
    mf_hf.kernel()
    hf_orbitals = mf_hf.mo_coeff

    ucc_ansatz_eval(mol, rand_orbitals, ncas=4, nelecas=4, label="NN")
    ucc_ansatz_eval(mol, hf_orbitals, ncas=4, nelecas=4, label="HF")

    sys.stdout.close()
    sys.stdout = sys.__stdout__

def run_all_calculations(pos_l, elements_l, proj_l, name_l, max_index=23):
    for i in range(max_index):
        process_index(i, pos_l, elements_l, proj_l, name_l)

def extract_log_data(log_files):
    pattern_steps = re.compile(r'CASSCF converged in\s+(\d+)\s+macro\s+\(\s*(\d+)\s+JK\s+(\d+)\s+micro')
    pattern_energy = re.compile(r'CASSCF energy\s*=\s*([-+]?\d*\.\d+(?:[eE][-+]?\d+)?)')
    
    results = []
    
    for log_file in log_files:
        with open(log_file, "r") as f:
            lines = f.readlines()

        # Extract steps
        matches_steps = [pattern_steps.search(line) for line in lines if "CASSCF converged" in line]
        matches_steps = [m for m in matches_steps if m]
        
        # Extract energies
        matches_energy = [pattern_energy.search(line) for line in lines if "CASSCF energy" in line]
        matches_energy = [m for m in matches_energy if m]

        if len(matches_steps) >= 2 and len(matches_energy) >= 2:
            nn_steps = tuple(map(int, matches_steps[0].groups()))
            hf_steps = tuple(map(int, matches_steps[1].groups()))
            nn_energy = float(matches_energy[0].group(1))
            hf_energy = float(matches_energy[1].group(1))
            results.append((log_file, nn_steps, hf_steps, nn_energy, hf_energy))
        else:
            print(f"Warning: {log_file} missing matches (Steps: {len(matches_steps)}, Energies: {len(matches_energy)})")
    
    return results

# Print results in a formatted table
def print_results_table(results):
    print(f"{'File':<15} {'NN_macro':>9} {'NN_JK':>6} {'NN_micro':>9} {'NN_energy':>14} | {'HF_macro':>9} {'HF_JK':>6} {'HF_micro':>9} {'HF_energy':>14}")
    print("-" * 95)
    for filename, nn_steps, hf_steps, nn_energy, hf_energy in results:
        print(f"{filename:<15} {nn_steps[0]:>9} {nn_steps[1]:>6} {nn_steps[2]:>9} {nn_energy:>14.8f} | "
              f"{hf_steps[0]:>9} {hf_steps[1]:>6} {hf_steps[2]:>9} {hf_energy:>14.8f}")


def organize_data_for_plotting(results, num_chains=25, bond_length_interval=0.1):
    lengths = [bond_length_interval * point for point in range(3, num_chains + 1)]
    
    nn_macro, nn_jk, nn_micro = [], [], []
    hf_macro, hf_jk, hf_micro = [], [], []
    nn_energy, hf_energy = [], []
    indices = []
    
    for filename, nn_steps, hf_steps, nn_en, hf_en in results:
        nn_macro.append(nn_steps[0])
        nn_jk.append(nn_steps[1]) 
        nn_micro.append(nn_steps[2])
        hf_macro.append(hf_steps[0])
        hf_jk.append(hf_steps[1])
        hf_micro.append(hf_steps[2])
        nn_energy.append(nn_en)
        hf_energy.append(hf_en)
        indices.append(int(re.search(r'\d+', filename).group()))
    
    return {
        'lengths': lengths,
        'nn_data': [nn_macro, nn_jk, nn_micro],
        'hf_data': [hf_macro, hf_jk, hf_micro],
        'nn_energy': nn_energy,
        'hf_energy': hf_energy,
        'indices': indices
    }


def plot_step_comparisons(data_dict, save_figs=True):
    mpl.rcParams['pdf.fonttype'] = 42
    
    lengths = data_dict['lengths']
    nn_data_all = data_dict['nn_data']
    hf_data_all = data_dict['hf_data']
    
    fig_titles = ['Macro Iterations', 'JK Steps', 'Micro Iterations']
    ylabels = ['Macro Steps', 'JK Steps', 'Micro Steps']
    y_limits = [12, 120, 25]

    for i in range(3):
        plt.figure()
        plt.plot(lengths[:-2], nn_data_all[i][:-2], label='BasisNN', marker='o', markerfacecolor='white')
        plt.plot(lengths[:-2], hf_data_all[i][:-2], label='HF', marker='o', markerfacecolor='white')
        plt.xlabel('Bond Length (Å)')
        plt.ylabel(ylabels[i])
        plt.title(f'CASSCF {fig_titles[i]} Comparison')
        plt.tick_params(axis='both', direction='in', length=6, width=1, labelsize=14, top=True, right=True)
        plt.xlim(0, 2.5)
        plt.ylim(0, y_limits[i])
        plt.legend()
        plt.tight_layout()
        
        if save_figs:
            plt.savefig(f'figs/casscf_{fig_titles[i].lower().replace(" ", "_")}.pdf', metadata={"TextAsShapes": False})

# This is part of Figure 4
def plot_energy_difference(data_dict, save_figs=True):
    lengths = data_dict['lengths']
    nn_energy = data_dict['nn_energy']
    hf_energy = data_dict['hf_energy']
    
    # Calculate energy difference in milli-Hartree
    energy_diff = np.array([nn - hf for nn, hf in zip(nn_energy, hf_energy)]) * 1e3
    print('Energy difference (mHa):', energy_diff)

    plt.figure()
    plt.plot(lengths[:-2], energy_diff[:-2], label='ΔE (NN - HF)', 
             marker='o', markerfacecolor='white', color='red')
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel('Bond Length (Å)')
    plt.ylabel('Energy Difference ($10^{-3}$ Ha)')
    plt.title('CASSCF Energy Difference (NN - HF)')
    plt.tick_params(axis='both', direction='in', length=6, width=1, labelsize=14, top=True, right=True)
    plt.xlim(0, 2.5)
    plt.ylim(-3, 0.5)
    plt.legend()
    plt.tight_layout()
    
    if save_figs:
        plt.savefig('figs/casscf_energy_difference.pdf', metadata={"TextAsShapes": False})


def plot_convergence_curve(log_file="log_4.txt", save_figs=True):
    try:
        with open(log_file, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"{log_file} not found in the current directory.")

    # Extract initial CASCI energy
    casci_lines = [line for line in lines if "CASCI E" in line]
    basisNN_E0 = float(re.search(r"CASCI E =\s*(-?\d+\.\d+)", casci_lines[0]).group(1))
    hf_E0 = float(re.search(r"CASCI E =\s*(-?\d+\.\d+)", casci_lines[2]).group(1))

    # Extract macro iteration energies
    macro_pattern = re.compile(r"CASSCF E =\s*(-?\d+\.\d+)")
    macro_lines = [line for line in lines if "macro iter" in line]
    basisNN_lines = macro_lines[:2]
    hf_lines = macro_lines[2:]

    basisNN_E = [basisNN_E0] + [float(macro_pattern.search(line).group(1)) for line in basisNN_lines]
    hf_E = [hf_E0] + [float(macro_pattern.search(line).group(1)) for line in hf_lines]

    # Plotting
    plt.figure(figsize=(6, 4))
    plt.plot(range(len(basisNN_E)), basisNN_E, label="BasisNN init", marker='o', markerfacecolor='white')
    plt.plot(range(len(hf_E)), hf_E, label="HF init", marker='o', markerfacecolor='white')
    plt.xlabel("Macro Iteration")
    plt.ylabel("CASSCF Energy (Ha)")
    plt.title(f"CASSCF Energy Convergence ({log_file})")
    plt.legend()
    plt.tick_params(axis='both', direction='in', length=6, width=1, labelsize=14, top=True, right=True)
    plt.xlim(0, 6)
    plt.ylim(-2.16, -2.12)
    plt.tight_layout()
    
    if save_figs:
        plt.savefig("figs/log4_casscf_energy_curve.pdf", metadata={"TextAsShapes": False})

    plt.show()


def main():
    print("=== Figure 4 Analysis ===")
    
    # Load molecular data
    pos_l, elements_l, proj_l, name_l = load_inference_data("inference.json")
    
    run_all_calculations(pos_l, elements_l, proj_l, name_l, max_index=23)
    
    # Extract data from log files
    log_files = sorted(glob.glob("log_*.txt"), key=lambda x: int(re.search(r'\d+', x).group()))
    results = extract_log_data(log_files)
    
    # Print results table
    print_results_table(results)
    
    # Organize data for plotting
    data_dict = organize_data_for_plotting(results)
    
    print("\n Plots:")
    plot_step_comparisons(data_dict, save_figs=True)
    plot_energy_difference(data_dict, save_figs=True) # This is for Figure 4
    plot_convergence_curve("log_4.txt", save_figs=True)
    
    return results, data_dict


if __name__ == "__main__":
    results, data_dict = main()
    
