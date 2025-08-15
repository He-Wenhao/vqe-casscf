import json
import numpy as np
import os
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pyscf import gto, scf, mcscf, fci
import scipy
from scipy.stats import gaussian_kde
from numpy.linalg import eigh, inv
from pyscf import gto, scf, dft
# from vqe_driver import perm_orca2pyscf, run_vqe_cas
from downfolding_methods_pytorch import nelec, norbs, fock_downfolding, Solve_fermionHam, perm_orca2pyscf, LambdaQ


mpl.rcParams['pdf.fonttype'] = 42
FOLDER = "data_H4/H4_hyb_diss_2"
FILE_PATH = "data_H4/inference.json"
RESULT_FOLDER = "results/fig2/"
num_chains = 25  # Number of chains
chain_length = 4  # Number of H atoms per chain
bond_length_interval = 0.1


def initialize_data_structures(result_folder=RESULT_FOLDER):
    methods = ["CASSCF", "HF", "B3LYP", "sto-3G", "basisNN", "ccpVDZ"]
    
    method_index = {
        'basisNN': 0,
        'CASSCF': 1,
        'HF': 2,
        "B3LYP": 3,
        'sto-3G': 4
    }
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    energy_data = {method: [] for method in methods}
    l_data = {}
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    with open(os.path.join(result_folder, "E_l_data.json"), "w") as f:
        json.dump({"energy_data": energy_data, "l_data": l_data}, f)
    
    return methods, method_index, colors, energy_data, l_data


def load_and_process_inference_data(file_path=FILE_PATH, result_folder=RESULT_FOLDER):
    with open(file_path, "r") as file:
        data = json.load(file)
    
    eigenvalue_l = []
    name_l = np.array(data["name"])
    
    print(name_l)
    print(len(data["proj"]))
    
    for i in range(len(data["proj"])):
        # Extract the first matrix under the 'proj' key
        i_matrix = np.array(data["proj"][i])
        
        # Compute the eigenvalues
        eigenvalues = np.linalg.eigvals(i_matrix)
        eigenvalues.sort()
        eigenvalue_l.extend(eigenvalues)
    
    print(len(eigenvalue_l))
    
    # write name_l to a E_l_data.json file
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    with open(os.path.join(result_folder, "E_l_data.json"), "r") as f:
        jdata = json.load(f)
    jdata["energy_data"]['name'] = name_l.tolist()
    with open(os.path.join(result_folder, "E_l_data.json"), "w") as f:
        json.dump(jdata, f, indent=4)
    
    return data, eigenvalue_l, name_l


def plot_distribution(eigenvalue_l, bins=50,result_folder=RESULT_FOLDER):
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    plt.figure(figsize=(8, 5))
    plt.hist(eigenvalue_l, bins=bins, edgecolor='black', alpha=0.7)
    
    # Add vertical dashed lines
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1)
    plt.axvline(x=1, color='blue', linestyle='--', linewidth=1)
    
    # Add text labels
    plt.text(0, plt.ylim()[1] * 0.9, 'eigen=0', color='red', ha='right', fontsize=12)
    plt.text(1, plt.ylim()[1] * 0.9, 'eigen=1', color='blue', ha='left', fontsize=12)
    
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.title('Eigenvalue Distribution')
    plt.grid(True, linestyle='--')
    plt.savefig(os.path.join(result_folder, "eigenvalue_distribution.pdf"))


def calc_basisNN_inp_file(inp_data):
    elements = inp_data['elements']
    coordinates = inp_data['coordinates']
    atoms = [(elements[i], coordinates[i]) for i in range(len(elements))]
    S = gto.M(
        atom=atoms,  # Atomic symbols and coordinates
        basis="cc-pVDZ"
    ).intor("int1e_ovlp")
    sqrtS = scipy.linalg.sqrtm(S).real
    perm = perm_orca2pyscf(
        atom=atoms,  # Atomic symbols and coordinates
        basis="cc-pVDZ"
    )
    
    proj = inp_data['proj']
    proj = perm @ proj @ perm.T
    proj = sqrtS @ proj @ sqrtS
    
    n_fold = norbs(atom=atoms,basis='sto-3g')
    ham = fock_downfolding(n_fold,('self-defined',-proj),False,atom=atoms, basis='cc-pVDZ')
    E = Solve_fermionHam(ham.Ham_const, ham.int_1bd, ham.int_2bd, nele=nelec(atom=atoms, basis='sto-3G'), method='FCI')[0]
    l = LambdaQ(ham.Ham_const,ham.int_1bd,ham.int_2bd).item()
    return E, l


def compute_basisNN_energies(data, result_folder = RESULT_FOLDER):
    print(len(data["proj"]))
    
    E_basisNN_l = []
    l_basisNN_l = []
    
    with tqdm(total=len(data["proj"]), desc="Processing", dynamic_ncols=True) as pbar:
        for proj, elements, pos in zip(data["proj"], data["elements"], data["pos"]):
            inp_data = {'elements': elements, 'coordinates': pos, 'proj': proj}
            basisNN_E, basisNN_l = calc_basisNN_inp_file(inp_data)
            E_basisNN_l.append(basisNN_E)
            l_basisNN_l.append(basisNN_l)
            
            pbar.set_postfix({"basisNN_E": f"{basisNN_E:.6f}"})  # Format to 6 decimals for better readability
            pbar.update(1)
            
    # Save results to JSON files
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    
    with open(os.path.join(result_folder, "E_l_data.json"), "r") as f:
        jdata = json.load(f)
    jdata["energy_data"]['basisNN'] = E_basisNN_l
    jdata["l_data"]['basisNN'] = l_basisNN_l
    with open(os.path.join(result_folder, "E_l_data.json"), "w") as f:
        json.dump(jdata, f, indent=4)
    print(f"Results saved to {result_folder}/E_l_data.json")


def collect_other_energies(name_l, folder=FOLDER, result_folder=RESULT_FOLDER):
    with open(os.path.join(result_folder, "E_l_data.json"), "r") as f:
        energy_data = json.load(f)
    energy_data["energy_data"]['CASSCF'] = []
    energy_data["energy_data"]['HF'] = []
    energy_data["energy_data"]['B3LYP'] = []
    energy_data["energy_data"]['sto-3G'] = []
    
    for name in name_l:
        # Load the JSON file
        file_path = os.path.join(folder, 'obs', name)
        with open(file_path, "r") as file:
            data = json.load(file)
        energy_data["energy_data"]['CASSCF'].append(data['E_opt']['E'])
        energy_data["energy_data"]['HF'].append(data['HF']['E'])
        energy_data["energy_data"]['B3LYP'].append(data['B3LYP']['E'])
        energy_data["energy_data"]['sto-3G'].append(data['sto-3G']['E'])
        energy_data["energy_data"]['ccpVDZ'].append(data['ccpVDZ']['E'])
    with open(os.path.join(result_folder, "E_l_data.json"), "w") as f:
        json.dump(energy_data, f, indent=4)
        


def collect_l_data(name_l, folder=FOLDER, output_folder=RESULT_FOLDER):
    # Other energies
    with open(os.path.join(output_folder, "E_l_data.json"), "r") as f:
        l_data = json.load(f)
    l_data['l_data']['l_opt'] = []
    l_data['l_data']['CASSCF'] = []
    l_data['l_data']['HF'] = []
    l_data['l_data']['B3LYP'] = []
    l_data['l_data']['sto-3G'] = []
    
    for name in name_l:
        # Load the JSON file
        file_path = os.path.join(folder, 'obs', name)
        with open(file_path, "r") as file:
            data = json.load(file)
        l_data['l_data']['l_opt'].append(data['l_opt']['l'])
        l_data['l_data']['CASSCF'].append(data['E_opt']['l'])
        l_data['l_data']['HF'].append(data['HF']['l'])
        l_data['l_data']['B3LYP'].append(data['B3LYP']['l'])
        l_data['l_data']['sto-3G'].append(data['sto-3G']['l'])
        
    with open(os.path.join(output_folder, "E_l_data.json"), "w") as f:
        json.dump(l_data, f, indent=4)
    print(f"L data saved to {output_folder}/E_l_data.json")
    






def main():
    # Initialize data structures
    initialize_data_structures()
    
    # Load and process inference data
    data, eigenvalue_l, name_l = load_and_process_inference_data()
    
    # Plot eigenvalue distribution
    plot_distribution(eigenvalue_l)
    
    # Compute basisNN energies
    compute_basisNN_energies(data)
    
    # Collect energies from other methods
    collect_other_energies(name_l, FOLDER)
    
    # Plot energy distribution
    #plot_energy_distribution(methods, energy_data, name_l, method_index, colors)
    
    # Collect L1 norm data
    collect_l_data(name_l, FOLDER)
    
    # Plot L1 norm distribution
    #plot_l_norm_distribution(l_data, name_l, method_index, colors, methods)
    
    # Plot molecule cartoon with error
    #ax1 = plot_cartoon_with_error(energy_data, name_l, method_index, methods)
    
    # Save legend separately
    #save_legend_separately(ax1)
    

if __name__ == "__main__":
    main()
