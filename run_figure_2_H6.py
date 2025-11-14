import json
import numpy as np
import os
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from pyscf.mcscf import avas
import numpy as np
from pyscf import gto, scf, mcscf, fci
import scipy
from scipy.stats import gaussian_kde
from numpy.linalg import eigh, inv
from pyscf import gto, scf, dft
from pyscf.fci.direct_spin1 import FCI as DirectSpinFCI
import torch
import json
from functools import reduce
from vqe_driver import perm_orca2pyscf

mpl.rcParams['pdf.fonttype'] = 42
FOLDER = "data_H6/H6_hyb_diss_2"
FILE_PATH = "data_H6/inference.json"
RESULT_FOLDER = "results/fig2_H6/"
num_chains = 25  # Number of chains
chain_length = 6  # Number of H atoms per chain
bond_length_interval = 0.1
device = 'cpu'

class Fermi_Ham:
    def __init__(self,Ham_const = None,int_1bd = None,int_2bd = None):
        self.Ham_const = Ham_const
        self.int_1bd = int_1bd
        self.int_2bd = int_2bd
    # initialize with molecule configuration
    def pyscf_init(self,**kargs):
        self.mol = gto.Mole()
        self.mol.build(**kargs)
    # calculate fermionic Hamiltonian in terms of atomic orbital
    # notice: this is not orthonormalized
    def calc_Ham_AO(self): 
        self._int_1bd_AO = self.mol.intor_symmetric('int1e_kin') + self.mol.intor_symmetric('int1e_nuc')
        self._int_2bd_AO = self.mol.intor('int2e')
        self.Ham_const = self.mol.energy_nuc()
        # tensorize
        self._int_1bd_AO = torch.tensor(self._int_1bd_AO,device=device)
        self._int_2bd_AO = torch.tensor(self._int_2bd_AO,device=device)
    # use a basis to othonormalize it
    # each base vector v is a column vector, basis=[v1,v2,...,vn]
    # we can only keep first n_cut basis vectors
    # basis must satisfy basis.T @ overlap @ basis = identity
    def calc_Ham_othonormalize(self,basis,ncut):
        cbasis = basis[:,:ncut]
        self.basis = cbasis
        self.int_1bd = torch.einsum('qa,ws,qw -> as',cbasis,cbasis,self._int_1bd_AO)
        
        self.int_2bd = torch.einsum('qa,ws,ed,rf,qwer -> asdf',cbasis,cbasis,cbasis,cbasis,self._int_2bd_AO) 
        
    def check_AO(self):
        print('AOs:',self.mol.ao_labels())

def LambdaQ(Ham_const,int_1bd,int_2bd,backend=torch):
    trace = backend.trace
    einsum = backend.einsum
    abs = backend.abs
    sum = backend.sum

    h = int_1bd
    g = int_2bd
    N = h.shape[0]

    # Î»_C = | sum_p h_pp + 1/2 sum_{p,r} g_pprr - 1/4 sum_{p,r} g_prrp |
    diag_h = trace(h)
    g_pprr = einsum('pprr->', g)
    g_prrp = einsum('prrp->', g)  # permute and sum over r â†’ g[p,r,r,p]
    lambda_C = abs(diag_h + 0.5 * g_pprr - 0.25 * g_prrp+Ham_const)

    # Î»_T = sum_{p,q} | h_pq + sum_r g_pqrr - 1/2 sum_r g_prrq |
    sum_r_g_pqrr = einsum('pqrr->pq', g)
    sum_r_g_prrq = einsum('prrq->pq', g)  # permute and sum over r â†’ g[p,r,*,q]
    T_term = h + sum_r_g_pqrr - 0.5 * sum_r_g_prrq
    lambda_T = sum(abs(T_term))

    # Î»_V' = 1/2 sum_{p>r, s>q} |g_pqrs - g_psrq| + 1/4 sum_{pqrs} |g_pqrs|
    indices = backend.arange(N)
    p, q, r, s = backend.meshgrid(indices, indices, indices, indices, indexing='ij')
    mask = (p > r) & (s > q)
    diff = g[p, q, r, s] - g[p, s, r, q]
    lambda_V_prime_1 = 0.5 * sum(abs(diff[mask]))
    lambda_V_prime_2 = 0.25 * sum(abs(g))
    lambda_V_prime = lambda_V_prime_1 + lambda_V_prime_2
    return lambda_C + lambda_T + lambda_V_prime

def fock_downfolding(n_folded,fock_method,QO,**kargs):
    ham = Fermi_Ham()
    # initilize with molecule configuration
    ham.pyscf_init(**kargs)
    # run HF, get Fock and overlap matrix
    ham.mol.verbose = 0
    # Function to diagonalize the Fock matrix
    def eig(h, s):
        d, t = np.linalg.eigh(s)
        x = t[:, d > 1e-8] / np.sqrt(d[d > 1e-8])
        xhx = reduce(np.dot, (x.T, h, x))
        e, c = np.linalg.eigh(xhx)
        c = np.dot(x, c)
        return e, c
    def eig_torch(h, s):
        d, t = torch.linalg.eigh(s)
        x = t[:, d > 1e-8] / torch.sqrt(d[d > 1e-8])
        xhx = reduce(torch.dot, (x.T, h, x))
        e, c = torch.linalg.eigh(xhx)
        c = torch.dot(x, c)
        return e, c

    # Perform RHF calculation as a starting point
    if fock_method == 'HF':
        myhf = scf.RHF(ham.mol)
        myhf.eig = eig
        myhf.kernel()
        fock_AO = myhf.get_fock()
    elif fock_method == 'B3LYP':
        myhf = scf.RKS(ham.mol)
        myhf.xc = 'B3LYP'
        myhf.eig = eig
        myhf.kernel()
        fock_AO = myhf.get_fock()
    elif fock_method == 'lda,vwn':
        myhf = scf.RKS(ham.mol)
        myhf.xc = 'lda,vwn'
        myhf.eig = eig
        myhf.kernel()
        fock_AO = myhf.get_fock()
    elif fock_method == 'EGNN':
        sys.path.append('/home/hewenhao/Documents/wenhaohe/research/VQE_downfold')
        from get_fock import get_NN_fock
        fock_AO = get_NN_fock(ham.mol)
    elif fock_method[0] == 'self-defined':
        fock_AO = fock_method[1]
    else:
        raise TypeError('fock_method ', fock_method, ' does not exist')
    overlap = ham.mol.intor('int1e_ovlp')
    # solve generalized eigenvalue problem, the generalized eigen vector is new basis
    _energy, basis = eig(fock_AO,overlap)
    basis = torch.tensor(basis,device=device)
    #print('my basis:\n',basis)
    #print('basic basis:\n',myhf.mo_coeff)
    # if quasi orbital is used
    if QO:
        half_nele = sum(ham.mol.nelec) // 2
        # filled orbitals
        fi_orbs = basis[:,:half_nele]
        # W matrix from Eq. (33) in https://journals.aps.org/prb/pdf/10.1103/PhysRevB.78.245112
        Wmat = (overlap - overlap @ fi_orbs @ fi_orbs.T @ overlap)
        # diagonalize W_mat, find maximal eigen states
        #Wmat_orth = overlap_mh @ Wmat @ overlap_mh
        _energy2, Weig = torch.linalg.eigh(-Wmat)
        print('energy2:',_energy2)
        emp_orbs = (torch.eye(overlap.shape[0]) - fi_orbs @ fi_orbs.T @ overlap ) @ Weig @ scipy.linalg.fractional_matrix_power(-torch.diag(_energy2), (-1/2)) 
        # build new basis
        QO_basis = torch.hstack( (fi_orbs , emp_orbs[:,:overlap.shape[0]-half_nele]) ) 
        basis = QO_basis
    # calculate downfolding hamiltonian accroding to basis
    ham.calc_Ham_AO()   # calculate fermionic hamiltonian on AO basis
    ham.calc_Ham_othonormalize(basis,n_folded) # fermionic hamiltonian on new basis
    # save the halmiltonian
    return ham

# return the total number of basis
def norbs(**kargs):
    ham = Fermi_Ham()
    # initilize with molecule configuration
    ham.pyscf_init(**kargs)
    # run HF, get Fock and overlap matrix
    overlap = ham.mol.intor('int1e_ovlp')
    return overlap.shape[0]

# return the total number of electrons
def nelec(**kargs):
    ham = Fermi_Ham()
    # initilize with molecule configuration
    ham.pyscf_init(**kargs)
    # run HF, get Fock and overlap matrix
    total_electrons = ham.mol.nelectron
    return total_electrons


def initialize_data_structures(result_folder=RESULT_FOLDER):
    methods = ["CASSCF", "HF", "B3LYP", "sto-3G", "basisNN", "ccpVDZ", "AVAS"]
    
    method_index = {
        'basisNN': 0,
        'CASSCF': 1,
        'HF': 2,
        "B3LYP": 3,
        'sto-3G': 4,
        'AVAS': 5
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

    mol = gto.M(atom=atoms, basis="cc-pVDZ")
    S = mol.intor("int1e_ovlp")
    S = 0.5*(S+S.T)

    s, Us = np.linalg.eigh(S); s = np.clip(s, 1e-12, None)
    sqrtS = Us @ np.diag(np.sqrt(s)) @ Us.T
    inv_sqrtS = Us @ np.diag(1.0/np.sqrt(s)) @ Us.T

    perm = perm_orca2pyscf(atom=atoms, basis="cc-pVDZ")
    P = inp_data['proj']
    P = perm @ P @ perm.T
    P = 0.5*(P+P.T)
    P_ortho = sqrtS @ P @ sqrtS
    P_ortho = 0.5*(P_ortho + P_ortho.T)

    ncas = norbs(atom=atoms, basis='sto-3g')

    ham = fock_downfolding(ncas, ('self-defined', -P_ortho), False, atom=atoms, basis='cc-pVDZ')

    def _to_np(x):
        try:
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
        except Exception:
            pass
        return np.asarray(x, dtype=float)

    H0 = float(_to_np(ham.Ham_const))
    h1 = _to_np(ham.int_1bd)
    g2 = _to_np(ham.int_2bd)

    tot = nelec(atom=atoms, basis='sto-3G')
    nele_tuple = (tot // 2, tot // 2)

    # Solve hamiltonian with PySCF
    fci_solver = DirectSpinFCI()
    fci_solver.spin = 0
    fci_solver.nroots = 1

    e_ci, ci_vec = fci_solver.kernel(h1, g2, ncas, nele_tuple)
    E = H0 + float(e_ci)

    l = float(LambdaQ(ham.Ham_const, ham.int_1bd, ham.int_2bd))

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



def compute_avas_fci_energies(data, result_folder=RESULT_FOLDER):
    """
    Runs an RHF->AVAS->FCI calculation for each geometry
    and saves the energies and L1 norms.
    """
    print("\n" + "="*60)
    print("Running AVAS-FCI calculations...")
    print("="*60)
    
    E_avas_l = []
    l_avas_l = []
    
    with tqdm(total=len(data["pos"]), desc="Processing AVAS-FCI", dynamic_ncols=True) as pbar:
        for pos, elements in zip(data["pos"], data["elements"]):
            atom = [[el, tuple(c)] for el, c in zip(elements, pos)]
            
            # --- 1. Build molecule ---
            mol = gto.M(atom=atom, basis='cc-pVDZ', spin=0, charge=0, verbose=0)
            
            # --- 2. Run RHF (needed for AVAS) ---
            mf = scf.RHF(mol).run()
            
            # --- 3. Run AVAS to get active space ---
            # This is the correct way to call AVAS
            ao_labels = ["H 1s"]
            ncas_avas, nelecas_avas, mo_avas = avas.kernel(mf, ao_labels, threshold=0.95)
            ncas_avas=4
            nelecas_avas=4
            # --- 4. Get Hamiltonian in AVAS-MO basis ---
            ham = Fermi_Ham()
            ham.pyscf_init(atom=atom, basis='cc-pVDZ')
            ham.calc_Ham_AO()
            # Project AO integrals into the AVAS-MO basis
            ham.calc_Ham_othonormalize(torch.tensor(mo_avas, device=device), ncut=ncas_avas)
            
            # --- 5. Run classical FCI on the small AVAS Hamiltonian ---
            h1 = ham.int_1bd.detach().cpu().numpy()
            g2 = ham.int_2bd.detach().cpu().numpy()
            nele_tuple = (nelecas_avas // 2, nelecas_avas // 2)

            fci_solver = DirectSpinFCI()
            fci_solver.spin = 0
            fci_solver.nroots = 1
            e_ci, ci_vec = fci_solver.kernel(h1, g2, ncas_avas, nele_tuple)
            
            E_total = ham.Ham_const + float(e_ci)
            E_avas_l.append(E_total)
            
            # --- 6. Calculate L1 norm (LambdaQ) ---
            l_avas = float(LambdaQ(ham.Ham_const, ham.int_1bd, ham.int_2bd))
            l_avas_l.append(l_avas)
            
            pbar.set_postfix({"AVAS_E": f"{E_total:.6f}"})
            pbar.update(1)

    # --- 7. Save results to the JSON file ---
    with open(os.path.join(result_folder, "E_l_data.json"), "r") as f:
        jdata = json.load(f)
    
    # Ensure the keys exist before trying to append
    if "AVAS" not in jdata["energy_data"]:
        jdata["energy_data"]['AVAS'] = []
    if "AVAS" not in jdata["l_data"]:
        jdata["l_data"]['AVAS'] = []
        
    jdata["energy_data"]['AVAS'] = E_avas_l
    jdata["l_data"]['AVAS'] = l_avas_l
    
    with open(os.path.join(result_folder, "E_l_data.json"), "w") as f:
        json.dump(jdata, f, indent=4)
    print(f"AVAS-FCI results saved to {result_folder}/E_l_data.json")


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

    compute_avas_fci_energies(data)
    
    # Collect energies from other methods
    collect_other_energies(name_l, FOLDER)
    
    # Collect L1 norm data
    collect_l_data(name_l, FOLDER)
    

if __name__ == "__main__":
    main()
