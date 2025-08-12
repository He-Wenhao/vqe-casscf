import os
import sys
# Path to exclude
exclude_path = '/opt/apps/intel19/impi19_0/python3/3.7.0/lib/python3.7/site-packages' 
# Remove the path if it exists in sys.path
if exclude_path in sys.path:
    sys.path.remove(exclude_path)
import numpy as np
from pyscf import gto
import openfermion
#from openfermion.transforms import get_fermion_operator, jordan_wigner, bravyi_kitaev
#from openfermion.linalg import get_sparse_operator
import scipy
import itertools
from pyscf import fci
from pyscf import gto, scf, ao2mo, cc
from scipy.linalg import expm
from numpy import linalg as LA

import sys
import torch
from torch import tensor
import time

import json
from functools import reduce

torch.set_printoptions(precision=10)
device = 'cpu'
# we describe fermionic hamiltonian as a 3 element tuple: (Ham_const, int_1bd,int_2bd):
# Hamiltonian = Ham_const + \sum_{ij}(int_1bd)_{ij}c_i^daggerc_j + \sum{ijkl}(int_2bd)_{}ijkl c_i^daggerc_jc_k^daggerc_l    (index order needed checked)

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
        


def add_spin_2bd(int_2bd):
    dim = int_2bd.shape[0]
    res = torch.zeros((2*dim,2*dim,2*dim,2*dim))
    for i1,i2,i3,i4 in itertools.product(range(dim), repeat=4):
        for s1,s2,s3,s4 in itertools.product(range(2), repeat=4):
            if s1 == s4 and s2 == s3:
                res[i1*2+s1,i2*2+s2,i3*2+s3,i4*2+s4] = int_2bd[i1,i2,i3,i4]
    return res

def add_spin_1bd(int_1bd):
    dim = int_1bd.shape[0]
    res = torch.zeros((2*dim,2*dim))
    for i1,i2 in itertools.product(range(dim), repeat=2):
        for s1,s2 in itertools.product(range(2), repeat=2):
            if s1 == s2:
                res[i1*2+s1,i2*2+s2] = int_1bd[i1,i2]
    return res


    

def basis_downfolding_init(fock_method = 'HF',**kargs):
    ham = Fermi_Ham()
    # initilize with molecule configuration
    ham.pyscf_init(**kargs)
    # run HF, get Fock and overlap matrix
    ham.mol.verbose = 0
    overlap = ham.mol.intor('int1e_ovlp')
    ham.calc_Ham_AO()   # calculate fermionic hamiltonian on AO basis
    overlap_mh = scipy.linalg.fractional_matrix_power(overlap, (-1/2)).real
    
    if fock_method == 'AO':
        h_orth = np.eye(overlap.shape[0])
    else:
        if fock_method == 'HF':
            myhf = ham.mol.RHF().run()
            fock_AO = myhf.get_fock()
        elif fock_method == 'B3LYP':
            myhf = ham.mol.RKS().run()
            myhf.xc = 'B3LYP'
            fock_AO = myhf.get_fock()
        elif fock_method == 'lda,vwn':
            myhf = ham.mol.RKS().run()
            myhf.xc = 'lda,vwn'
            fock_AO = myhf.get_fock()
        # solve generalized eigenvalue problem, the generalized eigen vector is new basis

        h_orth = overlap_mh @ fock_AO @ overlap_mh
    
    h_orth = torch.tensor(h_orth,device=device)
    overlap_mh = torch.tensor(overlap_mh,device=device)
    _energy, basis_orth = torch.linalg.eigh(h_orth)
    return ham,overlap_mh, basis_orth

# Many body Hamiltonian with downfolding technique
def basis_downfolding(ham:Fermi_Ham,overlap_mh, basis_orth, n_folded):
    basis = torch.einsum('ij,jk->ik',overlap_mh , basis_orth)
    #print('my basis:\n',basis)
    #print('basic basis:\n',myhf.mo_coeff)
    # calculate downfolding hamiltonian accroding to basis
    ham.calc_Ham_othonormalize(basis,n_folded) # fermionic hamiltonian on new basis
    # save the halmiltonian
    return ham

def E_optimized_basis(nbasis=2,**kargs):
    import numpy as np
    from scipy.optimize import minimize
    from scipy.linalg import qr

    # dimension before folding
    n_bf = norbs(**kargs)
    # initialize initial guess
    ham0,overlap_mh, basis_orth_init = basis_downfolding_init(**kargs)
    def cost_function(basis_orth_flat):
        t0 = time.time() 
        basis_orth_flat = torch.tensor(basis_orth_flat,device=device)
        t1 = time.time()  # Capture the end time
        basis_orth = basis_orth_flat.reshape((n_bf,nbasis))
        t2 = time.time()  # Capture the end time
        basis_orth, _R = torch.linalg.qr(basis_orth,mode='reduced')    # orthorgonalize basis_orth
        t3 = time.time()  # Capture the end time
        ham = basis_downfolding(ham0,overlap_mh, basis_orth, n_folded=nbasis)
        t4 = time.time()  # Capture the end time
        E, properties = Solve_fermionHam(ham.Ham_const,ham.int_1bd,ham.int_2bd,nele=sum([2,2]),method='FCI')
        t5 = time.time()  # Capture the end time
        rdm1, rdm2 = properties['rdm1'], properties['rdm2']
        print('fci energy:',E,'new_energy:',construct_torch_E(rdm1,rdm2,ham))
        #print("t1:{};t2:{};t3:{},t4:{},t5:{}".format(t1-t0,t2-t1,t3-t2,t4-t3,t5-t4))
        return E
    # minimize cost_func over a SO(n_bf) group
    
    # Initial guess (needs to be orthogonal)
    Q0 = basis_orth_init[:,:nbasis]
    Q0_flat = Q0.flatten()
    #print('initial guess:',basis_orth_init)
    # Optimization using Nelder-Mead
    result = minimize(cost_function, Q0_flat, method='Nelder-Mead',options = {'maxiter': 0})

    # Reshape the result back into a matrix
    Q_opt = result.x.reshape((n_bf, nbasis))
    Q_opt, _ = qr(Q_opt)  # Ensure the result is orthogonal
    #print(result)
    #print("Optimal matrix Q:", Q_opt)
    return result.fun

def E_optimized_basis_gradient_pytorch(nbasis=2,method='FCI',log_file='opt_log.txt',**kargs):
    import numpy as np
    from scipy.optimize import minimize
    from scipy.linalg import qr

    # dimension before folding
    n_bf = norbs(**kargs)
    nele = nelec(**kargs)
    assert nele %2 == 0
    # initialize initial guess
    ham0,overlap_mh, basis_orth_init = basis_downfolding_init(**kargs)
    def cost_function(basis_orth,method):
        t0 = time.time() 
        t1 = time.time()  # Capture the end time
        t2 = time.time()  # Capture the end time
        basis_orth, _R = torch.linalg.qr(basis_orth,mode='reduced')    # orthorgonalize basis_orth
        t3 = time.time()  # Capture the end time
        ham = basis_downfolding(ham0,overlap_mh, basis_orth, n_folded=nbasis)
        t4 = time.time()  # Capture the end time
        E, properties = Solve_fermionHam(ham.Ham_const,ham.int_1bd,ham.int_2bd,nele=sum([nele//2,nele//2]),method=method)
        t5 = time.time()  # Capture the end time
        rdm1, rdm2 = properties['rdm1'], properties['rdm2']
        #print('fci energy:',E,'new_energy:',construct_torch_E(rdm1,rdm2,ham))
        #print("t1:{};t2:{};t3:{},t4:{},t5:{}".format(t1-t0,t2-t1,t3-t2,t4-t3,t5-t4))
        return construct_torch_E(rdm1,rdm2,ham)
    # minimize cost_func over a SO(n_bf) group
    
    # Initial guess (needs to be orthogonal)
    Q = basis_orth_init.clone().detach()[:,:nbasis].requires_grad_(True)

    # Initialize the Adam optimizer
    optimizer = torch.optim.Adam([Q], lr=0.1)  # Learning rate is 0.1
    prev_loss = None
    # Optimization loop
    f = open(log_file, 'w')
    N_macro = 1000
    N_micro = 1
    for step in range(N_macro):
        optimizer.zero_grad()  # Reset the gradients to zero

        #loss = cost_function(Q,method)  # Compute the current loss
        t0 = time.time() 
        t1 = time.time()  # Capture the end time
        t2 = time.time()  # Capture the end time
        basis_orth, _R = torch.linalg.qr(Q,mode='reduced')    # orthorgonalize basis_orth
        t3 = time.time()  # Capture the end time
        ham = basis_downfolding(ham0,overlap_mh, basis_orth, n_folded=nbasis)
        t4 = time.time()  # Capture the end time
        E, properties = Solve_fermionHam(ham.Ham_const,ham.int_1bd,ham.int_2bd,nele=sum([nele//2,nele//2]),method=method)
        t5 = time.time()  # Capture the end time
        rdm1, rdm2 = properties['rdm1'], properties['rdm2']
        #print('fci energy:',E,'new_energy:',construct_torch_E(rdm1,rdm2,ham))
        #print("t1:{};t2:{};t3:{},t4:{},t5:{}".format(t1-t0,t2-t1,t3-t2,t4-t3,t5-t4))
        loss = construct_torch_E(rdm1,rdm2,ham)
        
        loss.backward()        # Perform backpropagation to compute the gradients
        optimizer.step()       # Update parameters using the computed gradients
        for i_micro in range(N_micro - 1):
            optimizer.zero_grad()
            basis_orth, _R = torch.linalg.qr(Q,mode='reduced')
            ham = basis_downfolding(ham0,overlap_mh, basis_orth, n_folded=nbasis)
            loss = construct_torch_E(rdm1,rdm2,ham)
        
            loss.backward()        # Perform backpropagation to compute the gradients
            optimizer.step()       # Update parameters using the computed gradients
          
        
        if step % 10 == 0:
            f.write(f"Step {step+1}, Loss: {loss.item()}  ")
            f.write(f"gradient: {abs(Q.grad).sum()}\n")
            f.flush()
        # Check if the change in loss is smaller than the threshold
        
        if abs(Q.grad).sum() < 1e-3:
            f.write(f"convergent at epoch {step+1}; gradient {abs(Q.grad).sum()} is below threshold")
            break
        prev_loss = loss.item()
    else:
        f.write(f"max iteration achieved")
    f.close()
    return torch.linalg.qr(Q,mode='reduced')[0]

def E_optimized_basis_gradient(nbasis=2,method='FCI',log_file='opt_log.txt',**kargs):
    assert method == 'FCI'
    from pyscf import gto, scf, mcscf
    import numpy as np
    
    from scipy.linalg import fractional_matrix_power

    # Define the H2 molecule
    mol = gto.M(**kargs)

    # Perform Hartree-Fock Calculation
    # Function to diagonalize the Fock matrix
    def eig(h, s):
        d, t = np.linalg.eigh(s)
        x = t[:, d > 1e-8] / np.sqrt(d[d > 1e-8])
        xhx = reduce(np.dot, (x.T, h, x))
        e, c = np.linalg.eigh(xhx)
        c = np.dot(x, c)
        return e, c

    # Perform RHF calculation as a starting point
    mf = scf.RHF(mol)
    mf.eig = eig
    mf.kernel()
    nele = nelec(**kargs)
    # Perform CASSCF(2,2) Calculation
    cas = mcscf.CASSCF(mf, nele, nbasis)
    cas.kernel()

    # Get the optimized MO coefficients
    mo_coeff = cas.mo_coeff[:,:nbasis ]  # Optimized Molecular Orbital Coefficients
    # Optionally, print in a more readable format


    f = open(log_file, 'w')
    f.write(f'energy:{cas.e_tot}')
    f.close()
    
    # do normalization
    # Compute the overlap matrix S
    S = mol.intor("int1e_ovlp")  # Overlap matrix

    # Compute S^(-1/2) using Löwdin orthogonalization
    S_inv_sqrt = fractional_matrix_power(S, 0.5)

    # Apply Löwdin transformation to the CASSCF orbitals
    mo_lowdin = S_inv_sqrt @ mo_coeff
    
    # ortho_check = mo_lowdin.T @  mo_lowdin
    # print("Orthonormality Check (should be identity):\n", ortho_check)

    return mo_lowdin.T

def S_optimized_basis(**kargs):
    import numpy as np
    from scipy.optimize import minimize
    from scipy.linalg import qr

    # dimension before folding
    n_bf = norbs(**kargs)
    # initialize initial guess
    ham0,overlap_mh, basis_orth_init = basis_downfolding_init(**kargs)
    def cost_function(basis_orth_flat):
        basis_orth = basis_orth_flat.reshape((n_bf,2))
        basis_orth, _R = qr(basis_orth)    # orthorgonalize basis_orth
        basis_orth = basis_orth[:,:2]
        ham = basis_downfolding(ham0,overlap_mh, basis_orth, n_folded=2)
        S,E, rdm, _ = entropy_entangle(ham.Ham_const,ham.int_1bd,ham.int_2bd,nele=sum([1,1]),method='FCI')
        print('rdm',rdm)
        print(E,'S:',S)
        return -S
    # minimize cost_func over a SO(n_bf) group
    
    # Initial guess (needs to be orthogonal)
    Q0 = basis_orth_init[:,:2]
    Q0_flat = Q0.flatten()
    #print('initial guess:',basis_orth_init)
    # Optimization using Nelder-Mead
    result = minimize(cost_function, Q0_flat, method='Nelder-Mead')

    # Reshape the result back into a matrix
    Q_opt = result.x.reshape((n_bf, 2))
    Q_opt, _ = qr(Q_opt)  # Ensure the result is orthogonal
    #print(result)
    #print("Optimal matrix Q:", Q_opt)
    return result.fun

def S_optimized_basis_constraint(fock_method='HF',**kargs):
    import numpy as np
    from scipy.optimize import minimize
    from scipy.linalg import qr

    # dimension before folding
    n_bf = norbs(**kargs)
    # initialize initial guess
    ham0,overlap_mh, basis_orth_init = basis_downfolding_init(fock_method=fock_method,**kargs)
    first_orb_const = (basis_orth_init[:,0:1]).flatten()
    def cost_function(basis_orth_flat):
        basis_orth_flat = torch.hstack((first_orb_const.reshape(10,1),basis_orth_flat.reshape(10,1))).flatten()
        basis_orth = basis_orth_flat.reshape((n_bf,2))
        basis_orth, _R = qr(basis_orth)    # orthorgonalize basis_orth
        basis_orth = basis_orth[:,:2]
        ham = basis_downfolding(ham0,overlap_mh, basis_orth, n_folded=2)
        S,E, _, _ = entropy_entangle(ham.Ham_const,ham.int_1bd,ham.int_2bd,nele=sum([1,1]),method='FCI')
        print(E)
        return S, E
    # minimize cost_func over a SO(n_bf) group
    
    # Initial guess (needs to be orthogonal)
    Q0 = basis_orth_init[:,1:2]
    Q0_flat = Q0.flatten()
    #print('initial guess:',basis_orth_init)
    # Optimization using Nelder-Mead
    result = minimize(lambda x:-cost_function(x)[0], Q0_flat, method='Nelder-Mead')

    # Reshape the result back into a matrix
    #Q_opt = result.x.reshape((n_bf, 2))
    #Q_opt, _ = qr(Q_opt)  # Ensure the result is orthogonal
    #print(result)
    #print("Optimal matrix Q:", Q_opt)
    return cost_function(result.x)[1]

def S_optimized_basis_constraint_multi_rounds(fock_method='HF',**kargs):
    import numpy as np
    from scipy.optimize import minimize
    from scipy.linalg import qr

    # dimension before folding
    n_bf = norbs(**kargs)
    # initialize initial guess
    ham0,overlap_mh, basis_orth_init = basis_downfolding_init(fock_method=fock_method,**kargs)
    first_orb_const = (basis_orth_init[:,0:1]).flatten()
    old_first_orb_const = torch.zeros(first_orb_const.shape)
    oldx = []
    while torch.linalg.norm( first_orb_const- old_first_orb_const) > 1e-8: # optimize first_orb_const
        #print('big cycle')
        def cost_function(basis_orth_flat,first_orb_const):
            basis_orth_flat = torch.hstack((first_orb_const.reshape(10,1),basis_orth_flat.reshape(10,1))).flatten()
            basis_orth = basis_orth_flat.reshape((n_bf,2))
            basis_orth, _R = qr(basis_orth)    # orthorgonalize basis_orth
            basis_orth = basis_orth[:,:2]
            ham = basis_downfolding(ham0,overlap_mh, basis_orth, n_folded=2)
            S,E, rdm, FCIvec = entropy_entangle(ham.Ham_const,ham.int_1bd,ham.int_2bd,nele=sum([1,1]),method='FCI')
            print(E,S)
            return S, E
        # minimize cost_func over a SO(n_bf) group
        
        # Initial guess (needs to be orthogonal)
        if len(oldx) == 0:
            Q0 = basis_orth_init[:,1:2]
            Q0_flat = Q0.flatten()
            oldx = Q0_flat
        #print('initial guess:',basis_orth_init)
        # Optimization using Nelder-Mead
        result = minimize(lambda x:-cost_function(x,first_orb_const)[0], oldx, method='Nelder-Mead')
        oldx = result.x
        basis_orth = torch.hstack((first_orb_const.reshape(10,1),result.x.reshape((n_bf,1))))
        basis_orth, _R = qr(basis_orth)    # orthorgonalize basis_orth
        basis_orth = basis_orth[:,:2]
        # get a optimized new_first_orb_const
        #sub_basis_orth_flat = torch.eye(2).flatten()
        #sub_basis_orth_flat = torch.array([1,2,3,4])
        def cost_function_2(theta,basis_orth):
            st = torch.sin(theta)
            ct = torch.cos(theta)
            sub_basis_orth = torch.array([[ct,st],[-st,ct]]).reshape((2,2))
            #sub_basis_orth, _R = qr(sub_basis_orth)    # orthorgonalize basis_orth
            n_basis_orth = basis_orth @ sub_basis_orth
            ham = basis_downfolding(ham0,overlap_mh, n_basis_orth, n_folded=2)
            S,E, rdm, FCIvec = entropy_entangle(ham.Ham_const,ham.int_1bd,ham.int_2bd,nele=sum([1,1]),method='FCI')
            #print('n_basis_orth:',n_basis_orth)
            #print('1bd mat:',ham.int_1bd)
            #print('rdm:',rdm)
            #print('FCIvec:',FCIvec)
            print('cycle 2:',E,S)
            #print(E,S)
            return S, E
        new_result = minimize(lambda x:cost_function_2(x,basis_orth)[0], 0, method='Nelder-Mead')
        old_first_orb_const = first_orb_const
        theta = new_result.x
        st = torch.sin(theta)
        ct = torch.cos(theta)
        sub_basis_orth = torch.array([[ct,st],[-st,ct]]).reshape((2,2))
        #sub_basis_orth, _R = qr(sub_basis_orth)
        first_orb_const = (basis_orth @ sub_basis_orth)[:,0:1]
    # Reshape the result back into a matrix
    #Q_opt = result.x.reshape((n_bf, 2))
    #Q_opt, _ = qr(Q_opt)  # Ensure the result is orthogonal
    #print(result)
    #print("Optimal matrix Q:", Q_opt)
    return cost_function(result.x,first_orb_const)[1]
# Many body Hamiltonian with downfolding technique
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

def perm_orca2pyscf(**kargs):# direct sum
    mol = gto.M(**kargs)
    mol.verbose = 0
    def direct_sum(A, B):
        if type(A) == str:
            return B
        if type(B) == str:
            return A
        # Create an output matrix with the appropriate shape filled with zeros
        result = np.zeros((A.shape[0] + B.shape[0], A.shape[1] + B.shape[1]))
        
        # Place A in the top-left corner
        result[:A.shape[0], :A.shape[1]] = A
        
        # Place B in the bottom-right corner
        result[A.shape[0]:, A.shape[1]:] = B
        
        return result
    # change basis order
    perm_block = {
        's':np.array([[1]]),
        'p':np.array([
                    [0,1,0],
                    [0,0,1],
                    [1,0,0]]),
        'd':np.array([
                    [0,0,0,0,1],
                    [0,0,1,0,0],
                    [1,0,0,0,0],
                    [0,1,0,0,0],
                    [0,0,0,1,0]]),
        }
    ind = 0
    perm_mat = 'None'
    while ind < len(mol.ao_labels()):
        l_val = mol.ao_labels()[ind][5]
        if l_val == 's':
            ind += 1
        elif l_val == 'p':
            ind += 3
        elif l_val == 'd':
            ind += 5
        else:
            raise TypeError('wrong l value')
        perm_mat = direct_sum(perm_mat,perm_block[l_val])
    return perm_mat

# Solve Fermionic Hamiltonian
# https://github.com/pyscf/pyscf/blob/master/examples/cc/40-ccsd_custom_hamiltonian.py
def Solve_fermionHam(Ham_const,int_1bd,int_2bd,nele,method):
    #raise TypeError('not debugged yet')
    t0 = time.time()
    mol = gto.M(verbose=0)
    n = int_1bd.shape[0]
    mol.nelectron = nele
    t1 = time.time()
    # Function to diagonalize the Fock matrix
    def eig(h, s):
        d, t = np.linalg.eigh(s)
        x = t[:, d > 1e-8] / np.sqrt(d[d > 1e-8])
        xhx = reduce(np.dot, (x.T, h, x))
        e, c = np.linalg.eigh(xhx)
        c = np.dot(x, c)
        return e, c
    mf = scf.RHF(mol)
    mf.eig = eig
    mf.get_hcore = lambda *args: int_1bd.detach().numpy()
    mf.get_ovlp = lambda *args: np.eye(n)
    mf._eri = ao2mo.restore(8, int_2bd.detach().numpy(), n)
    mf.kernel()
    mol.incore_anyway = True
    t2 = time.time()


    # In PySCF, the customized Hamiltonian needs to be created once in mf object.
    # The Hamiltonian will be used everywhere whenever possible.  Here, the model
    # Hamiltonian is passed to CCSD/FCI object via the mf object.
    if method == 'CCSD':
        mycc = cc.CCSD(mf)
        mycc.kernel()xf
        rdm1_mo = tensor(mycc.make_rdm1(),device=device)
        rdm2_mo = tensor(mycc.make_rdm2(),device=device)
    elif method == 'FCI':
        mycc = fci.FCI(mf).run()
        rdm1_mo = tensor(mycc.make_rdm1(mycc.ci,mycc.norb,mycc.nelec),device=device)
        rdm2_mo = tensor(mycc.make_rdm2(mycc.ci,mycc.norb,mycc.nelec),device=device)
    else:
        raise TypeError('method not found')
    t3 = time.time()
    mo_coeff = tensor(mf.mo_coeff,device=device)
    rdm1_ao = torch.einsum('qa,ws,as->qw',mo_coeff,mo_coeff,rdm1_mo)
    rdm2_ao = torch.einsum('qa,ws,ed,rf,asdf -> qwer',mo_coeff,mo_coeff,mo_coeff,mo_coeff,rdm2_mo)
    #mycc.kernel()
    #e,v = mycc.ipccsd(nroots=3)
    #print("t1:{};t2:{};t3:{}".format(t1-t0,t2-t1,t3-t2))
    properties = {'rdm1':rdm1_ao,'rdm2':rdm2_ao}
    return mycc.e_tot+Ham_const, properties

# construct pytorch E
def construct_torch_E(rdm1,rdm2,ham:Fermi_Ham):
    E = ham.Ham_const + torch.einsum('ij,ij->',rdm1,ham.int_1bd)+0.5*torch.einsum('ijkl,ijkl->',rdm2,ham.int_2bd)
    return E

# Solve Fermionic Hamiltonian
# https://github.com/pyscf/pyscf/blob/master/examples/cc/40-ccsd_custom_hamiltonian.py
def entropy_entangle(Ham_const,int_1bd,int_2bd,nele,method):
    assert method == 'FCI'
    #raise TypeError('not debugged yet')
    mol = gto.M(verbose=2)
    n = int_1bd.shape[0]
    mol.nelectron = nele

    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: int_1bd
    mf.get_ovlp = lambda *args: torch.eye(n)
    mf._eri = ao2mo.restore(8, int_2bd, n)
    mf.kernel()
    mol.incore_anyway = True


    # In PySCF, the customized Hamiltonian needs to be created once in mf object.
    # The Hamiltonian will be used everywhere whenever possible.  Here, the model
    # Hamiltonian is passed to CCSD/FCI object via the mf object.
    mycc = fci.FCI(mf,torch.array([[1.,0.],[0.,1.]])).run()
    FCIvec = mycc.ci
    waveFunc = torch.zeros((2,2,2,2))
    waveFunc[0,1,0,1] = FCIvec[0,0]
    waveFunc[0,1,1,0] = FCIvec[0,1]
    waveFunc[1,0,0,1] = FCIvec[1,0]
    waveFunc[1,0,1,0] = FCIvec[1,1]
    waveFunc = torch.reshape(waveFunc,(16,1))
    dm = waveFunc @ waveFunc.T
    dm = torch.reshape(dm,(2,2,2,2,2,2,2,2))
    rdm = torch.trace(dm,axis1=1,axis2=5)
    rdm = torch.trace(rdm,axis1=2,axis2=5)
    rdm = torch.reshape(rdm,(4,4))
    e, v = LA.eig(rdm)
    def entro(x):
        res = []
        for i in x:
            if i == 0.:
                continue
            else:
                res.append(-i*torch.log(i))
        return torch.sum(res)
    S = entro(e)
    #print('total energy:',mycc.e_tot+Ham_const,'; entropy:',S)
    return S ,mycc.e_tot+Ham_const,rdm , FCIvec


def test0():
    start_time = time.time() 
    #S_optimized_basis_constraint_multi_rounds(fock_method='B3LYP',atom='H2.xyz',basis='ccpVDZ')
    #S_optimized_basis_constraint(fock_method='HF',atom='H2.xyz',basis='ccpVDZ')
    #E_optimized_basis_gradient(nbasis=4,method='FCI',atom='H2.xyz',basis='ccpVDZ')
    #S_optimized_basis(atom='H2.xyz',basis='ccpVDZ')
    Q = E_optimized_basis_gradient(nbasis=4,method='FCI',atom='H2.xyz',basis='ccpVDZ')
    Q_list = Q.transpose(0,1).tolist()

    # Write the list into a JSON file
    output_file = "opt_basis.json"
    with open(output_file, "w") as f:
        json.dump(Q_list, f)
    end_time = time.time()  # Capture the end time
    total_time = end_time - start_time  # Calculate the total runtime

    print(f"The total running time of the script was: {total_time:.2f} seconds")

def LambdaQ_bk(Ham_const,int_1bd,int_2bd):
    """
    Calculate the LambdaQ value for a given Hamiltonian.
    
    Parameters:
    Ham_const : torch.Tensor
        The constant part of the Hamiltonian.
    int_1bd : torch.Tensor
        h_{pq}
        The one-body interaction terms of the Hamiltonian.
    int_2bd : torch.Tensor
        g_{pqrs}
        The two-body interaction terms of the Hamiltonian.
        
    Returns: float
        \lambda_T+\lambda_V
        where
            \lambda_T=\sum_{p q}^N\left|h_{p q}+\frac{1}{2} \sum_r^N\left(2 g_{p q r r}-g_{p r r q}\right)\right|
            \lambda_V=\frac{1}{2} \sum_{p q r s}^N\left|g_{p q r s}\right|
    """
    h = int_1bd
    g = int_2bd
    N = h.shape[0]

    # Compute sum_r (2 * g_{p q r r} - g_{p r r q})
    g_pqrr = torch.sum(torch.diagonal(g, dim1=2, dim2=3), dim=-1)  # shape: (N, N)
    g_prrq = torch.sum(torch.diagonal(g.permute(0, 2, 3, 1), dim1=1, dim2=2), dim=-1)  # shape: (N, N)

    T_term = h + 0.5 * (2 * g_pqrr - g_prrq)
    lambda_T = torch.sum(torch.abs(T_term))

    lambda_V = 0.5 * torch.sum(torch.abs(g))

    return (lambda_T + lambda_V).item()


def LambdaQ(Ham_const,int_1bd,int_2bd,backend=torch):
    """
    Calculate the LambdaQ value for a given Hamiltonian.
    
    Parameters:
    Ham_const : torch.Tensor
        E0
        The constant part of the Hamiltonian.
    int_1bd : torch.Tensor
        h_{pq}
        The one-body interaction terms of the Hamiltonian.
    int_2bd : torch.Tensor
        g_{pqrs}
        The two-body interaction terms of the Hamiltonian.
        
    Returns: float
        \lambda_C+\lambda_T+\lambda_V^{\prime}
        where
            \lambda_C=\left|E0+\sum_p^N h_{p p}+\frac{1}{2} \sum_{p r}^N g_{p p r r}-\frac{1}{4} \sum_{p r}^N g_{p r r p}\right|,
            \lambda_T=\sum_{p q}^N\left|h_{p q}+\sum_r^N g_{p q r r}-\frac{1}{2} \sum_r^N g_{p r r q}\right|,
            \lambda_V^{\prime}=\frac{1}{2} \sum_{p>r, s>q}^N\left|g_{p q r s}-g_{p s r q}\right|+\frac{1}{4} \sum_{p q r s}^N\left|g_{p q r s}\right| .
    """

    trace = backend.trace
    einsum = backend.einsum
    abs = backend.abs
    sum = backend.sum

    h = int_1bd
    g = int_2bd
    N = h.shape[0]

    # λ_C = | sum_p h_pp + 1/2 sum_{p,r} g_pprr - 1/4 sum_{p,r} g_prrp |
    diag_h = trace(h)
    g_pprr = einsum('pprr->', g)
    g_prrp = einsum('prrp->', g)  # permute and sum over r → g[p,r,r,p]
    lambda_C = abs(diag_h + 0.5 * g_pprr - 0.25 * g_prrp+Ham_const)

    # λ_T = sum_{p,q} | h_pq + sum_r g_pqrr - 1/2 sum_r g_prrq |
    sum_r_g_pqrr = einsum('pqrr->pq', g)
    sum_r_g_prrq = einsum('prrq->pq', g)  # permute and sum over r → g[p,r,*,q]
    T_term = h + sum_r_g_pqrr - 0.5 * sum_r_g_prrq
    lambda_T = sum(abs(T_term))

    # λ_V' = 1/2 sum_{p>r, s>q} |g_pqrs - g_psrq| + 1/4 sum_{pqrs} |g_pqrs|
    indices = backend.arange(N)
    p, q, r, s = backend.meshgrid(indices, indices, indices, indices, indexing='ij')
    mask = (p > r) & (s > q)
    diff = g[p, q, r, s] - g[p, s, r, q]
    lambda_V_prime_1 = 0.5 * sum(abs(diff[mask]))
    lambda_V_prime_2 = 0.25 * sum(abs(g))
    lambda_V_prime = lambda_V_prime_1 + lambda_V_prime_2
    return lambda_C + lambda_T + lambda_V_prime
    
def LER(Ham_const,int_1bd,int_2bd):
    """
    Calculate the LambdaQ value for a given Hamiltonian.
    
    Parameters:
    Ham_const : torch.Tensor
        The constant part of the Hamiltonian.
    int_1bd : torch.Tensor
        h_{pq}
        The one-body interaction terms of the Hamiltonian.
    int_2bd : torch.Tensor
        g_{pqrs}
        The two-body interaction terms of the Hamiltonian.
        
    Returns: float
        sum_i g_{iiii}
    """
    diag_indices = torch.arange(int_2bd.shape[0])
    return torch.sum(int_2bd[diag_indices, diag_indices, diag_indices, diag_indices])

def lambda_optimized_basis(nbasis=2,method='FCI',log_file='opt_log.txt',init_basis = 'HF',subspace=None,**kargs):
    import numpy as np
    from scipy.optimize import minimize
    from scipy.linalg import qr

    # dimension before folding
    n_bf = norbs(**kargs)
    nele = nelec(**kargs)
    assert nele %2 == 0
    # initialize initial guess
    ham0,overlap_mh, basis_orth_init = basis_downfolding_init(fock_method=init_basis,**kargs)

    def ler(basis_orth,method):
        t0 = time.time() 
        t1 = time.time()  # Capture the end time
        t2 = time.time()  # Capture the end time
        basis_orth, _R = torch.linalg.qr(basis_orth,mode='reduced')    # orthorgonalize basis_orth
        t3 = time.time()  # Capture the end time
        ham = basis_downfolding(ham0,overlap_mh, basis_orth, n_folded=nbasis)
        t4 = time.time()  # Capture the end time
        res = LER(ham.Ham_const,ham.int_1bd,ham.int_2bd)
        #E, properties = Solve_fermionHam(ham.Ham_const,ham.int_1bd,ham.int_2bd,nele=sum([nele//2,nele//2]),method=method)
        t5 = time.time()  # Capture the end time
        return res
    
    def l_1(basis_orth,method):
        t0 = time.time() 
        t1 = time.time()  # Capture the end time
        t2 = time.time()  # Capture the end time
        basis_orth, _R = torch.linalg.qr(basis_orth,mode='reduced')    # orthorgonalize basis_orth
        t3 = time.time()  # Capture the end time
        ham = basis_downfolding(ham0,overlap_mh, basis_orth, n_folded=nbasis)
        t4 = time.time()  # Capture the end time
        res = LambdaQ(ham.Ham_const,ham.int_1bd,ham.int_2bd)
        #E, properties = Solve_fermionHam(ham.Ham_const,ham.int_1bd,ham.int_2bd,nele=sum([nele//2,nele//2]),method=method)
        t5 = time.time()  # Capture the end time
        return res
    
    # Initial guess (needs to be orthogonal)
    Qs = torch.eye(nbasis,device=device,dtype=torch.float64).requires_grad_(True)
    if subspace == None:
        Q0 = basis_orth_init.clone().detach()[:,:nbasis]
    else:
        with open(subspace, 'r') as f:
            basis_orth_init = json.load(f)
        Q0 = torch.tensor(basis_orth_init,device=device,dtype=torch.float64).transpose(0,1)[:,:nbasis]

    # Initialize the Adam optimizer
    optimizer = torch.optim.Adam([Qs], lr=0.01)  # Learning rate is 0.1
    prev_loss = None
    # Optimization loop
    f = open(log_file, 'w')
    for step in range(1000):
        Q = Q0 @ Qs
        optimizer.zero_grad()  # Reset the gradients to zero

        loss = l_1(Q,method)  # Compute the current loss
        l_1_val = l_1(Q,method).item()

        loss.backward()        # Perform backpropagation to compute the gradients
        optimizer.step()       # Update parameters using the computed gradients
        if step % 10 == 0:
            f.write(f"Step {step+1}, Loss: {loss.item()}  ")
            f.write(f"gradient: {abs(Qs.grad).sum()}  ")
            f.write(f"l-1: {l_1_val}\n")
            f.flush()
        # Check if the change in loss is smaller than the threshold
        
        if abs(Qs.grad).sum() < 1e-3:
            f.write(f"convergent at epoch {step+1}; gradient {abs(Qs.grad).sum()} is below threshold")
            break
        prev_loss = loss.item()
    else:
        f.write(f"max iteration achieved")
    f.close()
    return torch.linalg.qr(Q.detach(),mode='reduced')[0]


if __name__=='__main__':
    start_time = time.time() 
    atoms = '''
    H 0.0 0.0 0.0
    H 0.0 0.0 1.5
    H 0.0 0.0 3.0
    H 0.0 0.0 4.5
    '''
    ba = 'ccpVDZ'
    Q = E_optimized_basis_gradient_pytorch(nbasis=4,atom=atoms,basis=ba)
    end_time = time.time()  # Capture the end time
    total_time = end_time - start_time  # Calculate the total runtime
    print(f"The total running time of the script was: {total_time:.2f} seconds")
    '''
    Q_list = Q.transpose(0,1).tolist()

    # Write the list into a JSON file
    output_file = "opt_basis.json"
    with open(output_file, "w") as f:
        json.dump(Q_list, f)

    '''