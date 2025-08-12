# README

## Overview

This project implements a Variational Quantum Eigensolver (VQE) framework using PennyLane and PySCF for quantum chemistry simulations. It includes utilities for converting between quantum states and configuration interaction (CI) representations, as well as a driver for running VQE and VQE-CASSCF calculations.

## Project Structure

- **`SCAN_V7_07.ipynb`**(to be simplified): currently contains the main implementation of the VQE algorithm using PennyLane. It includes functions for:

  - Converting between CI and qubit states.
  - Constructing the qubit Hamiltonian from molecular integrals.
  - Defining the UCCSD ansatz for VQE.
  - TBD: move functions to utils.py and vqe_driver.py
- **`checker.py`**:
  A script for performing sanity checks to ensure the correctness of the implemented code.
- **`utils.py`**:
  Contains helper functions such as `parity_from_occ` for computing permutation parity and other utility functions used across the project.
- **`vqe_driver.py`**:
  Provides a wrapper for running VQE and VQE-CASSCF calculations. It integrates the core VQE implementation with additional features for advanced quantum chemistry workflows.

## Requirements

- Python 3.8+
- Required libraries:
  - `pennylane`
  - `numpy`
  - `openfermion`
  - `pyscf`
  - `matplotlib`
  - `scipy`

Install dependencies using:

```bash
conda env create -f environment.yml
```

## Tests

* code for figure 2:```Figure_2.py```
* code for figure 3:```Figure_3.py```
* code for figure 4:```Figure_4.py```
* code for figure 5:```Figure_5.py```
