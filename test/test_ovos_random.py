"""
Test OVOS optimization with random unitary initialization on molecules.

Tests are done on molecules:
- CO
- H2O
- HF
- NH3

Random unitary rotations are applied to the virtual orbital space, 
and the best result is selected from multiple attempts.
"""

import numpy as np
from ovos import OVOS
from pyscf import gto, scf


def test_ovos_random_CO():
    # Carbon monoxide molecule
    mol = gto.Mole()
    mol.atom = 'C 0 0 0; O 0 0 1.128'
    mol.basis = '6-31G'
    mol.unit = 'Angstrom'
    mol.spin = 0
    mol.charge = 0
    mol.symmetry = False
    mol.verbose = 0
    mol.build()

    # UHF reference
    mf = scf.UHF(mol)
    mf.verbose = 0
    mf.kernel()

    # Run OVOS with random unitary initialization
    _run_ovos_random(mol, mf, num_opt_virtual_orbs=2, attempts=50)


def test_ovos_random_H2O():
    # Water molecule
    mol = gto.Mole()
    mol.atom = 'O 0.0000 0.0000  0.1173; H 0.0000    0.7572  -0.4692; H 0.0000   -0.7572 -0.4692'
    mol.basis = '6-31G'
    mol.unit = 'Angstrom'
    mol.spin = 0
    mol.charge = 0
    mol.symmetry = False
    mol.verbose = 0
    mol.build()

    # UHF reference
    mf = scf.UHF(mol)
    mf.verbose = 0
    mf.kernel()

    # Run OVOS with random unitary initialization
    _run_ovos_random(mol, mf, num_opt_virtual_orbs=2, attempts=50)


def test_ovos_random_HF():
    # Hydrogen fluoride molecule
    mol = gto.Mole()
    mol.atom = 'H 0 0 0; F 0 0 0.917'
    mol.basis = '6-31G'
    mol.unit = 'Angstrom'
    mol.spin = 0
    mol.charge = 0
    mol.symmetry = False
    mol.verbose = 0
    mol.build()

    # UHF reference
    mf = scf.UHF(mol)
    mf.verbose = 0
    mf.kernel()

    # Run OVOS with random unitary initialization
    _run_ovos_random(mol, mf, num_opt_virtual_orbs=2, attempts=50)


def test_ovos_random_NH3():
    # Ammonia molecule
    mol = gto.Mole()
    mol.atom = 'N 0 0 0; H 0 0 1.012; H 0 0.935 -0.262; H 0 -0.935 -0.262'
    mol.basis = '6-31G'
    mol.unit = 'Angstrom'
    mol.spin = 0
    mol.charge = 0
    mol.symmetry = False
    mol.verbose = 0
    mol.build()

    # UHF reference
    mf = scf.UHF(mol)
    mf.verbose = 0
    mf.kernel()

    # Run OVOS with random unitary initialization
    _run_ovos_random(mol, mf, num_opt_virtual_orbs=2, attempts=50)


def _run_ovos_random(mol, mf, num_opt_virtual_orbs=2, attempts=50):
    """
    Helper function to run OVOS with random unitary initialization.
    
    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule object
    mf : pyscf.scf.UHF
        UHF reference calculation
    num_opt_virtual_orbs : int
        Number of virtual orbitals to optimize (in spin-orbitals)
    attempts : int
        Number of random unitary attempts to try
    """
    
    # Get UHF orbitals and Fock matrix
    Fao = [mf.get_fock()[0], mf.get_fock()[1]]
    mo_coeffs_rhf = [mf.mo_coeff[0], mf.mo_coeff[1]]
    
    # Precompute integrals to avoid recomputation
    eri_4fold_ao = mol.intor('int2e_sph', aosym=1)
    S = mol.intor('int1e_ovlp')
    hcore_ao = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    
    # Get number of electrons and occupied orbitals
    num_electrons = mol.nelectron
    num_occupied_orbitals = num_electrons // 2
    total_spatial_orbitals = mo_coeffs_rhf[0].shape[1]
    num_virtual_orbitals = total_spatial_orbitals - num_occupied_orbitals
    
    best_E_corr = None
    best_mo_coeffs = None
    best_attempt = 0
    
    print(f"\n{'='*70}")
    print(f"Running OVOS with random unitary initialization")
    print(f"Molecule: {mol.atom}")
    print(f"Basis: {mol.basis}")
    print(f"Attempts: {attempts}")
    print(f"{'='*70}\n")
    
    for attempt in range(1, attempts + 1):
        # Generate random unitary matrix for virtual orbitals
        rand_matrix = np.random.rand(num_virtual_orbitals, num_virtual_orbitals)
        Q, R = np.linalg.qr(rand_matrix)  # QR decomposition to get unitary matrix
        
        # Apply random unitary rotation to virtual orbitals
        mo_coeffs = [np.copy(mo_coeffs_rhf[0]), np.copy(mo_coeffs_rhf[1])]
        for spin in [0, 1]:
            C_occ = mo_coeffs_rhf[spin][:, :num_occupied_orbitals]
            C_virt = mo_coeffs_rhf[spin][:, num_occupied_orbitals:]
            C_virt_rot = C_virt @ Q
            mo_coeffs[spin] = np.hstack((C_occ, C_virt_rot))
        
        mo_coeffs = np.array(mo_coeffs)
        
        # Run OVOS
        ovos = OVOS(
            mol=mol,
            scf=mf,
            Fao=Fao,
            num_opt_virtual_orbs=num_opt_virtual_orbs,
            mo_coeff=mo_coeffs,
            init_orbs="UHF",
            verbose=0,
            max_iter=1000,
            conv_energy=1e-8,
            conv_grad=1e-6
        )
        
        # Pass precomputed integrals to avoid recomputation
        ovos.eri_4fold_ao = eri_4fold_ao
        ovos.S = S
        ovos.hcore_ao = hcore_ao
        
        # Run optimization
        E_corr, E_corr_hist, E_corr_iter, E_corr_mo, E_corr_fock, stop_reason = ovos.run(mo_coeffs, fock_spin=None)
        
        # Track best result
        if best_E_corr is None or E_corr < best_E_corr:
            best_E_corr = E_corr
            best_mo_coeffs = E_corr_mo
            best_attempt = attempt
            print(f"Attempt {attempt:3d}/{attempts}: E_corr = {E_corr:.10f} Hartree *** NEW BEST ***")
        else:
            print(f"Attempt {attempt:3d}/{attempts}: E_corr = {E_corr:.10f} Hartree")
    
    print(f"\n{'='*70}")
    print(f"Best result from attempt {best_attempt}: E_corr = {best_E_corr:.10f} Hartree")
    print(f"{'='*70}\n")
