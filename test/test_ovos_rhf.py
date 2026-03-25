"""
Test OVOS optimization on a simple molecules using RHF orbitals as the initial guess.

Test are done on molecules:
- CO
- H2O
- HF
- NH3
"""

# Import OVOS
from ovos import OVOS
from pyscf import gto, scf

def test_ovos_rhf_CO():
    # Carbon monoxide molecule, minimal basis
    mol = gto.Mole()
    mol.atom = 'C 0 0 0; O 0 0 1.128'
    mol.basis = '6-31G'
    mol.unit = 'Angstrom'
    mol.spin = 0
    mol.charge = 0
    mol.symmetry = False
    mol.verbose = 0
    mol.build()

    # RHF reference
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()

    # Initial data (RHF orbitals)
    Fao = [mf.get_fock(), mf.get_fock()]
    mo_coeffs = [mf.mo_coeff, mf.mo_coeff]

    # Create OVOS object and run
    ovos = OVOS(
        mol=mol,
        scf=mf,
        Fao=Fao,
        num_opt_virtual_orbs=2,      # active virtual spin‑orbitals
        mo_coeff=mo_coeffs,
        init_orbs="RHF",
        verbose=1,
        max_iter=1000,
        conv_energy=1e-8,
        conv_grad=1e-6,
        keep_track_max=50
    )
    E_corr, E_corr_hist, E_corr_iter, E_corr_mo, E_corr_fock, stop_reason = ovos.run(mo_coeffs, fock_spin=None)

    print("\nOptimization finished. Final MP2 energy =", E_corr)


def test_ovos_rhf_H2O():
    # Water molecule, minimal basis
    mol = gto.Mole()
    mol.atom = 'O 0.0000 0.0000  0.1173; H 0.0000    0.7572  -0.4692; H 0.0000   -0.7572 -0.4692'
    mol.basis = '6-31G'
    mol.unit = 'Angstrom'
    mol.spin = 0
    mol.charge = 0
    mol.symmetry = False
    mol.verbose = 0
    mol.build()

    # RHF reference
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()

    # Initial data (RHF orbitals)
    Fao = [mf.get_fock(), mf.get_fock()]
    mo_coeffs = [mf.mo_coeff, mf.mo_coeff]

    # Create OVOS object and run
    ovos = OVOS(
        mol=mol,
        scf=mf,
        Fao=Fao,
        num_opt_virtual_orbs=2,      # active virtual spin‑orbitals
        mo_coeff=mo_coeffs,
        init_orbs="RHF",
        verbose=1,
        max_iter=1000,
        conv_energy=1e-8,
        conv_grad=1e-6,
        keep_track_max=50
    )
    E_corr, E_corr_hist, E_corr_iter, E_corr_mo, E_corr_fock, stop_reason = ovos.run(mo_coeffs, fock_spin=None)

    print("\nOptimization finished. Final MP2 energy =", E_corr)
    
def test_ovos_rhf_HF():
    # Hydrogen fluoride molecule, minimal basis
    mol = gto.Mole()
    mol.atom = 'H 0 0 0; F 0 0 0.917'
    mol.basis = '6-31G'
    mol.unit = 'Angstrom'
    mol.spin = 0
    mol.charge = 0
    mol.symmetry = False
    mol.verbose = 0
    mol.build()

    # RHF reference
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()

    # Initial data (RHF orbitals)
    Fao = [mf.get_fock(), mf.get_fock()]
    mo_coeffs = [mf.mo_coeff, mf.mo_coeff]

    # Create OVOS object and run
    ovos = OVOS(
        mol=mol,
        scf=mf,
        Fao=Fao,
        num_opt_virtual_orbs=2,      # active virtual spin‑orbitals
        mo_coeff=mo_coeffs,
        init_orbs="RHF",
        verbose=1,
        max_iter=1000,
        conv_energy=1e-8,
        conv_grad=1e-6,
        keep_track_max=50
    )
    E_corr, E_corr_hist, E_corr_iter, E_corr_mo, E_corr_fock, stop_reason = ovos.run(mo_coeffs, fock_spin=None)

    print("\nOptimization finished. Final MP2 energy =", E_corr)

def test_ovos_rhf_NH3():
    # Ammonia molecule, minimal basis
    mol = gto.Mole()
    mol.atom = 'N 0 0 0; H 0 0 1.012; H 0 0.935 -0.262; H 0 -0.935 -0.262'
    mol.basis = '6-31G'
    mol.unit = 'Angstrom'
    mol.spin = 0
    mol.charge = 0
    mol.symmetry = False
    mol.verbose = 0
    mol.build()

    # RHF reference
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()

    # Initial data (RHF orbitals)
    Fao = [mf.get_fock(), mf.get_fock()]
    mo_coeffs = [mf.mo_coeff, mf.mo_coeff]

    # Create OVOS object and run
    ovos = OVOS(
        mol=mol,
        scf=mf,
        Fao=Fao,
        num_opt_virtual_orbs=2,      # active virtual spin‑orbitals
        mo_coeff=mo_coeffs,
        init_orbs="RHF",
        verbose=1,
        max_iter=1000,
        conv_energy=1e-8,
        conv_grad=1e-6,
        keep_track_max=50
    )
    E_corr, E_corr_hist, E_corr_iter, E_corr_mo, E_corr_fock, stop_reason = ovos.run(mo_coeffs, fock_spin=None)

    print("\nOptimization finished. Final MP2 energy =", E_corr)