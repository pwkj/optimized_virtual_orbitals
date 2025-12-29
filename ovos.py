"""
OVOS class
"""

from typing import Tuple, List

import numpy as np
import scipy
import pyscf
from pyscf.cc.addons import spatial2spin



class OVOS:

	"""
	The OVOS algorithm minimizes the second-order correlation energy (MP2) using orbital rotations. 

	Implemenation is based on:
	https://pubs.aip.org/aip/jcp/article/86/11/6314/93345/Optimized-virtual-orbital-space-for-high-level

    Parameters
    ----------
    mol : pyscf.M
        PySCF molecule object.
    num_opt_virtual_orbs : int
        Number of optimized virtual spin orbitals.
    """

	def __init__(self, mol: pyscf.gto.Mole, num_opt_virtual_orbs: int) -> None:
		self.mol = mol
		self.num_opt_virtual_orbs = num_opt_virtual_orbs

		# Set up unrestricted Hartree-Fock calculation
		self.uhf = pyscf.scf.UHF(mol).run()
		self.e_rhf = self.uhf.e_tot
		self.h_nuc = mol.energy_nuc()

		# MO coefficients 
		self.mo_coeffs = self.uhf.mo_coeff

		# Fock matrix
		Fao = self.uhf.get_fock()
		Fmo_a = self.mo_coeffs[0].T @ Fao[0] @ self.mo_coeffs[0]
		Fmo_b = self.mo_coeffs[1].T @ Fao[1] @ self.mo_coeffs[1]
		Fmo = (Fmo_a, Fmo_b)
		eigval_a, eigvec_a = scipy.linalg.eig(Fmo_a)
		eigval_b, eigvec_b = scipy.linalg.eig(Fmo_b)
		sorting_a = np.argsort(eigval_a)
		sorting_b = np.argsort(eigval_b)
		mo_energy_a = np.real(eigval_a[sorting_a])
		mo_energy_b = np.real(eigval_b[sorting_b])
		self.orbital_energies = []
		for i in range(eigval_a.shape[0]):
			self.orbital_energies.append(float(mo_energy_a[i]))
			self.orbital_energies.append(float(mo_energy_b[i]))
		
		# Integrals in AO basis
		self.hcore_ao = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
		self.overlap = mol.intor('int1e_ovlp')
		self.eri_4fold_ao = mol.intor('int2e_sph', aosym=1)

		# Number of orbitals
		self.tot_num_spin_orbs = int(2*self.mo_coeffs.shape[1])
		
		# Number of electrons
		self.nelec = self.mol.nelec[0] + self.mol.nelec[1]

		# Build index lists of active and inactive spaces
		#I,J indices -> occupied spin orbitals
		self.active_occ_indices = [i for i in range(int(self.nelec))]
		#A, B indices -> inoccupied spin orbitals in active space
		self.active_inocc_indices = [i for i in range(self.active_occ_indices[-1]+1,int((self.num_opt_virtual_orbs+self.nelec)))]
		#actice + inactive space
		self.active_inactive_indices = [i for i in range(self.active_occ_indices[-1]+1,int((self.tot_num_spin_orbs)))]
		#print(self.active_inactive_indices)
		#print(self.active_inocc_indices)
		#print(int(self.num_opt_virtual_orbs+self.nelec))

		print("Total number of spin orbitals: ", self.tot_num_spin_orbs)
		print("Active space size: ", self.num_opt_virtual_orbs + self.nelec)
		print("Number of optimized spin orbitals: ", self.num_opt_virtual_orbs)
		print("Number of occupied spin orbitals: ", self.nelec)

		assert self.tot_num_spin_orbs >= self.num_opt_virtual_orbs+self.nelec, "Your space num_opt_virtual_orbs is too large"  
	
	def Fock_matrix(self, U) -> Tuple[np.ndarray, np.ndarray]:


		# Fmo  = mo_coeffs.T @ self.F_matrix @ mo_coeffs
		# eigval, eigvec = scipy.linalg.eig(Fmo)
		# sorting = np.argsort(eigval)
		# eigval = np.real(eigval[sorting])
		# eigvec = np.real(eigvec[:, sorting])

		return NotImplementedError 


	def MP2_energy(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
     
		"""
		MP2 energy for unrestricted orbitals 
		"""
		
		norb_alpha = self.mo_coeffs[0].shape[1]
		norb_beta = self.mo_coeffs[1].shape[1]

		# The ao2mo.kernel function can handle the four-index transformation for different orbital sets.
		# Note: PySCF stores 2e integrals in chemists' notation: (ij|kl) = <ik|jl> in physicists' notation.

		# --- (alpha alpha | alpha alpha) integrals ---
		eri_aaaa = pyscf.ao2mo.kernel(self.eri_4fold_ao, [self.mo_coeffs[0], self.mo_coeffs[0], self.mo_coeffs[0], self.mo_coeffs[0]], compact=False)
		#eri_aaaa = eri_aaaa.reshape(norb_alpha, norb_alpha, norb_alpha, norb_alpha)

		# --- (beta beta | beta beta) integrals ---
		eri_bbbb = pyscf.ao2mo.kernel(self.eri_4fold_ao, [self.mo_coeffs[1], self.mo_coeffs[1], self.mo_coeffs[1], self.mo_coeffs[1]], compact=False)
		#eri_bbbb = eri_bbbb.reshape(norb_beta, norb_beta, norb_beta, norb_beta)

		# --- (alpha alpha | beta beta) integrals ---
		# These are the (ij|kl) where i,j are alpha, k,l are beta
		eri_aabb = pyscf.ao2mo.kernel(self.eri_4fold_ao, [self.mo_coeffs[0], self.mo_coeffs[0], self.mo_coeffs[1], self.mo_coeffs[1]], compact=False)
		#eri_aabb = eri_aabb.reshape(norb_alpha, norb_alpha, norb_beta, norb_beta)

		# --- (beta beta | alpha alpha) integrals ---
		eri_bbaa = pyscf.ao2mo.kernel(self.eri_4fold_ao, [self.mo_coeffs[1], self.mo_coeffs[1], self.mo_coeffs[0], self.mo_coeffs[0]], compact=False)
		#eri_bbaa = eri_bbaa.reshape(norb_beta, norb_beta, norb_alpha, norb_alpha)

		norb_total = norb_alpha + norb_beta
		#eri_spin = np.zeros((norb_total, norb_total, norb_total, norb_total))

		#https://pyscf.org/_modules/pyscf/cc/addons.html#spatial2spin
		eri_spin = spatial2spin([eri_aaaa, eri_aabb, eri_bbbb], orbspin=None)

		MP1_amplitudes = np.zeros((norb_total, norb_total, norb_total, norb_total))

		E_corr = 0
		for I in self.active_occ_indices:
			for J in self.active_occ_indices:
				if I > J:
					for A in self.active_inocc_indices:
						for B in self.active_inocc_indices:
							if A > B:
								#MP2 correlation energy for restricted orbitals: 
								E_corr += -1.0*((eri_spin[A,I,B,J] - eri_spin[A,J,B,I])**2 
									/ (self.orbital_energies[A] + self.orbital_energies[B] - self.orbital_energies[I] - self.orbital_energies[J]) )

								#MP1 amplitudes:
								t1 =  -1.0*( (eri_spin[A,I,B,J] - eri_spin[A,J,B,I]) / (self.orbital_energies[A] + self.orbital_energies[B] - self.orbital_energies[I] - self.orbital_energies[J]) )
								MP1_amplitudes[A,I,B,J] = t1
		

		MP2 = self.uhf.MP2().run()
		assert np.abs(E_corr - MP2.e_corr) < 1e-6, "np.abs(E_corr - self.rhf.MP2().run().e_corr) < 1e-6"  
		return E_corr, MP1_amplitudes

	
	def orbital_optimization(self):

		"""
		First- and second-order derivatives of the second-order Hylleraas functional
		Equations 11a and 11b in https://pubs.aip.org/aip/jcp/article/86/11/6314/93345/Optimized-virtual-orbital-space-for-high-level
		"""

		E_corr, MP1_amplitudes = self.MP2_energy()

		norb_alpha = self.mo_coeffs[0].shape[1]
		norb_beta = self.mo_coeffs[1].shape[1]
		
		norb_total = norb_alpha + norb_beta

		#Gradient
		G = np.zeros((norb_total, norb_total))
		def gradient(E: int, A: int) -> float:
			for I in self.active_occ_indices:
				for J in self.active_occ_indices:
					if I > J:
						for B in self.active_inactive_indices:
							if E > B:
								pass
								




		return NotImplementedError



	
	def run_ovos(self,  mo_coeffs) -> float:

		# Two-electron integrals in MO basis
		eri_4fold_mo = pyscf.ao2mo.incore.full(self.eri_4fold_ao, mo_coeffs)

		J = 0

		raise NotImplementedError
		



# Molecule
atom = "Li .0 .0 .0; H .0 .0 1.595" 
#atom = "H .0 .0 .0; H .0 .0 0.74144"
#atom = """O 0.0000 0.0000  0.1173; H 0.0000    0.7572  -0.4692; H 0.0000   -0.7572 -0.4692;""" #Angstrom
basis = "STO-3G"
#basis = "6-31G"
unit="angstrom"
mol = pyscf.M(atom=atom, basis=basis, unit=unit)


rhf = pyscf.scf.RHF(mol).run()
mo_coeff = rhf.mo_coeff 

run_OVOS = OVOS(mol=mol, num_opt_virtual_orbs=8)
#run_OVOS.MP2_energy()
run_OVOS.orbital_optimization()






