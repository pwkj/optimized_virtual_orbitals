"""
OVOS class

The OVOS algorithm minimizes the second-order correlation energy (MP2)
using orbital rotations.

Implementation based on:
https://pubs.aip.org/aip/jcp/article/86/11/6314/93345
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

	def __init__(self, mol: pyscf.gto.Mole, num_opt_virtual_orbs: int, init_orbs = "UHF") -> None:
		self.mol = mol
		self.num_opt_virtual_orbs = num_opt_virtual_orbs
		self.init_orbs = init_orbs

		# Set up unrestricted Hartree-Fock calculation 
		self.uhf = pyscf.scf.UHF(mol).run()
		self.e_rhf = self.uhf.e_tot
		self.h_nuc = mol.energy_nuc()

		if self.init_orbs == "UHF":
			# MO coefficients (alpha, beta)
			self.mo_coeffs = self.uhf.mo_coeff

		# Fock matrix in AO basis 
		self.Fao = self.uhf.get_fock()
		
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
		self.inactive_indices = [i for i in range(self.active_inocc_indices[-1]+1,int((self.tot_num_spin_orbs)))]
		#print(self.active_inactive_indices)
		#print(self.active_inocc_indices)
		#print(int(self.num_opt_virtual_orbs+self.nelec))

		print("#### Active and inactive spaces ####")
		print("Total number of spin orbitals: ", self.tot_num_spin_orbs)
		print("Active occupied spin orbitals: ", self.active_occ_indices)
		print("Active unoccupied spin orbitals: ", self.active_inocc_indices)
		print("Inactive unoccupied spin orbitals: ", self.inactive_indices)

		# Also print number of orbital coefficients, R_EA

		assert self.tot_num_spin_orbs >= self.num_opt_virtual_orbs+self.nelec, "Your space 'num_opt_virtual_orbs' is too large"  


	def MP2_energy(self, mo_coeffs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
     
		"""
		MP2 energy for unrestricted orbitals 
		"""

		norb_alpha = mo_coeffs[0].shape[1]
		norb_beta = mo_coeffs[1].shape[1]

		# Fock matrix in MO basis 
		Fmo_a = mo_coeffs[0].T @ self.Fao[0] @ mo_coeffs[0]
		Fmo_b = mo_coeffs[1].T @ self.Fao[1] @ mo_coeffs[1]
		Fmo = (Fmo_a, Fmo_b)
		
		# Orbital energies (spin-orbital representation)
		eigval_a, eigvec_a = scipy.linalg.eig(Fmo_a)
		eigval_b, eigvec_b = scipy.linalg.eig(Fmo_b)
		sorting_a = np.argsort(eigval_a)
		sorting_b = np.argsort(eigval_b)
		mo_energy_a = np.real(eigval_a[sorting_a])
		mo_energy_b = np.real(eigval_b[sorting_b])
		orbital_energies = []
		for i in range(eigval_a.shape[0]):
			orbital_energies.append(float(mo_energy_a[i]))
			orbital_energies.append(float(mo_energy_b[i]))


		#PySCF stores 2e integrals in chemists' notation: (ij|kl) = <ik|jl> in physicists' notation.

		# (alpha alpha | alpha alpha) integrals
		eri_aaaa = pyscf.ao2mo.kernel(self.eri_4fold_ao, [mo_coeffs[0], mo_coeffs[0], mo_coeffs[0], mo_coeffs[0]], compact=False)
		#eri_aaaa = eri_aaaa.reshape(norb_alpha, norb_alpha, norb_alpha, norb_alpha)

		# (beta beta | beta beta) integrals
		eri_bbbb = pyscf.ao2mo.kernel(self.eri_4fold_ao, [mo_coeffs[1], mo_coeffs[1], mo_coeffs[1], mo_coeffs[1]], compact=False)
		#eri_bbbb = eri_bbbb.reshape(norb_beta, norb_beta, norb_beta, norb_beta)

		# (alpha alpha | beta beta) integrals
		# These are the (ij|kl) where i,j are alpha, k,l are beta
		eri_aabb = pyscf.ao2mo.kernel(self.eri_4fold_ao, [mo_coeffs[0], mo_coeffs[0], mo_coeffs[1], mo_coeffs[1]], compact=False)
		#eri_aabb = eri_aabb.reshape(norb_alpha, norb_alpha, norb_beta, norb_beta)

		# (beta beta | alpha alpha) integrals
		eri_bbaa = pyscf.ao2mo.kernel(self.eri_4fold_ao, [mo_coeffs[1], mo_coeffs[1], mo_coeffs[0], mo_coeffs[0]], compact=False)
		#eri_bbaa = eri_bbaa.reshape(norb_beta, norb_beta, norb_alpha, norb_alpha)

		norb_total = norb_alpha + norb_beta
		#eri_spin = np.zeros((norb_total, norb_total, norb_total, norb_total))

		#https://pyscf.org/_modules/pyscf/cc/addons.html#spatial2spin
		eri_spin = spatial2spin([eri_aaaa, eri_aabb, eri_bbbb], orbspin=None)
		Fmo_spin = spatial2spin([Fmo[0], Fmo[1]], orbspin=None)

		MP1_amplitudes = np.zeros((norb_total, norb_total, norb_total, norb_total))

		# E_corr = 0
		# for I in self.active_occ_indices:
		# 	for J in self.active_occ_indices:
		# 		if I > J:
		# 			for A in self.active_inocc_indices:
		# 				for B in self.active_inocc_indices:
		# 					if A > B:
		# 						#MP2 correlation energy for restricted orbitals: 
		# 						E_corr += -1.0*((eri_spin[A,I,B,J] - eri_spin[A,J,B,I])**2 
		# 							/ (orbital_energies[A] + orbital_energies[B] - orbital_energies[I] - orbital_energies[J]) )

		# 						#MP1 amplitudes:
		# 						t1 =  -1.0*( (eri_spin[A,I,B,J] - eri_spin[A,J,B,I]) / (orbital_energies[A] + orbital_energies[B] - orbital_energies[I] - orbital_energies[J]) )
		# 						MP1_amplitudes[A,I,B,J] = t1
		
		
	
		def t1(I,J,A,B):
			#MP1 amplitudes:
			t1 = -1.0*( (eri_spin[A,I,B,J] - eri_spin[A,J,B,I]) 
			/ (orbital_energies[A] + orbital_energies[B] 
			- orbital_energies[I] - orbital_energies[J]) )
			return t1


		J_2 = 0
		for I in self.active_occ_indices:
			for J in self.active_occ_indices:
				if I > J:
					first_term = 0
					for A in self.active_inocc_indices:
						for B in self.active_inocc_indices:
							if A > B:
								for C in self.active_inocc_indices:
									for D in self.active_inocc_indices:
										if C > D:
								
											t_abij = t1(I=I,J=J,A=A,B=B)
											t_cdij = t1(I=I,J=J,A=C,B=D)

											if B==D:
												first_term += t_abij*t_cdij*Fmo_spin[A,C]
											if A==C:
												first_term += t_abij*t_cdij*Fmo_spin[B,D]
											if B==C:
												first_term += -1.0*t_abij*t_cdij*Fmo_spin[A,D]
											if A==D:
												first_term += -1.0*t_abij*t_cdij*Fmo_spin[B,C]
											if A==C and B==D:
												first_term += -1.0*t_abij*t_cdij*(orbital_energies[I] + orbital_energies[J])
											if A==D and B==C:
												first_term += t_abij*t_cdij*(orbital_energies[I] + orbital_energies[J])

					second_term = 0
					for A in self.active_inocc_indices:
						for B in self.active_inocc_indices:
							if A > B:
								second_term += 2*t1(I=I,J=J,A=A,B=B)*(eri_spin[A,I,B,J] - eri_spin[A,J,B,I])
								MP1_amplitudes[A,I,B,J] = t1(I=I,J=J,A=A,B=B)
					
					J_2 += first_term+second_term

		#MP2 = self.uhf.MP2().run()
		#assert np.abs(J_2 - MP2.e_corr) < 1e-6, "|J_2 - E_corr_MP2| < 1e-6 !!!"  
		return J_2, MP1_amplitudes, eri_spin, Fmo_spin

	
	def orbital_optimization(self, mo_coeffs):

		"""
		First- and second-order derivatives of the second-order Hylleraas functional
		Equations 11a and 11b in https://pubs.aip.org/aip/jcp/article/86/11/6314/93345/Optimized-virtual-orbital-space-for-high-level
		"""

		E_corr, MP1_amplitudes, eri_spin, Fmo_spin = self.MP2_energy(mo_coeffs = mo_coeffs)

		norb_alpha = mo_coeffs[0].shape[1]
		norb_beta = mo_coeffs[1].shape[1]

		def second_order_density_matrix(A:int, B:int) -> float:
			#Equation 13
			D = 0
			for I in self.active_occ_indices:
				for J in self.active_occ_indices:
					if I > J:
						for C in self.active_inocc_indices:
							D += MP1_amplitudes[A,I,C,J]*MP1_amplitudes[B,I,C,J]
			return D


		def gradient(E: int, A: int) -> float:
			#Equation 12a
			first_term = 0
			for I in self.active_occ_indices:
				for J in self.active_occ_indices:
					if I > J:
						for B in self.active_inocc_indices:
							first_term += 2.0*MP1_amplitudes[A,I,B,J]*(eri_spin[E,I,B,J] - eri_spin[E,J,B,I])

			second_term = 0
			for B in self.active_inocc_indices:
				second_term += 2.0*second_order_density_matrix(A=A, B=B)*Fmo_spin[E,B]

			G_EA = first_term + second_term

			return G_EA

		
		idx = 0
		G = np.zeros((len(self.active_inocc_indices)*len(self.inactive_indices)))
		for idx2, E in enumerate(self.inactive_indices):
			for idx1, A in enumerate(self.active_inocc_indices):
				idx += 1
				print(idx, ":", idx2, idx1)
				#print()
				#print(gradient(E,A))
				G[idx-1] = gradient(E,A)

		
		


		

				

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


uhf = pyscf.scf.UHF(mol).run()
mo_coeff = uhf.mo_coeff 

run_OVOS = OVOS(mol=mol, num_opt_virtual_orbs=6)
#run_OVOS.MP2_energy(mo_coeff)
run_OVOS.orbital_optimization(mo_coeff)






