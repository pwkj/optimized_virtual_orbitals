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

		# Set up unrestricted Hartree-Fock  calculation
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
		self.mo_energy_a = np.real(eigval_a[sorting_a])
		self.mo_energy_b = np.real(eigval_b[sorting_b])
		#print(eigval_a)
		#print(eigval_b)
		#print(self.uhf.mo_energy)

		# Integrals in AO basis
		self.hcore_ao = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
		self.overlap = mol.intor('int1e_ovlp')
		self.eri_4fold_ao = mol.intor('int2e_sph', aosym=1)

		# Number of orbitals
		self.tot_num_spin_orbs = int(2*self.mo_coeffs.shape[1])
		
		# Number of electrons
		self.nelec = self.mol.nelec[0] + self.mol.nelec[1]

		# Build index lists of active and inactive spaces
		#i,j indices -> occupied orbitals
		self.active_occ_indices = [i for i in range(int(self.nelec/2))]
		#a, b indices -> inoccupied orbitals in active space
		self.active_inocc_indices = [i for i in range(self.active_occ_indices[-1]+1,int((self.num_opt_virtual_orbs+self.nelec)/2))]
		#print(self.active_occ_indices)
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


	def MP2_energy(self, E_rhf) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
     
		"""
		MP2 energy for unrestricted orbitals 
		"""
		
		# # Transform AO integrals to MO integrals
		# eri_mo_aa = pyscf.ao2mo.incore.full(self.eri_4fold_ao, self.mo_coeffs[0], compact=False)
		# eri_mo_bb = pyscf.ao2mo.incore.full(self.eri_4fold_ao, self.mo_coeffs[1])
		# eri_mo_ab = pyscf.ao2mo.incore.general(self.eri_4fold_ao, (self.mo_coeffs[0], self.mo_coeffs[0], self.mo_coeffs[1], self.mo_coeffs[1]))

		# # This will result in separate alpha-alpha, alpha-beta, and beta-beta spatial ERI tensors
		# eri_aa = pyscf.ao2mo.incore.full(self.eri_4fold_ao,  self.mo_coeffs[0])
		# eri_bb = pyscf.ao2mo.incore.full(self.eri_4fold_ao,  self.mo_coeffs[1])
		# eri_ab = pyscf.ao2mo.incore.general(self.eri_4fold_ao, (self.mo_coeffs[0], self.mo_coeffs[0], self.mo_coeffs[1], self.mo_coeffs[1]))

		# # Reshape from 2D (compact) to 4D (full) array if needed (compact=False does this automatically)
		norb_alpha = self.mo_coeffs[0].shape[1]
		norb_beta = self.mo_coeffs[1].shape[1]

		# eri_aa = eri_aa.reshape(norb_alpha, norb_alpha, norb_alpha, norb_alpha)
		# eri_bb = eri_bb.reshape(norb_beta, norb_beta, norb_beta, norb_beta)
		# eri_ab = eri_ab.reshape(norb_alpha, norb_alpha, norb_beta, norb_beta)
		# eri_ba = eri_ab.transpose(2, 3, 0, 1) # eri_ba is the transpose of eri_ab

		# # 4. Construct the full spin-orbital ERI tensor
		# # The spin-orbitals are typically ordered as [alpha_1, ..., alpha_N, beta_1, ..., beta_M]
		# # The full tensor will have dimensions of (N+M) x (N+M) x (N+M) x (N+M)
		# norb_total = norb_alpha + norb_beta
		# eri_spin = np.zeros((norb_total, norb_total, norb_total, norb_total))

		# eri_spin[:norb_alpha, :norb_alpha, :norb_alpha, :norb_alpha] = eri_aa
		# eri_spin[norb_alpha:, norb_alpha:, norb_alpha:, norb_alpha:] = eri_bb
		# eri_spin[:norb_alpha, :norb_alpha, norb_alpha:, norb_alpha:] = eri_ab
		# eri_spin[norb_alpha:, norb_alpha:, :norb_alpha, :norb_alpha] = eri_ba


		# The ao2mo.kernel function can handle the four-index transformation for different orbital sets.
		# Note: PySCF stores 2e integrals in chemists' notation: (ij|kl) = <ik|jl> in physicists' notation.

		# --- (alpha alpha | alpha alpha) integrals ---
		eri_aaaa = pyscf.ao2mo.kernel(self.eri_4fold_ao, [self.mo_coeffs[0], self.mo_coeffs[0], self.mo_coeffs[0], self.mo_coeffs[0]], compact=False)
		eri_aaaa = eri_aaaa.reshape(norb_alpha, norb_alpha, norb_alpha, norb_alpha)

		# --- (beta beta | beta beta) integrals ---
		eri_bbbb = pyscf.ao2mo.kernel(self.eri_4fold_ao, [self.mo_coeffs[1], self.mo_coeffs[1], self.mo_coeffs[1], self.mo_coeffs[1]], compact=False)
		eri_bbbb = eri_bbbb.reshape(norb_beta, norb_beta, norb_beta, norb_beta)

		# --- (alpha alpha | beta beta) integrals ---
		# These are the (ij|kl) where i,j are alpha, k,l are beta
		eri_aabb = pyscf.ao2mo.kernel(self.eri_4fold_ao, [self.mo_coeffs[0], self.mo_coeffs[0], self.mo_coeffs[1], self.mo_coeffs[1]], compact=False)
		eri_aabb = eri_aabb.reshape(norb_alpha, norb_alpha, norb_beta, norb_beta)

		# --- (beta beta | alpha alpha) integrals ---
		eri_bbaa = pyscf.ao2mo.kernel(self.eri_4fold_ao, [self.mo_coeffs[1], self.mo_coeffs[1], self.mo_coeffs[0], self.mo_coeffs[0]], compact=False)
		eri_bbaa = eri_bbaa.reshape(norb_beta, norb_beta, norb_alpha, norb_alpha)

		norb_total = norb_alpha + norb_beta
		eri_spin = np.zeros((norb_total, norb_total, norb_total, norb_total))
		eri_spin[:norb_alpha, :norb_alpha, :norb_alpha, :norb_alpha] = eri_aaaa
		eri_spin[norb_alpha:, norb_alpha:, norb_alpha:, norb_alpha:] = eri_bbbb
		eri_spin[:norb_alpha, :norb_alpha, norb_alpha:, norb_alpha:] = eri_aabb
		eri_spin[norb_alpha:, norb_alpha:, :norb_alpha, :norb_alpha] = eri_bbaa

		for I in [i for i in range(int(self.nelec))]:
			for J in [i for i in range(int(self.nelec))]:
				if I > J:
					for A in [i for i in range(self.nelec,int((self.num_opt_virtual_orbs+self.nelec)))]:
						for B in [i for i in range(self.nelec,int((self.num_opt_virtual_orbs+self.nelec)))]:
							if A > B:
								print(I,A,"-",J,B,"->",eri_spin[I,J,A,B])



		# MP2 in spin-orbital basis, Eq. 14.2.53 in Molecular electronic-structure theory book
		# I,J -> occupied spin orbitals in active space
		# A,B -> empty spins orbitals in active space

		E_MP2_corr = 0
		for I in self.active_occ_indices:
			for J in self.active_occ_indices:
				if I > J:
					for A in self.active_inocc_indices:
						for B in self.active_inocc_indices:
							if A > B:
								E_MP2_corr += -1.0*( (eri_aaaa[A,I,B,J] - eri_aaaa[A,J,B,I])**2 
									/ (self.mo_energy_a[A] + self.mo_energy_a[B] - self.mo_energy_a[I] - self.mo_energy_a[J]) )

								E_MP2_corr += -1.0*( (eri_bbbb[A,I,B,J] - eri_bbbb[A,J,B,I])**2 
									/ (self.mo_energy_b[A] + self.mo_energy_b[B] - self.mo_energy_b[I] - self.mo_energy_b[J]) )

								# E_MP2_corr += -1.0*( (eri_ab[A,I,B,J] - eri_ab[A,J,B,I])**2 
								# 	/ (self.mo_energy_a[A] + self.mo_energy_b[B] - self.mo_energy_a[I] - self.mo_energy_b[J]) )
		
								#print(I,J,A,B,E_MP2_corr)
								# #print(self.mo_energy_a[A] + self.mo_energy_a[B] - self.mo_energy_a[I] - self.mo_energy_a[J])
								#print(I,J,A,B,eri_aa[A,I,B,J] - eri_aa[A,J,B,I], self.mo_energy_a[A] + self.mo_energy_a[B] - self.mo_energy_a[I] - self.mo_energy_a[J])
								# print(I,J,A,B,eri_bb[A,B,I,J] - eri_bb[A,B,J,I])
								# print(E_MP2_corr)
								# print()

								if abs(eri_aaaa[A,I,B,J] - eri_aaaa[A,J,B,I])>1e-3:
									print(I,J,A,B,eri_aaaa[A,I,B,J] - eri_aaaa[A,J,B,I])

								if abs(eri_bbbb[A,I,B,J] - eri_bbbb[A,J,B,I])>1e-3:
									print(I,J,A,B,eri_bbbb[A,I,B,J] - eri_bbbb[A,J,B,I])


		#eri_4fold_spin_mo = spatial2spin(eri_ab, orbspin=None)
		for I in [i for i in range(int(self.nelec))]:
			#if I % 2 == 0:
			for J in [i for i in range(int(self.nelec))]:
				#if J % 2 != 0:
				if I > J:
					for A in [i for i in range(self.nelec,int((self.num_opt_virtual_orbs+self.nelec)))]:
						#if A % 2 == 0:
						for B in [i for i in range(self.nelec,int((self.num_opt_virtual_orbs+self.nelec)))]:
							#if B % 2 != 0:
							if A > B:
								if I % 2 == 0 and A % 2 == 0 and J % 2 != 0 and B % 2 != 0:
									# print(I,J,A,B)
									# print(int(I/2), int((J-1)/2), int(A/2), int((B-1)/2))
									# print()
									I_ = int(I/2)
									J_ = int((J-1)/2)
									A_ = int(A/2)
									B_ = int((B-1)/2)
									# print(I,"->",I_)
									# print(J,"->",J_)
									# print()

									E_MP2_corr += -1.0*( (eri_ab[A_,I_,B_,J_] - eri_ab[A_,J_,B_,I_])**2 
									/ (self.mo_energy_a[A_] + self.mo_energy_b[B_] - self.mo_energy_a[I_] - self.mo_energy_b[J_]) )
									#print(I_, J_, A_, B_)


									if abs(eri_ab[A_,I_,B_,J_] - eri_ab[A_,J_,B_,I_])>1e-3:
										print(I,J,A,B,eri_ab[A_,I_,B_,J_] - eri_ab[A_,J_,B_,I_])



			


		

		#a, b indices -> inoccupied orbitals in active space
		self.active_inocc_indices = [i for i in range(self.active_occ_indices[-1]+1,int((self.num_opt_virtual_orbs+self.nelec)/2))]


		print("E_MP2_corr = ", E_MP2_corr)
		
		E_MP2 = E_rhf + E_MP2_corr

		MP2 = self.uhf.MP2().run()
		assert np.abs(E_MP2_corr - MP2.e_corr) < 1e-6, "np.abs(E_MP2_corr - self.rhf.MP2().run().e_corr) < 1e-6"  
		assert np.abs(E_MP2 - MP2.e_tot) < 1e-6, "np.abs(E_MP2 - MP2.e_tot) < 1e-6"  

		return E_MP2, t1_tensor 

	
	def orbital_rotation(self, mo_coeffs):

		"""
		First- and second-order derivatives of the second-order Hylleraas functional
		Equations 11a and 11b in https://pubs.aip.org/aip/jcp/article/86/11/6314/93345/Optimized-virtual-orbital-space-for-high-level
		"""

		#Gradient
		G = np.zeros((self.n_orbs, self.n_orbs))

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
run_OVOS.MP2_energy(E_rhf = rhf.e_tot)






