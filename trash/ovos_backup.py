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
    num_vir_ops : int
        Number of optimized virtual orbitals.
    """

	def __init__(self, mol: pyscf.gto.Mole, num_active_orbs: int) -> None:
		self.mol = mol
		self.num_active_orbs = num_active_orbs

		# Restricted Hartree-Fock calculation
		self.rhf = pyscf.scf.RHF(mol).run()
		self.e_rhf = self.rhf.e_tot
		self.h_nuc = mol.energy_nuc()

		# Build fock matrix in AO basis
		self.F_matrix  = self.rhf.get_fock()

		# Integrals in AO basis
		self.hcore_ao = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
		self.overlap = mol.intor('int1e_ovlp')
		self.eri_4fold_ao = mol.intor('int2e_sph', aosym=1)

		# Number of orbitals
		self.n_orbs = int(self.rhf.mo_coeff.shape[0])
		
		# Number of electrons
		self.nelec = self.mol.nelec[0] + self.mol.nelec[1]

		# Number of inactive orbitals
		#i,j indices (occupied orbitals)
		self.active_spin_occ_indices = [i for i in range(int(self.nelec))]
		#a, b indices (inoccupied orbitals in active space)
		self.active_spin_inocc_indices = [i for i in range(self.active_spin_occ_indices[-1]+1,self.active_spin_occ_indices[-1]+1+int(2*self.num_active_orbs-self.nelec))]
		#print(self.active_spin_occ_indices)
		#print(self.active_spin_inocc_indices)

		assert self.n_orbs >= self.num_active_orbs, "Your num_active_orbs is too large"  

		#build initial Fock matrix
		# Fmo  = mo_coeffs.T @ self.F_matrix @ mo_coeffs
		# self.F_spin = spatial2spin(Fmo)
		# eigval, eigvec = scipy.linalg.eig(self.F_spin)
		# sorting = np.argsort(eigval)
		# self.eigval = np.real(eigval[sorting])
		# self.eigvec = np.real(eigvec[:, sorting])

	
	def Fock_matrix(self, U) -> Tuple[np.ndarray, np.ndarray]:


		# Fmo  = mo_coeffs.T @ self.F_matrix @ mo_coeffs
		# eigval, eigvec = scipy.linalg.eig(Fmo)
		# sorting = np.argsort(eigval)
		# eigval = np.real(eigval[sorting])
		# eigvec = np.real(eigvec[:, sorting])

		return NotImplementedError 


	def MP2_energy(self, mo_coeffs, E_rhf, spin_orbital_basis: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
     
		"""
		MP2 energy for restricted orbitals 
		"""

		Fmo  = mo_coeffs.T @ self.F_matrix @ mo_coeffs
		eigval, eigvec = scipy.linalg.eig(Fmo)
		sorting = np.argsort(eigval)
		eigval = np.real(eigval[sorting])
		eigvec = np.real(eigvec[:, sorting])


		# i,j -> occupied orbitals 
		# a,b -> empty orbitals in active space

		eri_4fold_mo = pyscf.ao2mo.incore.full(self.eri_4fold_ao, mo_coeffs)

		E_corr = 0

		# MP2 in spin-orbital basis, Eq. 14.2.53 in Molecular electronic-structure theory book				
		if spin_orbital_basis:
			eri_4fold_spin_mo = spatial2spin(eri_4fold_mo, orbspin=None)
			print(eri_4fold_spin_mo[0,0])
			t1_tensor = np.zeros((2*self.n_orbs,2*self.n_orbs,2*self.n_orbs,2*self.n_orbs))
			eigval_spin_mo = []
			for i in eigval:
				for rep in range(2):
					eigval_spin_mo.append(float(i))

			for I in self.active_spin_occ_indices :
				for J in self.active_spin_occ_indices :
					if I > J:
						for A in self.active_spin_inocc_indices:
							for B in self.active_spin_inocc_indices:
								if A > B:
									#print(I,A,"-",J,B,"->",eri_4fold_spin_mo[I,J,A,B])
									#MP2 correlation energy for restricted orbitals: 
									E_corr += -1.0*((eri_4fold_spin_mo[A,I,B,J] - eri_4fold_spin_mo[A,J,B,I])**2 
										/ (eigval_spin_mo[A] + eigval_spin_mo[B] - eigval_spin_mo[I] - eigval_spin_mo[J]) )

									#MP1 amplitudes:
									t1 =  -1.0*( (eri_4fold_spin_mo[A,I,B,J] - eri_4fold_spin_mo[A,J,B,I]) / (eigval_spin_mo[A] + eigval_spin_mo[B] - eigval_spin_mo[I] - eigval_spin_mo[J]) )
									t1_tensor[A,I,B,J] = t1
									#print(t1)
									#if I == 2 and J == 0 and A == 11 and B == 4:
									#	print(I,J,A,B,eri_4fold_spin_mo[A,I,B,J] - eri_4fold_spin_mo[A,J,B,I])
									#if I % 2 != 0 and J % 2 != 0 and A % 2 != 0 and B % 2 != 0:
										#print(I,J,A,B,eri_4fold_spin_mo[A,I,B,J] - eri_4fold_spin_mo[A,J,B,I])
									#if I % 2 == 0 and J % 2 != 0 and A % 2 == 0 and B % 2 != 0:
										#print(I,J,A,B,eri_4fold_spin_mo[A,I,B,J] - eri_4fold_spin_mo[A,J,B,I])
									# if abs(eri_4fold_spin_mo[A,I,B,J] - eri_4fold_spin_mo[A,J,B,I])>1e-3:
									# 	print(I,J,A,B,eri_4fold_spin_mo[A,I,B,J] - eri_4fold_spin_mo[A,J,B,I])

		

		# MP2 in spatial orbital basis, Equation 14.4.56 in Molecular electronic-structure theory book
		if spin_orbital_basis is False:
			t1_tensor = np.zeros((self.n_orbs,self.n_orbs,self.n_orbs,self.n_orbs))
			for i in range(int(self.nelec/2)):
				for j in range(int(self.nelec/2)):
					for a in range(int(self.nelec/2),self.n_orbs):
						for b in range(int(self.nelec/2),self.n_orbs):
							#print("i,j = ",i,j)
							#print("a,b = ",a,b, "\n")

							#MP2 correlation energy for restricted closed-shell: 
							E_corr += -1.0*(eri_4fold_mo[a,i,b,j]*(2*eri_4fold_mo[i,a,j,b] - eri_4fold_mo[i,b,j,a]) / 
								(eigval[a] + eigval[b] - eigval[i] - eigval[j]) )

							#MP1 amplitudes:
							t1 =  -1.0*(eri_4fold_mo[a,i,b,j] / (eigval[a] + eigval[b] - eigval[i] - eigval[j]) )
							t1_tensor[a,i,b,j] = t1
							#print(t1)


		print("E_corr = ",E_corr)
		E_MP2 = E_rhf + E_corr

		MP2 = self.rhf.MP2().run()
		assert np.abs(E_corr - MP2.e_corr) < 1e-6, "np.abs(E_corr - self.rhf.MP2().run().e_corr) < 1e-6"  
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

run_OVOS = OVOS(mol=mol, num_active_orbs=6)
run_OVOS.MP2_energy(mo_coeffs = mo_coeff, E_rhf = rhf.e_tot)






