"""
OVOS class
"""

from typing import Tuple, List

import numpy as np
import scipy
import pyscf


class OVOS:

	"""
	Implemenation is based on:
	https://pubs.aip.org/aip/jcp/article/86/11/6314/93345/Optimized-virtual-orbital-space-for-high-level

    Parameters
    ----------
    mol : pyscf.M
        PySCF molecule object.
    num_vir_ops : int
        Number of optimized virtual orbitals.
    """

	def __init__(self, mol: pyscf.gto.Mole, num_vir_ops: int) -> None:
		self.mol = mol
		self.num_vir_ops = num_vir_ops

		# Restricted Hartree-Fock calculation
		self.rhf = pyscf.scf.RHF(mol).run()
		self.h_nuc = mol.energy_nuc()

		# Build fock matrix in AO basis
		self.F_matrix  = self.rhf.get_fock()

		# Integrals in AO basis
		self.hcore_ao = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
		self.overlap = mol.intor('int1e_ovlp')
		self.eri_4fold_ao = mol.intor('int2e_sph', aosym=1)

		# Number of orbitals
		self.n_orbs = int(self.rhf.mo_coeff.shape[0])
		
		# Number of orbitals
		self.nelec = self.mol.nelec


	def _t1(self, mo_coeffs) -> float:
     
		"""
		MP1 amplitudes
		"""

		Fmo  = mo_coeffs.T @ self.F_matrix @ mo_coeffs
		eigval, eigvec = scipy.linalg.eig(Fmo)
		sorting = np.argsort(eigval)
		eigval = np.real(eigval[sorting])
		eigvec = np.real(eigvec[:, sorting])

		eri_4fold_mo = pyscf.ao2mo.incore.full(self.eri_4fold_ao, mo_coeffs)

		# I,J -> occupied orbitals in the HF state 
		# A,B -> empty orbitals in the HF state
		t1 = 0
		nelec_ = self.nelec[0] + self.nelec[1]
		for I in range(int(nelec_/2)):
			for J in range(int(nelec_/2)):
				for A in range(nelec_,self.n_orbs):
					for B in range(nelec_,self.n_orbs):
						print("I,J = ",I,J)
						print("A,B = ",A,B, "\n")
						t1 += eri_4fold_mo[A,I,B,J] - eri_4fold_mo[A,J,B,I]
						#t1 += eri_4fold_mo[A,I,B,J] - eri_4fold_mo[A,J,B,I]
						
		print(t1)
		print(self.n_orbs)
		print(self.rhf.MP2().run())





		# ASSERT
		#print(self.rhf.mo_energy, "\n")
		#print(eigval)



		return None

	
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

run_OVOS = OVOS(mol=mol, num_vir_ops=3)
run_OVOS._t1(mo_coeffs = mo_coeff)






