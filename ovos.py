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
		
		# Number of orbitals
		self.nelec = self.mol.nelec


	def _t1(self, mo_coeffs, e_rhf) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
     
		"""
		MP1 amplitudes
		"""

		Fmo  = mo_coeffs.T @ self.F_matrix @ mo_coeffs
		eigval, eigvec = scipy.linalg.eig(Fmo)
		sorting = np.argsort(eigval)
		eigval = np.real(eigval[sorting])
		eigvec = np.real(eigvec[:, sorting])

		eri_4fold_mo = pyscf.ao2mo.incore.full(self.eri_4fold_ao, mo_coeffs)

		# i,j -> occupied orbitals in the HF state 
		# a,b -> empty orbitals in the HF state

		t1_tensor = np.zeros((self.n_orbs,self.n_orbs,self.n_orbs,self.n_orbs))
		E_corr = 0
		nelec_ = self.nelec[0] + self.nelec[1]
		for i in range(int(nelec_/2)):
			for j in range(int(nelec_/2)):
				for a in range(int(nelec_/2),self.n_orbs):
					for b in range(int(nelec_/2),self.n_orbs):
						#print("i,j = ",i,j)
						#print("a,b = ",a,b, "\n")

						E_corr += -1.0*(eri_4fold_mo[a,i,b,j]*(2*eri_4fold_mo[i,a,j,b] - eri_4fold_mo[i,b,j,a]) / 
							(eigval[a] + eigval[b] - eigval[i] - eigval[j]) )

					
						t1 =  -1.0*(eri_4fold_mo[a,i,b,j] / (eigval[a] + eigval[b] - eigval[i] - eigval[j]) )
						t1_tensor[a,i,b,j] = t1
						#print(t1)
						
		E_MP2 = e_rhf + E_corr
		
		
		MP2 = self.rhf.MP2().run()
		assert np.abs(E_corr - MP2.e_corr) < 1e-6, "np.abs(E_corr - self.rhf.MP2().run().e_corr) < 1e-6"  
		assert np.abs(E_MP2 - MP2.e_tot) < 1e-6, "np.abs(E_corr - self.rhf.MP2().run().e_corr) < 1e-6"  

		return E_MP2, t1_tensor 

	
	def _orbital_rotation(self,):

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

run_OVOS = OVOS(mol=mol, num_vir_ops=3)
run_OVOS._t1(mo_coeffs = mo_coeff, e_rhf = rhf.e_tot)






