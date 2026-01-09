"""
COVO class
"""

from typing import Tuple, List

import numpy as np
import pyscf

# slowquant imports
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS
from slowquant.unitary_coupled_cluster.operators import hamiltonian_0i_0a
from slowquant.unitary_coupled_cluster.fermionic_operator import FermionicOperator
from slowquant.unitary_coupled_cluster.util import iterate_t1_sa_generalized, iterate_pair_t2_generalized
from slowquant.unitary_coupled_cluster.operators import G1_sa, G2_sa
from slowquant.unitary_coupled_cluster.operator_state_algebra import expectation_value, propagate_state


class COVO:

	"""
    Parameters
    ----------
    mol : pyscf.M
        PySCF molecule object.
    num_covos : int
        Number of 'covo orbitals'.
    """

	def __init__(self, mol: pyscf.gto.Mole, num_covos: int) -> None:
		self.mol = mol
		self.num_covos = num_covos

		# Restricted Hartree-Fock calculation
		self.rhf = pyscf.scf.RHF(mol).run()
		self.h_nuc = mol.energy_nuc()

		# Integrals in AO basis
		self.hcore_ao = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
		self.overlap = mol.intor('int1e_ovlp')
		self.eri_4fold_ao = mol.intor('int2e_sph', aosym=1)

		# Number of spatial AOs (or orbitals)
		self.n_orbs = int(self.rhf.mo_coeff.shape[0])


	def _wavefunction_object(self, mo_coeffs: np.ndarray) -> WaveFunctionUPS:
		
		"""Build WaveFunction object."""

		WF = WaveFunctionUPS(
		num_elec = mol.nelectron,
		cas = (2, self.num_covos+1),
		mo_coeffs = mo_coeffs,
		h_ao = self.hcore_ao,
		g_ao = self.eri_4fold_ao,
		ansatz = "fUCCSD",)

		return WF


	def _get_ci_hamiltonian(self, idx_e: int, mo_coeffs: np.ndarray) -> np.ndarray:
        
		"""
		Build the CI Hamiltonian matrix in the basis of operators constructed below.
		Returns a symmetric matrix H_CI of shape (3, 3).
		"""

		WF = self._wavefunction_object(mo_coeffs)

	    # Hamiltonian
		H = hamiltonian_0i_0a(WF.h_mo, WF.g_mo, WF.num_inactive_orbs, WF.num_active_orbs)
		
		# Make lists
		l_g = list(range(WF.num_inactive_orbs + 1)) # [0...g]
		l_e = list(range(WF.num_inactive_orbs)) + [idx_e] # [0...e]
		l = list(range(WF.num_inactive_orbs)) # [0...g-1]

		# Make operator pool
		operators = []

		# This line is adding the identity operator to the pool.
		operators.append(FermionicOperator({"": 1.0}))

		# Build operators to generate the states |psi_e> and |psi_m>.
		# |psi_m> (single excitation):
		idx_g = l_g[-1] #fixed
		E_1 = G2_sa(i=idx_g, j=idx_g, a=idx_e, b=idx_e, case=1, return_anti_hermitian = False) 
		# |psi_e> (double excitation):
		E_2 = G1_sa(i=idx_g, a=idx_e, return_anti_hermitian = False)

		operators.append(E_1)
		operators.append(E_2)

		# Building Hamiltonian matrix of the type <HF|O_pq^dagger H O_rs|HF>
		H_type1 = np.zeros((len(operators), len(operators)))

		# Building CI Hamiltonian matrix of the type <HF|O_pq^dagger U^dagger H U O_rs|HF>
		for j, GJ in enumerate(operators):
			HGJ = propagate_state(
			      	[H, GJ],
			        WF.csf_coeffs, # HF reference
			        WF.ci_info,
			        WF.thetas,
			        WF.ups_layout,
			    )
			for i, GI in enumerate(operators[j:], j):
				# <CSF| GId H GJ | CSF>
				val = expectation_value(
				WF.csf_coeffs, # HF reference
				[GI.dagger],
				HGJ,
				WF.ci_info,
				WF.thetas,
				WF.ups_layout,
				)
				H_type1[i,j] = H_type1[j,i] = val

		# Add nuclear repulsion energy to diagonal
		H_CI = H_type1 + np.identity(len((H_type1)))*self.h_nuc
		
		# ---- Sanity checks ----

		# Matrix elements computed from notes 
		
		# Integrals in MO basis
		hcore_mo = np.einsum('pi,pq,qj->ij', mo_coeffs, self.hcore_ao, mo_coeffs)
		eri_4fold_mo = pyscf.ao2mo.incore.full(self.eri_4fold_ao, mo_coeffs)
		
		# <psi_g|H|psi_g> (which is the same as the Hartree-Fock energy)
		h = 0
		J = 0
		K = 0
		for i in l_g:
			h = h + 2*hcore_mo[i,i]
			for j in l_g:
				#Coulomb integral
				J = J + 2*eri_4fold_mo[i,i,j,j]
				#Exchange integral
				K = K + eri_4fold_mo[i,j,j,i]

		E_g = h + J - K + self.h_nuc
		assert np.abs(E_g - H_CI[0,0]) < 1e-6, "<psi_g|H|psi_g> is not correct!"
		#print(np.abs(E_g - H_CI[0,0]) < 1e-6, "<gHg>")

		# <psi_e|H|psi_e>
		h = 0
		J = 0
		K = 0
		for i in l_e:
			h = h + 2*hcore_mo[i,i]
			for j in l_e:
				#Coulomb integral
				J = J + 2*eri_4fold_mo[i,i,j,j]
				#Exchange integral
				K = K + eri_4fold_mo[i,j,j,i]

		E_e = h + J - K + self.h_nuc
		assert np.abs(E_e - H_CI[1,1]) < 1e-6, "<psi_e|H|psi_e> is not correct!"

		# <psi_m|H|psi_m> 
		h = 0
		J = 0
		J_g = 0
		J_e = 0
		K = 0
		K_g = 0
		K_e = 0
		for i in l:
			h = h + 2*hcore_mo[i,i]
			J_g = J_g + 2*eri_4fold_mo[l_g[-1],l_g[-1],i,i]
			K_g = K_g + eri_4fold_mo[l_g[-1],i,i,l_g[-1]]

			J_e = J_e + 2*eri_4fold_mo[l_e[-1],l_e[-1],i,i]
			K_e = K_e + eri_4fold_mo[l_e[-1],i,i,l_e[-1]]
			for j in l:
				#Coulomb integral
				J = J + 2*eri_4fold_mo[i,i,j,j]
				#Exchange integral
				K = K + eri_4fold_mo[i,j,j,i]


		h = h + hcore_mo[l_g[-1],l_g[-1]] #g-orbital
		h = h + hcore_mo[l_e[-1],l_e[-1]] #e-orbital
		g_ggee = eri_4fold_mo[l_g[-1],l_g[-1],l_e[-1],l_e[-1]]
		g_geeg = eri_4fold_mo[l_g[-1],l_e[-1],l_e[-1], l_g[-1]]

		E_m = h+(J-K)+(J_g-K_g)+(J_e-K_e)+g_ggee+g_geeg+self.h_nuc

		assert np.abs(E_m - H_CI[2,2]) < 1e-6, "<psi_m|H|psi_m> is not correct!"
		#print(np.abs(E_m - H_CI[2,2]) < 1e-6, "<mHm>")


		# <psi_g|H|psi_e> 
		E_ge = eri_4fold_mo[l_g[-1],l_e[-1],l_g[-1],l_e[-1]]

		assert np.abs(E_ge - H_CI[0,1]) < 1e-6, "<psi_g|H|psi_e> is not correct!"
		#print(np.abs(E_ge - H_CI[0,1]) < 1e-6, "<gHe>")

		# <psi_g|H|psi_m>
		J = 0
		K = 0
		for i in l:
			J = J + 2*eri_4fold_mo[l_g[-1],l_e[-1],i,i]
			K = K + eri_4fold_mo[l_g[-1],i,i,l_e[-1]]

		E_gm = 2/np.sqrt(2)*(hcore_mo[l_g[-1],l_e[-1]] + (J-K) + eri_4fold_mo[l_g[-1],l_g[-1],l_e[-1],l_g[-1]])

		assert np.abs(E_gm - H_CI[0,2]) < 1e-6, "<psi_g|H|psi_m> is not correct!"
		#print(np.abs(E_gm - H_CI[0,2]) < 1e-6, "<gHm>")

		# <psi_e|H|psi_m> 
		J = 0
		K = 0
		for i in l:
			J = J + 2*eri_4fold_mo[l_e[-1],l_g[-1],i,i]
			K = K + eri_4fold_mo[l_e[-1],i,i,l_g[-1]]

		E_em = 2/np.sqrt(2)*(hcore_mo[l_e[-1],l_g[-1]] + J - K + eri_4fold_mo[l_e[-1],l_g[-1],l_e[-1],l_e[-1]])

		assert np.abs(E_em - H_CI[1,2]) < 1e-6, "<psi_e|H|psi_m> is not correct!"
		#print(np.abs(E_em - H_CI[1,2]) < 1e-6, "<eHm>")

		return H_CI

	@staticmethod
	def _diagonalization(H_CI: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		
		"""Diagonalize CI Hamiltonian"""

		# Compute eigenvalues and eigenvectors
		eigenvalues, eigenvectors = np.linalg.eig(H_CI)

		# Get the indices that would sort the eigenvalues in ascending order
		sort_indices = eigenvalues.argsort()

		# Sort the eigenvalues using these indices
		sorted_eigenvalues = eigenvalues[sort_indices]

		# Reorder the eigenvectors using these same indices (by column)
		sorted_eigenvectors = eigenvectors[:, sort_indices]

		return sorted_eigenvalues, sorted_eigenvectors

	
	def _build_matrices(self, c_e: np.ndarray, mo_coeffs: np.ndarray, idx_e: int) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Build matrix A and vector b used in the optimization routine.
		"""

		WF = self._wavefunction_object(mo_coeffs)
		H_CI = self._get_ci_hamiltonian(idx_e, mo_coeffs)

		_, eigvecs = self._diagonalization(H_CI)
		a_g, a_e, a_m = eigvecs[0, 0], eigvecs[1, 0], eigvecs[2, 0]
		#print(a_e**2, a_m**2)

		l = list(range(WF.num_inactive_orbs)) # [0...g-1]
		l_g = list(range(WF.num_inactive_orbs + 1)) # [0...g]
		A_ee = np.zeros((self.n_orbs, self.n_orbs))
		A_mm = np.zeros((self.n_orbs, self.n_orbs))
		A_em = np.zeros((self.n_orbs, self.n_orbs))
		A_ge = np.zeros((self.n_orbs, self.n_orbs))
		b_gm = np.zeros((self.n_orbs,))
		b_em = np.zeros((self.n_orbs,))

		# Build A_ee 
		for p in range(self.n_orbs):
			for q in range(self.n_orbs):
				tmp = 0.0
				for i in l:
					for rho in range(self.n_orbs):
						for sigma in range(self.n_orbs):
							tmp += 4.0*(mo_coeffs[rho,i]*mo_coeffs[sigma,i] * 
								 	(2.0*self.eri_4fold_ao[p,q,rho,sigma] - 
									self.eri_4fold_ao[p,rho,sigma,q]))
				_tmp = 0.0
				for rho in range(self.n_orbs):
					for sigma in range(self.n_orbs):
						_tmp += 4.0*(c_e[rho]*c_e[sigma]*self.eri_4fold_ao[p,rho,sigma,q])

				A_ee[p,q] = 4.0*self.hcore_ao[p,q] + tmp + _tmp

		
		# Build A_mm
		for p in range(self.n_orbs):
			for q in range(self.n_orbs):
				tmp = 0.0
				for i in l:
					for rho in range(self.n_orbs):
						for sigma in range(self.n_orbs):
							tmp += 2.0*(mo_coeffs[rho,i]*mo_coeffs[sigma,i] * 
								 	(2.0*self.eri_4fold_ao[p,q,rho,sigma] - 
									self.eri_4fold_ao[p,rho,sigma,q]))
				_tmp = 0.0
				for rho in range(self.n_orbs):
					for sigma in range(self.n_orbs):
						_tmp += 2.0*(mo_coeffs[rho,l_g[-1]]*mo_coeffs[sigma,l_g[-1]] * 
							(self.eri_4fold_ao[p,q,rho,sigma] + self.eri_4fold_ao[p,rho,q,sigma]))
				
				A_mm[p,q] = 2.0*self.hcore_ao[p,q] + tmp + _tmp
		
		# Build A_em
		for p in range(self.n_orbs):
			for q in range(self.n_orbs):
				tmp = 0.0
				for rho in range(self.n_orbs):
					for sigma in range(self.n_orbs):
						tmp += 12.0/np.sqrt(2.0)*(mo_coeffs[rho,l_g[-1]]*c_e[sigma] * 
							self.eri_4fold_ao[p,rho,q,sigma])
			
				A_em[p,q] = 4.0/np.sqrt(2.0)*self.hcore_ao[p,q] + tmp

		# Build A_ge
		for p in range(self.n_orbs):
			for q in range(self.n_orbs):
				tmp = 0.0
				for rho in range(self.n_orbs):
					for sigma in range(self.n_orbs):
						tmp += 4.0*(mo_coeffs[rho,l_g[-1]]*mo_coeffs[sigma,l_g[-1]] * 
							self.eri_4fold_ao[p,rho,sigma,q])
			
				A_ge[p,q] = tmp

		# Build b_gm
		for p in range(self.n_orbs):
			tmp = 0.0
			for i in l:
				for rho in range(self.n_orbs):
					for sigma in range(self.n_orbs):
						for tau in range(self.n_orbs):
							tmp += -4.0/np.sqrt(2.0)*(mo_coeffs[rho,i]*mo_coeffs[sigma,i]*mo_coeffs[tau,l[-1]+1] *
								 (2.0*self.eri_4fold_ao[tau,p,rho,sigma] - self.eri_4fold_ao[tau,rho,sigma,p]))
			_tmp = 0.0
			for rho in range(self.n_orbs):
				for sigma in range(self.n_orbs):
					for tau in range(self.n_orbs):
						_tmp += -4.0/np.sqrt(2.0)*(mo_coeffs[rho,l_g[-1]]*mo_coeffs[sigma,l_g[-1]]*mo_coeffs[tau,l_g[-1]] *
							 self.eri_4fold_ao[rho,p,sigma,tau])

			__tmp = 0.0
			for rho in range(self.n_orbs):
				__tmp += -4.0/np.sqrt(2.0)*mo_coeffs[rho,l_g[-1]]*self.hcore_ao[rho,p]

			b_gm[p] = tmp + _tmp + __tmp

		# Build b_em
		for p in range(self.n_orbs):
			tmp = 0.0
			for i in l:
				for rho in range(self.n_orbs):
					for sigma in range(self.n_orbs):
						for tau in range(self.n_orbs):
							tmp += -4.0/np.sqrt(2.0)*(mo_coeffs[rho,i]*mo_coeffs[sigma,i]*mo_coeffs[tau,l[-1]+1] *
								 (2.0*self.eri_4fold_ao[p,tau,rho,sigma] - self.eri_4fold_ao[p,rho,sigma,tau]))
			b_em[p] = tmp

		return a_e**2*A_ee + a_m**2*A_mm + a_e*a_m*A_em + a_g*a_e*A_ge, a_e*a_m*b_em+a_g*a_m*b_gm

			
	def _optimization_of_vir_orb(self, c0_e: np.ndarray, mo_coeffs: np.ndarray, idx_e: int,
                      max_iterations: int = 1000, tol: float = 1e-3) -> Tuple[np.ndarray, List[np.ndarray]]:

		"""
        Fixed-point optimization for virtual orbital coefficients.

	    Parameters
	    ----------
	    c0_e : np.ndarray
	        Initial guess for the virtual orbital coefficients.
	    mo_coeffs : np.ndarray
	        Molecular orbital coefficients matrix.
	    idx_e : int
	        Index of the virtual orbital being optimized.
	    max_iterations : int, optional
	        Maximum number of iterations (default = 1000).
	    tol : float, optional
	        Convergence tolerance for iteration (default = 1e-3).
        
	    Returns
	    -------

        """

        # Initial A and b
		A, b = self._build_matrices(c_e=c0_e, mo_coeffs=mo_coeffs, idx_e=idx_e)

		x_old = c0_e.copy()
		history = [x_old.copy()]
		for iteration in range(1, max_iterations + 1):
			''' Start with Ax=b. A simple rearrangement is to write it as x=x-Ax+b, 
			which becomes x=(I-A)x+b, where I is the identity matrix. This is a fixed-point iteration of the form x_{k+1}=T(x_{k})+c
			'''
			x_new = (np.identity(self.n_orbs)-A)@x_old+b

			#Try to add the Gram Schmidt orthonormalization here!

			error = np.linalg.norm(x_new - x_old)
			print(f"{iteration:3d} iters and error = {error}")

			history.append(x_new.copy())
			if error <= tol:
				print(f"Converged after {iteration:3d} iters and error = {error}")
				print("Gram Schmidt orthonormalization ...") 
				
				# Insert the new orbital and compute overlap
				mo_coeffs_new = np.insert(
				np.delete(mo_coeffs, idx_e, axis=1), idx_e, x_new, axis=1
				)
				overlap_mo_coeffs = mo_coeffs_new.T @ self.overlap @ mo_coeffs_new
				
				# Orthogonalize the updated orbital
				x = 0
				for i in range(self.n_orbs):
					if i!=idx_e:
						x -= overlap_mo_coeffs[idx_e,i]*mo_coeffs[:,i]
				x += x_new	

				# Normalize the virtual orbital
				mo_coeffs_non_norm = np.insert(np.delete(mo_coeffs, idx_e, axis=1), idx_e, x, axis=1)
				overlap_mo_coeffs_non_norm = mo_coeffs_non_norm.T@self.overlap@mo_coeffs_non_norm
				norm_x = np.sqrt(overlap_mo_coeffs_non_norm[idx_e, idx_e])
				norm_const = 1/norm_x
				x *= norm_const
				if norm_x < 1e-10:  
					raise ValueError("Input vectors are not linearly independent.")

				# Final normalized coefficients
				mo_coeffs_final = np.insert(np.delete(mo_coeffs, idx_e, axis=1), idx_e, x, axis=1)
				
				if not np.allclose(mo_coeffs_final.T@self.overlap@mo_coeffs_final, np.eye(self.n_orbs)):
					raise ValueError("The virtual orbital is not orthogonal!")
				
				print()
				print(history[0])
				print()
				print(x)
				#print()
				print(mo_coeffs[:,idx_e]==x)
				print()
				print(mo_coeffs[:,idx_e])
				#print(history[-1])
					
				return mo_coeffs_final, history

			x_old = x_new
			A, _ = self._build_matrices(c_e=x_new, mo_coeffs=mo_coeffs, idx_e=idx_e)

		print("Warning: maximum iterations reached without convergence.")
		return x_old, history
	
	def _optimization_of_ci_amplitudes(self,):
		

		raise NotImplementedError

	
	def run_COVO(self,):


		WF = self._wavefunction_object(self.rhf.mo_coeff)
		l_g = list(range(WF.num_inactive_orbs + 1)) # [0...g]

		idx_e = l_g[-1]+1

		H_CI = self._get_ci_hamiltonian(idx_e = idx_e, mo_coeffs = self.rhf.mo_coeff)
		E0, _ = self._diagonalization(H_CI)
		print("E0",E0)

		mo_coeffs_opt, _ = self._optimization_of_vir_orb(c0_e = mo_coeff[:,idx_e] + np.random.rand(len(mo_coeff)), mo_coeffs = self.rhf.mo_coeff, idx_e=idx_e) #np.random.rand(len(mo_coeff))
		#H_CI = self._get_ci_hamiltonian(idx_e = idx_e, mo_coeffs = mo_coeffs_opt)
		#E0, _ = self._diagonalization(H_CI)
		#print("E0",E0)




# Molecule
#atom = "Li .0 .0 .0; H .0 .0 1.595"
#atom = "H .0 .0 .0; H .0 .0 0.74144"
atom = """O 0.0000 0.0000  0.1173; H 0.0000    0.7572  -0.4692; H 0.0000   -0.7572 -0.4692;""" #Angstrom
basis = "STO-3G"
unit="angstrom"
mol = pyscf.M(atom=atom, basis=basis, unit=unit)

rhf = pyscf.scf.RHF(mol).run()
mo_coeff = rhf.mo_coeff 

run_COVO = COVO(mol=mol, num_covos=3)
#H_CI = run_COVO._get_ci_hamiltonian(idx_e=2, mo_coeffs = mo_coeff)
#run_COVO._diagonalization(H_CI = H_CI)
#run_COVO._build_matrices(c = mo_coeff[2], mo_coeffs = mo_coeff, idx_e=2)
#run_COVO._optimization_of_vir_orb(c0_e = mo_coeff[2], mo_coeffs = mo_coeff, idx_e=2)
run_COVO.run_COVO()


