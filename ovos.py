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
		print(f"Number of orbitals: {self.n_orbs}")
		
		# Number of electrons
		self.nelec = self.mol.nelec

	def _space_selection(self) -> Tuple[List[int], List[int]]:
		"""
		Step (iii): Define active and inactive orbitals

		The choice is based upon the contribution from each individual 
		virtual orbital to the second-order correlation energy. The 
		contribution is calculated as a sum of the diagonal and a half 
		of the off-diagonal part.

		Occupied orbitals are not considered for the selection.
		Virtual orbitals of a,b,... are ranked according to their contribution
		to the MP2 correlation energy, and the top `num_vir_ops` orbitals
		are selected as active orbitals, a,b,... and the rest as inactive orbitals, e,f,...

		Returns
		-------
		List[int], List[int]
			Indices of active and inactive orbitals
		"""

		# Get MP2 energy contribution tensor
		_, E_corr_tensor = self._MP2_energy(mo_coeffs=self.rhf.mo_coeff, E_rhf=self.e_rhf, spin_orbital_basis=False)
		
		nelec_ = self.nelec[0] + self.nelec[1]
		n_occ = int(nelec_ / 2)
		n_virt = self.n_orbs - n_occ
		
		print(f"Number of occupied orbitals: {n_occ}")
		print(f"Number of virtual orbitals: {n_virt}")
		print(f"Requested active virtual orbitals: {self.num_vir_ops}")
		
		# Check if we have enough virtual orbitals for the requested active space
		if self.num_vir_ops > n_virt:
			raise ValueError(
				f"Insufficient virtual orbitals: requested {self.num_vir_ops} active virtuals "
				f"but only {n_virt} virtual orbitals available. "
				f"Use a larger basis set or reduce num_vir_ops."
			)
		
		# Ensure we have at least 1 inactive virtual orbital for rotation
		if self.num_vir_ops == n_virt:
			raise ValueError(
				f"Need at least 1 inactive virtual orbital for OVOS optimization. "
				f"Current: {n_virt} virtuals, requested {self.num_vir_ops} active. "
				f"Use a larger basis set or set num_vir_ops < {n_virt}."
			)
		
		# Calculate contribution from each VIRTUAL orbital to MP2 correlation energy
		# Only consider virtual orbitals (a >= n_occ)
		virt_contributions = np.zeros(n_virt)
		
		for a_idx, a in enumerate(range(n_occ, self.n_orbs)):
			contribution = 0.0
			for i in range(n_occ):
				for b in range(n_occ, self.n_orbs):
					for j in range(n_occ):
						if a == b:
							# Diagonal contribution (full)
							contribution += E_corr_tensor[a, i, b, j]
						else:
							# Off-diagonal contribution (half to avoid double counting)
							contribution += 0.5 * E_corr_tensor[a, i, b, j]
			virt_contributions[a_idx] = contribution
		
		# Rank virtual orbitals by their contribution (descending order)
		sorted_virt_indices = np.argsort(virt_contributions)[::-1]
		
		# Select top num_vir_ops virtual orbitals as active (a, b, ...)
		active_virt_indices = sorted_virt_indices[:self.num_vir_ops]
		# Remaining virtual orbitals are inactive (e, f, ...)
		inactive_virt_indices = sorted_virt_indices[self.num_vir_ops:]
		
		# Convert back to absolute orbital indices
		active_virt_indices = [n_occ + idx for idx in active_virt_indices]
		inactive_virt_indices = [n_occ + idx for idx in inactive_virt_indices]

		return active_virt_indices, inactive_virt_indices
	


	def _t1(self, mo_coeffs, active_virt_indices, spin_orbital_basis: bool = True) -> np.ndarray:
		"""
		Step (iv): MP1 amplitudes
		
		calculate t1 amplitudes for active virtual orbitals only

		Returns
		-------
		np.ndarray
			Shape (n_orbs, n_orbs, n_orbs, n_orbs) array of t1 amplitudes
			Only amplitudes for active virtual orbitals are non-zero
		"""

		# Transform Fock matrix to MO basis
		Fmo  = mo_coeffs.T @ self.F_matrix @ mo_coeffs
		eigval, eigvec = scipy.linalg.eig(Fmo)
		
		# Sort eigenvalues and eigenvectors
		sorting = np.argsort(eigval)
		eigval = np.real(eigval[sorting])
		eigvec = np.real(eigvec[:, sorting])

		# Two-electron integrals in MO basis
		eri_4fold_mo = pyscf.ao2mo.incore.full(self.eri_4fold_ao, mo_coeffs)

		# i,j -> occupied orbitals
		# a,b -> active virtual orbitals
		
		nelec_ = self.nelec[0] + self.nelec[1]
		n_occ = int(nelec_ / 2)

		t1_tensor = np.zeros((self.n_orbs,self.n_orbs,self.n_orbs,self.n_orbs))

		# MP2 in spin-orbital basis, Eq. 14.2.53 in Molecular electronic-structure theory book				
		if spin_orbital_basis:
			eri_4fold_spin_mo = spatial2spin(eri_4fold_mo, orbspin=None)
			
			eigval_spin_mo = []
			for i in eigval:
				for rep in range(2):
					eigval_spin_mo.append(float(i))

			# Convert active virtual indices to spin-orbital basis
			active_spin_indices = []
			for idx in active_virt_indices:
				active_spin_indices.append(2*idx)      # alpha spin
				active_spin_indices.append(2*idx + 1)  # beta spin

			# Build t1 amplitudes only for active virtual orbitals
			for I in range(int(nelec_)):
				for J in range(int(nelec_)):
					if I > J:
						for A in active_spin_indices:
							for B in active_spin_indices:
								if A > B:

									# Calculate MP1 amplitudes, t1, for each combination of (A,I,B,J)
									t1 =  -1.0*( (eri_4fold_spin_mo[A,I,B,J] - eri_4fold_spin_mo[A,J,B,I]) / (eigval_spin_mo[A] + eigval_spin_mo[B] - eigval_spin_mo[I] - eigval_spin_mo[J]) )
									t1_tensor[A,I,B,J] = t1

		if spin_orbital_basis is False:
			# Build t1 amplitudes only for active virtual orbitals
			for i in range(n_occ):
				for j in range(n_occ):
					for a in active_virt_indices:
						for b in active_virt_indices:

							# Calculate MP1 amplitudes, t1, for each combination of (a,i,b,j)
							t1 =  -1.0*(eri_4fold_mo[a,i,b,j] / (eigval[a] + eigval[b] - eigval[i] - eigval[j]) )
							t1_tensor[a,i,b,j] = t1


		return t1_tensor 
	

	
	def _MP2_energy(self, mo_coeffs, E_rhf, spin_orbital_basis: bool = True) -> Tuple[float, np.ndarray]: 
     
		"""
		MP2 energy

		Returns
		-------
		float
			MP2 total energy, E_MP2 = E_RHF + E_corr
		"""

		# Transform Fock matrix to MO basis
		Fmo  = mo_coeffs.T @ self.F_matrix @ mo_coeffs
		eigval, eigvec = scipy.linalg.eig(Fmo)
		
		# Sort eigenvalues and eigenvectors
		sorting = np.argsort(eigval)
		eigval = np.real(eigval[sorting])
		eigvec = np.real(eigvec[:, sorting])

		# Two-electron integrals in MO basis
		eri_4fold_mo = pyscf.ao2mo.incore.full(self.eri_4fold_ao, mo_coeffs)

		# i,j -> occupied orbitals
		# a,b -> virtual orbitals
		
		nelec_ = self.nelec[0] + self.nelec[1]
		
		E_corr_sum = 0.0
		E_corr_tensor = np.zeros((self.n_orbs,self.n_orbs,self.n_orbs,self.n_orbs))

		# MP2 in spin-orbital basis, Eq. 14.2.53 in Molecular electronic-structure theory book				
		if spin_orbital_basis:
			eri_4fold_spin_mo = spatial2spin(eri_4fold_mo, orbspin=None)
			
			eigval_spin_mo = []
			for i in eigval:
				for rep in range(2):
					eigval_spin_mo.append(float(i))

			# Build correlation energy
			for I in range(int(nelec_)):
				for J in range(int(nelec_)):
					if I > J:
						for A in range(int(nelec_),2*self.n_orbs):
							for B in range(int(nelec_),2*self.n_orbs):
								if A > B:
									
									# Calculate correlation energy contribution for each combination of (a,i,b,j)
									E_corr = -1.0*((eri_4fold_spin_mo[A,I,B,J] - eri_4fold_spin_mo[A,J,B,I])**2 
										/ (eigval_spin_mo[A] + eigval_spin_mo[B] - eigval_spin_mo[I] - eigval_spin_mo[J]) )		
									E_corr_sum += E_corr

									# Store individual contributions in tensor
									E_corr_tensor[A,I,B,J] = E_corr

		if spin_orbital_basis is False:
			# Build correlation energy
			for i in range(int(nelec_/2)):
				for j in range(int(nelec_/2)):
					for a in range(int(nelec_/2),self.n_orbs):
						for b in range(int(nelec_/2),self.n_orbs):
							
							# Calculate correlation energy contribution for each combination of (a,i,b,j)
							E_corr = -1.0*(eri_4fold_mo[a,i,b,j]*(2*eri_4fold_mo[i,a,j,b] - eri_4fold_mo[i,b,j,a]) / 
								(eigval[a] + eigval[b] - eigval[i] - eigval[j]) )
							E_corr_sum += E_corr

							# Store individual contributions in tensor
							E_corr_tensor[a,i,b,j] = E_corr
													
		E_MP2 = E_rhf + E_corr
		
		if False:
			# Verify with PySCF MP2		
			MP2 = self.rhf.MP2().run()
			assert np.abs(E_corr - MP2.e_corr) < 1e-6, "np.abs(E_corr - self.rhf.MP2().run().e_corr) < 1e-6"  
			assert np.abs(E_MP2 - MP2.e_tot) < 1e-6, "np.abs(E_corr - self.rhf.MP2().run().e_corr) < 1e-6"  

		return E_MP2, E_corr_tensor
	


	def _compute_gradient_hessian(self, mo_coeffs, active_virt_indices, inactive_virt_indices, t1_tensor) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Step (v): Compute gradient and Hessian

		Gradient and Hessian of the MP2 energy with respect to orbital rotations
		between active and inactive orbitals.

		Expressions
		----------
		Gradient:
		G_ea = 2 Σ_i>j Σ_b t_ij^ab ⟨ij|eb⟩ + 2 Σ_b Σ_i>j Σ_c t_ij^ac t_ij^bc f_eb
		Hessian:
		H_ea,fb = 2 Σ_i>j t_ij^ab ⟨ij|eb⟩ - Σ_i>j Σ_c (t_ij^ac ⟨ij|bc⟩ - t_ij^cb ⟨ij|ca⟩) delta_ef + Σ_i>j Σ_c t_ij^ac t_ij^bc (f_aa - f_bb) delta_ef + Σ_i>j Σ_c t_ij^ac t_ij^bc f_ef (1 - delta_ef)

		Parameters
		----------
		mo_coeffs : np.ndarray
			Molecular orbital coefficients
		active_indices : List[int]
			Indices of active orbitals
		inactive_indices : List[int]
			Indices of inactive orbitals

		Returns
		-------
		np.ndarray, np.ndarray
			Gradient and Hessian matrices
		"""

		nelec_ = self.nelec[0] + self.nelec[1]
		n_occ = int(nelec_/2)
		
		# Get Fock matrix eigenvalues (orbital energies)
		Fmo = mo_coeffs.T @ self.F_matrix @ mo_coeffs
		eigval = np.real(scipy.linalg.eigh(Fmo)[0])
		
		# Get ERIs in MO basis
		eri_4fold_mo = pyscf.ao2mo.incore.full(self.eri_4fold_ao, mo_coeffs)
		
		n_pairs = len(active_virt_indices) * len(inactive_virt_indices)
		gradient = np.zeros(n_pairs)
		hessian = np.zeros((n_pairs, n_pairs))

		# Compute gradient and Hessian
		idx_ea = 0
		for e in inactive_virt_indices:
			for a in active_virt_indices:
				# Gradient G_ea
				grad_ea_term1 = 0.0
				grad_ea_term2 = 0.0
				
				# First term: 2 Σ_i>j Σ_b t_ij^ab ⟨ij|eb⟩
				for i in range(n_occ):
					for j in range(i+1, n_occ):  # i > j
						for b in active_virt_indices:
							t_ijab = t1_tensor[a, i, b, j]
							eri_ijeb = eri_4fold_mo[i, j, e, b]
							grad_ea_term1 += 2.0 * t_ijab * eri_ijeb
				
				# Second term: 2 Σ_b Σ_i>j Σ_c t_ij^ac t_ij^bc f_eb
				for b in active_virt_indices:
					for i in range(n_occ):
						for j in range(i+1, n_occ):
							for c in active_virt_indices:
								t_ijac = t1_tensor[a, i, c, j]
								t_ijbc = t1_tensor[b, i, c, j]
								f_eb = Fmo[e, b]
								grad_ea_term2 += 2.0 * t_ijac * t_ijbc * f_eb
				
				gradient[idx_ea] = grad_ea_term1 + grad_ea_term2
				
				# Hessian H_ea,fb (diagonal approximation for now)
				# For diagonal: H_ea,ea ≈ 2(f_ee - f_aa)
				hessian[idx_ea, idx_ea] = 2.0 * (eigval[e] - eigval[a])
				
				idx_ea += 1

		return gradient, hessian

	def _transform_mo_coeffs(self, mo_coeffs, active_virt_indices, inactive_virt_indices, rotatio_params) -> np.ndarray:
		"""
		Step (iv): Transform MO coefficients

		Expressions
		-----------
		a -> a' = a + Σ_e κ_ea e - 1/2 Σ_e Σ_b κ_ea κ_eb b + ...

		Parameters
		----------
		mo_coeffs : np.ndarray
			Molecular orbital coefficients
		active_indices : List[int]
			Indices of active orbitals
		inactive_indices : List[int]
			Indices of inactive orbitals
		
		Returns
		-------
		np.ndarray
			Transformed MO coefficients
		"""

		return NotImplementedError



	def _rotate_mo_coeffs(self, mo_coeffs, active_virt_indices, inactive_virt_indices, rotation_unitary) -> np.ndarray:
		"""
		Step (viii): Rotate MO coefficients

		Expressions
		-----------
		a -> a' = Σ_b U_ba b + Σ_e U_ea e

		Parameters
		----------
		mo_coeffs : np.ndarray
			Molecular orbital coefficients
		active_indices : List[int]
			Indices of active orbitals
		inactive_indices : List[int]
			Indices of inactive orbitals
		rotation_unitary : np.ndarray
			Unitary rotation matrix U

		Returns
		-------
		np.ndarray
			Rotated MO coefficients
		"""

		return NotImplementedError

	def _compute_rotation_unitary(self, active_virt_indices, inactive_virt_indices, rotation_params) -> np.ndarray:
		"""
		Step (vii): Generate Unitary rotation matrix U

		Expressions
		----------
		U = exp(κ) = X cosh(d) X^T + κ X sinh(d)d^-1 X^T
		d^2 = X^T κ^2 X

		Parameters
		----------
		active_virt_indices : List[int]
			Indices of active virtual orbitals
		inactive_virt_indices : List[int]
			Indices of inactive virtual orbitals
		rotation_params : np.ndarray
			Rotation parameters from Newton-Raphson solution

		Returns
		-------
		np.ndarray
			Unitary rotation matrix U
		"""

		return NotImplementedError


	def _Fock_matrix(self, rotation_unitary) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Step (viii): Construct Fock matrix for rotated active space and diagonalize

		Expressions
		-----------
		F' = U^T F U

		Parameters
		----------
		rotation_unitary : np.ndarray
			Unitary rotation matrix U
		
		Returns
		-------
		np.ndarray, np.ndarray
			Eigenvalues and eigenvectors of rotated Fock matrix
		"""

		# Fmo  = mo_coeffs.T @ self.F_matrix @ mo_coeffs
		# eigval, eigvec = scipy.linalg.eig(Fmo)
		# sorting = np.argsort(eigval)
		# eigval = np.real(eigval[sorting])
		# eigvec = np.real(eigvec[:, sorting])

		return NotImplementedError 



	def run_OVOS(self):
		"""
		Run OVOS procedure to obtain optimized virtual orbitals
		
		Returns
		-------
		np.ndarray
			Optimized virtual orbital coefficients
		"""

		# Step (i): Compute SCF solution
			# Get initial MO coefficients from RHF
		mo_coeffs = self.rhf.mo_coeff 
		print(f"Initial: {mo_coeffs}")

		# Step (ii): Record structure of integrals

		# Step (iii): Define active and inactive orbitals
		active_virt_indices, inactive_virt_indices = self._space_selection()
		print(f"Active virtual indices: {active_virt_indices}")
		print(f"Inactive virtual indices: {inactive_virt_indices}")

		# Iterative procedure
		max_iterations = 1
		for iteration in range(max_iterations):
			print(f"--- Iteration {iteration+1} ---")

			# Step (iv):
				# Transform integrals, a -> a'
					# Initial iteration: no transformation needed
			if iteration == 0:
				pass
					# Iterative procedure: Transform integrals using previous rotation matrix
			else:
				mo_coeffs = self._transform_mo_coeffs(self, mo_coeffs, active_virt_indices, inactive_virt_indices, rotation_params)

				# Compute t1 amplitudes
			t1_tensor = self._t1(mo_coeffs=mo_coeffs,
						 active_virt_indices=active_virt_indices,
						 spin_orbital_basis=False)

			# Step (v): Compute gradient and Hessian
			Grad, Hess = self._compute_gradient_hessian(mo_coeffs=mo_coeffs,
										 active_virt_indices=active_virt_indices,
										 inactive_virt_indices=inactive_virt_indices,
										 t1_tensor=t1_tensor)
			print(f"Gradient: {Grad}")
			print(f"Hessian: {Hess}")

			# Step (vi): Solve Newton-Raphson equations to get rotation parameters
			rotation_params = -np.linalg.solve(Hess, Grad)
			print(f"Rotation parameters: {rotation_params}")

			# Step (vii): Generate Unitary rotation matrix U
			#rotation_unitary = self._compute_rotation_unitary(active_virt_indices,
			#												 inactive_virt_indices,
			#												 rotation_params)
			#print(f"Unitary rotation matrix: {rotation_unitary}")

			# Step (viii): Construct Fock matrix for the rotated active space (occupied + active virtuals)
				# and diagonalize the Fock matrix to generate new canonical active orbitals
			#mo_coeffs = self._diagonalize_fock_matrix(self,rotation_unitary)

			# Step (ix): Calculate MP2 correlation energy with new canonical active orbitals
				# If energy is converged, exit loop
			#if converged:
			#	break

				# Else, go back to step (iv)

		return None


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
run_OVOS.run_OVOS()






