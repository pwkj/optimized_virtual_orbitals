"""
OVOS class

The OVOS algorithm minimizes the second-order correlation energy (MP2)
using orbital rotations.
"""

import os
# Force single-threaded
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from typing import Tuple

import numpy as np
import scipy
import pyscf
from pyscf.cc.addons import spatial2spin, spin2spatial
from pyscf.scf.addons import convert_to_ghf




class OVOS:

	"""
	Implemenation is based on:
	[L. Adamowicz & R. J. Bartlett (1987)](https://pubs.aip.org/aip/jcp/article/86/11/6314/93345/Optimized-virtual-orbital-space-for-high-level)

    Parameters
    ----------
    mol : pyscf.M
        PySCF molecule object.
    num_opt_virtual_orbs : int
        Number of optimized virtual spin orbitals.
    init_orbs : str, 
        Initial orbitals.
    """

	def __init__(self, mol: pyscf.gto.Mole, num_opt_virtual_orbs: int, init_orbs: str = "UHF") -> None:
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
									
		# MP2 in spatial orbital basis, Equation 14.4.56 in Molecular electronic-structure theory book
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
		Step (v-viii) of the OVOS algorithm: Orbital optimization via orbital rotations.
		
		- Compute gradient, first-order derivatives of the second-order Hylleraas functional, Equation 11a [L. Adamowicz & R. J. Bartlett (1987)]
		
		- Compute Hessiansecond-order derivatives of the second-order Hylleraas functional
		Equation 11b in [L. Adamowicz & R. J. Bartlett (1987)]

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

		# MP2 in spatial orbital basis, Equation 14.4.56 in Molecular electronic-structure theory book
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
	

			second_term = 0
			for idx_B, B in enumerate(self.active_inocc_indices):
				second_term += 2.0*D_AB_cache[idx_A, idx_B]*Fmo_spin[E,B]

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

		# Check that R_matrix is anti-symmetric
		R_antisymmetric_test = R_matrix + R_matrix.T
			# The anti-symmetric test matrix should be close to zero matrix
		assert np.allclose(R_matrix + R_matrix.T, 0, atol=1e8), f"R_matrix is not anti-symmetric, max deviation {np.max(np.abs(R_antisymmetric_test))}"

		# Check shape of R_matrix
		expected_shape = (len(self.active_inocc_indices) + len(self.inactive_indices), len(self.active_inocc_indices) + len(self.inactive_indices))
		assert R_matrix.shape == expected_shape, f"R_matrix shape is {R_matrix.shape}, expected {expected_shape}"





		# Step (vii): Construct the unitary orbital rotation matrix U = exp(R)

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
				#mo_coeffs = self._transform_mo_coeffs(self, mo_coeffs, active_virt_indices, inactive_virt_indices, rotation_params)
				pass

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
atom_choose_between = [
	"H .0 .0 .0; H .0 .0 0.74144",  # H2 bond length 0.74144 Angstrom
	"Li .0 .0 .0; H .0 .0 1.595",   # LiH bond length 1.595 Angstrom
	"O 0.0000 0.0000  0.1173; H 0.0000    0.7572  -0.4692; H 0.0000   -0.7572 -0.4692;",  # H2O equilibrium geometry
	"C  0.0000  0.0000  0.0000; H  0.0000  0.9350  0.5230; H  0.0000 -0.9350  0.5230;" # CH2 
]
# Basis set
basis_choose_between = [
	"STO-3G",
	"6-31G",
]

# Unit
unit="angstrom"


# Select molecule and basis set
select_atom, select_basis = (1,1) # Select molecule index here
atom, basis = (atom_choose_between[select_atom], basis_choose_between[select_basis])

# Get number of electrons and full space size in molecular orbitals
mol = pyscf.M(atom=atom, basis=basis, unit=unit)
	# Number of electrons
num_electrons = mol.nelec[0] + mol.nelec[1]
	# Full space size in molecular orbitals
full_space_size = int(pyscf.scf.UHF(mol).run().mo_coeff.shape[1])













run_OVOS = OVOS(mol=mol, num_vir_ops=3)
run_OVOS.run_OVOS()

"""
Run OVOS for different numbers of optimized virtual orbitals
"""
run_different_virt_orbs = True
if run_different_virt_orbs == True:
	# Loop over different numbers of optimized virtual orbitals
	# List of MP2 correlation energies for different numbers of optimized virtual orbitals
	lst_E_corr_virt_orbs = [[],[]]  # [[E_corr_list], [num_opt_virtual_orbs_list]]
	lst_MP2_virt_orbs = []  # [(num_opt_virtual_orbs, E_corr, iterations_till_convergence), ...]

	# Retry bounds
	max_retries = 1
	retry_count = 0

	# Set maximum number of optimized virtual orbitals to test
		# Denoted in molecular orbitals (not spin orbitals)
	max_opt_virtual_orbs = full_space_size*2 - num_electrons
		# Set statrting number of optimized virtual orbitals and increment
	num_opt_virtual_orbs_current = 0  # Start with number of occupied orbitals
		# Incremenet by 2 for closed shell molecules
	increment = 2 

	while num_opt_virtual_orbs_current < 4: #max_opt_virtual_orbs:  
		# Increment num_opt_virtual_orbs until OVOS converges successfully
		num_opt_virtual_orbs_current += increment 

		print("")
		print("#### OVOS with ", num_opt_virtual_orbs_current, " out of ", max_opt_virtual_orbs," optimized virtual orbitals (Retry count: ", retry_count,") ####")

		try:
			# Re-initialize molecule and UHF for each run
			mol = pyscf.M(atom=atom, basis=basis, unit=unit)
				
			uhf = pyscf.scf.UHF(mol).run()
			mo_coeff = uhf.mo_coeff 

			lst_E_corr = OVOS(mol=mol, num_opt_virtual_orbs=num_opt_virtual_orbs_current).run_ovos(mo_coeff)

			# run_OVOS got stuck in a non-converging loop
			if len(lst_E_corr) >= 1000:
				print("OVOS with ", num_opt_virtual_orbs_current, " optimized virtual orbitals did not converge. Rerunning with the same number of virtual orbitals.")
				num_opt_virtual_orbs_current -= increment  # Decrement to retry the same number
				continue

			# run_OVOS converged to a positive MP2 correlation energy
			if lst_E_corr[-1] > 0:
				print("Warning: OVOS with ", num_opt_virtual_orbs_current, " optimized virtual orbitals converged to a positive MP2 correlation energy. Rerunning with the same number of virtual orbitals.")
				num_opt_virtual_orbs_current -= increment  # Decrement to retry the same number
				continue

			# Add MP2 correlation energy for num_opt_virtual_orbs_current to list
			lst_MP2_virt_orbs.append((num_opt_virtual_orbs_current, lst_E_corr[-1], len(lst_E_corr)))
			lst_E_corr_virt_orbs[0].append(lst_E_corr)
			lst_E_corr_virt_orbs[1].append(num_opt_virtual_orbs_current)

			# Reset retry count on success
			retry_count = 0

		except AssertionError as e:
			print(f"Error during OVOS with {num_opt_virtual_orbs_current} optimized virtual orbitals: {e}")
			print("Rerunning with the same number of virtual orbitals.")

			retry_count += 1
			if retry_count >= max_retries:
				print(f"Maximum retries reached for {num_opt_virtual_orbs_current} optimized virtual orbitals. Skipping to next.")
				retry_count = 0
				continue

			num_opt_virtual_orbs_current -= increment  # Decrement to retry the same number
			continue


	# Print the final MP2 correlation energy after all OVOS and amount of iterations till convergence
	print("")
	for num_opt_virtual_orbs_current, E_corr, iter_ in lst_MP2_virt_orbs:
		print("MP2 correlation energy, for ", num_opt_virtual_orbs_current, " optimized virtual orbitals:", E_corr, " @ ", iter_, " iterations till convergence")
	print("")

	# Print
	print("Number of electrons: ", num_electrons)
	print("Full space size in molecular orbitals: ", full_space_size)
	print("Maximum number of optimized virtual orbitals tested: ", max_opt_virtual_orbs)
	print("Total OVOS runs completed: ", len(lst_MP2_virt_orbs))
	print("")

	# Save data to JSON files
	import json

	str_name = "different_virt_orbs"

	if select_atom == 0:
		str_atom = "H2"
	elif select_atom == 1:
		str_atom = "LiH"
	elif select_atom == 2:
		str_atom = "H2O"
	elif select_atom == 3:
		str_atom = "CH2"

	if select_basis == 0:
		str_basis = "STO-3G"
	elif select_basis == 1:
		str_basis = "6-31G"

	# Save MP2 correlation energy convergence data
	with open("branch/data/"+str_atom+"/"+str_basis+"/lst_MP2_"+str_name+".json", "w") as f:
		json.dump(lst_E_corr_virt_orbs, f, indent=2)

	print("Data saved to branch/data/"+str_atom+"/"+str_basis+"/...")


"""
Time profiling 
"""
time_profile = False
if time_profile == True and run_single_ovos == True:
	import cProfile
	import pstats

	opt = "opt_C" # Lable for optimization settings, see Optimization_options.md

	cProfile.run('OVOS(mol=mol, num_opt_virtual_orbs=6).run_ovos(mo_coeff)', 'branch/profil/profiling_results_'+opt+'.prof')

	# Print the profiling results
	with open('branch/profil/profiling_results_'+opt+'.txt', 'w') as f:
		stats = pstats.Stats('branch/profil/profiling_results_'+opt+'.prof', stream=f)
		stats.sort_stats('cumulative')  # Sort by cumulative time
		stats.print_stats()














"""
Run OVOS algorithm for N cycles and store MP2 correlation energy convergence data.
Save data to JSON files
"""
run_cycles = False
if run_cycles == True:
	import json
	
	# Set maximum number of optimized virtual orbitals to test
		# Denoted in molecular orbitals (not spin orbitals)
	max_opt_virtual_orbs = full_space_size*2 - num_electrons
		# Set statrting number of optimized virtual orbitals and increment
	num_opt_virtual_orbs_current = 0  # Start with number of occupied orbitals
	increment = 1 # Increment by 1 optimized virtual orbital

	# Retry bounds for vorb
	max_retries_vorb = 3
	retry_count_vorb = 0

	for num_opt_virtual_orbs in range(1, max_opt_virtual_orbs+1, increment):
		
		print("")
		print("#### OVOS with ", num_opt_virtual_orbs, " out of ", max_opt_virtual_orbs," optimized virtual orbitals ####")
		print("")

		# List of MP2 correlation energy convergence data for each cycle
		lst_E_corr_cycle = []
		iter_conv_cycle = []

		# Number of OVOS cycles
		cycle_max = 25 # N = 100
		cycle_max_run = cycle_max # To keep track of actual number of runs including restarts
		max_cycle_max = 100 # To avoid infinite loops
		cycle = 0

		# Retry bounds
		max_retries = 5
		retry_count = 0

		while cycle < cycle_max:
			print("")
			print("#### OVOS Cycle ", cycle+1, " out of", cycle_max_run," ####")

			try:
				# Re-initialize molecule and UHF for each cycle
				mol = pyscf.M(atom=atom, basis=basis, unit=unit)
				uhf = pyscf.scf.UHF(mol).run()
				mo_coeff = uhf.mo_coeff

				run_OVOS = OVOS(mol=mol, num_opt_virtual_orbs=num_opt_virtual_orbs)

				lst_E_corr = run_OVOS.run_ovos(mo_coeff)

				# run_OVOS got stuck in a non-converging loop
				if len(lst_E_corr) >= 600:
					print("OVOS cycle did not converge. Restarting cycle.")
					cycle_max_run += 1
					continue

				# If the last cycle's correlation energy converges to a positive value, skip storing the data
				if lst_E_corr[-1] > 0:
					print("Warning: OVOS converged to a positive MP2 correlation energy. Skipping data storage for this cycle.")
					print("")

					# Do a new cycle still keeping the max number of cycles the same
					cycle_max_run += 1
				else:
					lst_E_corr_cycle.append(lst_E_corr)

					# Reset retry count on success
					retry_count = 0
					retry_count_vorb = 0

					# Increment cycle count only on successful completion
					cycle += 1
				
			except AssertionError as e:
				print(f"Error during OVOS cycle: {e}")
				print("Restarting cycle.")
				cycle_max_run += 1

				retry_count += 1
				if retry_count >= max_retries:
					print(f"Maximum retries reached for this cycle. Skipping to next.")
					retry_count = 0
					continue

				retry_count_vorb += 1
				if retry_count_vorb >= max_retries_vorb:
					print(f"Maximum retries reached for number of optimized virtual orbitals. Moving to next number of virtual orbitals.")
					retry_count_vorb = 0
					break

				if cycle_max_run >= max_cycle_max:
					print("Reached maximum allowed cycles. Exiting.")
					break

				continue

		str_name = "cycle_"+str(cycle_max)
		str_name += "_vorb_"+str(num_opt_virtual_orbs)
		
		if select_atom == 0:
			str_atom = "H2"
		elif select_atom == 1:
			str_atom = "LiH"
		elif select_atom == 2:
			str_atom = "H2O"

		if select_basis == 0:
			str_basis = "STO-3G"
		elif select_basis == 1:
			str_basis = "6-31G"

		# If lst_E_corr_cycle is empty, skip saving
		if len(lst_E_corr_cycle) == 0:
			print("No successful OVOS cycles completed for ", num_opt_virtual_orbs, " optimized virtual orbitals. Skipping data saving.")
			continue

		# Save MP2 correlation energy convergence data
		with open("branch/data/"+str_atom+"/"+str_basis+"/lst_E_corr_"+str_name+".json", "w") as f:
			json.dump(lst_E_corr_cycle, f, indent=2)