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
	The OVOS algorithm minimizes the second-order correlation energy (MP2) using orbital rotations. 

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
		self.tot_num_spin_orbs = int(2*self.mo_coeffs.shape[1])

		# Check that orbitals are unrestricted
		assert self.mo_coeffs[0].shape[1] == self.mo_coeffs[1].shape[1], "Number of alpha and beta orbitals must be equal for OVOS."
		
		# Number of electrons
		self.nelec = self.mol.nelec[0] + self.mol.nelec[1]

		# build spin orbital coefficients
			# [0,1,0,1,...] for alpha and beta spin orbitals
		self.orbspin = np.array([0,1]*self.tot_num_spin_orbs)

		# Build index lists of active and inactive spaces
		#I,J indices -> occupied spin orbitals
		self.active_occ_indices = [i for i in range(int(self.nelec))]
		#A, B indices -> inoccupied spin orbitals in active space
		self.active_inocc_indices = [i for i in range(self.active_occ_indices[-1]+1,int((self.num_opt_virtual_orbs+self.nelec)))]
		#actice + inactive space
		self.inactive_indices = [i for i in range(self.active_inocc_indices[-1]+1,int((self.tot_num_spin_orbs)))]
		
		# Check that the active spaces are correctly built
		for I in self.active_occ_indices:
			assert I < self.nelec, f"I={I} not less than number of electrons {self.nelec}"
		for A in self.active_inocc_indices:
			assert A >= self.nelec, f"A={A} not greater than or equal to number of electrons {self.nelec}"
			assert A < self.nelec + self.num_opt_virtual_orbs, f"A={A} not less than number of electrons + num_opt_virtual_orbs {self.nelec + self.num_opt_virtual_orbs}"
		for E in self.inactive_indices:
			assert E >= self.nelec + self.num_opt_virtual_orbs, f"E={E} not greater than or equal to number of electrons + num_opt_virtual_orbs {self.nelec + self.num_opt_virtual_orbs}"
			assert E < self.tot_num_spin_orbs, f"E={E} not less than total number of spin orbitals {self.tot_num_spin_orbs}"


		# Precompute valid I>J, A>B, C>D combinations to avoid redundant calculations
			# Not implemented yet!
		self.active_occ_indices_valid = [(I, J) for I in self.active_occ_indices for J in self.active_occ_indices if I > J]
		self.active_inocc_indices_valid = [(A, B) for A in self.active_inocc_indices for B in self.active_inocc_indices if A > B]
		self.inactive_indices_valid = [(E, F) for E in self.inactive_indices for F in self.inactive_indices if E > F]
			# This lets us transform nested loops like:
			# for A in self.active_inocc_indices:
			# 	for B in self.active_inocc_indices:
			# 		if A > B:
			# into:
			# for (A, B) in self.active_inocc_indices_valid:
			#
			# For different num_opt_virtual_orbs, see below:
			# num_opt_virtual_orbs = 2 → 1 pair (A,B) with A>B
			# num_opt_virtual_orbs = 3 → 3 pairs (A,B) with A>B
			# num_opt_virtual_orbs = 4 → 6 pairs (A,B) with A>B
			# num_opt_virtual_orbs = 5 → 10 pairs (A,B) with A>B
			# num_opt_virtual_orbs = 6 → 15 pairs (A,B) with A>B
			# num_opt_virtual_orbs = 7 → 21 pairs (A,B) with A>B

		# Check if valid indices are correctly built
		for (I, J) in self.active_occ_indices_valid:
			assert I in self.active_occ_indices, f"I={I} not in active_occ_indices"
			assert J in self.active_occ_indices, f"J={J} not in active_occ_indices"
			assert I > J, f"I={I} not greater than J={J}"


		# Print information about the spaces
		print()
		print("#### Active and inactive spaces ####")
		print("Total number of spin-orbitals: ", self.tot_num_spin_orbs)
		print("Active occupied spin-orbitals: ", self.active_occ_indices)
		print("Active unoccupied spin-orbitals: ", self.active_inocc_indices)
		print("Inactive unoccupied spin-orbitals: ", self.inactive_indices)
		print()



		# Check that the number of optimized virtual orbitals is not too large
		# S = mol.intor('int1e_ovlp')
		# cond = np.linalg.cond(S)
		# assert cond < 1e12, "Overlap matrix is ill-conditioned. Consider using a better basis set."



		# Check that the number of optimized virtual orbitals is not too large
		assert self.tot_num_spin_orbs >= self.num_opt_virtual_orbs+self.nelec, "Your space 'num_opt_virtual_orbs' is too large"  


	def MP2_energy(self, mo_coeffs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
		"""
		MP2 correlation energy for unrestricted orbitals 

		Returns
		-------
		E_corr : float
			MP2 correlation energy.
		t1_amplitudes : ndarray
			First-order MP amplitudes.
		eri_spin : ndarray
			Spin-orbital two-electron integrals.
		Fmo_spin : ndarray
			Spin-orbital Fock matrix.
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
		
		# Print orbital energies for checking
			# For orbital index i, energy is orbital_energies[i]
		# print("Orbital energies (before sorting): ", np.concatenate((np.real(eigval_a), np.real(eigval_b))))
			# Sorted orbital energies
		# print("Orbital energies (spin-orbital basis): ", orbital_energies)
			# For active occupied and inoccupied spaces
		# print("Active occupied orbital energies: ", [orbital_energies[i] for i in self.active_occ_indices])
		# print("Active unoccupied orbital energies: ", [orbital_energies[i] for i in self.active_inocc_indices])
		# print("Inactive unoccupied orbital energies: ", [orbital_energies[i] for i in self.inactive_indices])

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

		norb_total = norb_alpha + norb_beta
		#eri_spin = np.zeros((norb_total, norb_total, norb_total, norb_total))

		#See https://pyscf.org/_modules/pyscf/cc/addons.html#spatial2spin
		eri_spin = spatial2spin([eri_aaaa, eri_aabb, eri_bbbb], orbspin=None)
		Fmo_spin = spatial2spin([Fmo[0], Fmo[1]], orbspin=None)

		# Initialize MP1 amplitudes array
		MP1_amplitudes = np.zeros((norb_total, norb_total, norb_total, norb_total))

		# Define t1 function for MP1 amplitudes
		def t1(I,J,A,B) -> float:
			#MP1 amplitudes:
			t1 = -1.0*( (eri_spin[A,I,B,J]- eri_spin[A,J,B,I])
			/ (orbital_energies[A] + orbital_energies[B] 
			- orbital_energies[I] - orbital_energies[J]) )
			return t1 

		# Compute MP2 correlation energy
			# Equation 10: J_2 = ...
				# Term 1: \sum_{I>J} \sum_{A>B} \sum_{C>D} t_ABIJ t_CDIJ * ( F_AC \delta_BD + F_BD \delta_BC - F_AD \delta_BC - F_BC \delta_AD - (e_I + e_J) (\delta_AC \delta_BD - \delta_AD \delta_BC) )
				# Term 2: \sum_{I>J} \sum_{A>B} 2 t_ABIJ ( <AI|BJ> - <AJ|BI> )

		J_2 = 0
		for (I, J) in self.active_occ_indices_valid:
			first_term = 0
			for (A, B) in self.active_inocc_indices_valid:
				for (C, D) in self.active_inocc_indices_valid:

					t_abij = t1(I=I,J=J,A=A,B=B)
					t_cdij = t1(I=I,J=J,A=C,B=D)

					if B==D: # delta_BD
						first_term += t_abij*t_cdij*Fmo_spin[A,C]
					if A==C: # delta_AC
						first_term += t_abij*t_cdij*Fmo_spin[B,D]
					if B==C: # delta_BC
						first_term += -1.0*t_abij*t_cdij*Fmo_spin[A,D]
					if A==D: # delta_AD
						first_term += -1.0*t_abij*t_cdij*Fmo_spin[B,C]
					if A==C and B==D: # delta_AC delta_BD
						first_term += -1.0*t_abij*t_cdij*(
							orbital_energies[I] + orbital_energies[J])
					if A==D and B==C: # delta_AD delta_BC
						first_term += t_abij*t_cdij*(
							orbital_energies[I] + orbital_energies[J])

			second_term = 0
			for (A, B) in self.active_inocc_indices_valid:
				second_term += 2*t1(I=I,J=J,A=A,B=B)*(eri_spin[A,I,B,J]- eri_spin[A,J,B,I])
				MP1_amplitudes[A,I,B,J] = t1(I=I,J=J,A=A,B=B)
			
			J_2 += first_term+second_term

		# MP2 = self.uhf.MP2().run()
		# assert np.abs(J_2 - MP2.e_corr) < 1e-6, "|J_2 - E_corr_MP2| < 1e-6 !!!"  

		# Check that MP1_amplitudes is not all zeros
		assert not np.allclose(MP1_amplitudes, 0), "MP1_amplitudes is all zeros!"
		# Check shape of MP1_amplitudes
		expected_shape = (norb_total, norb_total, norb_total, norb_total)
		assert MP1_amplitudes.shape == expected_shape, f"MP1_amplitudes shape is {MP1_amplitudes.shape}, expected {expected_shape}"
		# Check if MP1_amplitudes has NaN or Inf values
		assert np.all(np.isfinite(MP1_amplitudes)), "MP1_amplitudes contains NaN or Inf values"
		# Check if MP1_amplitudes values are within reasonable range
		max_amp = np.max(np.abs(MP1_amplitudes))
		assert max_amp < 1e3, f"MP1_amplitudes has unreasonably large values: max amplitude = {max_amp}"


		# Check that eri_spin is not all zeros
		assert not np.allclose(eri_spin, 0), "eri_spin is all zeros!"
		# Check shape of eri_spin
		assert eri_spin.shape == expected_shape, f"eri_spin shape is {eri_spin.shape}, expected {expected_shape}"


		# Check that Fmo_spin is not all zeros
		assert not np.allclose(Fmo_spin, 0), "Fmo_spin is all zeros!"
		# Check shape of Fmo_spin
		expected_shape_Fmo = (norb_total, norb_total)
		assert Fmo_spin.shape == expected_shape_Fmo, f"Fmo_spin shape is {Fmo_spin.shape}, expected {expected_shape_Fmo}"


		# Ensure float64 for numerical stability
		J_2 = np.float64(J_2)
		MP1_amplitudes = MP1_amplitudes.astype(np.float64)
		eri_spin = eri_spin.astype(np.float64)
		Fmo_spin = Fmo_spin.astype(np.float64)

		return J_2, MP1_amplitudes, eri_spin, Fmo_spin, orbital_energies
	





	def orbital_optimization(self, mo_coeffs, MP1_amplitudes, eri_spin, Fmo_spin) -> np.ndarray:

		"""
		Step (v-viii) of the OVOS algorithm: Orbital optimization via orbital rotations.
		
		- Compute gradient, first-order derivatives of the second-order Hylleraas functional, Equation 11a [L. Adamowicz & R. J. Bartlett (1987)]
		
		- Compute Hessiansecond-order derivatives of the second-order Hylleraas functional
		Equation 11b in [L. Adamowicz & R. J. Bartlett (1987)]

		- Use the Newton-Raphson method to minimize the second-order Hylleraas functional, Equations 14 in [L. Adamowicz & R. J. Bartlett (1987)]

		- Construct the unitary orbital rotation matrix U = exp(R), Equation 15 in [L. Adamowicz & R. J. Bartlett (1987)]

		First- and second-order derivatives of the second-order Hylleraas functional
		Equations 11a and 11b in https://pubs.aip.org/aip/jcp/article/86/11/6314/93345/Optimized-virtual-orbital-space-for-high-level
		"""

		# Step (v): Compute the gradient and Hessian of the second-order Hylleraas functional
		
		# Precompute D_AB values for all A,B in active_inocc_indices
			# Equation 13: D_AB^(2) = \sum_{I>J} \sum_C t_ABIJ t_CBJI
		# This avoids recalculating the same D_AB values multiple times in hessian()
		n_active_inocc = len(self.active_inocc_indices)
		D_AB_cache = np.zeros((n_active_inocc, n_active_inocc))
		
		for idx_A, A in enumerate(self.active_inocc_indices):
			for idx_B, B in enumerate(self.active_inocc_indices):
				D = 0.0
				for (I, J) in self.active_occ_indices_valid:
					for C in self.active_inocc_indices:
						D += MP1_amplitudes[A,I,C,J] * MP1_amplitudes[B,I,C,J]
				D_AB_cache[idx_A, idx_B] = D

		# Ensure D_AB_cache is float64 for numerical stability
		D_AB_cache = D_AB_cache.astype(np.float64)


		def gradient(E: int, A: int, idx_A: int) -> float:
			#Equation 12a: G_EA = ...
				# Term 1: 2 \sum_{I>J} \sum_B t_ABIJ ( <EI|BJ> - <EJ|BI> )
				# Term 2: + 2 \sum_B D_AB F_EB

			first_term = 0
			for (I, J) in self.active_occ_indices_valid:
				for B in self.active_inocc_indices:
					first_term += 2.0*MP1_amplitudes[A,I,B,J]*(eri_spin[E,I,B,J] - eri_spin[E,J,B,I])

			second_term = 0
			for idx_B, B in enumerate(self.active_inocc_indices):
				second_term += 2.0*D_AB_cache[idx_A, idx_B]*Fmo_spin[E,B]

			return first_term + second_term

		
		def hessian(E: int, A: int, F: int, B: int, idx_A: int, idx_B: int) -> float:
			#Equation 12b: H_EA,FB = ...
				# Term 1: 2 \sum_{I>J} t_ABIJ ( <EI|FJ> - <EJ|FI> )
				# Term 2: - 1.0 * \sum_{I>J} \sum_C [ t_ACIJ ( <BI|CJ> - <BJ|CI> ) + t_CBIJ ( <CI|AJ> - <CJ|AI> ) ] \delta_EF
				# Term 3: + D_AB ( F_AA - F_BB ) \delta_EF 
				# Term 4: + D_AB F_EF
				# Term 5: - 1.0 * D_AB F_EF \delta_EF

			first_term = 0
			for (I, J) in self.active_occ_indices_valid:
				first_term += 2.0*MP1_amplitudes[A,I,B,J]*(eri_spin[E,I,F,J] - eri_spin[E,J,F,I])

			D_AB = D_AB_cache[idx_A, idx_B]

			second_term = 0			
			for (I, J) in self.active_occ_indices_valid:
				for C in self.active_inocc_indices:
					if E==F:
						second_term +=-1.0*(
							  MP1_amplitudes[A,I,C,J]*(eri_spin[B,I,C,J] - eri_spin[B,J,C,I]) 
							+ MP1_amplitudes[C,I,B,J]*(eri_spin[C,I,A,J] - eri_spin[C,J,A,I])
						)
			
			third_term = 0
			fifth_term = 0
			if E==F:
				third_term += D_AB*(Fmo_spin[A,A] - Fmo_spin[B,B])
				fifth_term += -1.0*D_AB*Fmo_spin[E,F]

			forth_term = 0
			forth_term = D_AB*Fmo_spin[E,F]

			return first_term + second_term + third_term + forth_term + fifth_term
		



		# Step (vi): Use the Newton-Raphson method to minimize the second-order Hylleraas functional

		# solve for rotation parameters
			# Original direct inversion method
				# equation 14: R = - G H^-1 
				# R = -G @ np.linalg.inv(H)

		use_RLE = True  # Set to True to use Reduced Linear Equation method
		if not use_RLE:
			# Build full gradient vector G and Hessian matrix H
			G = np.zeros((len(self.active_inocc_indices)*len(self.inactive_indices),))
			H = np.zeros((len(self.active_inocc_indices)*len(self.inactive_indices), len(self.active_inocc_indices)*len(self.inactive_indices)))
			for i_E, E in enumerate(self.inactive_indices):
				for idx_A, A in enumerate(self.active_inocc_indices):
					# Build G
					idx_G = i_E * len(self.active_inocc_indices) + idx_A
					G[idx_G] = gradient(E, A, idx_A)

					for j_F, F in enumerate(self.inactive_indices):
						for idx_B, B in enumerate(self.active_inocc_indices):
							# Build H
							idx_H_row = i_E * len(self.active_inocc_indices) + idx_A
							idx_H_col = j_F * len(self.active_inocc_indices) + idx_B
							H[idx_H_row, idx_H_col] = hessian(E, A, F, B, idx_A, idx_B)
			
			# Ensure float64 for numerical stability
			H = H.astype(np.float64)
			G = G.astype(np.float64)

			# Solve for R
			R = -G @ np.linalg.inv(H)
			#R = -scipy.linalg.solve(H.T, G.T).T

			# Check Hessian eigenvalues for stability
			eigvals = np.linalg.eigvalsh(H)
			num_neg_eigvals = np.sum(eigvals < 0)
				# If negative eigenvalues exist: 
				# Energy surface is saddle-shaped, not bowl-shaped. 
				# Newton-Raphson will go uphill in those directions, 
				# causing divergence.
				

			# Values for checking 
			norm_R = np.linalg.norm(R)
			norm_H = np.linalg.norm(H)
			norm_G = np.linalg.norm(G)

			# Number of eignevalues that are negative for E==F
			sum_eigvals_EqF = 0
			for i_E, E in enumerate(self.inactive_indices):
				# Build H_block directly
				H_block = np.zeros((len(self.active_inocc_indices), len(self.active_inocc_indices)))

				for idx_A, A in enumerate(self.active_inocc_indices):
					for idx_B, B in enumerate(self.active_inocc_indices):
						# Build H_block (only diagonal block H_{ea,eb} with E fixed)
						H_block[idx_A, idx_B] = hessian(E, A, E, B, idx_A, idx_B)
				
				# Ensure float64 for numerical stability
				H_block = H_block.astype(np.float64)

				# Check Hessian eigenvalues for stability
				eigvals_block = np.linalg.eigvalsh(H_block)
				num_neg_eigvals_block = np.sum(eigvals_block < 0)

				# Accumulate number of negative eigenvalues
				sum_eigvals_EqF += num_neg_eigvals_block

			# Print full hessian, if needed
			# if num_neg_eigvals > len(eigvals)//4 :
			# 	np.set_printoptions(precision=3, suppress=True)
			# 	print("Full Hessian H: \n", H)

			print(f"  Non-block: ||H_block||={norm_H:.2e} (Neg. eigval: {num_neg_eigvals}/{len(eigvals)} [{sum_eigvals_EqF}]), ||G_block||={norm_G:.2e}, ||R_block||={norm_R:.2e}")



		# Try the option of implementing the RLE for R,
			# as described in the paper, instead of direct inversion of H.
		if use_RLE:	
			# The reduced linear equation (RLE) method
				# Each H_{ea,eb} block dominates the contribution
			R = np.zeros((len(self.active_inocc_indices)*len(self.inactive_indices),))

			for i_E, E in enumerate(self.inactive_indices):
				
				# Build H_block and G_block directly
				H_block = np.zeros((len(self.active_inocc_indices), len(self.active_inocc_indices)))
				G_block = np.zeros((len(self.active_inocc_indices),))

				for idx_A, A in enumerate(self.active_inocc_indices):
					# Build G_block
					G_block[idx_A] = gradient(E, A, idx_A)

					for idx_B, B in enumerate(self.active_inocc_indices):
						# Build H_block (only diagonal block H_{ea,eb} with E fixed)
						H_block[idx_A, idx_B] = hessian(E, A, E, B, idx_A, idx_B)
				
				# Ensure float64 for numerical stability
				H_block = H_block.astype(np.float64)
				G_block = G_block.astype(np.float64)

				# Check Hessian eigenvalues for stability
				eigvals_block = np.linalg.eigvalsh(H_block)

					# Large R occurs when H_block is ill-conditioned or G_block is large
				# Check condition number of H_block
				cond_H_block = np.linalg.cond(H_block)
				# if cond_H_block > 100: # Catch ill-conditioning earlier
				# 	    # Stronger regularization for small spaces
				# 	if cond_H_block > 1000:
				# 		reg_param = 1e-4  # Strong for severe ill-conditioning
				# 	elif cond_H_block > 500:
				# 		reg_param = 1e-5  # Medium for small spaces
				# 	else:
				# 		reg_param = 1e-6  # Light

				# 	H_block += reg_param * np.eye(len(H_block))
				# 	print(f"Warning: Hessian block ill-conditioned for E={E} (κ={cond_H_block:.2e}), applying regularization of {reg_param:.1e}")

				# Solve for R_{ea}
				R_block = -G_block @ np.linalg.inv(H_block)
				
				# Typecast to float64 for numerical stability
				R_block = R_block.astype(np.float64)

				# limit maximum rotation per step
				# norm_R_block = np.linalg.norm(R_block)
				# max_rotation = 0.5  # Maximum allowed rotation magnitude per step
				# if norm_R_block > max_rotation:
				# 	R_block = R_block * (max_rotation / norm_R_block)
				# 	print(f"Warning: Large rotation for E={E} (||R_block||={norm_R_block:.2e}), limiting to {max_rotation:.2e}")

				# Get rotation magnitude
				norm_R_block = np.linalg.norm(R_block)

				# Get Hessian magnitude
				norm_H_block = np.linalg.norm(H_block)

				# Get if Gradient is close to zero vector
				norm_G_block = np.linalg.norm(G_block)

				# Print values for checking
				print(f"For E={E}: κ={cond_H_block:.2e}, ||H_block||={norm_H_block:.2e} (Neg. eignval: {np.sum(eigvals_block < 0)}/{len(eigvals_block)}), ||G_block||={norm_G_block:.2e}, ||R_block||={norm_R_block:.2e}")

				# Store in R
				for idx_A, A in enumerate(self.active_inocc_indices):
					idx_R = i_E * len(self.active_inocc_indices) + idx_A
					R[idx_R] = R_block[idx_A]			

		
		# Ensure float64 for numerical stability
		R = R.astype(np.float64)

		# Check that R is not NaN or Inf
		assert np.all(np.isfinite(R)), "Rotation parameters contain NaN or Inf values"
		
		# Check shape of R
		expected_R_shape = (len(self.active_inocc_indices)*len(self.inactive_indices),)
		assert R.shape == expected_R_shape, f"R shape is {R.shape}, expected {expected_R_shape}"

		# Check norm of R
		norm_R = np.linalg.norm(R)
		if norm_R > 1e2:
			print(f"Warning: Large rotation (||R||={norm_R:.2e}), possible divergence")

		# build rotation matrix
		idx = 0
		R_matrix = np.zeros((len(self.active_inocc_indices)+len(self.inactive_indices), len(self.active_inocc_indices)+len(self.inactive_indices)))
		for E in range(len(self.inactive_indices)):
			for A in range(len(self.active_inocc_indices)):
				R_matrix[E, len(self.inactive_indices) + A] = -1.0*R[idx]
				R_matrix[len(self.inactive_indices) + A, E] = -1.0*R_matrix[E, len(self.inactive_indices) + A]

				idx += 1

		# Check that R_matrix is anti-symmetric
		R_antisymmetric_test = R_matrix + R_matrix.T
			# The anti-symmetric test matrix should be close to zero matrix
		assert np.allclose(R_matrix + R_matrix.T, 0, atol=1e8), f"R_matrix is not anti-symmetric, max deviation {np.max(np.abs(R_antisymmetric_test))}"

		# Check shape of R_matrix
		expected_shape = (len(self.active_inocc_indices) + len(self.inactive_indices), len(self.active_inocc_indices) + len(self.inactive_indices))
		assert R_matrix.shape == expected_shape, f"R_matrix shape is {R_matrix.shape}, expected {expected_shape}"





		# Step (vii): Construct the unitary orbital rotation matrix U = exp(R)

		# Unitary rotation matrix
		U = scipy.linalg.expm(R_matrix)
	
		# Check shape of U, does this shape make sense?
		expected_shape = (len(self.active_inocc_indices) + len(self.inactive_indices), len(self.active_inocc_indices) + len(self.inactive_indices))
		assert U.shape == expected_shape, f"U shape is {U.shape}, expected {expected_shape}"

		# Check that U is orthogonal
			# Note, sometimes due to numerical errors (e-11) U@U.T is not exactly identity, but very close
				# If the difference is too large, raise an error
		diff = np.linalg.norm(U@U.T - np.eye(len(U)))
		assert diff < 1e-6, f"U is not orthogonal, ||U@U.T - I|| = {diff}"




		# Step (viii): Rotate the orbitals

		# rotate orbitals, mo_coeffs (6,6), (6,6) --> (12,12)
			# convert to spin orbital basis
		mo_coeffs_spin = spatial2spin([mo_coeffs[0], mo_coeffs[1]],orbspin=self.orbspin)
			# rotate only the active_inocc space oribtals not occupied or inactive
		mo_coeffs_spin_new = mo_coeffs_spin.copy()
		mo_coeffs_spin_new[len(self.active_occ_indices):, len(self.active_occ_indices):] = mo_coeffs_spin[len(self.active_occ_indices):, len(self.active_occ_indices):] @ U
			
		# check if rotation worked
		diff_rot = np.linalg.norm(mo_coeffs_spin_new - mo_coeffs_spin)
		if diff_rot > 1e2:
			print(f"Warning: Large orbital rotation (||C_new - C_old||={diff_rot:.2e}), possible divergence")


					# convert back to spatial orbital basis
		mo_coeffs_rot = spin2spatial(mo_coeffs_spin_new, orbspin=self.orbspin)
		
		return mo_coeffs_rot


	





	def run_ovos(self,  mo_coeffs):
		"""
		Run the OVOS algorithm.
		"""

		converged = False
		max_iter = 1000
		iter = 0

		while not converged and iter < max_iter:
			iter += 1
			print("#### OVOS Iteration ", iter, " ####")
			
			E_corr, MP1_amplitudes, eri_spin, Fmo_spin, E_orb = self.MP2_energy(mo_coeffs = mo_coeffs)
			print("MP2 correlation energy: ", E_corr)

			# Step (ix): check convergence
			# convergence criterion: change in correlation energy < 1e-6 Hartree
			if iter > 1:
				threshold = 1e-8 if len(self.active_inocc_indices) < 4 else 1e-6
				if np.abs(E_corr - lst_E_corr[-1]) < threshold:
					converged = True
					print("OVOS converged in ", iter, " iterations.")
				else:
					lst_E_corr.append(E_corr)
			else:
				lst_E_corr = []
				lst_E_corr.append(E_corr)

			# If MP2 goes positive, stop the optimization
			#if E_corr > 0:
				#print("Warning: MP2 correlation energy is positive. Stopping OVOS optimization.")
				#break

			mo_coeffs = self.orbital_optimization(mo_coeffs, MP1_amplitudes=MP1_amplitudes, eri_spin=eri_spin, Fmo_spin=Fmo_spin)

		# Print information about the spaces
		print()
		print("#### Active and inactive spaces ####")
		print("Total number of spin-orbitals: ", self.tot_num_spin_orbs)
		print("Active occupied spin-orbitals: ", self.active_occ_indices)
		print("Active unoccupied spin-orbitals: ", self.active_inocc_indices)
		print("Inactive unoccupied spin-orbitals: ", self.inactive_indices)
		print()

		# Check orbital near-degeneracies in active_inocc space
		E_orb_active_inocc = [E_orb[i] for i in self.active_inocc_indices]
		E_diff = np.abs(np.diff(sorted(E_orb_active_inocc)))
		min_diff = np.min(E_diff) if len(E_diff) > 0 else None
		if min_diff is not None and min_diff < 1e-4:
			print(f"Warning: Near-degeneracy detected in active unoccupied orbitals, minimum energy difference = {min_diff:.2e} Hartree")
			print()

		# Check orbital near-degeneracies in inactive space
		E_orb_inactive = [E_orb[i] for i in self.inactive_indices]
		E_diff_inactive = np.abs(np.diff(sorted(E_orb_inactive)))
		min_diff_inactive = np.min(E_diff_inactive) if len(E_diff_inactive) > 0 else None
		if min_diff_inactive is not None and min_diff_inactive < 1e-4:
			print(f"Warning: Near-degeneracy detected in inactive unoccupied orbitals, minimum energy difference = {min_diff_inactive:.2e} Hartree")
			print()
		
		# Return list of correlation energies
		# print("List of MP2 correlation energies during OVOS optimization, w. length ", len(lst_E_corr) ,": ", lst_E_corr)

		# Check if OVOS converged
		if not converged:
			assert False, "OVOS did not converge within the maximum number of iterations."

		return lst_E_corr
	
	
	
		
# CTRL + ' (thingy under ENTER) to comment/uncomment multiple lines







"""
Objective:
- Get LiH STO-3G molecule working with OVOS at different numbers of optimized virtual orbitals
	* Point might be to instead solve the Newton-Raphson equation for the orbital rotation parameters R
		with a different method than direct inversion of the Hessian, e.g. conjugate gradient
		The other proposed method in the paper is:
			"A block generalization of the reduced linear equation 
			technique of Purvis and Bartlett is used. 
			It is recognized that the Hessian matrix is dominated 
			by square blocks located diagonally. A particular block is 
			related to the rotation of all possible active orbitals with one 
			nonactive orbital (Hea,eb)' Only inverses of all diagonally 
			located blocks are kept in the core."
		It is called the "reduced linear equation (RLE) method" in the following paper:
			The reduced linear equation method in coupled cluster theory,
				by George D. Purvis III and Rodney J. Bartlett,
				at Battelle Columbus Laboratories. Columbus, Ohio 43201 
				(Received 24 September 1980; accepted 14 April 1981) 


"""








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