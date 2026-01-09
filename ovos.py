"""
OVOS class

The OVOS algorithm minimizes the second-order correlation energy (MP2)
using orbital rotations.
"""

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
		#print(self.active_inactive_indices)
		#print(self.active_inocc_indices)
		#print(int(self.num_opt_virtual_orbs+self.nelec))

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

		# Print information about the spaces
		print()
		print("#### Active and inactive spaces ####")
		print("Total number of spin-orbitals: ", self.tot_num_spin_orbs)
		print("Active occupied spin-orbitals: ", self.active_occ_indices)
		print("Active unoccupied spin-orbitals: ", self.active_inocc_indices)
		print("Inactive unoccupied spin-orbitals: ", self.inactive_indices)
		print()

		# Also print number of orbital coefficients, R_EA

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



		def t1(I,J,A,B) -> float:
			#MP1 amplitudes:
			t1 = -1.0*( (eri_spin[A,I,B,J] - eri_spin[A,J,B,I]) 
			/ (orbital_energies[A] + orbital_energies[B] 
			- orbital_energies[I] - orbital_energies[J]) )
			return t1

		J_2 = 0
		for (I, J) in self.active_occ_indices_valid:
			first_term = 0
			for (A, B) in self.active_inocc_indices_valid:
				for (C, D) in self.active_inocc_indices_valid:

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
			for (A, B) in self.active_inocc_indices_valid:
				second_term += 2*t1(I=I,J=J,A=A,B=B)*(eri_spin[A,I,B,J] - eri_spin[A,J,B,I])
				MP1_amplitudes[A,I,B,J] = t1(I=I,J=J,A=A,B=B)
			
			J_2 += first_term+second_term

		#MP2 = self.uhf.MP2().run()
		#assert np.abs(J_2 - MP2.e_corr) < 1e-6, "|J_2 - E_corr_MP2| < 1e-6 !!!"  
		return J_2, MP1_amplitudes, eri_spin, Fmo_spin
	
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


		def gradient(E: int, A: int, idx_A: int) -> float:
			#Equation 12a
			first_term = 0
			for (I, J) in self.active_occ_indices_valid:
				for B in self.active_inocc_indices:
					first_term += 2.0*MP1_amplitudes[A,I,B,J]*(eri_spin[E,I,B,J] - eri_spin[E,J,B,I])

			second_term = 0
			for idx_B, B in enumerate(self.active_inocc_indices):
				second_term += 2.0*D_AB_cache[idx_A, idx_B]*Fmo_spin[E,B]

			return first_term + second_term

		
		def hessian(E: int, A: int, F: int, B: int, idx_A: int, idx_B: int) -> float:
			#Equation 12b
			first_term = 0
			for (I, J) in self.active_occ_indices_valid:
				first_term += 2.0*MP1_amplitudes[A,I,B,J]*(eri_spin[E,I,F,J] - eri_spin[E,J,F,I])

			second_term = 0
			D_AB = D_AB_cache[idx_A, idx_B]
			
			for (I, J) in self.active_occ_indices_valid:
				for C in self.active_inocc_indices:
					if E==F:
						second_term +=-1.0*(MP1_amplitudes[A,I,B,J]*(eri_spin[B,I,C,J] - eri_spin[B,J,C,I]) 
							+ MP1_amplitudes[C,I,B,J]*(eri_spin[C,I,A,J] - eri_spin[C,J,A,I])
							+ D_AB*(Fmo_spin[A,A] - Fmo_spin[B,B])
							- D_AB*Fmo_spin[E,F])

					second_term += D_AB*Fmo_spin[E,F]

			return first_term + second_term


		# build the matrices (gradient and Hessian)
		idx = 0
		G = np.zeros((len(self.active_inocc_indices)*len(self.inactive_indices)))
		for E in self.inactive_indices:
			for idx_A, A in enumerate(self.active_inocc_indices):
				G[idx] = gradient(E, A, idx_A)
				idx += 1

		H = np.zeros((len(G), len(G)))
		idx = 0
		
		for i_E, E in enumerate(self.inactive_indices):
			for idx_A, A in enumerate(self.active_inocc_indices):
				idx1 = i_E * len(self.active_inocc_indices) + idx_A
				
				for i_F, F in enumerate(self.inactive_indices):
					for idx_B, B in enumerate(self.active_inocc_indices):
						idx2 = i_F * len(self.active_inocc_indices) + idx_B
						
						# Only compute upper triangle (including diagonal)
						if idx2 >= idx1:
							H[idx1, idx2] = hessian(E, A, F, B, idx_A, idx_B)
		
		# Mirror to lower triangle (Hessian is symmetric)
		H = H + H.T - np.diag(np.diag(H))
			
		# Step (vi): Use the Newton-Raphson method to minimize the second-order Hylleraas functional

		# solve for rotation parameters
			# equation 14
		R = -1.0*G@np.linalg.inv(H)

		# build rotation matrix
		idx = 0
		R_matrix = np.zeros((len(G),len(G)))
		for i in range(len(self.inactive_indices)):
			for j in range(len(self.active_inocc_indices)):
				R_matrix[i,j+len(self.active_inocc_indices)] = -1.0*R[idx]
				R_matrix[j+len(self.active_inocc_indices),i] = -1.0*R_matrix[i,j+len(self.active_inocc_indices)]

				idx += 1

		# Check that R_matrix is anti-symmetric
		assert np.allclose(R_matrix + R_matrix.T, 0), "R_matrix is not anti-symmetric"

		# Step (vii): Construct the unitary orbital rotation matrix U = exp(R)

		U = scipy.linalg.expm(R_matrix)

		# Check that U is orthogonal
		assert np.allclose(U@U.T, np.eye(len(U))), "U is not orthogonal"
		
		# Step (viii): Rotate the orbitals

		# rotate orbitals, mo_coeffs (6,6), (6,6) --> (12,12)
			# convert to spin orbital basis
		mo_coeffs_spin = spatial2spin([mo_coeffs[0], mo_coeffs[1]],orbspin=self.orbspin)
			# rotate
		mo_coeffs_spin_new = mo_coeffs_spin@U
			# convert back to spatial orbital basis
		mo_coeffs_rot = spin2spatial(mo_coeffs_spin_new, orbspin=self.orbspin)
		
		return mo_coeffs_rot


	
	def run_ovos(self,  mo_coeffs):
		"""
		Run the OVOS algorithm.
		"""

		converged = False
		max_iter = 100000
		iter = 0

		while not converged and iter < max_iter:
			iter += 1
			print("#### OVOS Iteration ", iter, " ####")
			
			E_corr, MP1_amplitudes, eri_spin, Fmo_spin = self.MP2_energy(mo_coeffs = mo_coeffs)
			print("MP2 correlation energy: ", E_corr)

			# Step (ix): check convergence
			# convergence criterion: change in correlation energy < 1e-6 Hartree
			if iter > 1:
				if np.abs(E_corr - lst_E_corr[-1]) < 1e-6:
					converged = True
					print("OVOS converged in ", iter, " iterations.")
				else:
					lst_E_corr.append(E_corr)
			else:
				lst_E_corr = []
				lst_E_corr.append(E_corr)

			# If MP2 goes positive, stop the optimization
			if E_corr > 0:
				print("Warning: MP2 correlation energy is positive. Stopping OVOS optimization.")
				break

			mo_coeffs = self.orbital_optimization(mo_coeffs, MP1_amplitudes=MP1_amplitudes, eri_spin=eri_spin, Fmo_spin=Fmo_spin)

		if not converged:
			print("OVOS did not converge in ", max_iter, " iterations.")

		return lst_E_corr, iter
	
	
		



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
# run_OVOS = OVOS(mol=mol, num_opt_virtual_orbs=6)
# run_OVOS.run_ovos(mo_coeff)

# Calculate the full space MP2 correlation energy for reference
mp2_full = OVOS(mol=mol, num_opt_virtual_orbs=8).MP2_energy(mo_coeffs=mo_coeff)[0]
print("Full space MP2 correlation energy: ", mp2_full)
print("")	




"""
Run OVOS algorithm for N cycles and store MP2 correlation energy convergence data.
"""

import time

lst_E_corr_cycle = []
iter_conv_cycle = []

cycle_max = 0 # N = 100
cycle_max_run = cycle_max

start_time = time.time()

for cycle in range(cycle_max_run):
	print("")
	print("#### OVOS Cycle ", cycle+1, " ####")

	# You can change the number of optimized virtual orbitals here
	num_opt_virtual_orbs = 6
	run_OVOS = OVOS(mol=mol, num_opt_virtual_orbs=num_opt_virtual_orbs)

	lst_E_corr, iter_conv = run_OVOS.run_ovos(mo_coeff)
	
	# If the last cycle's correlation energy converges to a positive value, skip storing the data
	if lst_E_corr[-1] > 0:
		print("Warning: OVOS converged to a positive MP2 correlation energy. Skipping data storage for this cycle.")
		print("")

		# Do a new cycle still keeping the max number of cycles the same
		cycle_max_run += 1
	else:
		lst_E_corr_cycle.append(lst_E_corr)
		iter_conv_cycle.append(iter_conv)

elapsed_time = time.time() - start_time
minutes = elapsed_time / 60
print(f"Cycle {len(lst_E_corr_cycle)} completed in {minutes:.2f} min. ({elapsed_time:.2f} sec.)")
print("")

# Times taken for full cycles:
# cycle_max = 100 --> 73.63 minutes (No optimizations)
# cycle_max = 100 --> ... minutes (With optimizations, ofc. randomness affects times)



"""
Time profiling
"""
time_profile = True
if time_profile == True and cycle_max == 0:
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
Save data to JSON files
"""
save_data = False
if save_data == True:
	import json

	cycle_max_str = str(cycle_max)

	# Save iteration convergence data
	with open("branch/data/iter_conv_cycle_"+cycle_max_str+".json", "w") as f:
		json.dump(iter_conv_cycle, f, indent=2)

	# Save MP2 correlation energy convergence data
	with open("branch/data/lst_E_corr_cycle_"+cycle_max_str+".json", "w") as f:
		json.dump(lst_E_corr_cycle, f, indent=2)

	print("Data saved to branch/data/iter_conv_cycle.json and branch/data/lst_E_corr_cycle.json")


