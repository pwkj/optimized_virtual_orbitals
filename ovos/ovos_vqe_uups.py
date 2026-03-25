"""
Try OVOS with VQE and see if it can find the ground state energy of a molecule.
"""
    # Math and general imports
import numpy as np
import time
    # SlowQuant imports
import slowquant.SlowQuant as sq
from slowquant.unitary_coupled_cluster.unrestricted_ups_wavefunction import UnrestrictedWaveFunctionUPS
    # Pyscf imports
from pyscf import gto, scf
    # OVOS imports
from ovos import OVOS


# # Qiskit
#     # Qiskit imports
# from qiskit_aer import AerSimulator
# from qiskit_ibm_runtime import SamplerV2 as SamplerV2IBM
# from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
#     # Qiskit Setup
# aer = AerSimulator()
# sampler = SamplerV2IBM(mode=aer)
# mapper = ParityMapper(num_particles=(1, 1))
#     # Initialize Quantum Interface
# QI = QuantumInterface(sampler, "fUCCSD", mapper, shots=10)

# To get iterations and energy from SlowQuant
import sys
import io
import re

class Dee:
    """Write to multiple streams simultaneously."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()

class Tee(io.StringIO):
    """
    A writeable stream that writes to both an original stream (e.g. the terminal)
    and to an internal StringIO buffer.
    """
    def __init__(self, original_stream):
        super().__init__()
        self.original_stream = original_stream

    def write(self, s):
        # Write to the buffer (for later retrieval)
        super().write(s)
        # Write to the original stream (the terminal) and flush immediately
        self.original_stream.write(s)
        self.original_stream.flush()

    def flush(self):
        super().flush()
        self.original_stream.flush()
def run_ucc_and_get_stats(wf, str_, orbital_optimization):
    # Save the original streams
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Create tee streams that write to both the terminal and a buffer
    tee_stdout = Tee(original_stdout)
    tee_stderr = Tee(original_stderr)

    # Replace the system streams with our tees
    sys.stdout = tee_stdout
    sys.stderr = tee_stderr

    try:
        # Run the optimization – output will appear on the terminal AND be captured
        
        # Run the optimization – output will appear on the terminal AND be captured
        wf.run_wf_optimization_1step(str_, orbital_optimization=orbital_optimization, tol=1e-6, maxiter=5000)
    finally:
        # Restore the original streams
        sys.stdout = original_stdout
        sys.stderr = original_stderr

    # Combine the captured output from both streams
    output = tee_stdout.getvalue() + tee_stderr.getvalue()

    # --- Now parse the output exactly as before ---
    lines = output.splitlines()
    stats = {
        'iterations': None,
        'function_evaluations': None,
        'gradient_evaluations': None,
        'final_energy': None,
        'success': None,
    }

    for line in lines:
        line_stripped = line.strip()

        if 'Optimization terminated successfully' in line_stripped:
            stats['success'] = True
        elif 'Optimization failed.' in line_stripped or 'Unsuccessful' in line_stripped:
            stats['success'] = False

        match = re.search(r'Current function value:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line_stripped)
        if match:
            stats['final_energy'] = float(match.group(1))

        match = re.search(r'Iterations:\s*(\d+)', line_stripped)
        if match:
            stats['iterations'] = int(match.group(1))

        match = re.search(r'Function evaluations:\s*(\d+)', line_stripped)
        if match:
            stats['function_evaluations'] = int(match.group(1))

        match = re.search(r'Gradient evaluations:\s*(\d+)', line_stripped)
        if match:
            stats['gradient_evaluations'] = int(match.group(1))

    # Fallback to progress table if summary missing
    if stats['iterations'] is None:
        iteration_numbers = []
        for line in lines:
            if '|' in line and 'Iteration #' not in line:
                parts = line.split('|')
                if len(parts) >= 1:
                    first_col = parts[0].strip().lstrip('-').strip()
                    try:
                        iteration_numbers.append(int(first_col))
                    except ValueError:
                        pass
        if iteration_numbers:
            stats['iterations'] = max(iteration_numbers)

    return stats



def VQE_OVOS(atom, basis, num_opt_virtual_orbs, include_active_kappa):
    molecule = str(atom.split()[0])  # Get the first element symbol for naming
    if molecule == "H":
        molecule = "HF"
    elif molecule == "C":
        molecule = "CO"
    elif molecule == "N":
        molecule = "NH3"
    elif molecule == "O":
        molecule = "H2O"

    with open(f"branch/data/{molecule}/{basis}/VQE/OVOS_{molecule}_{basis}_VQE_opt_num_{num_opt_virtual_orbs}_{include_active_kappa}_output.txt", "w") as f:
        sys.stdout = Dee(sys.__stdout__, f)
        try:
            print (f"\nRunning OVOS with VQE optimization for {atom} in basis {basis} with {num_opt_virtual_orbs*100:.0f}% active virtual orbitals and include_active_kappa = {include_active_kappa}...")

            # Water molecule, minimal basis
            mol = gto.Mole()
            mol.atom = atom
            mol.basis = basis
            mol.unit = 'Angstrom'
            mol.spin = 0
            mol.charge = 0
            mol.symmetry = False
            mol.verbose = 0
            mol.build()
                # Get one- and two-electron integrals
            h_core = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
            g_eri = mol.intor("int2e")

                # Number of electrons and orbitals
            num_electrons = mol.nelectron
            num_orbitals = mol.nao_nr() 
            print(f"Number of electrons: {num_electrons}, Number of orbitals: {num_orbitals}")

            # Create OVOS object and run
                # Start Timer OVOS
            start_ovos = time.time()

                # RHF reference for OVOS
            mf = scf.RHF(mol)
            mf.verbose = 0
            mf.kernel()

                # Initial data (RHF orbitals)
            Fao = [mf.get_fock(), mf.get_fock()]
            mo_coeffs = [mf.mo_coeff, mf.mo_coeff]
                    # Check if mo_coeffs are unrestricted or restricted
            if np.isclose(mo_coeffs[0], mo_coeffs[1], atol=1e-12).all():
                print("Initial MO coefficients are restricted (RHF-like).")
            else:
                print("Initial MO coefficients are unrestricted (UHF-like).")

                # Set up OVOS
            num_opt_virtual_orbs = int(num_opt_virtual_orbs * (num_orbitals - num_electrons//2))  # Convert fraction to actual number of orbitals
            print(f"Optimizing {num_opt_virtual_orbs} active virtual orbitals (out of {num_orbitals - num_electrons//2} total virtual orbitals).")
            ovos = OVOS(
                mol=mol,
                scf=mf,
                Fao=Fao,
                num_opt_virtual_orbs=num_opt_virtual_orbs*2,      # active virtual spin‑orbitals
                mo_coeff=mo_coeffs,
                init_orbs="RHF",
                verbose=1,
                max_iter=1000,
                conv_energy=1e-8,
                conv_grad=1e-4,
                keep_track_max=50
            )
                # Run OVOS
            E_corr, E_corr_hist, E_corr_iter, E_corr_mo, E_corr_fock, stop_reason = ovos.run(mo_coeffs, fock_spin=None)
            E_corr = E_corr[-1]  # Final correlation energy
            E_tot = E_corr + mf.e_tot
            print(f"\nOptimization finished. Final MP2 energy = {E_tot} Hartree. (ΔE_corr = {E_corr} Hartree)")
            # Check if mo_coeffs are unrestricted or restricted
            if np.isclose(E_corr_mo[0], E_corr_mo[1], atol=1e-12).all():
                print("OVOS optimized orbitals are restricted (RHF-like).")
            else:
                print("OVOS optimized orbitals are unrestricted (UHF-like).")
            print()

            # SlowQuant
                # Initialize for UPS wave function with OVOS-optimized orbitals
            WF_ovos = UnrestrictedWaveFunctionUPS(
                mol.nelectron,
                ((mol.nelectron//2 + 1,mol.nelectron//2 - 1), num_electrons//2+num_opt_virtual_orbs),                   # CAS(2,2) for H2O in 6-31G
                E_corr_mo,  # Use OVOS-optimized orbitals
                h_core,
                g_eri,
                "utups",
                {"n_layers":1},
                include_active_kappa=include_active_kappa,
            )
                # To ensure reproducibility, set random seed for SlowQuant optimizations
            np.random.seed(42)
            thetas = (2*np.pi*np.random.random(len(WF_ovos.thetas)) - np.pi).tolist()   
            WF_ovos.thetas = thetas
            
                # Optimize WF
            stats_ovos_opt = run_ucc_and_get_stats(WF_ovos, "BFGS", True)
                    # Get optimization iterations/evaluations
            iter_ovos_opt = stats_ovos_opt['iterations']
            eval_ovos_opt = [stats_ovos_opt['function_evaluations'], stats_ovos_opt['gradient_evaluations']]
                    # Get optimized energy
            E_ovos_opt = stats_ovos_opt['final_energy']
            print(f"OVOS optimized energy = {E_ovos_opt} Hartree @ iterations {iter_ovos_opt}.")
                # End Timer OVOS w. WF optimization
            end_ovos_w_opt = time.time()
            print(f"OVOS optimization took {end_ovos_w_opt - start_ovos:.2f} seconds.")
            print()

            # Compare with UHF reference
                # UHF reference for comparison
            mf_uhf = scf.UHF(mol)
            mf_uhf.verbose = 0
            mf_uhf.kernel()

                # Start Timer UHF
            start_uhf = time.time()

            #     # Initualize for UPS wave function with UHF orbitals
            WF_uhf = UnrestrictedWaveFunctionUPS(
                mol.nelectron, # 10, 
                ((mol.nelectron//2 + 1,mol.nelectron//2 - 1), num_electrons//2+num_opt_virtual_orbs),                   # CAS(2,2) for H2O in 6-31G
                mf_uhf.mo_coeff,  
                h_core,
                g_eri,
                "utups",
                {"n_layers":1},
                include_active_kappa=include_active_kappa,
            )
            WF_uhf.thetas = thetas
                # Optimize WF
            stats_uhf_opt = run_ucc_and_get_stats(WF_uhf, "BFGS", True)
                    # Get optimization iterations/evaluations
            iter_uhf_opt = stats_uhf_opt['iterations']
            eval_uhf_opt = [stats_uhf_opt['function_evaluations'], stats_uhf_opt['gradient_evaluations']]
                    # Get optimized energy
            E_uhf_opt = stats_uhf_opt['final_energy']
            print(f"UHF optimized energy = {E_uhf_opt} Hartree @ iterations {iter_uhf_opt}.")

                # End Timer UHF w. WF optimization
            end_uhf_w_opt = time.time()
            print(f"UHF initialization took {end_uhf_w_opt - start_uhf:.2f} seconds.")

            # # Diff of OVOS and UHF MO coefficients
            # print("\nDifference in MO coefficients (OVOS - UHF):")
            # print(E_corr_mo - mf.mo_coeff)

            # Summary of results
            print("\nSummary of results:")
            print(f"Molecule: {atom} with basis set {basis} for a total of {num_electrons} electrons and {num_orbitals} orbitals.")
            print(f"OVOS correlation energy: {E_tot:.6f} Hartree (Active unocc. orbitals: {num_opt_virtual_orbs})")
            print(f"OVOS optimized energy: {E_ovos_opt:.6f} Hartree  @ iterations {iter_ovos_opt} (Eval. func. {eval_ovos_opt[0]}, grad. {eval_ovos_opt[0]} |Time: {end_ovos_w_opt - start_ovos:.2f} seconds | Energy RDM: {WF_ovos.energy_elec_RDM})")
            print(f"UHF optimized energy: {E_uhf_opt:.6f} Hartree  @ iterations {iter_uhf_opt} (Eval. func. {eval_uhf_opt[0]}, grad. {eval_uhf_opt[0]} |Time: {end_uhf_w_opt - start_uhf:.2f} seconds | Energy RDM: {WF_uhf.energy_elec_RDM})")

        finally:
            sys.stdout = sys.__stdout__


# Define the molecule
    # HF, CO, NH3, H2O
atom_1 = "H 0 0 0; F 0 0 0.917" # HF bond length 0.917 Angstrom
atom_2 = "N 0 0 0; H 0 0 1.012; H 0 0.935 -0.262; H 0 -0.935 -0.262" # NH3 equilibrium geometry	
atom_3 = "O 0.0000 0.0000  0.1173; H 0.0000    0.7572  -0.4692; H 0.0000   -0.7572 -0.4692;"  # H2O equilibrium geometry
atom_4 = "C 0 0 0; O 0 0 1.128" # CO bond length 1.128 Angstrom

basis = "6-31G" # 6-31G, cc-pVDZ

    # Whether to include active kappa parameters in the UCC optimization (True/False)
include_active_kappa = True

for atom in [atom_2]: # atom_3, atom_4
    for basis in [basis]:
        for num_opt_virtual_orbs in [0.75]: # 0.25,0.5,0.75
            for include_active_kappa in [include_active_kappa]:
                print(f"\nRunning VQE with OVOS optimization for {atom} in basis {basis} with {num_opt_virtual_orbs*100:.0f}% active virtual orbitals and include_active_kappa = {include_active_kappa}...")
                VQE_OVOS(atom, basis, num_opt_virtual_orbs, include_active_kappa)

# Table for presenting results:
#       I use 
# Step 1: Run with include_active_kappa = False
# | Results Summary for 6-31G with VQE Optimization (include_active_kappa = False) |
# | Molecule | Basis Set | # Electrons | # Orbitals | OVOS Num. of Act. Unocc. | OVOS Corr. Energy  | OVOS Opt. Energy @ Iterations | UHF Optimized Energy @ Iterations | UHF Opt. Energy Diff  |  Same or diff. Theta? |
# |          |           |             |            |      25%, 50%, 75%       |     [Hartree]      |           [Hartree]           |             [Hartree]             |       [Hartree]       |                       |
# |----------|-----------|-------------|------------|--------------------------|--------------------|-------------------------------|-----------------------------------|-----------------------|-----------------------|
# | H2O      | 6-31G     | 10          | 7          | 2 (4)  [RHF]             | -76.026963 (33.4%) | -84.896027  @ 40              | -84.904256  @ 38                  | -0.008229             | Same                  |
# | H2O      | 6-31G     | 10          | 9          | 4 (8)  [UHF]             | -76.101365 (91.1%) | -84.858729  @ 43              | -84.891800  @ 39                  | -0.033071             | Same                  |
# | H2O      | 6-31G     | 10          | 11         | 6 (12) [UHF]             | -76.111242 (98.8%) | -84.840472  @ 27              | -84.861685  @ 25                  | -0.021213             | Same                  |
# | H2O      | cc-pVDZ   | 10          | 8          | 4 (8)  [RHF]             | -76.142620 (56.8%) | -84.892488  @ 80              | -84.958209  @ 84                  | -0.065721             | Same                  |
# |..........|...........|.............|............|..........................|....................|...............................|...................................|.......................|.......................|
# | HF       | 6-31G     | 10          | 6          | 1 (2)  [RHF]             | -100.00188 (14.4%) | -104.825261 @ 27              | -104.829580 @ 25                  | -0.004319             | Same                  |
# | HF       | 6-31G     | 10          | 8          | 3 (6)  [UHF]             | -100.07082 (67.9%) | -104.802017 @ 46              | -104.805675 @ 34                  | -0.003658             | Same                  |
# | HF       | 6-31G     | 10          | 11         | 4 (8)  [UHF]             | -100.10745 (96.3%) | -104.775620 @ 43              | -104.789234 @ 18                  | -0.013614             | Same                  |
# | HF       | cc-pVDZ   | 10          | 8          | 3 (6)  [RHF]             | -100.10733 (33.6%) | -104.840054 @ 116             | -104.845022 @ 74                  | -0.004968             | Same                  |
# |..........|...........|.............|............|..........................|....................|...............................|...................................|.......................|.......................|
# | CO       | 6-31G     | 14          | 9          | 2 (4)  [UHF]             | -112.742906 (35.7%)| -134.989406 @ 125             | -134.988008 @ 142                 | +0.001398             | Same                  | 
# | CO       | 6-31G     | 14          | 12         | 5 (10) [UHF]             | -112.842877 (82.8%)| -134.981360 @ 66              | -134.967924 @ 72                  | +0.013436             | Same                  |                 
# | CO       | 6-31G     | 14          | 15         | 8 (16) [UHF]             | -112.873902 (97.4%)| ...                           | ...                               | ...                   | Same                  |
# |..........|...........|.............|............|..........................|....................|...............................|...................................|.......................|.......................|
# | NH3      | 6-31G     | 10          | 7          | 2 (4)  [RHF]             | -56.173502 (26.4%) | -68.156893  @ 47              | -68.159458  @ 37                  | -0.002565             | Same                  |
# | NH3      | 6-31G     | 10          | 10         | 5 (10) [UHF]             | -56.244012 (88.0%) | -68.127669  @ 41              | -68.143925  @ 38                  | -0.016256             | Same                  |
# | NH3      | 6-31G     | 10          | 12         | 7 (14) [UHF]             | -56.255257 (97.9%) | -68.075272  @ 39              | -68.136328  @ 37                  | -0.061056             | Same                  |
# | NH3      | cc-pVDZ   | 10          | 11         | 6 (12) [RHF]             | -56.276992 (59.4%) | -68.166951  @ 66              | -68.187273  @ 56                  | -0.020322             | Same                  |
# |----------|-----------|-------------|------------|--------------------------|--------------------|-------------------------------|-----------------------------------|-----------------------|-----------------------|
#
# Run each configuration with a few times with different random seeds to get an idea of the variability in the results (e.g. due to optimization getting stuck in local minima).
# Get mean and standard deviation of the energies and iterations for each configuration.
# 
# Plot: Energy vs. iteration for each configuration to visualize convergence behavior. (OVOS vs. UHF initializations, different numbers of active virtual orbitals, etc.)
#    Compare initial energies (OVOS vs. UHF) to see how much OVOS improves the starting point for VQE optimization. 
#
#
# Step 2: Run with include_active_kappa = True
# | Results Summary for H2O in 6-31G with VQE Optimization (include_active_kappa = True) |
# | Molecule | Basis Set | # Electrons | # Orbitals | OVOS Num. of Act. Unocc. | OVOS Corr. Energy (Hartree) | OVOS Opt. Energy (Hartree) @ Iterations | UHF Optimized Energy (Hartree) @ Iterations | UHF Opt. Energy Diff (Hartree) |
# |----------|-----------|-------------|------------|--------------------------|-----------------------------|-----------------------------------------|---------------------------------------------|--------------------------------|
# | H2O      | 6-31G     | 10          | 7          | 2 (4)  
# | H2O      | 6-31G     | 10          | 7          | 4 (8)  
# | H2O      | 6-31G     | 10          | 7          | 6 (12)  
# |..........|...........|.............|............|..........................|.............................|.........................................|.............................................|................................|
# | HF       | 6-31G     | 10          | 6          | 1 (2)
# | HF       | 6-31G     | 10          | 6          | 3 (6)
# | HF       | 6-31G     | 10          | 6          | 4 (8)
# | HF       | cc-PVDZ   | ...
# |..........|...........|.............|............|..........................|.............................|................................
# | ...
#
#
#


# # SlowQuant
#     # Initialize for UPS wave function with OVOS-optimized orbitals
# WF_ovos = UnrestrictedWaveFunctionUPS(
#     mol.nelectron,  # e.g. 10 electrons
#     ((6,4), num_electrons//2+num_opt_virtual_orbs),                   
#     E_corr_mo,          # Use OVOS-optimized orbitals
#     h_core,
#     g_eri,
#     "utups",
#     {"n_layers":1},
#     include_active_kappa=False,
# )
#     # Initialize thetas randomly for reproducibility
# np.random.seed(42)
# thetas = (2*np.pi*np.random.random(len(WF_ovos.thetas)) - np.pi).tolist()   
# WF_ovos.thetas = thetas
# WF_ovos.run_wf_optimization_1step("BFGS", orbital_optimization=True, tol=1e-6, maxiter=5000)

#     # Initualize for UPS wave function with UHF orbitals
# WF_uhf = UnrestrictedWaveFunctionUPS(
#     mol.nelectron,  # e.g. 10 electrons
#     ((6,4), num_electrons//2+num_opt_virtual_orbs),                  
#     mf_uhf.mo_coeff,  # Use UHF orbitals
#     h_core,
#     g_eri,
#     "utups",
#     {"n_layers":1},
#     include_active_kappa=False,
# )
#     # Use same random seed and initial thetas for fair comparison
# WF_uhf.thetas = thetas
# WF_uhf.run_wf_optimization_1step("BFGS", orbital_optimization=True, tol=1e-6, maxiter=5000)