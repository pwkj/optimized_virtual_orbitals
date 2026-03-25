"""
OVOS (Optimized Virtual Orbital Space) method

Minimizes the second-order correlation energy (MP2) by rotating virtual orbitals.

Spaces:
    Active occupied   (I,J): "Inactive"
    Active virtual    (A,B): "Active"
    Inactive virtual  (E,F): "Virtual"

Reference:
    L. Adamowicz & R. J. Bartlett (1987), J. Chem. Phys. 86, 6314
    https://pubs.aip.org/aip/jcp/article/86/11/6314/93345
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import scipy.linalg
import pyscf
from pyscf import ao2mo

# Limit OpenBLAS threads to avoid oversubscription in parallel runs
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

np.set_printoptions(precision=4, suppress=False, linewidth=200)


class OVOS:
    """
    Minimizes the MP2 correlation energy via orbital rotations (OVOS algorithm).

    Parameters
    ----------
    mol : pyscf.gto.Mole
    scf : RHF or UHF object
    Fao : list of np.ndarray
        AO Fock matrices [alpha, beta].
    num_opt_virtual_orbs : int
        Number of optimized virtual spin orbitals (active virtuals).
    mo_coeff : list of np.ndarray
        Initial MO coefficients [alpha, beta].
    init_orbs : str, optional
        'RHF' or 'UHF' (unused, kept for compatibility).
    verbose : int, optional
        Verbosity level. 0 = silent, 1 = normal output.
    max_iter : int, optional
        Maximum number of OVOS iterations.
    conv_energy : float, optional
        Convergence threshold on absolute change in MP2 energy.
    conv_grad : float, optional
        Convergence threshold on gradient norm.
    keep_track_max : int, optional
        Number of consecutive converged iterations required to stop.
    lambda_init : float, optional
        Initial Levenberg‑Marquardt damping parameter.
    lambda_max : float, optional
        Maximum allowed damping parameter.
    trust_radius : float, optional
        Initial trust radius for step size control.
    hessian_reg : float, optional
        Regularization added to Hessian diagonal when singular.
    """

    def __init__(self, mol, scf, Fao, num_opt_virtual_orbs, mo_coeff,
                 init_orbs="RHF", verbose=1,
                 max_iter=1000, conv_energy=1e-8, conv_grad=1e-4,
                 keep_track_max=50,
                 lambda_init=1e-4, lambda_max=1e4,
                 trust_radius=0.3, hessian_reg=1e-8):

        self.verbose = verbose
        self.mol = mol
        self.num_opt_virtual_orbs = num_opt_virtual_orbs
        self.init_orbs = init_orbs
        self.mo_coeffs = mo_coeff
        self.Fao = Fao

        self.scf = scf
        self.e_rhf = scf.e_tot
        self.h_nuc = mol.energy_nuc()

        # Convergence parameters
        self.max_iter = max_iter
        self.conv_energy = conv_energy
        self.conv_grad = conv_grad
        self.keep_track_max = keep_track_max

        # Optimization parameters
        self.lambda_init = lambda_init
        self.lambda_max = lambda_max
        self.trust_radius = trust_radius
        self.hessian_reg = hessian_reg

        # AO integrals
        self.S = mol.intor('int1e_ovlp')
        self.hcore_ao = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
        self.eri_4fold_ao = mol.intor('int2e_sph', aosym=1)

        # Orbital counts
        n_spatial = self.mo_coeffs[0].shape[0]
        self.tot_num_spin_orbs = 2 * n_spatial
        assert self.mo_coeffs[0].shape[0] == self.mo_coeffs[1].shape[0], \
            "Alpha and beta MO coeff matrices must have same number of rows"

        self.nelec = mol.nelec[0] + mol.nelec[1]

        # Index lists for subspaces
        self.active_occ_indices = list(range(self.nelec))
        self.active_inocc_indices = list(range(
            self.active_occ_indices[-1] + 1,
            self.nelec + self.num_opt_virtual_orbs
        ))
        self.virtual_inocc_indices = list(range(
            self.active_inocc_indices[-1] + 1,
            self.tot_num_spin_orbs
        ))
        self.inactive_indices = self.virtual_inocc_indices
        self.full_indices = list(range(self.tot_num_spin_orbs))

        assert self.tot_num_spin_orbs >= self.num_opt_virtual_orbs + self.nelec, \
            f"'num_opt_virtual_orbs' ({num_opt_virtual_orbs}) too large for basis set"

        if self.verbose:
            self._print()
            self._print("#### Active and inactive spaces ####")
            self._print("Total number of spin-orbitals:   ", self.tot_num_spin_orbs)
            self._print("Active occupied spin-orbitals:   ", self.active_occ_indices)
            self._print("Active unoccupied spin-orbitals: ", self.active_inocc_indices)
            self._print("Inactive unoccupied spin-orbs:   ", self.inactive_indices)
            self._print()

    # -------------------------------------------------------------------------
    # Print helper
    # -------------------------------------------------------------------------
    def _print(self, *args, **kwargs):
        """Print with CRLF line endings and immediate flush."""
        kwargs.setdefault('end', '\r\n')
        kwargs.setdefault('flush', True)
        print(*args, **kwargs)

    # -------------------------------------------------------------------------
    # Spin‑orbital conversion helpers
    # -------------------------------------------------------------------------
    def _spatial_to_spin_energies(self, mo_energy_a: np.ndarray,
                                   mo_energy_b: np.ndarray) -> np.ndarray:
        """Interleave alpha/beta spatial MO energies into spin‑orbital order."""
        eps = np.empty(2 * len(mo_energy_a))
        eps[0::2] = mo_energy_a
        eps[1::2] = mo_energy_b
        return eps

    def _build_spin_fock(self, Fmo_a: np.ndarray, Fmo_b: np.ndarray) -> np.ndarray:
        """Build block‑diagonal spin‑orbital Fock matrix (even alpha, odd beta)."""
        n_spin = 2 * self.Fao[0].shape[0]
        Fspin = np.zeros((n_spin, n_spin))
        Fspin[0::2, 0::2] = Fmo_a
        Fspin[1::2, 1::2] = Fmo_b
        return Fspin

    # -------------------------------------------------------------------------
    # Integral transformation
    # -------------------------------------------------------------------------
    def _eri_vovo_antisym(self, mo_coeffs: List[np.ndarray]) -> np.ndarray:
        """
        Transform (vir,occ|vir,occ) AO integrals to MO basis, assemble spin‑orbital
        and antisymmetrize. Returns <ab||ij> in physicist notation (a,b,i,j).
        """
        nocc_a, nocc_b = self.mol.nelec
        C_occ_a = mo_coeffs[0][:, :nocc_a]
        C_occ_b = mo_coeffs[1][:, :nocc_b]
        C_vir_a = mo_coeffs[0][:, nocc_a:]
        C_vir_b = mo_coeffs[1][:, nocc_b:]
        nvir_a = C_vir_a.shape[1]
        nvir_b = C_vir_b.shape[1]

        def _ao2mo(C_left, C_row, C_right, C_col):
            # returns (left,row|right,col) in chemist notation
            return ao2mo.kernel(self.eri_4fold_ao,
                                [C_left, C_row, C_right, C_col],
                                compact=False)

        # Chemist blocks (vir,occ|vir,occ)
        vovo_aaaa = _ao2mo(C_vir_a, C_occ_a, C_vir_a, C_occ_a).reshape(
            nvir_a, nocc_a, nvir_a, nocc_a)
        vovo_bbbb = _ao2mo(C_vir_b, C_occ_b, C_vir_b, C_occ_b).reshape(
            nvir_b, nocc_b, nvir_b, nocc_b)
        vovo_aabb = _ao2mo(C_vir_a, C_occ_a, C_vir_b, C_occ_b).reshape(
            nvir_a, nocc_a, nvir_b, nocc_b)

        nvir_spin = 2 * nvir_a
        nocc_spin = nocc_a + nocc_b

        # Assemble spin‑orbital chemist tensor
        vovo_spin = np.zeros((nvir_spin, nocc_spin, nvir_spin, nocc_spin))
        vovo_spin[0::2, 0::2, 0::2, 0::2] = vovo_aaaa
        vovo_spin[1::2, 1::2, 1::2, 1::2] = vovo_bbbb
        vovo_spin[0::2, 0::2, 1::2, 1::2] = vovo_aabb
        vovo_spin[1::2, 1::2, 0::2, 0::2] = vovo_aabb.transpose(2, 3, 0, 1)

        # Convert to physicist (a,i,b,j) -> (a,b,i,j) and antisymmetrize
        eri_phys = vovo_spin.transpose(0, 2, 1, 3)
        eri_as = eri_phys - eri_phys.transpose(0, 1, 3, 2)
        return eri_as

    # -------------------------------------------------------------------------
    # MP2 energy and amplitude computation
    # -------------------------------------------------------------------------
    def _mp1_amplitudes(self, fock_diag: np.ndarray,
                         eri_as: np.ndarray) -> np.ndarray:
        """
        Compute MP1 amplitudes t_ij^ab = -<ab||ij> / (f_aa + f_bb - f_ii - f_jj).
        Only the active occupied and active virtual subspaces are used.
        """
        occ_idx = self.active_occ_indices
        vir_idx = self.active_inocc_indices
        nvir_act = len(vir_idx)

        eps_occ = fock_diag[occ_idx]
        eps_vir = fock_diag[vir_idx]

        denom = (eps_vir[:, None, None, None] +
                 eps_vir[None, :, None, None] -
                 eps_occ[None, None, :, None] -
                 eps_occ[None, None, None, :])

        return -eri_as[:nvir_act, :nvir_act, :, :] / denom

    def _mp2_energy(self, fock_spin: np.ndarray, t_abij: np.ndarray,
                    eri_as: np.ndarray) -> float:
        """
        MP2 correlation energy using full virtual‑virtual Fock block to handle
        non‑canonical orbitals correctly.
        """
        occ_abs = np.array(self.active_occ_indices)
        vir_abs = np.array(self.active_inocc_indices)
        nocc = len(occ_abs)
        nvir = len(vir_abs)

        if nocc == 0 or nvir == 0:
            return 0.0

        a_loc, b_loc = np.triu_indices(nvir, k=1)          # a < b
        F_virt = fock_spin[np.ix_(vir_abs, vir_abs)]
        chunk_size = min(500, len(a_loc))
        n_pairs = len(a_loc)

        # Pre‑compute all (i,j) with i > j
        occ_pairs = [(i, j) for i in range(nocc) for j in range(i)]

        energy = 0.0
        for loc_i, loc_j in occ_pairs:
            abs_i = occ_abs[loc_i]
            abs_j = occ_abs[loc_j]
            eps_ij_sum = fock_spin[abs_i, abs_i] + fock_spin[abs_j, abs_j]

            t_ab = t_abij[a_loc, b_loc, loc_i, loc_j]       # shape (n_pairs,)
            g_ab = eri_as[a_loc, b_loc, loc_i, loc_j]       # shape (n_pairs,)

            # Diagonal part (2 * t · g)
            term2 = 2.0 * np.dot(t_ab, g_ab)

            # Off‑diagonal part
            term1 = 0.0
            for r0 in range(0, n_pairs, chunk_size):
                r1 = min(r0 + chunk_size, n_pairs)
                t_row = t_ab[r0:r1, None]
                ar, br = a_loc[r0:r1], b_loc[r0:r1]

                for c0 in range(0, n_pairs, chunk_size):
                    c1 = min(c0 + chunk_size, n_pairs)
                    t_col = t_ab[None, c0:c1]
                    ac, bc = a_loc[c0:c1], b_loc[c0:c1]

                    # Masks for index equalities
                    d_ac = ar[:, None] == ac[None, :]
                    d_bd = br[:, None] == bc[None, :]
                    d_ad = ar[:, None] == bc[None, :]
                    d_bc = br[:, None] == ac[None, :]

                    bracket = np.zeros((r1 - r0, c1 - c0))
                    if np.any(d_bd):
                        bracket += F_virt[ar[:, None], ac[None, :]] * d_bd
                    if np.any(d_ac):
                        bracket += F_virt[br[:, None], bc[None, :]] * d_ac
                    if np.any(d_bc):
                        bracket -= F_virt[ar[:, None], bc[None, :]] * d_bc
                    if np.any(d_ad):
                        bracket -= F_virt[br[:, None], ac[None, :]] * d_ad

                    mask_ac_bd = d_ac & d_bd
                    if np.any(mask_ac_bd):
                        bracket[mask_ac_bd] -= eps_ij_sum
                    mask_ad_bc = d_ad & d_bc
                    if np.any(mask_ad_bc):
                        bracket[mask_ad_bc] += eps_ij_sum

                    term1 += np.sum(t_row * bracket * t_col)

            energy += term1 + term2

        return energy

    def _compute_density(self, t_abij: np.ndarray) -> np.ndarray:
        """
        Virtual‑virtual density matrix D_ab = ∑_{i>j,c} t_ac^ij t_bc^ij.
        """
        nocc = t_abij.shape[2]
        i_idx, j_idx = np.tril_indices(nocc, k=-1)
        t_ij = t_abij[:, :, i_idx, j_idx]          # (nvir, nvir, n_pairs)
        D_ab = np.einsum('acp,bcp->ab', t_ij, t_ij, optimize=True)
        D_ab[np.abs(D_ab) < 1e-12] = 0.0
        return D_ab

    # -------------------------------------------------------------------------
    # Gradient and Hessian
    # -------------------------------------------------------------------------
    def _gradient(self, t_abij: np.ndarray, eri_as: np.ndarray,
                  D_ab: np.ndarray, fock_spin: np.ndarray) -> np.ndarray:
        """
        Gradient G[a,e] = 2∑_{i>j,b} t_ab^ij <eb||ij> + 2∑_b D_ab F_be
        """
        vir = np.array(self.active_inocc_indices)
        inactive = np.array(self.inactive_indices)
        nvir = len(vir)
        ninact = len(inactive)

        if ninact == 0:
            return np.zeros((nvir, 0))

        nocc = t_abij.shape[2]
        i_idx, j_idx = np.tril_indices(nocc, k=-1)

        # <eb||ij> for e in inactive, b in active, i>j
        eri_eb = eri_as[nvir:, :nvir, :, :][:, :, i_idx, j_idx]  # (ninact, nvir, n_pairs)
        t_ij = t_abij[:, :, i_idx, j_idx]                         # (nvir, nvir, n_pairs)

        G = 2.0 * np.einsum('abp,ebp->ae', t_ij, eri_eb, optimize=True)
        F_be = fock_spin[np.ix_(vir, inactive)]                   # (nvir, ninact)
        G += 2.0 * D_ab @ F_be
        return G

    def _hessian(self, t_abij: np.ndarray, eri_as: np.ndarray,
                 D_ab: np.ndarray, fock_spin: np.ndarray) -> np.ndarray:
        """
        Hessian H_{ae,bf} = 2∑_{i>j} t_ab^ij <ef||ij>
                          - [∑_{c,i>j} (t_ac^ij <bc||ij> + t_cb^ij <ca||ij>)] δ_ef
                          + D_ab (f_aa + f_bb) δ_ef
                          + D_ab f_ef (1 - δ_ef)
        """
        vir = np.array(self.active_inocc_indices)
        inactive = np.array(self.inactive_indices)
        nvir = len(vir)
        ninact = len(inactive)

        if ninact == 0:
            return np.zeros((0, 0))

        nocc = t_abij.shape[2]
        i_idx, j_idx = np.tril_indices(nocc, k=-1)

        fock_vir_diag = np.diag(fock_spin)[vir]                     # (nvir,)
        fock_inact = fock_spin[np.ix_(inactive, inactive)]          # (ninact, ninact)
        fock_inact_offdiag = fock_inact.copy()
        np.fill_diagonal(fock_inact_offdiag, 0.0)

        t_ij = t_abij[:, :, i_idx, j_idx]                            # (nvir, nvir, n_pairs)
        eri_ef = eri_as[nvir:, nvir:, :, :][:, :, i_idx, j_idx]      # (ninact, ninact, n_pairs)
        eri_ab = eri_as[:nvir, :nvir, :, :][:, :, i_idx, j_idx]      # (nvir, nvir, n_pairs)

        # Term 1: 2 t_ab^p <ef||ij>^p  -> reshape to (a,e,b,f)
        term1 = 2.0 * np.einsum('abp,efp->aebf', t_ij, eri_ef, optimize=True)
        H = term1.reshape(nvir * ninact, nvir * ninact)

        # Terms 2 & 3 (diagonal in e=f)
        term2 = -(
            np.einsum('acp,bcp->ab', t_ij, eri_ab, optimize=True) +
            np.einsum('cbp,cap->ab', t_ij, eri_ab, optimize=True)
        )
        term3 = D_ab * (fock_vir_diag[:, None] + fock_vir_diag[None, :])
        H += np.kron(term2 + term3, np.eye(ninact)).reshape(nvir * ninact, nvir * ninact)

        # Term 4 (off‑diagonal e≠f): D_ab f_ef
        H += np.kron(D_ab, fock_inact_offdiag).reshape(nvir * ninact, nvir * ninact)

        return H

    # -------------------------------------------------------------------------
    # Orbital rotation and canonicalization
    # -------------------------------------------------------------------------
    def _rotate_orbitals(self, mo_coeffs: List[np.ndarray],
                         fock_spin: np.ndarray,
                         R_vec: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Apply rotation R_vec (active‑inactive mixings) to MO coefficients and Fock.
        R_vec is a flat array of length nvir * ninact.
        """
        nvir = len(self.active_inocc_indices)
        ninact = len(self.inactive_indices)
        # R is a vector of length nvir*ninact, reshape to matrix form for orbital rotations
        R_2d = R_vec.reshape(nvir, ninact)
			# Matrix: nvir+ninact x nvir+ninact
        R_full = np.zeros((nvir+ninact, nvir+ninact), dtype=np.float64)
			# Fill the anti-symmetric R_matrix with R_2d and its negative transpose
        for i, a in enumerate(np.arange(nvir)): # Local indices for active virtuals
            for j, e in enumerate(np.arange(ninact) + nvir): # Local indices for inactive orbitals
                R_ae = R_2d[i, j]
                R_full[e, a] = R_ae 			# Note the order of indices for correct placement
                R_full[a, e] = -R_ae			# Ensure anti-symmetry

        U_sub = scipy.linalg.expm(R_full)

        n_occ = len(self.active_occ_indices)
        n_full = len(self.full_indices)
        U_full = np.eye(n_full)
        U_full[n_occ:, n_occ:] = U_sub

        # Extract spatial rotations for alpha and beta
        U_alpha = U_full[0::2, 0::2]
        U_beta = U_full[1::2, 1::2]

        mo_rot = [mo_coeffs[0] @ U_alpha, mo_coeffs[1] @ U_beta]
        fock_rot = U_full.T @ fock_spin @ U_full

        # Sanity checks
        assert np.allclose(U_full.T @ U_full, np.eye(n_full), atol=1e-6), "Unitary broken"
        assert np.allclose(fock_rot, fock_rot.T, atol=1e-8), "Rotated Fock not Hermitian"

        return mo_rot, fock_rot

    def _canonicalize_active(self, mo_coeffs: List[np.ndarray],
                              fock_spin: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Diagonalize the active‑virtual block of the Fock matrix to restore
        a canonical representation. This does not change the energy.
        """
        nocc_a, nocc_b = self.mol.nelec
        nact_spatial = self.num_opt_virtual_orbs // 2

        # Build full spin transformation that diagonalizes active virtual blocks
        U_alpha_full = np.eye(fock_spin.shape[0] // 2)
        U_beta_full = np.eye(fock_spin.shape[0] // 2)

        # Alpha active block
        sl_a = slice(nocc_a, nocc_a + nact_spatial)
        if nact_spatial > 0:
            _, eigvecs = np.linalg.eigh(fock_spin[0::2, 0::2][sl_a, sl_a])
            U_alpha_full[sl_a, sl_a] = eigvecs

        # Beta active block
        sl_b = slice(nocc_b, nocc_b + nact_spatial)
        if nact_spatial > 0:
            _, eigvecs = np.linalg.eigh(fock_spin[1::2, 1::2][sl_b, sl_b])
            U_beta_full[sl_b, sl_b] = eigvecs

        # Build spin transformation
        U_spin = np.zeros_like(fock_spin)
        U_spin[0::2, 0::2] = U_alpha_full
        U_spin[1::2, 1::2] = U_beta_full

        mo_canon = [mo_coeffs[0] @ U_alpha_full, mo_coeffs[1] @ U_beta_full]
        fock_canon = U_spin.T @ fock_spin @ U_spin

        # Eigenvalues must remain unchanged (unitary transformation)
        assert np.allclose(np.sort(np.linalg.eigvalsh(fock_canon)),
                           np.sort(np.linalg.eigvalsh(fock_spin)), atol=1e-8)
        return mo_canon, fock_canon

    # -------------------------------------------------------------------------
    # Newton step solver with trust region / Levenberg‑Marquardt
    # -------------------------------------------------------------------------
    def _newton_step(self, G: np.ndarray, H: np.ndarray,
                     iteration: int) -> np.ndarray:
        """
        Solve H·R = -G with adaptive Levenberg‑Marquardt damping.
        """
        g_vec = G.flatten()
        dim = len(g_vec)

        if iteration <= 5:
            # Pure Newton with fallback
            try:
                R = np.linalg.solve(H, -g_vec)
            except np.linalg.LinAlgError:
                H_reg = H + self.hessian_reg * np.eye(dim)
                R = np.linalg.solve(H_reg, -g_vec)
        else:
            # Adaptive Levenberg‑Marquardt
            lambda_lm = self.lambda_init
            best_R = None
            best_norm = np.inf
            for _ in range(20):
                try:
                    H_reg = H + lambda_lm * np.eye(dim)
                    R_trial = np.linalg.solve(H_reg, -g_vec)
                except np.linalg.LinAlgError:
                    lambda_lm *= 2
                    continue

                step_norm = np.linalg.norm(R_trial)
                if step_norm < self.trust_radius:
                    # Accept step and reduce lambda for next time
                    best_R = R_trial
                    self.lambda_init = max(lambda_lm / 2, 1e-12)
                    break
                else:
                    if step_norm < best_norm:
                        best_norm = step_norm
                        best_R = R_trial
                    lambda_lm *= 2
                    if lambda_lm > self.lambda_max:
                        break

            if best_R is None:
                # Ultimate fallback: steepest descent scaled to trust radius
                g_norm = np.linalg.norm(g_vec)
                if g_norm > 0:
                    best_R = - (self.trust_radius / g_norm) * g_vec
                else:
                    best_R = np.zeros_like(g_vec)
            R = best_R

        return R

    # -------------------------------------------------------------------------
    # Main driver
    # -------------------------------------------------------------------------
    def run(self, mo_coeffs: List[np.ndarray],
            fock_spin: Optional[np.ndarray] = None) -> dict:
        """
        Run the OVOS optimization loop.

        Parameters
        ----------
        mo_coeffs : list of np.ndarray
            Initial MO coefficients [alpha, beta].
        fock_spin : np.ndarray or None
            Initial spin‑orbital Fock matrix. If None, built from mo_coeffs.

        Returns
        -------
        dict with keys:
            energy_history : list of float
            converged_energy_history : list of float (only converged part)
            iteration_history : list of int
            final_mo_coeffs : list of np.ndarray
            final_fock_spin : np.ndarray
            stop_reasons : list of str
        """
        converged = False
        start_counting = False
        iter_count = 0
        keep_track = 0

        # Storage
        energy_hist = []
        iter_hist = []
        stop_reasons = []
        mo_hist = []
        fock_hist = []
        grad_norm_hist = []

        # Initial Fock if not provided
        if fock_spin is None:
            Fmo_a = mo_coeffs[0].T @ self.Fao[0] @ mo_coeffs[0]
            Fmo_b = mo_coeffs[1].T @ self.Fao[1] @ mo_coeffs[1]
            fock_spin = self._build_spin_fock(Fmo_a, Fmo_b)
            eig_a = scipy.linalg.eigh(Fmo_a, eigvals_only=True)
            eig_b = scipy.linalg.eigh(Fmo_b, eigvals_only=True)
            self.eps = self._spatial_to_spin_energies(np.real(eig_a), np.real(eig_b))
        else:
            self.eps = np.diag(fock_spin)

        while iter_count < self.max_iter:
            iter_count += 1

            if self.verbose:
                self._print()
                self._print(f"#### OVOS Iteration {iter_count} ####")

            # MP2 calculation
            eri_as = self._eri_vovo_antisym(mo_coeffs)
            t_abij = self._mp1_amplitudes(self.eps, eri_as)
            assert np.allclose(t_abij, t_abij.transpose(1, 0, 3, 2), atol=1e-10), \
                "t_abij antisymmetry broken"
            E_corr = self._mp2_energy(fock_spin, t_abij, eri_as)

            energy_hist.append(E_corr)
            iter_hist.append(iter_count)
            mo_hist.append(mo_coeffs)
            fock_hist.append(fock_spin)

            if self.verbose:
                nact = len(self.active_inocc_indices)
                ntot = nact + len(self.inactive_indices)
                self._print(f"    [{nact}/{ntot}]: MP2 energy = {E_corr:.12f}")

            if iter_count == 1:
                stop_reasons.append("Initial")
                if len(self.virtual_inocc_indices) == 0:
                    break
            else:
                # Check convergence
                dE = abs(energy_hist[-1] - energy_hist[-2])
                self.dE = energy_hist[-1] - energy_hist[-2]

                D_ab = self._compute_density(t_abij)
                G = self._gradient(t_abij, eri_as, D_ab, fock_spin)
                grad_norm = np.linalg.norm(G.flatten())
                grad_norm_hist.append(grad_norm)
                if len(grad_norm_hist) > 5:
                    grad_norm_hist.pop(0)
                dgrad = abs(grad_norm_hist[-1] - grad_norm_hist[-2]) if len(grad_norm_hist) > 1 else None

                if self.verbose and dgrad is not None:
                    flag = "(energy increased!)" if self.dE > 0 else ""
                    self._print(f"            ΔE = {self.dE:.2e}  ‖grad‖ = {grad_norm:.2e}  {flag}")
                
                if (dE < self.conv_energy and grad_norm < self.conv_grad) or dE < 1e-12:
                    stop_reasons.append("Convergence")
                    start_counting = True
                elif dE > 1e-12:
                    stop_reasons.append("Non‑converged")
                    start_counting = False
                    keep_track = 0

                if start_counting: # (dE < self.conv_energy and grad_norm < self.conv_grad) or dE < 1e-12:
                    keep_track += 1
                    print(f"Keep track: {keep_track}/{self.keep_track_max}")
                    if keep_track >= self.keep_track_max:
                        if self.verbose:
                            self._print(f"OVOS converged after {iter_count} iterations")
                        # Trim the extra tracked steps
                        trim = self.keep_track_max
                        energy_hist = energy_hist[:-trim]
                        iter_hist = iter_hist[:-trim]
                        mo_hist = mo_hist[:-trim]
                        fock_hist = fock_hist[:-trim]
                        stop_reasons = stop_reasons[:-trim]
                        converged = True
                        break
                else:
                    keep_track = 0

            D_ab = self._compute_density(t_abij)
            G = self._gradient(t_abij, eri_as, D_ab, fock_spin)
            H = self._hessian(t_abij, eri_as, D_ab, fock_spin)

            # Solve Newton step
            R_vec = self._newton_step(G, H, iter_count)

            # Apply rotation
            mo_coeffs, fock_spin = self._rotate_orbitals(mo_coeffs, fock_spin, R_vec)

            # Re‑canonicalize the active virtual block
            mo_coeffs, fock_spin = self._canonicalize_active(mo_coeffs, fock_spin)

            # Update diagonal elements for next MP1 denominator (approximate, but ok)
            self.eps = np.diag(fock_spin)

            if iter_count >= self.max_iter:
                if self.verbose:
                    self._print(f"Reached maximum iterations ({self.max_iter})")
                break

        # --- Summary ---
        if converged and self.verbose:
            self._print()
            self._print("#### OVOS Summary ####")
            self._print(f"Initial MP2 energy: {energy_hist[0]:.12f} Ha")
            self._print(f"Final MP2 energy:   {energy_hist[-1]:.12f} Ha")
            delta = energy_hist[-1] - energy_hist[0]
            self._print(f"Total change:       {delta:.12f} Ha")
            if delta < 0:
                self._print("OVOS lowered the correlation energy.")
            else:
                self._print("WARNING: OVOS increased the correlation energy.")
            # Unrestricted or restricted final orbitals
            if np.allclose(mo_coeffs[0], mo_coeffs[1], atol=1e-6):
                self._print("Final orbitals are effectively restricted.")
            else:
                self._print("Final orbitals are unrestricted.")
            self._print(f"Total iterations: {iter_count}")

        # Select best result (lowest energy)
        best_idx = int(np.argmin(energy_hist))
        result = [
            energy_hist[:best_idx+1],
            energy_hist,
            iter_hist[:best_idx+1],
            mo_hist[best_idx],
            fock_hist[best_idx],
            stop_reasons[:best_idx+1],
        ]
        return result


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    from pyscf import gto, scf

    # Water molecule, minimal basis
    mol = gto.Mole()
    mol.atom = 'O 0 0 0; H 0 0 1; H 0 1 0'
    mol.basis = '6-31G'
    mol.unit = 'Angstrom'
    mol.spin = 0
    mol.charge = 0
    mol.symmetry = False
    mol.verbose = 0
    mol.build()

    # RHF reference
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()

    # Initial data (RHF orbitals)
    Fao = [mf.get_fock(), mf.get_fock()]
    mo_coeffs = [mf.mo_coeff, mf.mo_coeff]

    # Create OVOS object and run
    ovos = OVOS(
        mol=mol,
        scf=mf,
        Fao=Fao,
        num_opt_virtual_orbs=2,      # active virtual spin‑orbitals
        mo_coeff=mo_coeffs,
        init_orbs="RHF",
        verbose=1,
        max_iter=1000,
        conv_energy=1e-8,
        conv_grad=1e-6,
        keep_track_max=50
    )
    E_corr, E_corr_hist, E_corr_iter, E_corr_mo, E_corr_fock, stop_reason = ovos.run(mo_coeffs, fock_spin=None)

    print("\nOptimization finished. Final MP2 energy =", E_corr)