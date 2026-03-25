# Optimized Virtual Orbital Space (OVOS)

Implementation of the Optimized Virtual Orbital Space (OVOS) method for reducing computational complexity in correlated electronic structure calculations.

## Overview

This package implements the OVOS algorithm based on the seminal work by Adamowicz and Bartlett (1987). The method optimizes a reduced virtual orbital space to minimize the second-order correlation energy (MP2), enabling efficient high-level quantum chemistry calculations.

**Key features:**
- Reduces virtual orbital space while preserving correlation energy
- Newton-Raphson orbital optimization using the Hylleraas functional
- UHF and RHF orbital support
- Multiple initialization strategies (UHF, RHF, random unitary)
- Compatible with PySCF

## Installation

```bash
pip install -e .
```

## Quick Start

### Basic UHF Example

```python
from ovos import OVOS
from pyscf import gto, scf

# Define molecule
mol = gto.Mole()
mol.atom = 'O 0 0 0; H 0 0 1; H 0 1 0'
mol.basis = '6-31G'
mol.build()

# Run UHF reference
mf = scf.UHF(mol)
mf.kernel()

# Initialize and run OVOS
ovos = OVOS(
    mol=mol,
    scf=mf,
    Fao=[mf.get_fock()[0], mf.get_fock()[1]],
    num_opt_virtual_orbs=2,
    mo_coeff=[mf.mo_coeff[0], mf.mo_coeff[1]],
    init_orbs="UHF"
)
E_corr, E_corr_hist, E_corr_iter, E_corr_mo, E_corr_fock, stop_reason = ovos.run(
    [mf.mo_coeff[0], mf.mo_coeff[1]], 
    fock_spin=None
)

print(f"MP2 correlation energy: {E_corr}")
```

### RHF Example

See `test/test_ovos_rhf.py` for a complete RHF implementation example.

### Random Unitary Initialization

For exploring different virtual orbital choices, use the random unitary initialization approach:

```python
# See test/test_ovos_random.py for full example with multiple attempts
```

## Tests

Run the test suite to verify installation:

```bash
pytest test/
```

Available tests:
- `test_ovos_uhf.py` - UHF initialization on CO, H2O, HF, and NH3
- `test_ovos_rhf.py` - RHF initialization on water molecule
- `test_ovos_random.py` - Random unitary virtual orbital selection

## Citation

If you use this code, please cite the original work:

**Adamowicz, L. & Bartlett, R. J.**  
"Optimized virtual orbital space for high-level correlated calculations"  
*J. Chem. Phys.* **86**, 6314-6324 (1987)  
DOI: [10.1063/1.452468](https://doi.org/10.1063/1.452468)

## References

- Original paper: https://pubs.aip.org/aip/jcp/article/86/11/6314/93345/Optimized-virtual-orbital-space-for-high-level
- PySCF documentation: https://pyscf.org/
