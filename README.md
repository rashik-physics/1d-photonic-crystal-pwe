# 1D Photonic Crystal Band Structure in Python

An educational computational photonics project exploring how a periodic pair of dielectric layers produces photonic bands and frequency gaps. The notebook connects a one-dimensional Maxwell eigenproblem to a small plane-wave expansion (PWE) implementation using NumPy and Matplotlib.

This project demonstrates Fourier analysis, Bloch periodicity, Hermitian eigenvalue problems, and numerical convergence checks. It is a learning project, not a publication or an experimentally validated device design. No COMSOL work is included or required.

## Files

| File | Purpose |
| --- | --- |
| `1D_Photonic_Crystal_PWE_Beginners.ipynb` | Step-by-step derivations, code, and plotting workflow |
| `photonic_crystal_pwe.py` | The notebook's code cells exported as a runnable script |
| `requirements.txt` | Versions used for the verification run |
| `validation.txt` | Captured numerical output from that run |

## Model and method

The model is an infinite, one-dimensional stack of two alternating dielectric layers at normal incidence. Materials have real, positive, scalar relative permittivity, unit relative permeability, and no dispersion or absorption.

The solver expands the inverse relative permittivity and transverse magnetic field in Fourier harmonics. It assembles the dimensionless matrix

$$B_{mm'}=(q+m)(q+m')\eta_{m-m'},$$

where $q=ka/(2\pi)$, $\eta_p$ is a Fourier coefficient of $1/\epsilon_r$, and the retained field indices are $m=-M,\ldots,M$. Eigenvalues are $\Omega^2$, with normalized frequency $\Omega=\omega a/(2\pi c)$. The code uses `numpy.linalg.eigvalsh` and takes the square root to obtain the bands.

Analytic Fourier coefficients are compared with sampled FFT coefficients. Candidate gaps are found by comparing the maximum of each lower band with the minimum of the next band across the sampled Brillouin zone.

## Default example

| Parameter | Value |
| --- | --- |
| Refractive indices | 1.50 and 1.00 |
| Filling fraction of material 1 | 0.50 |
| Period | 500 nm |
| Main band plot | 41 plane waves, 201 wavevectors, 5 bands |
| First-gap convergence study | 21, 41, 81, and 161 plane waves |
| Refined wavevector grid | 401 points |

These are illustrative nondispersive teaching values, not measured material data. The notebook generates the dielectric profile, a Fourier-coefficient comparison, and a band diagram with candidate gaps.

## Run

Use Python 3.12, the version used for verification, and install the dependencies:

```sh
python -m pip install -r requirements.txt
python photonic_crystal_pwe.py
```

For the explanatory notebook, open `1D_Photonic_Crystal_PWE_Beginners.ipynb` in Jupyter or upload it to Google Colab, then run all cells in order. Jupyter is an optional interface and is not included in the two-library requirements file. Colab package versions may differ from the verified local environment. No GPU or external dataset is required.

The script opens Matplotlib figures; close each plot window to continue if the plotting backend blocks execution. The convergence calculation may take longer on slower machines. Change `n1`, `n2`, `f`, or `a` to explore the model, and rerun from the beginning: function defaults capture parameter values when defined.

## Verification and example output

On 30 October 2026, all 10 notebook code cells executed sequentially with Python 3.12.14, NumPy 2.5.4, and Matplotlib 3.11.2 using the noninteractive Agg backend. All existing assertions passed. The script is an export of those same code cells; it was not separately tested with an interactive plotting backend.

Checks cover the Fourier mean, conjugate symmetry, the analytic coefficient formula, FFT sampling error, matrix Hermiticity, rejection of materially negative eigenvalues, unit conversion, the zero-frequency mode, wavevector symmetry, and uniform-medium dispersion.

The refined first-gap estimate for the default example was:

- Normalized frequency: **0.3507871–0.4483371**.
- Frequency for a 500 nm period: **210.33–268.82 THz**.
- Vacuum wavelength: **1115.23–1425.37 nm**.

Increasing the cutoff from M=40 to M=80 changed the first-gap edges by at most approximately **2.406e-4** in normalized frequency. Refining 101 to 401 wavevectors at M=80 produced no change in the reported extrema for this example. These observations are empirical convergence evidence, not rigorous error bounds or proof of exact values. See `validation.txt` for the complete output.

## Scope and limitations

- This is a classical electromagnetic calculation; it does not simulate quantum states or establish quantum-mechanical research results.
- Only propagation normal to the layers is modeled. The two transverse polarizations are degenerate under these assumptions; the reported gaps are not omnidirectional gaps.
- The structure is infinite. Finite-stack transmission, reflection, defects, losses, and fabrication effects are not computed.
- Finite Fourier truncation and sampled wavevectors limit accuracy. The detailed convergence study targets the first gap; higher gaps need their own study.
- FFT agreement verifies material coefficients, not the complete physical model. No independent solver, COMSOL, or experimental benchmark is supplied.
- The numerical example is not a claim of novel materials performance, publication, or a validated optical device.

## Project context

The notebook is an explanatory learning artifact from the Photonic Crystal Research Paper project. This repository packages that notebook and its exported code for review and reproducibility; the accompanying project reference document is not redistributed. The README describes the implementation and verified checks without asserting a publication or sole authorship of all underlying ideas.
