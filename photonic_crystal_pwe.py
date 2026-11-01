# Exported code cells from the accompanying educational notebook.

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.figsize': (9, 4.5), 'font.size': 11,
                     'axes.grid': True, 'grid.alpha': 0.25})
c0 = 299_792_458.0  # vacuum speed of light, m/s
print('NumPy version:', np.__version__)


# Edit these four physical inputs, then rerun all cells below.
n1, n2 = 1.50, 1.00
f = 0.50                  # filling fraction, not frequency
a = 500e-9                # period in metres
eps1, eps2 = n1**2, n2**2
eta1, eta2 = 1/eps1, 1/eps2
assert a > 0 and n1 > 0 and n2 > 0 and 0 < f < 1
print(f'epsilon_1={eps1:.4f}, epsilon_2={eps2:.4f}')
print(f'Layer thicknesses: {f*a*1e9:.1f} nm and {(1-f)*a*1e9:.1f} nm')

u = np.linspace(0, 2, 2001, endpoint=False)  # u=x/a; two periods
eps_x = np.where(np.mod(u, 1) < f, eps1, eps2)
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 5))
axes[0].plot(u, eps_x); axes[0].set_ylabel(r'$\epsilon_r(x)$')
axes[1].plot(u, 1/eps_x); axes[1].set_ylabel(r'$\eta(x)=1/\epsilon_r(x)$')
axes[1].set_xlabel(r'Position $x/a$')
fig.suptitle('A binary unit cell repeated twice'); fig.tight_layout(); plt.show()


def eta_coeff(p, eps_a=eps1, eps_b=eps2, fill=f):
    """Analytic coefficient eta_p; accepts a scalar or array of integer indices."""
    p = np.asarray(p)
    contrast = 1/eps_a - 1/eps_b
    return ((p == 0)/eps_b + contrast*fill*np.sinc(p*fill)
            * np.exp(-1j*np.pi*p*fill))

p_demo = np.arange(-4, 5)
for p, value in zip(p_demo, eta_coeff(p_demo)):
    print(f'p={p:2d}: {value.real:+.7f} {value.imag:+.7f}j')
assert np.allclose(eta_coeff(0), f/eps1+(1-f)/eps2)
assert np.allclose(eta_coeff(-p_demo), np.conj(eta_coeff(p_demo)))
pnz = p_demo[p_demo != 0]
direct = (eta1-eta2)*(1-np.exp(-2j*np.pi*pnz*f))/(2j*np.pi*pnz)
assert np.allclose(eta_coeff(pnz), direct)
print('Mean, conjugate symmetry, and nonzero-index formula: PASS')


p_check = np.arange(-30, 31)
exact = eta_coeff(p_check)
fft_errors = []
sample_sizes = [256, 1024, 4096, 16384]
for N in sample_sizes:
    assert 2*np.max(np.abs(p_check)) < N  # avoid Nyquist ambiguity
    x_over_a = np.arange(N)/N
    samples = np.where(x_over_a < f, eta1, eta2)
    fft_coeffs = np.fft.fft(samples)/N
    estimated = fft_coeffs[p_check % N]
    err = np.max(np.abs(estimated-exact))
    fft_errors.append(err)
    print(f'N={N:5d}: max absolute coefficient error = {err:.3e}')
# A conservative finite-grid check, scaled to the material contrast.
assert fft_errors[-1] < 10*abs(eta1-eta2)/sample_sizes[-1] + 1e-12
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(p_check, exact.real, 'o', label='analytic real')
axes[0].plot(p_check, estimated.real, '+', label='FFT real')
axes[0].plot(p_check, exact.imag, '.', label='analytic imaginary')
axes[0].plot(p_check, estimated.imag, 'x', label='FFT imaginary')
axes[0].set(xlabel='Coefficient index p', ylabel='Coefficient'); axes[0].legend()
axes[1].loglog(sample_sizes, np.maximum(fft_errors, 1e-16), 'o-')
axes[1].set(xlabel='FFT samples N', ylabel='Maximum absolute error')
fig.tight_layout(); plt.show()


M = 20                      # 41 plane waves
num_bands = 5                # lowest five eigenfrequencies
q_values = np.linspace(-0.5, 0.5, 201)  # includes q=0 and zone edges
m = np.arange(-M, M+1)
G = 2*np.pi*m/a              # physical reciprocal vectors, rad/m
k_values = 2*np.pi*q_values/a
assert num_bands < len(m)
print(f'{len(m)} plane waves; {len(q_values)} Bloch wavevectors')
print(f'First Brillouin zone: [{-np.pi/a:.3e}, {np.pi/a:.3e}] rad/m')


def pwe_matrix(q, cutoff, eps_a=eps1, eps_b=eps2, fill=f):
    indices = np.arange(-cutoff, cutoff+1)
    differences = indices[:, None] - indices[None, :]
    eta_matrix = eta_coeff(differences, eps_a, eps_b, fill)
    momenta = q + indices
    return momenta[:, None]*eta_matrix*momenta[None, :]

def solve_bands(q_grid, cutoff, count, eps_a=eps1, eps_b=eps2, fill=f):
    assert 1 <= count <= 2*cutoff+1
    result = np.empty((len(q_grid), count))
    for j, q in enumerate(q_grid):
        B = pwe_matrix(q, cutoff, eps_a, eps_b, fill)
        assert np.allclose(B, B.conj().T, rtol=1e-12, atol=1e-12)
        eigenvalues = np.linalg.eigvalsh(B)  # ascending real eigenvalues
        tolerance = 1e-11*max(1.0, np.max(np.abs(eigenvalues)))
        if eigenvalues[0] < -tolerance:
            raise ValueError('Unexpected negative eigenvalue: check matrix construction')
        result[j] = np.sqrt(np.maximum(eigenvalues[:count], 0.0))
    return result

bands = solve_bands(q_values, M, num_bands)
print('Array shape (k points, bands):', bands.shape)
# Check the physical matrix and the dimensionless matrix at the same q.
q_test = 0.23
k_test = q_test*2*np.pi/a
A = ((k_test+G)[:, None]*(k_test+G)[None, :]
     *eta_coeff(m[:, None]-m[None, :]))
assert np.allclose(A*(a/(2*np.pi))**2, pwe_matrix(q_test, M))
print('Hermiticity, positivity, and unit conversion: PASS')


zero_index = np.argmin(np.abs(q_values))
assert bands[zero_index, 0] < 1e-6
assert np.allclose(bands, bands[::-1], atol=1e-6)
q_uniform = np.array([-0.5, -0.17, 0.0, 0.31, 0.5])
n_uniform = 1.5
uniform = solve_bands(q_uniform, M, num_bands, n_uniform**2, n_uniform**2)
expected = np.sort(np.abs(q_uniform[:, None]+m[None, :])/n_uniform, axis=1)[:, :num_bands]
assert np.allclose(uniform, expected, atol=1e-7)
print('Zero-frequency mode, k symmetry, and uniform-medium dispersion: PASS')


def find_gaps(values, threshold=1e-5):
    gaps = []
    for b in range(values.shape[1]-1):
        lower = values[:, b].max()
        upper = values[:, b+1].min()
        if upper-lower > threshold:
            gaps.append((b+1, lower, upper))  # human band numbers start at 1
    return gaps

gaps = find_gaps(bands)
print('Band pair    lower       upper       width       relative width')
for b, lower, upper in gaps:
    print(f'{b:2d}-{b+1:<2d}      {lower:.6f}    {upper:.6f}    '
          f'{upper-lower:.6f}    {100*(upper-lower)/((upper+lower)/2):.2f}%')
if not gaps:
    print('No gaps detected between these bands on this grid.')

fig, ax = plt.subplots(figsize=(9, 6))
for b in range(num_bands):
    ax.plot(q_values, bands[:, b], label=f'Band {b+1}')
for j, (b, lower, upper) in enumerate(gaps):
    ax.axhspan(lower, upper, color='gold', alpha=0.3,
               label='Candidate gap' if j == 0 else None)
ax.set(xlabel=r'Bloch wavevector $q=ka/(2\pi)$',
       ylabel=r'Normalized frequency $\Omega=\omega a/(2\pi c)$',
       title=f'1D H-field PWE: n1={n1:g}, n2={n2:g}, f={f:g}, M={M}',
       xlim=(-0.5, 0.5), ylim=(0, bands.max()*1.04))
ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5]); ax.legend(loc='upper right', fontsize=9)
fig.tight_layout(); plt.show()


q_convergence = np.linspace(-0.5, 0.5, 101)
previous = None
convergence_edges = []
for cutoff in [10, 20, 40, 80]:
    trial = solve_bands(q_convergence, cutoff, 2)
    edges = np.array([trial[:, 0].max(), trial[:, 1].min()])
    change = np.nan if previous is None else np.max(np.abs(edges-previous))
    print(f'M={cutoff:2d}, waves={2*cutoff+1:3d}: '
          f'edges={edges[0]:.7f}, {edges[1]:.7f}; max change={change:.3e}')
    convergence_edges.append(edges)
    previous = edges

q_fine = np.linspace(-0.5, 0.5, 401)
fine = solve_bands(q_fine, 80, 2)
fine_edges = np.array([fine[:, 0].max(), fine[:, 1].min()])
print('101 -> 401 k points, M=80: edge change =', np.max(np.abs(fine_edges-previous)))
print('M=20 plot vs M=80 refined first-gap edges:',
      np.array([bands[:, 0].max(), bands[:, 1].min()])-fine_edges)


lower, upper = fine_edges
if upper > lower + 1e-5 and lower > 0:
    print(f'Refined first gap: Omega = {lower:.7f} to {upper:.7f}')
    print(f'Frequency interval: {lower*c0/a/1e12:.2f} to {upper*c0/a/1e12:.2f} THz')
    print(f'Vacuum wavelength interval: {a/upper*1e9:.2f} to {a/lower*1e9:.2f} nm')
else:
    print('No positive first gap to convert.')
