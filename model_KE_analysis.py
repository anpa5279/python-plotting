"""
Inner-scale / convergence diagnostics for an implicit-LES (WENO) plume run, across resolutions.

Per case:
  1. 1D spectra of velocity fluctuations computed on the RAW STAGGERED fields (no centering:
     averaging faces to centres multiplies the spectrum by cos^2(k dx/2) and kills Nyquist).
     Longitudinal = u along x and v along y; transverse = w along x and y.
  2. Spectral collapse between successive runs: E_fine/E_coarse at wavelengths both resolve.
     Ratio ~ 1 means converged at that scale. This is the most direct convergence evidence here.
  3. eps from the inertial range, eps from u'^3/L11, and nu_eff = eps/(2<s'_ij s'_ij>), eta_eff.
     These are only computed if the fit window is >= res_cells*dx (otherwise printed as nan).
  4. Effective resolution (roll-off of k^(5/3)E) and a dx estimate for a target wavelength.

ASSUMPTIONS: lazy_field(n)[it] -> raw staggered (nx,ny,nz) [u,v] and (nx,ny,nz+1) [w]; x,y periodic;
uniform dz; same domain size for all cases; cases listed coarse -> fine.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# USER SETTINGS
# ==========================================================
base = '/Users/annapauls/Documents/Github repositories/3d_langmuir_gpu/localoutputs/scheme-tests/longer/WENO9'
cases = ['dx2.0', 'dx1.0', 'dx0.5']          # coarse -> fine

t_slice = slice(None)          # e.g. slice(10, None) to skip spin-up
periodic_xy = True
band_frac = 0.3                # z-band: levels where horizontally averaged w_rms >= band_frac * max
z_band_manual = None           # or (zmin, zmax) in metres
use_plot_format = False        # your plot_format() makes the fonts huge for this 3-panel figure

res_cells = 6                  # scales smaller than res_cells*dx are treated as NOT resolved
fit_wavelengths = (3.0, 6.0)   # shared inertial-range window (m); only used for runs that resolve it
C1 = 0.5                       # 1D longitudinal Kolmogorov constant
C1_T = 4.0 / 3.0 * C1          # transverse
roll_frac = 0.5
lam_target = 1.0               # smallest wavelength (m) you want the -5/3 range to reach

trapz = getattr(np, 'trapezoid', None) or np.trapz


# ==========================================================
# HELPERS
# ==========================================================
def spec1d(f, axis, dl, periodic=True):
    """One-sided 1D power spectrum along `axis`, averaged over other axes; sum(E)*dk = mean(f^2)."""
    n = f.shape[axis]
    w = np.ones(n) if periodic else np.hanning(n)
    shp = [1] * f.ndim
    shp[axis] = n
    F = np.fft.rfft(f * w.reshape(shp), axis=axis) / n
    P = np.abs(F) ** 2 / np.mean(w ** 2)
    P = np.moveaxis(P, axis, 0)
    P = P.reshape(P.shape[0], -1).mean(axis=1)
    if n % 2 == 0:
        P[1:-1] *= 2
    else:
        P[1:] *= 2
    k = 2 * np.pi * np.fft.rfftfreq(n, d=dl)
    return k[1:], P[1:] / k[1]


def mean_strain_sq(u, v, w, dx, dy, dz):
    """<s'_ij s'_ij> on the staggered grid. u:(nx,ny,nb) at (F,C,C), v at (C,F,C), w:(nx,ny,nb+1) at (C,C,F).
    Periodic in x,y. Diagonal terms at cell centres, off-diagonals at edges (no interpolation)."""
    nb = u.shape[2]
    s2 = (np.mean(((np.roll(u, -1, 0) - u) / dx) ** 2)
          + np.mean(((np.roll(v, -1, 1) - v) / dy) ** 2)
          + np.mean((np.diff(w, axis=2) / dz) ** 2))
    s12 = 0.5 * ((u - np.roll(u, 1, 1)) / dy + (v - np.roll(v, 1, 0)) / dx)
    wi = w[:, :, 1:nb]
    s13 = 0.5 * ((u[:, :, 1:] - u[:, :, :-1]) / dz + (wi - np.roll(wi, 1, 0)) / dx)
    s23 = 0.5 * ((v[:, :, 1:] - v[:, :, :-1]) / dz + (wi - np.roll(wi, 1, 1)) / dy)
    return s2 + 2 * (np.mean(s12 ** 2) + np.mean(s13 ** 2) + np.mean(s23 ** 2))


def _c(a):
    return np.asarray(a.compute() if hasattr(a, 'compute') else a)


# ==========================================================
# PER-CASE ANALYSIS (reads data)
# ==========================================================
def analyze(name):
    from reader import OceananigansData

    rd = OceananigansData(os.path.join(base, name), salinity=True, with_halos=True)
    dxs = np.asarray(rd.dx, float)[:3]
    dx, dy, dz = dxs
    fields = {n: rd.lazy_field(n) for n in 'uvw'}
    idx = np.arange(rd.nt)[t_slice]
    nt_used = len(idx)

    def snap(it):
        u, v, w = (_c(fields[n][it]) for n in 'uvw')
        if w.shape[2] == u.shape[2]:                       # top face not stored: w=0 there
            w = np.concatenate([w, np.zeros_like(w[:, :, :1])], axis=2)
        return [u, v, w]

    # ---- pass 1: time mean and w variance profile (on w faces) ----
    mean, w2z = None, 0.0
    for it in idx:
        s = snap(it)
        mean = s if mean is None else [m + q for m, q in zip(mean, s)]
        w2z = w2z + (s[2] ** 2).mean(axis=(0, 1))
    mean = [m / nt_used for m in mean]
    w2z = w2z / nt_used
    nx, ny, nz = mean[0].shape
    rms_f = np.sqrt(np.maximum(w2z - (mean[2] ** 2).mean(axis=(0, 1)), 0))
    rms_z = 0.5 * (rms_f[:-1] + rms_f[1:])

    z = np.asarray(rd.z, float)
    if len(z) != nz:
        print(f'[{name}] WARNING: len(z)={len(z)} != nz={nz} (halos?). Using index-based z.')
        z = (np.arange(nz) + 0.5) * dz
    m = ((z >= z_band_manual[0]) & (z <= z_band_manual[1])) if z_band_manual else rms_z >= band_frac * rms_z.max()
    k0, k1 = np.where(m)[0][[0, -1]]
    print(f'[{name}] {nt_used} snapshots, grid {nx}x{ny}x{nz}, z-band [{z[k0]:.1f}, {z[k1]:.1f}] m')

    # ---- pass 2: spectra and strain ----
    EL = Ew = S2 = 0.0
    k = None
    for it in idx:
        fl = [q - m_ for q, m_ in zip(snap(it), mean)]
        u, v = fl[0][:, :, k0:k1 + 1], fl[1][:, :, k0:k1 + 1]
        w = fl[2][:, :, k0:k1 + 2]                          # faces k0..k1+1 (nb+1 levels)
        kx, Eu = spec1d(u, 0, dx, periodic_xy)
        ky, Ev = spec1d(v, 1, dy, periodic_xy)
        _, Ewx = spec1d(w, 0, dx, periodic_xy)
        kyw, Ewy = spec1d(w, 1, dy, periodic_xy)
        EL = EL + 0.5 * (Eu + np.interp(kx, ky, Ev))
        Ew = Ew + 0.5 * (Ewx + np.interp(kx, kyw, Ewy))
        k = kx
        S2 += mean_strain_sq(u, v, w, dx, dy, dz)
    return dict(name=name, dx=dx, k=k, EL=EL / nt_used, Ew=Ew / nt_used, S2=S2 / nt_used,
                domain=np.array([nx, ny, nz]) * dxs)


# ==========================================================
# POST-PROCESSING (no data access)
# ==========================================================
def eps_from_spec(k, E, C):
    return (E * k ** (5 / 3) / C) ** 1.5


def postprocess(r):
    k, EL, Ew = r['k'], r['EL'], r['Ew']
    kl, kh = 2 * np.pi / fit_wavelengths[1], 2 * np.pi / fit_wavelengths[0]
    win = (k >= kl) & (k <= kh)
    r['eps_kL'], r['eps_kW'] = eps_from_spec(k, EL, C1), eps_from_spec(k, Ew, C1_T)
    ok = fit_wavelengths[0] >= res_cells * r['dx'] and win.any()
    r['win_ok'] = ok
    r['eps_L'] = np.median(r['eps_kL'][win]) if ok else np.nan
    r['eps_W'] = np.median(r['eps_kW'][win]) if ok else np.nan

    comp = EL * k ** (5 / 3)
    plateau = np.median(comp[win]) if ok else np.nan
    above = np.where((k > kh) & (comp < roll_frac * plateau))[0] if ok else []
    r['lam_eff'] = 2 * np.pi / k[above[0]] if len(above) else np.nan
    r['lam_eff_dx'] = r['lam_eff'] / r['dx']

    u2 = trapz(EL, k)
    r['L11'] = 0.5 * np.pi * trapz(EL / k, k) / u2
    r['eps_scale'] = u2 ** 1.5 / r['L11']
    r['nu_eff'] = r['eps_L'] / (2 * r['S2'])
    r['eta_eff'] = (r['nu_eff'] ** 3 / r['eps_L']) ** 0.25
    r['eta_over_dx'] = r['eta_eff'] / r['dx']
    return r


def collapse(results):
    print('\nSpectral collapse, E_fine / E_coarse (1 = converged at that scale). '
          f'Only wavelengths >= {res_cells}*dx_coarse are compared.')
    print(f"{'pair':>14} {'lambda (m)':>11} {'E_long':>8} {'E_w':>8}")
    for c, f in zip(results[:-1], results[1:]):
        n = min(len(c['k']), len(f['k']))
        if not np.allclose(c['k'][:n], f['k'][:n]):
            print(f"{c['name']}->{f['name']}: wavenumber grids differ (different domains?) - skipped")
            continue
        k = c['k'][:n]
        for lam in (32, 16, 8, 4, 2):
            if lam < res_cells * c['dx']:
                continue
            kc = 2 * np.pi / lam
            sel = (k >= kc / np.sqrt(2)) & (k <= kc * np.sqrt(2))
            if not sel.any():
                continue
            rl = np.median(f['EL'][:n][sel] / c['EL'][:n][sel])
            rw = np.median(f['Ew'][:n][sel] / c['Ew'][:n][sel])
            print(f"{c['name'] + '->' + f['name']:>14} {lam:11.1f} {rl:8.2f} {rw:8.2f}")


def report(results):
    print('\n' + '=' * 96)
    print(f"{'case':>8} {'dx':>7} {'lam_eff':>8} {'/dx':>5} {'eps_L':>10} {'eps_w':>10} "
          f"{'eps_scal':>10} {'<s2>':>9} {'nu_eff':>10} {'eta_eff':>8} {'/dx':>5}")
    for r in results:
        print(f"{r['name']:>8} {r['dx']:7.3f} {r['lam_eff']:8.3f} {r['lam_eff_dx']:5.1f} "
              f"{r['eps_L']:10.2e} {r['eps_W']:10.2e} {r['eps_scale']:10.2e} {r['S2']:9.2e} "
              f"{r['nu_eff']:10.2e} {r['eta_eff']:8.3f} {r['eta_over_dx']:5.2f}")
    print('=' * 96)
    print(f'nan = window {fit_wavelengths} m is smaller than {res_cells}*dx for that run (not resolved).')
    print('eps_scal = u\'^3/L11 (rough, uses the whole spectrum, no inertial range needed).')
    collapse(results)
    good = [r for r in results if np.isfinite(r['lam_eff_dx'])]
    if good:
        a = np.median([r['lam_eff_dx'] for r in good])
        dx_new = lam_target / a
        print(f'\nEffective resolution ~ {a:.1f} dx. To reach {lam_target} m you need dx ~ {dx_new:.3f} m '
              f'(~{np.prod(results[-1]["domain"] / dx_new):.2e} cells).')


def plot(results):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(results)))
    for r, c in zip(results, cols):
        k = r['k']
        ax[0].loglog(k, r['EL'], color=c, label=r['name'])
        ax[0].loglog(k, r['Ew'], color=c, ls=':', lw=1)
        ax[1].loglog(k, r['eps_kL'], color=c, label=r['name'])
        ax[2].loglog(k, r['eps_kW'], color=c, label=r['name'])
        for a in ax:
            a.axvline(np.pi / r['dx'], color=c, ls='--', lw=0.8)      # Nyquist
    kk = results[-1]['k']
    ref = next((r['eps_L'] for r in results[::-1] if np.isfinite(r['eps_L'])), None)
    if ref:
        ax[0].loglog(kk, C1 * ref ** (2 / 3) * kk ** (-5 / 3), 'k--', lw=1, label='-5/3')
    for a in ax:
        a.axvspan(2 * np.pi / fit_wavelengths[1], 2 * np.pi / fit_wavelengths[0], color='gray', alpha=0.15)
        a.set_xlabel('k (rad/m)')
        a.legend(fontsize=7)
    ax[0].set_ylabel('E(k)')
    ax[0].set_title('(a) solid: u,v long.; dotted: w')
    ax[1].set_ylabel('implied eps')
    ax[1].set_title('(b) longitudinal')
    ax[2].set_title('(c) transverse (w)')
    plt.show()


if __name__ == '__main__':
    if use_plot_format:
        from plotting_general import plot_format
        plot_format(fontsize=10)
    results = [postprocess(analyze(c)) for c in cases]
    report(results)
    plot(results)