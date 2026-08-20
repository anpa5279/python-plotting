import os
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import colors

def plot_frequency_spectrum(omega, P_omega, N_profile, z, z_ref=None, n_peaks=5, ax=None,):
    """
    Plot the spatially averaged vertical-velocity frequency spectrum.

    Parameters
    ----------
    omega : 1D array
        Angular frequency [rad/s].
    P_omega : 1D array
        Spatially averaged power spectrum.
    N_profile : 1D array
        Brunt-Vaisala frequency N(z) [rad/s].
    z : 1D array
        Vertical coordinates [m].
    z_ref : float, optional
        Depth at which to plot N as a reference value.
    n_peaks : int
        Number of dominant peaks to identify.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    P = np.asarray(P_omega).copy()
    omega = np.asarray(omega)

    # Ignore zero-frequency component
    P[0] = 0.0

    # Dominant frequencies
    peak_indices = np.argsort(P)[-n_peaks:]
    peak_indices = np.sort(peak_indices)

    ax.semilogy(omega, P, linewidth=1.8, label=r'$\langle |\hat{w}|^2 \rangle_{y,z}$')

    # Mark dominant peaks
    ax.scatter(omega[peak_indices], P[peak_indices], zorder=5)

    for i in peak_indices:
        ax.annotate(fr'$\omega={omega[i]:.3e}$', (omega[i], P[i]), xytext=(5, 5), textcoords='offset points', fontsize=9)

    # Reference N
    if z_ref is not None:
        N_ref = np.interp(z_ref, z, N_profile)

        ax.axvline(N_ref, linestyle='--', linewidth=1.5, label=fr'$N(z={z_ref:g}\,\mathrm{{m}})$')

    ax.set_xlabel(r'$\omega$ [rad s$^{-1}$]')
    ax.set_ylabel(r'Power spectral density of $w^\prime$')
    ax.set_title('Internal-wave frequency spectrum at domain edge')

    ax.set_xlim(left=0)

    ax.legend()

    return ax, peak_indices
def plot_frequency_maps(omega, power, y, z, N_profile=None, peak_indices=None, n_peaks=4, log_power=True,):
    """
    Plot spatial maps of vertical-velocity spectral power
    at dominant frequencies on the x = xmax edge plane.
    """

    omega = np.asarray(omega)

    if peak_indices is None:

        P_omega = np.mean(power, axis=(1, 2))

        P_search = P_omega.copy()
        P_search[0] = 0.0

        peak_indices = np.argsort(P_search)[-n_peaks:]

        peak_indices = np.sort(peak_indices)

    nplot = len(peak_indices)

    ncols = min(2, nplot)
    nrows = int(np.ceil(nplot / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4.5 * nrows), squeeze=False, constrained_layout=True)

    axes = axes.ravel()

    for j, i in enumerate(peak_indices):

        ax = axes[j]

        P = power[i].copy()

        if log_power:
            P_plot = np.log10(P + np.finfo(float).eps)
            label = r'$\log_{10} |\hat{w}|^2$'
        else:
            P_plot = P
            label = r'$|\hat{w}|^2$'

        im = ax.imshow(P_plot.T, origin='lower', extent=[
                y[0], y[-1], z[0], z[-1]
            ], aspect='auto', interpolation='none')

        # Plot N(z) = omega boundary
        if N_profile is not None:

            N_map = np.broadcast_to(N_profile, (len(y), len(z)))

            try:
                ax.contour(y,     z,     N_map.T,     levels=[omega[i]],     linewidths=2, )
            except ValueError:
                pass

        ax.set_xlabel(r'$y$ [m]')
        ax.set_ylabel(r'$z$ [m]')

        ax.set_title(fr'$\omega={omega[i]:.4e}$ rad s$^{{-1}}$')

        cbar = fig.colorbar(im, ax=ax)

        cbar.set_label(label)

    # Remove unused axes
    for j in range(nplot, len(axes)):
        axes[j].remove()

    fig.suptitle(r'Internal-wave spectral power at $x=x_{\max}$', fontsize=15)

    return fig, axes

def plot_N_profile(z, N_profile, omega_peaks=None, ax=None,):
    """
    Plot Brunt-Vaisala frequency profile and optionally
    dominant wave frequencies.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 7))

    z = np.asarray(z)
    N_profile = np.asarray(N_profile)

    ax.plot(N_profile, z, linewidth=2, label=r'$N(z)$')

    if omega_peaks is not None:

        for j, omega_peak in enumerate(omega_peaks):

            ax.axvline(omega_peak, linestyle='--', linewidth=1.3, label=(fr'$\omega_{j+1}='
                    fr'{omega_peak:.3e}$'))

    ax.set_xlabel(r'$N$ [rad s$^{-1}$]')

    ax.set_ylabel(r'$z$ [m]')

    ax.set_title('Background stratification and wave frequencies')

    ax.set_xlim(left=0)

    ax.legend()

    return ax

def plot_wave_energy(u_fluc, v_fluc, w_fluc, b_fluc, N_profile, y, z, rho0=1026.0, ax=None,):
    """
    Plot mean internal-wave energy density on the edge plane.

    Parameters
    ----------
    u_fluc, v_fluc, w_fluc : ndarray
        Fluctuating velocity fields with shape (time, y, z).
    b_fluc : ndarray
        Buoyancy fluctuation with shape (time, y, z).
    N_profile : ndarray
        Brunt-Vaisala frequency N(z).
    y, z : ndarray
        Edge-plane coordinates.
    rho0 : float
        Reference density [kg/m^3].
    """

    N2 = N_profile**2

    # Avoid division by zero
    N2_safe = np.maximum(N2, np.finfo(float).tiny)

    # Kinetic energy
    E_K = 0.5 * rho0 * (u_fluc**2
        + v_fluc**2
        + w_fluc**2)

    # Available potential energy
    E_P = (0.5
        * rho0
        * b_fluc**2
        / N2_safe[None, None, :])

    E_total = E_K + E_P

    # Time-average
    E_K_mean = np.mean(E_K, axis=0)
    E_P_mean = np.mean(E_P, axis=0)
    E_total_mean = np.mean(E_total, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    # ------------------------------------------------------
    # Spatial energy map
    # ------------------------------------------------------

    ax0 = axes[0]

    E_plot = np.log10(E_total_mean + np.finfo(float).tiny)

    im = ax0.imshow(E_plot.T, origin='lower', extent=[
            y[0], y[-1], z[0], z[-1]
        ], aspect='auto', interpolation='none')

    ax0.set_xlabel(r'$y$ [m]')
    ax0.set_ylabel(r'$z$ [m]')
    ax0.set_title(r'Time-mean wave energy '
        r'$\log_{10}(E_{\mathrm{wave}})$')

    cbar = fig.colorbar(im, ax=ax0)

    cbar.set_label(r'$\log_{10}$ [J m$^{-3}$]')

    # ------------------------------------------------------
    # Horizontally averaged vertical profile
    # ------------------------------------------------------

    ax1 = axes[1]

    E_K_z = np.mean(E_K_mean, axis=0)

    E_P_z = np.mean(E_P_mean, axis=0)

    E_total_z = np.mean(E_total_mean, axis=0)

    ax1.semilogx(E_K_z, z, label=r'$E_K$')

    ax1.semilogx(E_P_z, z, label=r'$E_P$')

    ax1.semilogx(E_total_z, z, linewidth=2, label=r'$E_{\mathrm{wave}}$')

    ax1.set_xlabel(r'Energy density [J m$^{-3}$]')

    ax1.set_ylabel(r'$z$ [m]')

    ax1.set_title('Wave-energy vertical structure')

    ax1.legend()

    return {
        "figure": fig, "E_K": E_K_mean, "E_P": E_P_mean, "E_total": E_total_mean, }