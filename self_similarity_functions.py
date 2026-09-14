import os
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from plotting_general import save_frame

# ----------------------------------------------------------------------
# 1. Centerline velocity and half-width
# ----------------------------------------------------------------------

def _half_width_1d(r, wprof, wc_val, threshold=0.5):
    """
    Radius at which a single 1D profile wprof(r) first crosses
    threshold*wc_val, via linear interpolation between the bracketing
    grid points. Returns np.nan if no crossing is found.
    """
    target = threshold * wc_val
    if not np.isfinite(wc_val) or wc_val == 0:
        return np.nan

    below = np.where(wprof <= target)[0]
    if below.size == 0:
        return np.nan  # profile never drops to half the centerline value

    i1 = below[0]
    if i1 == 0:
        # first grid point already at/below target -> can't bracket
        return r[0]

    i0 = i1 - 1
    w0, w1 = wprof[i0], wprof[i1]
    if w1 == w0:
        return r[i1]

    frac = (target - w0) / (w1 - w0)
    return r[i0] + frac * (r[i1] - r[i0])

def compute_delta(w, r, wc, threshold=0.5):
    """
    Half-width delta(t, z): the radius where w(r, z, t) = threshold * w_c(t, z).

    w  : ndarray, shape (nt, nr, nz)
    r  : ndarray, shape (nr,)
    wc : ndarray, shape (nt, nz) 

    Returns
    -------
    delta : ndarray, shape (nt, nz)
    """
    if w.ndim == 3:
        nt, nr, nz = w.shape
        delta = np.empty((nt, nz))
        for t in range(nt):
            for k in range(nz):
                delta[t, k] = _half_width_1d(r, w[t, :, k], wc[t, k], threshold)
        return delta
    else:
        nr, nz = w.shape
        delta = np.empty((nz))
        for k in range(nz):
            delta[k] = _half_width_1d(r, w[:, k], wc[k], threshold)
        return delta

# ----------------------------------------------------------------------
# 2. Power-law fit via curve_fit in log-log space
# ----------------------------------------------------------------------

def _linear_model(log_z, log_C, alpha):
    """log(y) = log(C) + alpha * log(z) -- linear in log-log space."""
    return log_C + alpha * log_z


def fit_power_law(z, y):
    """
    Fit y = C * |z|^alpha by linear regression of log(y) vs log(|z|)
    using scipy.optimize.curve_fit.

    z, y : 1D arrays (same length). Non-finite, non-positive, or
    zero-crossing entries are dropped before fitting.

    Returns
    -------
    C, alpha, C_err, alpha_err
    """
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)

    zabs = np.abs(z)
    mask = np.isfinite(zabs) & np.isfinite(y) & (zabs > 0) & (y > 0)
    if mask.sum() < 2:
        return np.nan, np.nan, np.nan, np.nan

    log_z = np.log(zabs[mask])
    log_y = np.log(y[mask])

    popt, pcov = curve_fit(_linear_model, log_z, log_y, p0=[log_y[0], -1.0])
    log_C, alpha = popt
    log_C_err, alpha_err = np.sqrt(np.diag(pcov))

    C = np.exp(log_C)
    C_err = C * log_C_err  # propagate error through exp()
    return C, alpha, C_err, alpha_err


def fit_power_law_all_times(z, Y):
    """
    Apply fit_power_law independently for each time index.

    z : ndarray, shape (nz,)
    Y : ndarray, shape (nt, nz)   (e.g. wc or delta)

    Returns
    -------
    C, alpha, C_err, alpha_err : ndarrays, shape (nt,)
    """
    nt = Y.shape[0]
    C = np.empty(nt)
    alpha = np.empty(nt)
    C_err = np.empty(nt)
    alpha_err = np.empty(nt)
    for t in range(nt):
        C[t], alpha[t], C_err[t], alpha_err[t] = fit_power_law(z, Y[t, :])
    return C, alpha, C_err, alpha_err


# ----------------------------------------------------------------------
# 3. Plotting: log(w_c) vs log(z) and log(delta) vs log(z), with fit
# ----------------------------------------------------------------------

def plot_loglog_fit(outdir, z, y, C, alpha, ylabel, leg_label, title, range, it = ''):
    zabs = np.abs(z)
    mask = np.isfinite(zabs) & np.isfinite(y) & (zabs > 0) & (y > 0)

    size_in = (6, 4)
    fig, ax = plt.subplots()

    ax.scatter(zabs[mask], y[mask], s=20, label="data")

    if np.isfinite(alpha):
        z_fit = np.linspace(zabs[mask].min(), zabs[mask].max(), 100)
        y_fit = C * z_fit ** alpha
        ax.plot(z_fit, y_fit, "r-",
                label=leg_label)

    ax.set_xlabel("z [m]")
    ax.set_ylabel(ylabel)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(range)
    ax.set_title(title)
    ax.legend(loc = 'upper left')
    if isinstance(it, str):
        frame_path = os.path.join(outdir, it + '.png')
        plt.savefig(frame_path, dpi = 200)
    else:
        save_frame(fig, outdir, it, size_in)