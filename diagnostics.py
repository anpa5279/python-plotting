import numpy as np
import dask.array as da
import os 
import h5py

from interpolation import velocities_to_center
from physics import rms
import scipy.sparse as sp

### -------------------------COLLECTING COMPARISON CASE INFO----------------- ###
def comparison_info(variations, universal_folder):
    wp = -0.001
    if variations == 'WENO':
        folder_names = ['S0 = 0.1 dTdz = 0.01 MLD = 60', 'S0 = 0.1 dTdz = 0.01 MLD = 60 WENO mod', 'S0 = 0.1 dTdz = 0.01 MLD = 60 WENO mod callback']
        case_names = [r'Default', r'WENO modified', r'WENO modified with callback 0 function']
        num_cases = len(folder_names)
        dTdz = 0.01 * np.ones(num_cases) # background temperature gradient in K/m
        mld = 60 * np.ones(num_cases)
        F_s = wp * 0.1 * np.ones(num_cases)
    else:
        print("Variation type not recognized.")
        return None # user defined specific case not defined here
    folder_names = [os.path.join(universal_folder, folder) for folder in folder_names]
    # Set up folder and simulation parameters
    fig_folder = os.path.join(universal_folder, 'comparison figures', variations + ' comparison figures')
    case_info = {
        "folder_names": folder_names,
        "fig_folder": fig_folder,
        "case_names": case_names,
        "num_cases": num_cases,
        "dTdz": dTdz,
        "mld": mld,
        "F_s": F_s
    }
    return case_info
### -------------------------BINNING----------------------------------------- ###
def azimuthal_avg(var, X, Y, dx_scale = None, return_r = False):
    if dx_scale is None:
        X_bin = X
        dx_scale = max([np.diff(X)[0, 0], np.diff(Y, axis=0)[0, 0]]) 
    else:
        x_bin = np.arange(X.min(), X.max()+dx_scale, dx_scale)
        y_bin = np.arange(Y.min(), Y.max()+dx_scale, dx_scale)
        X_bin, Y_bin = np.meshgrid(x_bin, y_bin)

    nx = X_bin.shape
    ncirc = min(nx[0], nx[1])
    if ncirc % 2 == 0:
        ncirc = ncirc//2      # full circular shells
    else:
        ncirc = ncirc//2+1      # full circular shells
    r_bin = np.sqrt((X/dx_scale)**2 + (Y/dx_scale)**2).astype(int)
    counts = np.bincount(r_bin.flat)  # number of points in each radial shell, including corners
    print(f'r_bin shape: {r_bin.shape}, var shape: {var.shape}')
    bin_var = np.bincount(r_bin.flat, weights=var.flat)

    # cut off the corners that aren't full circles.
    bin_var = (1 / counts[:ncirc]) * bin_var[:ncirc]
    if return_r:
        r = np.arange(np.min(np.abs([X_bin, Y_bin]))/2, ncirc*dx_scale, dx_scale)
        return r, bin_var
    return bin_var

def radial_binning_matrix(x, y, dx_scale, nr):
    """Sparse (nr, nx*ny) operator that azimuthally averages a flattened (nx, ny) plane."""
    X, Y = np.meshgrid(x, y, indexing='ij')
    r_bin = np.sqrt((X / dx_scale)**2 + (Y / dx_scale)**2).astype(int).ravel()
    counts = np.bincount(r_bin)[:nr]
    keep = np.flatnonzero(r_bin < nr)
    rows = r_bin[keep]
    vals = 1.0 / counts[rows]
    return sp.csr_matrix((vals, (rows, keep)), shape=(nr, r_bin.size))


def binning_oc_multi(reader, names, center=(0.0, 0.0), nt_max=None):
    """
    Bin several variables in one pass, loading each raw field once per time step.
    names: subset of ['T', 'S', 'w', 'ur', 'utheta']
    Returns dict of arrays with shape (nt, nr, nz).
    """
    nx = reader.nx
    t_save = reader.t_save if nt_max is None else reader.t_save[:nt_max]
    nt = len(t_save)
    dx_scale = np.max(reader.dx[:-1])
    x = reader.x - center[0]
    y = reader.y - center[1]
    nr = np.min(nx[:-1]) // 2

    M = radial_binning_matrix(x, y, dx_scale, nr)
    X2 = x[:, None, None]
    Y2 = y[None, :, None]

    out = {n: np.empty((nt, nr, nx[2])) for n in names}
    need_vel = 'ur' in names or 'utheta' in names

    for it, t in enumerate(t_save):
        print(f'Binning time step {it+1}/{nt}', flush=True)
        fields = {}
        if need_vel:
            u = velocities_to_center(reader.lazy_field('u', t).compute(), axis=-3)
            v = velocities_to_center(reader.lazy_field('v', t).compute(), axis=-2)
            ur, utheta = vel_to_cylindrical(u, v, X2, Y2)   # broadcasts, no 3D meshgrid
            fields['ur'], fields['utheta'] = ur, utheta
            del u, v
        for n in ('T', 'S'):
            if n in names:
                fields[n] = reader.lazy_field(n, t).compute()
        if 'w' in names:
            fields['w'] = velocities_to_center(reader.lazy_field('w', t).compute(), axis=-1)

        for n in names:
            out[n][it] = M @ fields[n].reshape(nx[0] * nx[1], nx[2])   # (nr, nz)
        del fields
    return out
### -------------------------CYLINDRICAL POLAR COORDINATES------------------- ###
def vel_to_cylindrical(u, v, X, Y):
    dist = np.sqrt(X**2 + Y**2)
    ur = u*X/dist + v*Y/dist
    utheta = -u*Y/dist + v*X/dist
    return ur, utheta