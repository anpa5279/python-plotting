import numpy as np
import dask.array as da
import os 
import h5py

from interpolation import velocities_to_center
from physics import rms

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
def binning_oc(var, reader, center=(0.0, 0.0)):
    nx = reader.nx
    t_save = reader.t_save
    nt = len(t_save)

    dx_scale = np.max(reader.dx[:-1]) # not including dz
    x = reader.x - center[0]
    y = reader.y - center[1]
    z = reader.z
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    nr = np.min(nx[:-1])//2
    var_rz = np.empty((nr, nx[2], nt))

    for it, t in enumerate(t_save):
        if var=='ur' or var=='utheta':
            u = reader.lazy_field('u', t).compute()
            u = velocities_to_center(u, axis=-3)
            v = reader.lazy_field('v', t).compute()
            v = velocities_to_center(u, axis=-2)
            # u and v to cylindrical polar coordinates
            ur, utheta = vel_to_cylindrical(u, v, X, Y)
            if var=='ur':
                var_field = ur
            else:
                var_field = utheta
            del u, v
        else:
            # Load data from files
            var_field = reader.lazy_field(var, t).compute()
        if var =='u':
            var_field = velocities_to_center(var_field, axis=-3)
        elif var == 'v':
            var_field = velocities_to_center(var_field, axis=-2)
        elif var == 'w':
            var_field = velocities_to_center(var_field, axis=-1)
        for k in range(nx[2]):
            var_rz[:, k, it] = azimuthal_avg(var_field[:, :, k], X[:, :, k], Y[:, :, k], dx_scale=dx_scale)
        del var_field # to be memory efficient
    return var_rz

### -------------------------CYLINDRICAL POLAR COORDINATES------------------- ###
def vel_to_cylindrical(u, v, X, Y):
    dist = np.sqrt(X**2 + Y**2)
    ur = u*X/dist + v*Y/dist
    utheta = -u*Y/dist + v*X/dist
    return ur, utheta