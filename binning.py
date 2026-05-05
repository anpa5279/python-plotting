import os
import numpy as np
import h5py

from reader import OceananigansData
from physics import buoyancy
from interpolation import velocities_to_center
from plotting_functions import plot_binning
# set flags
binning_flag = True

# Set up folder and simulation parameters
folder = '/Users/annapauls/Documents/TESLa /Simulations/Oceananigans/dense plume/salinity and temperature/no noise circle inlet/S0 = 0.2 dTdz = 0.01 MLD = 60'
output_folder = os.path.join(folder, 'binning')
os.makedirs(output_folder, exist_ok=True)
file_path = os.path.join(output_folder, 'binning_rtz.h5')

if binning_flag:

    reader = OceananigansData(folder)

    # inital paramreters
    rho0 = 1026
    T0 = 25
    coeffs = reader.load_equation_of_state(True)
    # grid info
    reader.load_grid()
    x, y, z = reader.x, reader.y, reader.z
    nx = reader.nx
    dx = reader.dx
    hx = reader.hx
    lx = reader.lx
    # load time and equation of state info
    time, t_save = reader.load_time()

    X, Y = np.meshgrid(x, y)
    dist = np.sqrt(X**2 + Y**2)
    dx_scale = max(dx[:-1])
    r = np.arange(dx[0]/2, lx[0]/2, dx_scale)
    r_bin = np.sqrt((X/dx_scale)**2 + (Y/dx_scale)**2).astype(int)
    r_max = r_bin.max() + 1 
    counts = np.bincount(r_bin.flat)      # number of points in each radial shell, including corners
    ncirc = max(nx[0], nx[1])//2      # full circular shells

    nt = len(t_save)
    S_rz = np.empty((counts.size, nx[2], nt)) 
    T_fluc_rz = np.empty((counts.size, nx[2], nt)) 
    T_rz = np.empty((counts.size, nx[2], nt)) 
    ur_rz = np.empty((counts.size, nx[2], nt))
    w_rz = np.empty((counts.size, nx[2], nt))
    b_fluc_rz = np.empty((counts.size, nx[2], nt))

    for it, t in enumerate(reader.t_save):
        # Load data from files
        T = reader.lazy_field('T', t)
        S = reader.lazy_field('S', t)
        u = reader.lazy_field('u', t)
        v = reader.lazy_field('v', t)
        w = reader.lazy_field('w', t)

        u, v, w = velocities_to_center(u, v, w)
        # u and v 
        ur = u*X/dist + v*Y/dist
        T_avg = np.mean(T, axis=(-3, -2), keepdims=True)
        T_fluc = T - T_avg

        # calculate b and b_fluc
        bs = buoyancy(T, rho0, coeffs, T0, S)
        b = bs['b_total']
        b_avg = np.mean(b, axis=(-3, -2), keepdims=True)
        b_fluc = b - b_avg
        for k in range(nx[2]):
            S_rz[:, k, it] = np.bincount(r_bin.flat, weights=S[:, :, k].flat) 
            T_fluc_rz[:, k, it] = np.bincount(r_bin.flat, weights=T_fluc[:, :, k].flat) 
            T_rz[:, k, it] = np.bincount(r_bin.flat, weights=T[:, :, k].flat) 
            ur_rz[:, k, it] = np.bincount(r_bin.flat, weights=ur[:, :, k].flat)
            w_rz[:, k, it] = np.bincount(r_bin.flat, weights=w[:, :, k].flat) 
            b_fluc_rz[:, k, it] = np.bincount(r_bin.flat, weights=b_fluc[:, :, k].flat)

    # cut off the corners that aren't full circles.
    S_rz = (1 / counts[:ncirc, None, None]) * S_rz[:ncirc, :, :]
    T_fluc_rz = (1 / counts[:ncirc, None, None]) * T_fluc_rz[:ncirc, :, :]
    T_rz = (1 / counts[:ncirc, None, None]) * T_rz[:ncirc, :, :]
    w_rz = (1 / counts[:ncirc, None, None]) * w_rz[:ncirc, :, :]
    ur_rz = (1 / counts[:ncirc, None, None]) * ur_rz[:ncirc, :, :]
    b_fluc_rz = (1 / counts[:ncirc, None, None]) * b_fluc_rz[:ncirc, :, :]
    # write to file 
    with h5py.File(file_path, "a") as f:
        f.create_dataset("ccc/dimensions/r_bin", data = r)
        f.create_dataset("ccc/dimensions/z", data=z)
        f.create_dataset("ccc/dimensions/time", data=time)
        f.create_dataset("ccc/S_rz", data=S_rz)
        f.create_dataset("ccc/T'_rz", data=T_fluc_rz)
        f.create_dataset("ccc/T_rz", data=T_rz)
        f.create_dataset("ccc/b'_rz", data=b_fluc_rz)
        f.create_dataset("ccc/horizontal velocity", data=ur_rz)
        f.create_dataset("ccc/w_rz", data=w_rz)
    f.close()
else:
    with h5py.File(file_path, "r") as f:
        r = f["ccc/dimensions/r_bin"][:]
        z = f["ccc/dimensions/z"][:]
        time = f["ccc/dimensions/time"][:]
        S_rz = f["ccc/S_rz"][:]
        T_fluc_rz = f["ccc/T'_rz"][:]
        T_rz = f["ccc/T_rz"][:]
        b_fluc_rz = f["ccc/b'_rz"][:]
        ur_rz = f["ccc/horizontal velocity"][:]
        w_rz = f["ccc/w_rz"][:]
    f.close()

plot_binning(S_rz, T_fluc_rz, T_rz, ur_rz, w_rz, b_fluc_rz, r, z, time, output_folder)