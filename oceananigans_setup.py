import os
import numpy as np
import h5py

from reader import OceananigansData
from diagnostics import compute_temporal_averages, write_temporal_averages, compute_fluct_averages
from interpolation import velocities_to_center

# set flags
compute_temporal_averages_flag = True # computes temporal averages of S and w at the default contour value and writes to file
binning_flag = True # creates binning of S, T, u, w in r-z space with the S and w contour values
contour_flag = True # calculates radius of contour at each depth and time that is not in the default
planelsice_flag = True # creates plane slices of S, T, u, v, w at x = 0 for all time steps
fluc_flag = True # calculates turbulent statistics from binning information

salinity = True

# Set up folder and simulation parameters
folder = '/glade/derecho/scratch/apauls/outputs/horizontal-domain/sj0.1-mld60-dTdz0.01-lx-400-nx320'
file_path = os.path.join(folder, 'binning_rtz.h5')

g = 9.80665
T0 = 25.0

reader = OceananigansData(folder, salinity = salinity)
reader.load_grid()
# grid info
nx = reader.nx
dx = reader.dx
hx = reader.hx
lx = reader.lx
time, t_save = reader.load_time()
nt = len(t_save)
reader.load_equation_of_state()
if compute_temporal_averages_flag:
    data_temp = compute_temporal_averages(reader)
    # compute radius of plume 
    data = {
        'S_value': data_temp['S_value'],
        'w_value': data_temp['w_value'], 
    }
    write_temporal_averages(file_path, data)

if binning_flag:
    x, y, z = reader.x, reader.y, reader.z
    X, Y, Z= np.meshgrid(x, y, z)
    dist = np.sqrt(X**2 + Y**2)
    dx_scale = max(dx[:-1])
    r = np.arange(dx[0]/2, lx[0]/2, dx_scale)
    r_bin = np.sqrt((X[:, :, 0]/dx_scale)**2 + (Y[:, :, 0]/dx_scale)**2).astype(int)
    r_max = r_bin.max() + 1 
    counts = np.bincount(r_bin.flat)      # number of points in each radial shell, including corners
    ncirc = max(nx[0], nx[1])//2      # full circular shells

    S_rz = np.empty((counts.size, nx[2], nt)) 
    T_rz = np.empty((counts.size, nx[2], nt)) 
    ur_rz = np.empty((counts.size, nx[2], nt))
    utheta_rz = np.empty((counts.size, nx[2], nt))
    w_rz = np.empty((counts.size, nx[2], nt))

    for it, t in enumerate(reader.t_save):
        # Load data from files
        T = np.array(reader.lazy_field('T', t))
        S = np.array(reader.lazy_field('S', t))
        u = np.array(reader.lazy_field('u', t))
        v = np.array(reader.lazy_field('v', t))
        w = np.array(reader.lazy_field('w', t))

        u = velocities_to_center(u, axis=0)
        v = velocities_to_center(v, axis=1)
        w = velocities_to_center(w, axis=2)

        # u and v 
        ur = u*X/dist + v*Y/dist
        utheta = -u*Y/dist + v*X/dist

        for k in range(nx[2]):
            S_rz[:, k, it] = np.bincount(r_bin.flat, weights=S[:, :, k].flat) 
            T_rz[:, k, it] = np.bincount(r_bin.flat, weights=T[:, :, k].flat) 
            utheta_rz[:, k, it] = np.bincount(r_bin.flat, weights=utheta[:, :, k].flat)
            ur_rz[:, k, it] = np.bincount(r_bin.flat, weights=ur[:, :, k].flat)
            w_rz[:, k, it] = np.bincount(r_bin.flat, weights=w[:, :, k].flat) 
    # cut off the corners that aren't full circles.
    S_rz = (1 / counts[:ncirc, None, None]) * S_rz[:ncirc, :, :]
    T_rz = (1 / counts[:ncirc, None, None]) * T_rz[:ncirc, :, :]
    w_rz = (1 / counts[:ncirc, None, None]) * w_rz[:ncirc, :, :]
    utheta_rz = (1 / counts[:ncirc, None, None]) * utheta_rz[:ncirc, :, :]
    ur_rz = (1 / counts[:ncirc, None, None]) * ur_rz[:ncirc, :, :]
    # write to file 
    with h5py.File(file_path, "a") as f:
        if "ccc/dimensions/r_bin" in f:
            del f["ccc/dimensions/r_bin"]
        if "ccc/dimensions/z" in f:
            del f["ccc/dimensions/z"]
        if "ccc/dimensions/time" in f:
            del f["ccc/dimensions/time"]
        if "ccc/S_rz" in f:
            del f["ccc/S_rz"]
        if "ccc/T_rz" in f:
            del f["ccc/T_rz"]
        if "ccc/horizontal velocity" in f:
            del f["ccc/horizontal velocity"]
        if "ccc/rotation velocity" in f:
            del f["ccc/rotation velocity"]
        if "ccc/w_rz" in f:
            del f["ccc/w_rz"]
        f.create_dataset("ccc/dimensions/r_bin", data = r)
        f.create_dataset("ccc/dimensions/z", data=z)
        f.create_dataset("ccc/dimensions/time", data=time)
        f.create_dataset("ccc/S_rz", data=S_rz)
        f.create_dataset("ccc/T_rz", data=T_rz)
        f.create_dataset("ccc/horizontal velocity", data=ur_rz)
        f.create_dataset("ccc/rotation velocity", data=utheta_rz)
        f.create_dataset("ccc/w_rz", data=w_rz)
    f.close()

if contour_flag: # calculate radius of contour at each depth and time that is not in the default
    contours = np.array([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05])
    S_value, w_value = reader.load_contour_temporal_averages(file_path)
    if not binning_flag:
        S_rz = reader.load_binning_var('S_rz')
    for contour in contours:
        r_contour = np.empty((nx[2], nt))
        for it, t in enumerate(reader.t_save):
            plume = S_rz[:, :, it] >= S_value * contour  # shape: (nr, nz)
            
            radius_tracer = np.zeros(nx[2])
            for k in range(nx[2]):
                plume_at_depth = plume[:, k]
                if np.any(plume_at_depth):
                    radius_tracer[k] = r[np.max(np.where(plume_at_depth))]
                # else stays 0
            
            r_contour[:, it] = radius_tracer
        with h5py.File(file_path, "a") as f:
            if f"r given contour/contour = {contour}" in f:
                del f[f"r given contour/contour = {contour}"]
            f.create_dataset(f"r given contour/contour = {contour}", data = r_contour)
    f.close()

if planelsice_flag:
    file_path = os.path.join(folder, 'plane_slice.h5')
    T = reader.field_slice('T')
    S = reader.field_slice('S')
    u = reader.field_slice('u')
    v = reader.field_slice('v')
    w = reader.field_slice('w')
    with h5py.File(file_path, "a") as f:
        if "YZ/x = 0/S" in f:
            del f["YZ/x = 0/S"]
        if "YZ/x = 0/T" in f:
            del f["YZ/x = 0/T"]
        if "YZ/x = 0/u" in f:
            del f["YZ/x = 0/u"]
        if "YZ/x = 0/v" in f:
            del f["YZ/x = 0/v"]
        if "YZ/x = 0/w" in f:
            del f["YZ/x = 0/w"]
        f.create_dataset("YZ/x = 0/S", data = S)
        f.create_dataset("YZ/x = 0/T", data=T)
        f.create_dataset("YZ/x = 0/u", data=u)
        f.create_dataset("YZ/x = 0/v", data=v)
        f.create_dataset("YZ/x = 0/w", data=w)
    f.close()

if fluc_flag:
    file_path = os.path.join(folder, 'fluctuations.h5')
    data = compute_fluct_averages(reader)
    with h5py.File(file_path, "a") as f:
        if "fluctuations/T_fluc" in f:
            del f["fluctuations/T_fluc"]
        if "fluctuations/S_fluc" in f:
            del f["fluctuations/S_fluc"]
        f.create_dataset("fluctuations/T_fluc", data=data['T_fluc'])
        f.create_dataset("fluctuations/S_fluc", data=data['S_fluc'])
        f.create_dataset("fluctuations/ur_fluc", data=data['ur_fluc'])
        f.create_dataset("fluctuations/utheta_fluc", data=data['utheta_fluc'])
        f.create_dataset("fluctuations/w_fluc", data=data['w_fluc'])
        f.create_dataset("fluctuations/b_fluc", data=data['b_fluc'])
        f.create_dataset("fluctuations/bur_fluc", data=data['bu_fluc'])
        f.create_dataset("fluctuations/butheta_fluc", data=data['bv_fluc'])
        f.create_dataset("fluctuations/bw_fluc", data=data['bw_fluc'])
    f.close()