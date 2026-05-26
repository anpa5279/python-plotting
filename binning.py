import os
import numpy as np
import h5py

from reader import OceananigansData
from physics import buoyancy
from interpolation import velocities_to_center
from plotting_planes import plot_binning
# set flags
binning_flag = True
plot_flag = False
contour_flag = True
salinity = True

# Set up folder and simulation parameters
folder = '/Users/annapauls/Documents/TESLa /Simulations/Oceananigans/dense plume/salinity and temperature/no noise circle inlet/domain resolution testing/horizontal resolution/sj0.1-mld60-dTdz0.01-lx320-nx384'
output_folder = os.path.join(folder, 'binning')
os.makedirs(output_folder, exist_ok=True)
file_path = os.path.join(output_folder, 'binning_rtz.h5')

reader = OceananigansData(folder, salinity = salinity)
if contour_flag:
    contours = np.array([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05])
    S_value, w_value = reader.load_contour_temporal_averages('interp_temporal_averages.h5')

if binning_flag:
    reader.load_equation_of_state()
    # grid info
    reader.load_grid()
    x, y, z = reader.x, reader.y, reader.z
    nx = reader.nx
    dx = reader.dx
    hx = reader.hx
    lx = reader.lx
    # load time and equation of state info
    time, t_save = reader.load_time()

    X, Y, Z= np.meshgrid(x, y, z)
    dist = np.sqrt(X**2 + Y**2)
    dx_scale = max(dx[:-1])
    r = np.arange(dx[0]/2, lx[0]/2, dx_scale)
    r_bin = np.sqrt((X[:, :, 0]/dx_scale)**2 + (Y[:, :, 0]/dx_scale)**2).astype(int)
    r_max = r_bin.max() + 1 
    counts = np.bincount(r_bin.flat)      # number of points in each radial shell, including corners
    ncirc = max(nx[0], nx[1])//2      # full circular shells

    nt = len(t_save)
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
            utheta_rz[:, k, it] = np.bincount(r_bin.flat, weights=ur[:, :, k].flat)
            w_rz[:, k, it] = np.bincount(r_bin.flat, weights=w[:, :, k].flat) 

    # cut off the corners that aren't full circles.
    S_rz = (1 / counts[:ncirc, None, None]) * S_rz[:ncirc, :, :]
    T_rz = (1 / counts[:ncirc, None, None]) * T_rz[:ncirc, :, :]
    w_rz = (1 / counts[:ncirc, None, None]) * w_rz[:ncirc, :, :]
    utheta_rz = (1 / counts[:ncirc, None, None]) * utheta_rz[:ncirc, :, :]
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
else:
    r, z, time, S_rz, T_rz, ur_rz, w_rz = reader.load_binning()

if contour_flag: # calculate radius of contour at each depth and time that is not in the default
    reader.load_grid()
    reader.load_time()
    nx = reader.nx
    nt = len(reader.t_save)
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
        """
        for it, t in enumerate(reader.t_save):
            plume = S_rz[:, :, it] >= S_value*contour
            ri, zi = np.where(plume)
            plume_index = [ri, zi]
            counts = np.bincount(zi, minlength=nx[2])

            r_values = r[ri]
            sums   = np.bincount(zi, weights=r_values, minlength=nx[2])

            radius_tracer = np.zeros(nx[2])
            mask = counts > 0
            if np.any(mask):
                radius_tracer[mask] = sums[mask] / counts[mask]
            else:
                radius_tracer = np.zeros(nx[2])
            r_contour[:, it] = radius_tracer
            """
        with h5py.File(file_path, "a") as f:
            if f"r given contour/contour = {contour}" in f:
                del f[f"r given contour/contour = {contour}"]
            f.create_dataset(f"r given contour/contour = {contour}", data = r_contour)
    f.close()
if plot_flag:
    plot_binning(S_rz, T_rz, ur_rz, w_rz, r, z, time, output_folder)