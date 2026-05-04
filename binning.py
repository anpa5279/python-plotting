import os
import numpy as np
import h5py

from reader import OceananigansData
from interpolation import velocities_to_center
from plotting_functions import plot_binning

# Set up folder and simulation parameters
folder = '/Users/annapauls/Library/CloudStorage/OneDrive-UCB-O365/CU-Boulder/TESLa/Carbon Sequestration/Simulations/Oceananigans/NBP/salinity and temperature/no noise circle inlet/S0 = 0.1 dTdz = 0.01 MLD = 60 WENO mod callback'
output_folder = os.path.join(folder, 'binning')
os.makedirs(output_folder, exist_ok=True)
file_path = os.path.join(output_folder, 'binning_rtz.h5')

reader = OceananigansData(folder)
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
hor_vel_rz = np.empty((counts.size, nx[2], nt))
w_rz = np.empty((counts.size, nx[2], nt))

for it, t in enumerate(reader.t_save):
    # Load data from files
    T = reader.lazy_field('T', t)
    S = reader.lazy_field('S', t)
    u = reader.lazy_field('u', t)
    v = reader.lazy_field('v', t)
    w = reader.lazy_field('w', t)

    u, v, w = velocities_to_center(u, v, w)
    # u and v 
    u[:nx[0]//2 - 1, :, :] = -u[:nx[0]//2 - 1, :, :]
    v[:, :nx[1]//2 - 1, :] = -v[:, :nx[1]//2 - 1, :]
    u_sign = np.sign(u)
    v_sign = np.sign(v)
    hor_vel = np.sqrt(u**2 + v**2)
    hor_vel = -hor_vel[(u_sign + v_sign)== -2]
    T_avg = np.mean(T, axis=(-3, -2), keepdims=True)
    T_fluc = T - T_avg
    for k in range(nx[2]):
        S_rz[:, k, it] = np.bincount(r_bin.flat, weights=S[:, :, k].flat) 
        T_fluc_rz[:, k, it] = np.bincount(r_bin.flat, weights=T_fluc[:, :, k].flat) 
        T_rz[:, k, it] = np.bincount(r_bin.flat, weights=T[:, :, k].flat) 
        hor_vel_rz[:, k, it] = np.bincount(r_bin.flat, weights=hor_vel[:, :, k].flat)
        w_rz[:, k, it] = np.bincount(r_bin.flat, weights=w[:, :, k].flat) 

# cut off the corners that aren't full circles.
S_rz = (1 / counts[:ncirc, None, None]) * S_rz[:ncirc, :, :]
T_fluc_rz = (1 / counts[:ncirc, None, None]) * T_fluc_rz[:ncirc, :, :]
T_rz = (1 / counts[:ncirc, None, None]) * T_rz[:ncirc, :, :]
w_rz = (1 / counts[:ncirc, None, None]) * w_rz[:ncirc, :, :]
hor_vel_rz = (1 / counts[:ncirc, None, None]) * hor_vel_rz[:ncirc, :, :]
# write to file 
with h5py.File(file_path, "a") as f:
    f.create_dataset("ccc/dimensions/r_bin", data = r)
    f.create_dataset("ccc/dimensions/z", data=z)
    f.create_dataset("ccc/dimensions/time", data=time)
    f.create_dataset("ccc/S_rz", data=S_rz)
    f.create_dataset("ccc/T'_rz", data=T_fluc_rz)
    f.create_dataset("ccc/T_rz", data=T_rz)
    f.create_dataset("ccc/horizontal velocity", data=hor_vel_rz)
    f.create_dataset("ccc/w_rz", data=w_rz)
f.close()

plot_binning(S_rz, T_fluc_rz, T_rz, hor_vel_rz, w_rz, r, z, time, output_folder)