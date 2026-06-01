import os
import numpy as np
import h5py

from reader import OceananigansData
from diagnostics import compute_fluct_averages, compute_rms
from interpolation import velocities_to_center

# set flags
contour_flag = False # calculates radius of contour at each depth and time that is not in the default
planelsice_flag = True # creates plane slices of S, T, u, v, w at x = 0 for all time steps
fluc_flag = False # calculates turbulent statistics from binning information
rms_flag = False # calculates RMS from 3D fields

salinity = True
idx_slice = False

# Set up folder and simulation parameters
folder = '/glade/derecho/scratch/apauls/outputs/version109/horizontal-domain/coarse2'
#'/Users/annapauls/Documents/TESLa /Simulations/Oceananigans/dense plume/salinity and temperature/no noise circle inlet/domain testing/Lz = 160m/S0 = 0.1 dTdz = 0.01 MLD = 60'
print(f"Reading data from {folder}")
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

if planelsice_flag:
    file_path = os.path.join(folder, 'plane_slice.h5')
    if idx_slice:
        x_opt = reader.nx[0]//2
        x_save = reader.x[x_opt]
        T = reader.field_slice('T', N = x_opt)
        S = reader.field_slice('S', N = x_opt)
        u = reader.field_slice('u', N = x_opt)
        v = reader.field_slice('v', N = x_opt)
        w = reader.field_slice('w', N = x_opt)
    else:
        x_save = 0.0
        T = reader.field_slice('T', loc = x_save)
        S = reader.field_slice('S', loc = x_save)
        u = reader.field_slice('u', loc = x_save)
        v = reader.field_slice('v', loc = x_save)
        w = reader.field_slice('w', loc = x_save)
    with h5py.File(file_path, "a") as f:
        if f"YZ/x = {x_save}/S" in f:
            del f[f"YZ/x = {x_save}/S"]
        if f"YZ/x = {x_save}/T" in f:
            del f[f"YZ/x = {x_save}/T"]
        if f"YZ/x = {x_save}/u" in f:
            del f[f"YZ/x = {x_save}/u"]
        if f"YZ/x = {x_save}/v" in f:
            del f[f"YZ/x = {x_save}/v"]
        if f"YZ/x = {x_save}/w" in f:
            del f[f"YZ/x = {x_save}/w"]
        f.create_dataset(f"YZ/x = {x_save}/S", data = S)
        f.create_dataset(f"YZ/x = {x_save}/T", data=T)
        f.create_dataset(f"YZ/x = {x_save}/u", data=u)
        f.create_dataset(f"YZ/x = {x_save}/v", data=v)
        f.create_dataset(f"YZ/x = {x_save}/w", data=w)
    f.close()
    print(f"Saved plane slices to {file_path}")

if fluc_flag:
    file_path = os.path.join(folder, 'fluctuations.h5')
    data = compute_fluct_averages(reader)
    with h5py.File(file_path, "a") as f:
        if "fluctuations/T_fluc" in f:
            del f["fluctuations/T_fluc"]
        if "fluctuations/S_fluc" in f:
            del f["fluctuations/S_fluc"]
        if "fluctuations/ur_fluc" in f:
            del f["fluctuations/ur_fluc"]
        if "fluctuations/utheta_fluc" in f:
            del f["fluctuations/utheta_fluc"]
        if "fluctuations/w_fluc" in f:
            del f["fluctuations/w_fluc"]
        if "fluctuations/b_fluc" in f:
            del f["fluctuations/b_fluc"]
        if "fluctuations/bur_fluc" in f:
            del f["fluctuations/bur_fluc"]
        if "fluctuations/butheta_fluc" in f:
            del f["fluctuations/butheta_fluc"]
        if "fluctuations/bw_fluc" in f:
            del f["fluctuations/bw_fluc"]
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
    print(f"Saved fluctuations to {file_path}")

if rms_flag:
    file_path = os.path.join(folder, 'fluctuations.h5')
    rms_values = compute_rms(reader)
    with h5py.File(file_path, "a") as f:
        if "rms/u" in f:
            del f["rms/u"]
        if "rms/v" in f:
            del f["rms/v"]
        if "rms/w" in f:
            del f["rms/w"]
        f.create_dataset("rms/u", data=rms_values['u_rms'])
        f.create_dataset("rms/v", data=rms_values['v_rms'])
        f.create_dataset("rms/w", data=rms_values['w_rms'])
    f.close()
    print(f"Saved RMS to {file_path}")