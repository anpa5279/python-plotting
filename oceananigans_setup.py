import os
import numpy as np
import h5py

from reader import OceananigansData
from diagnostics import compute_temporal_averages, write_temporal_averages, compute_fluct_averages, compute_rms, binning_oc
from interpolation import interp1d_axis, vertical_line

# set flags
compute_temporal_averages_flag = True # computes temporal averages of S and w at the default contour value and writes to file
binning_flag = True # creates binning of S, T, u, w in r-z space with the S and w contour values
contour_flag = True # calculates radius of contour at each depth and time that is not in the default
centerline_flag = True # creates vertical line of S, T, u, w at x = 0, y = 0 for all time steps
planelsice_flag = True # creates plane slices of S, T, u, v, w at x = 0 for all time steps
fluc_flag = True # calculates turbulent statistics from binning information
rms_flag = True # calculates RMS from 3D fields
buoyancy_flag = True

salinity = True

# Set up folder and simulation parameters
folder = '/glade/derecho/scratch/apauls/outputs/version109/square-inlet/coarse2'
#'/Users/annapauls/Documents/TESLa /Simulations/Oceananigans/dense plume/salinity and temperature/no noise circle inlet/domain testing/Lz = 160m/S0 = 0.2 dTdz = 0.01 MLD = 60'
#

print(f"Reading data from {folder}")
file_path = os.path.join(folder, 'binning_rtz.h5')

g = 9.80665
T0 = 25.0

reader = OceananigansData(folder, salinity = salinity)
reader.load_grid(grid_specs = False)
# grid info
nx = reader.nx
dx = reader.dx
lx = reader.lx
time, t_save = reader.load_time()
nt = len(t_save)
reader.load_equation_of_state()

dx_scale = max(dx[:-1]) # not including dz
r = np.arange(dx[0]/2, lx[0]/2, dx_scale)
x, y, z = reader.x, reader.y, reader.z
X, Y, Z = np.meshgrid(x, y, z)
dist = np.sqrt(X**2 + Y**2)
ncirc = min(nx[0], nx[1])//2      # full circular shells
if buoyancy_flag:
    buoyancy_file = os.path.join(folder, 'buoyancy_profile.h5')
    alpha = reader.alpha
    if salinity:
        beta = reader.beta
    T = reader.lazy_field('T').compute()
    if salinity:
        beta = reader.beta
        S = reader.lazy_field('S').compute()
        b_profile = g * alpha * (T - T0) - g * beta * S
    else:
        b_profile = g * alpha * (T - T0)
    b_avg = np.mean(b_profile, axis=(-3, -2))
    b_fluc = b_profile - b_avg[:, None, None, :]
    b_rms = np.mean(b_fluc**2, axis=(-3, -2))**0.5
    with h5py.File(buoyancy_file, "a") as f:
        f.create_dataset("z", data = z)
        f.create_dataset("b_rms", data = b_rms)
        f.create_dataset("b_avg", data = b_avg)
    print(f"Saved buoyancy information to {buoyancy_file}")
if compute_temporal_averages_flag:
    data_temp = compute_temporal_averages(reader)
    # compute radius of plume 
    data = {
        'S_value': data_temp['S_value'],
        'w_value': data_temp['w_value'], 
    }
    write_temporal_averages(file_path, data)

if binning_flag:
    S_rz, T_rz, ur_rz, utheta_rz, w_rz = binning_oc(reader)
    # write to file 
    with h5py.File(file_path, "a") as f:
        if "ccc/dimensions/r_bin" in f:
            del f["ccc/dimensions/r_bin"]
        if "ccc/dimensions/z" in f:
            del f["ccc/dimensions/z"]
        if "ccc/dimensions/time" in f:
            del f["ccc/dimensions/time"]
        if "ccc/S" in f:
            del f["ccc/S"]
        if "ccc/T" in f:
            del f["ccc/T"]
        if "ccc/horizontal velocity" in f:
            del f["ccc/horizontal velocity"]
        if "ccc/rotation velocity" in f:
            del f["ccc/rotation velocity"]
        if "ccc/w" in f:
            del f["ccc/w"]
        f.create_dataset("ccc/dimensions/r_bin", data = r)
        f.create_dataset("ccc/dimensions/z", data=z)
        f.create_dataset("ccc/dimensions/time", data=time)
        f.create_dataset("ccc/S", data=S_rz)
        f.create_dataset("ccc/T", data=T_rz)
        f.create_dataset("ccc/horizontal velocity", data=ur_rz)
        f.create_dataset("ccc/rotation velocity", data=utheta_rz)
        f.create_dataset("ccc/w", data=w_rz)
    print(f"Saved binning to {file_path}")

if contour_flag:
    contours = np.array([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05])
    S_value = reader.load_S_temporal_avg(file_path)
    if not binning_flag:
        S_rz = reader.load_binning_var('S')

    for contour in contours:
        r_contour = np.zeros((nx[2], nt))

        for it in range(nt):
            radius_tracer = np.zeros(nx[2])
            level = S_value * contour

            for k in range(nx[2]):
                S_radial = S_rz[:ncirc, k, it]

                # Guard 1: level not reached at this depth/time
                if np.max(S_radial) < level:
                    continue

                # Guard 2: ensure monotonically decreasing outward
                if S_radial[0] < S_radial[-1]:
                    S_radial = S_radial[::-1]
                    r_search = r[::-1]
                else:
                    r_search = r

                # Guard 3: trim to only the region that brackets the level
                # avoids flat tails (divide by zero) and irrelevant outer zeros
                above = np.where(S_radial >= level)[0]
                if len(above) == 0:
                    continue
                # keep one point beyond the last above-threshold index
                i_last = above[-1]
                i_end = min(i_last + 2, len(S_radial))
                S_trimmed = S_radial[:i_end]
                r_trimmed = r_search[:i_end]

                # Guard 4: need at least 2 points and a sign change
                if len(S_trimmed) < 2:
                    continue
                if not np.any(np.diff(S_trimmed) != 0):
                    # flat profile — take the outermost above-threshold r directly
                    radius_tracer[k] = r_search[i_last]
                    continue

                r_interp = interp1d_axis(S_trimmed, r_trimmed, f_new=level)
                r_val = np.max(r_interp) if np.ndim(r_interp) > 0 else float(r_interp)
                radius_tracer[k] = r_val

            r_contour[:, it] = radius_tracer

        with h5py.File(file_path, "a") as f:
            key = f"r given contour/contour = {contour}"
            if key in f:
                del f[key]
            f.create_dataset(key, data=r_contour)

    print(f"Saved contours to {file_path}")
if centerline_flag:
    file_path = os.path.join(folder, 'centerline.h5')
    T = reader.field_centerline('T')
    S = reader.field_centerline('S')
    u = reader.field_centerline('u')
    v = reader.field_centerline('v')
    w = reader.field_centerline('w')
    with h5py.File(file_path, "a") as f:
        if "centerline/S" in f:
            del f["centerline/S"]
        if "centerline/T" in f:
            del f["centerline/T"]
        if "centerline/u" in f:
            del f["centerline/u"]
        if "centerline/v" in f:
            del f["centerline/v"]
        if "centerline/w" in f:
            del f["centerline/w"]
        f.create_dataset("centerline/S", data=S)
        f.create_dataset("centerline/T", data=T)
        f.create_dataset("centerline/u", data=u)
        f.create_dataset("centerline/v", data=v)
        f.create_dataset("centerline/w", data=w)
    
    print(f"Saved centerlines to {file_path}")
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
    
    print(f"Saved RMS to {file_path}")