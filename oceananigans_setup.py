import os
import numpy as np
import h5py
import math

from reader import OceananigansData
from diagnostics import compute_temporal_averages, compute_fluct_averages, compute_rms, binning_oc
from interpolation import vertical_line

# set flags
binning_flag = True # creates binning of S, T, u, w in r-z space with the S and w contour values
centerline_flag = True # creates vertical line of S, T, u, w at x = 0, y = 0 for all time steps
planelsice_flag = True # creates plane slices of S, T, u, v, w at x = 0 for all time steps
buoyancy_flag = True
fluc_flag = True # calculates turbulent statistics from binning information
rms_flag = True # calculates RMS from 3D fields
compute_temporal_averages_flag = False # computes temporal averages of S and w at the default contour value and writes to file
contour_flag = True # calculates radius of contour at each depth and time that is not in the default
mass_flag = True

# model options
with_halos = True
salinity = True

# update flags if salinity is False
if not salinity:
    compute_temporal_averages_flag = False
    contour_flag = False
    mass_flag = False

# Set up folder and simulation parameters
folder = '/glade/derecho/scratch/apauls/outputs/version109/square-inlet/open-bottom-BC/AR1/dxi025'

print(f"Reading data from {folder}")
bin_path = os.path.join(folder, 'binning_rtz.h5')

g = 9.80665
T0 = 25.0

reader = OceananigansData(folder, salinity = salinity, with_halos=with_halos, Sval=0.1)
# grid info
nx = reader.nx
nt = reader.nt
dx = reader.dx
lx = reader.lx
time = reader.t


dx_scale = max(dx[:-1]) # not including dz
r = np.arange(dx[0]/2, lx[0]/2, dx_scale)
x, y, z = reader.x, reader.y, reader.z
ncirc = min(nx[0], nx[1])//2      # full circular shells

###------------APPLYING AZIMUTHAL AVERAGING TO DATA-----------------###
if binning_flag:
    # write to file 
    with h5py.File(bin_path, "a") as f:
        if "ccc/dimensions/r_bin" in f:
            del f["ccc/dimensions/r_bin"]
        if "ccc/dimensions/z" in f:
            del f["ccc/dimensions/z"]
        if "ccc/dimensions/time" in f:
            del f["ccc/dimensions/time"]
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
    if reader.salinity:
        S_rz = binning_oc('S', reader)
        with h5py.File(bin_path, "a") as f:
            if "ccc/S" in f:
                del f["ccc/S"]
            f.create_dataset("ccc/S", data=S_rz)
    T_rz= binning_oc('T', reader)
    with h5py.File(bin_path, "a") as f:
        f.create_dataset("ccc/T", data=T_rz)
    del T_rz
    ur_rz = binning_oc('ur', reader)
    with h5py.File(bin_path, "a") as f:
        f.create_dataset("ccc/horizontal velocity", data=ur_rz)
    del ur_rz
    utheta_rz = binning_oc('utheta', reader)
    with h5py.File(bin_path, "a") as f:
        f.create_dataset("ccc/rotation velocity", data=utheta_rz)
    del utheta_rz
    w_rz = binning_oc('w', reader)
    with h5py.File(bin_path, "a") as f:
        f.create_dataset("ccc/w", data=w_rz)
    del w_rz
    print(f"Saved binning to {bin_path}")
    reader.binning = True
    reader.bin_file = 'binning_rtz.h5'
###------------INTERPOLATION TO CENTERLINE--------------------------###
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
    del S, T, u, v, w
    print(f"Saved centerlines to {file_path}")
    reader.centerline = True
    reader.centerline_file = 'centerline.h5'
###------------INTERPOLATION TO PLANESLICE--------------------------###
if planelsice_flag:
    xy = True
    yz = True
    xz = False
    file_path = os.path.join(folder, 'plane_slice.h5')
    if xy:
        z_locs = [-reader.dx[-1]/2, 0.0]#[0.0, ]#
        for z_loc in z_locs:
            T = reader.field_slice('T', plane = 'XY', loc = z_loc)
            u = reader.field_slice('u', plane = 'XY', loc = z_loc)
            v = reader.field_slice('v', plane = 'XY', loc = z_loc)
            w = reader.field_slice('w', plane = 'XY', loc = z_loc)
            if reader.salinity:
                S = reader.field_slice('S', plane = 'XY', loc = z_loc)
            with h5py.File(file_path, "a") as f:
                if f"XY/z = {z_loc}/T" in f:
                    del f[f"XY/z = {z_loc}/T"]
                if f"XY/z = {z_loc}/u" in f:
                    del f[f"XY/z = {z_loc}/u"]
                if f"XY/z = {z_loc}/v" in f:
                    del f[f"XY/z = {z_loc}/v"]
                if f"XY/z = {z_loc}/w" in f:
                    del f[f"XY/z = {z_loc}/w"]
                if reader.salinity:
                    if f"XY/z = {z_loc}/S" in f:
                        del f[f"XY/z = {z_loc}/S"]
                    f.create_dataset(f"XY/z = {z_loc}/S", data = S)
                f.create_dataset(f"XY/z = {z_loc}/T", data=T)
                f.create_dataset(f"XY/z = {z_loc}/u", data=u)
                f.create_dataset(f"XY/z = {z_loc}/v", data=v)
                f.create_dataset(f"XY/z = {z_loc}/w", data=w)
                del S, T, u, v, w
    if yz:
        T = reader.field_slice('T')
        u = reader.field_slice('u')
        v = reader.field_slice('v')
        w = reader.field_slice('w')
        if reader.salinity:
            S = reader.field_slice('S')
        with h5py.File(file_path, "a") as f:
            if "YZ/x = 0/T" in f:
                del f["YZ/x = 0/T"]
            if "YZ/x = 0/u" in f:
                del f["YZ/x = 0/u"]
            if "YZ/x = 0/v" in f:
                del f["YZ/x = 0/v"]
            if "YZ/x = 0/w" in f:
                del f["YZ/x = 0/w"]
            if reader.salinity:
                if "YZ/x = 0/S" in f:
                    del f["YZ/x = 0/S"]
                f.create_dataset("YZ/x = 0/S", data = S)
            f.create_dataset("YZ/x = 0/T", data=T)
            f.create_dataset("YZ/x = 0/u", data=u)
            f.create_dataset("YZ/x = 0/v", data=v)
            f.create_dataset("YZ/x = 0/w", data=w)
        del S, T, u, v, w
    
    print(f"Saved plane slices to {file_path}")
###------------BUOYANCY CALCULATIONS--------------------------------###
if buoyancy_flag:
    buoyancy_file = os.path.join(folder, 'buoyancy_profile.h5')
    alpha = reader.alpha
    T = reader.lazy_field('T').compute()
    b_profile = g * alpha * (T - T0)
    if reader.salinity:
        beta = reader.beta
        S = reader.lazy_field('S').compute()
        b_profile += - g * beta * S
        del S
    del T
    b_avg = np.mean(b_profile, axis=(-3, -2))
    b_fluc = b_profile - b_avg[:, None, None, :]
    b_rms = np.mean(b_fluc**2, axis=(-3, -2))**0.5
    if reader.averaging:
        T_avg = reader.load_averages('T')
        b_avg = g * alpha * (T_avg - T0) 
        if reader.salinity:
            S_avg = reader.load_averages('S')
            beta = reader.beta
            b_avg += - g * beta * S_avg
            del S_avg
        del T_avg
    if not reader.centerline:
        b_centerline = vertical_line(b_profile, x = reader.x, y = reader.y)
        b_fluc_centerline = vertical_line(b_fluc, x = reader.x, y = reader.y)
    with h5py.File(buoyancy_file, "a") as f:
        if "z" in f:
            del f["z"]
        if "b_rms" in f:
            del f["b_rms"]
        if "b_avg" in f:
            del f["b_avg"]
        f.create_dataset("z", data = z)
        f.create_dataset("b_rms", data = b_rms)
        f.create_dataset("b_avg", data = b_avg)
        if not reader.centerline:
            f.create_dataset("centerline/b", data = b_centerline)
            f.create_dataset("centerline/b_fluc", data = b_fluc_centerline)
    del b_profile
    print(f"Saved buoyancy information to {buoyancy_file}")
###------------FLUCTUATION AVERAGES---------------------------------###
if fluc_flag:
    file_path = os.path.join(folder, 'fluctuations.h5')
    data = compute_fluct_averages(reader)
    with h5py.File(file_path, "a") as f:
        if "fluctuations/T_fluc" in f:
            del f["fluctuations/T_fluc"]
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
        f.create_dataset("fluctuations/ur_fluc", data=data['ur_fluc'])
        f.create_dataset("fluctuations/utheta_fluc", data=data['utheta_fluc'])
        f.create_dataset("fluctuations/w_fluc", data=data['w_fluc'])
        f.create_dataset("fluctuations/b_fluc", data=data['b_fluc'])
        f.create_dataset("fluctuations/bur_fluc", data=data['bur_fluc'])
        f.create_dataset("fluctuations/butheta_fluc", data=data['butheta_fluc'])
        f.create_dataset("fluctuations/bw_fluc", data=data['bw_fluc'])
        if salinity:
            if "fluctuations/S_fluc" in f:
                del f["fluctuations/S_fluc"]
            f.create_dataset("fluctuations/S_fluc", data=data['S_fluc'])
    print(f"Saved fluctuations to {file_path}")
###------------ROOT MEAN SQUARE-------------------------------------###
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
    del rms_values
    print(f"Saved RMS to {file_path}")
###------------TEMPORAL AVERAGES------------------------------------###
if compute_temporal_averages_flag:
    start = 10
    if reader.averaging:
        start = start*100
    data_temp = compute_temporal_averages(reader, start = start)
    # compute radius of plume 
    data = {
        'S_value': data_temp['S_value'],
        'w_value': data_temp['w_value'], 
    }
    folder_contour = f"contour temporal averages"

    with h5py.File(bin_path, "a") as f:
        if folder_contour in f:
            del f[folder_contour]
        f.create_group(f'{folder_contour}')
        f.create_dataset(f'{folder_contour}/S', data=data['S_value'])
        f.create_dataset(f'{folder_contour}/w', data=data['w_value'])
    f.close()
    del data_temp
    print(f"Saved temporal averages to {bin_path}")
###------------PLUME CONTOURS---------------------------------------###
if contour_flag:
    contours = np.array([0.001, 0.005, 0.01, 0.05])
    S_value = reader.load_S_temporal_avg()
    S_rz = reader.load_binning_var('S')

    for contour in contours:
        r_contour = np.zeros((nx[2], nt))
        level = S_value * contour

        for it in range(nt):
            radius_tracer = np.zeros(nx[2])

            for k in range(nx[2]):
                S_radial = S_rz[:ncirc, k, it]

                # Guard 1: level not reached at this depth/time
                if np.max(S_radial) < level:
                    continue

                # Orient so r is ascending and S trends downward outward
                if S_radial[0] < S_radial[-1]:
                    S_radial = S_radial[::-1]
                    r_search = r[::-1]
                else:
                    r_search = r

                # Guard 2: trim to the bracketing region around the crossing
                above = np.where(S_radial >= level)[0]
                if len(above) == 0:
                    continue
                i_last = above[-1]
                i_end = min(i_last + 2, len(S_radial))
                S_trimmed = S_radial[:i_end]
                r_trimmed = r_search[:i_end]

                if len(S_trimmed) < 2:
                    radius_tracer[k] = r_trimmed[-1] if len(S_trimmed) else 0.0
                    continue

                # If we never drop below `level` in the trimmed window,
                # take the last (outermost) sample as the best estimate
                above_mask = S_trimmed >= level
                if above_mask.all():
                    radius_tracer[k] = r_trimmed[-1]
                    continue

                # Find the first index where S drops below level; the
                # crossing is bracketed by (i1, i2) = (last above, first below)
                idx_below = np.where(~above_mask)[0]
                i2 = idx_below[0]
                i1 = i2 - 1

                if i1 < 0:
                    # Level exceeded already at the first trimmed point
                    radius_tracer[k] = r_trimmed[0]
                    continue

                S1, S2 = S_trimmed[i1], S_trimmed[i2]
                r1, r2 = r_trimmed[i1], r_trimmed[i2]

                if S1 == S2:
                    radius_tracer[k] = r1
                else:
                    frac = (level - S1) / (S2 - S1)
                    radius_tracer[k] = r1 + frac * (r2 - r1)

            # Sanity clip: radius can never be negative or exceed grid extent
            radius_tracer = np.clip(radius_tracer, 0.0, r.max())
            r_contour[:, it] = radius_tracer

        with h5py.File(bin_path, "a") as f:
            key = f"r given contour/contour = {contour}"
            if key in f:
                del f[key]
            f.create_dataset(key, data=r_contour)

    print(f"Saved contours to {bin_path}")
###------------MASS CALCULATIONS------------------------------------###
if mass_flag:
    rho0 = 1026 # kg/m^3
    S = reader.lazy_field('S').compute()
    vol = math.prod(reader.lx)
    dims = (1, 2, 3)
    Smin = np.min(S, axis = dims)
    Smax = np.max(S, axis = dims)
    # volume integral of S value in domain
    S_mass = np.mean(S, axis = dims)*vol*rho0
    del S
    dmdt = np.gradient(S_mass, time)
    with h5py.File(bin_path, "a") as f:
        if "S mass" in f:
            del f["S mass"]
        if "time gradient of S mass" in f:
            del f["time gradient of S mass"]
        if "max of S" in f:
            del f["max of S"]
        if "min of S" in f:
            del f["min of S"]
        f.create_dataset("S mass", data=S_mass)
        f.create_dataset("time gradient of S mass", data=dmdt)
        f.create_dataset("max of S", data=Smax)
        f.create_dataset("min of S", data=Smin)
    print(f"Saved mass calculations to {bin_path}")