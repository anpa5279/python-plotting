import os
import numpy as np
import h5py

from reader import OceananigansData

# set flags
binning_flag = False # creates binning of S, T, u, w in r-z space with the S and w contour values
centerline_flag = False # creates vertical line of S, T, u, w at x = 0, y = 0 for all time steps
planelsice_flag = False # creates plane slices of S, T, u, v, w at x = 0 for all time steps
buoyancy_flag = False
fluc_flag = False # calculates turbulent statistics from binning information
rms_flag = False # calculates RMS from 3D fields
compute_temporal_averages_flag = True # computes temporal averages of S and w at the default contour value and writes to file
contour_flag = True # calculates radius of contour at each depth and time that is not in the default
mass_flag = False

# model options
with_halos = False
salinity = True

# update flags if salinity is False
if not salinity:
    compute_temporal_averages_flag = False
    contour_flag = False
    mass_flag = False

# Set up folder and simulation parameters
folder = '/glade/derecho/scratch/apauls/outputs/version109/square-inlet/openBC/dx025/shorter'

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
reader.load_equation_of_state()

dx_scale = max(dx[:-1]) # not including dz
r = np.arange(dx[0]/2, lx[0]/2, dx_scale)
x, y, z = reader.x, reader.y, reader.z
X, Y, Z = np.meshgrid(x, y, z)
ncirc = min(nx[0], nx[1])//2      # full circular shells

###------------APPLYING AZIMUTHAL AVERAGING TO DATA-----------------###
if binning_flag:
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
    print(f"Deleted binning from {bin_path}")
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
    print(f"Deleted centerlines from {file_path}")
###------------INTERPOLATION TO PLANESLICE--------------------------###
if planelsice_flag:
    xy = False
    yz = True
    xz = False
    file_path = os.path.join(folder, 'plane_slice.h5')
    if xy:
        z_locs = [-reader.lx[-1] + reader.dx[-1]/2, -reader.lx[-1]/2, 0.0]
        for z_loc in z_locs:
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
    if yz:
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
        del S, T, u, v, w
    
    print(f"Deleted plane slices from {file_path}")
###------------BUOYANCY CALCULATIONS--------------------------------###
if buoyancy_flag:
    buoyancy_file = os.path.join(folder, 'buoyancy_profile.h5')
    with h5py.File(buoyancy_file, "a") as f:
        if "z" in f:
            del f["z"]
        if "b_rms" in f:
            del f["b_rms"]
        if "b_avg" in f:
            del f["b_avg"]
    print(f"Deleted buoyancy information from {buoyancy_file}")
###------------FLUCTUATION AVERAGES---------------------------------###
if fluc_flag:
    file_path = os.path.join(folder, 'fluctuations.h5')
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
        if salinity:
            if "fluctuations/S_fluc" in f:
                del f["fluctuations/S_fluc"]
    print(f"Deleted fluctuations from {file_path}")
###------------ROOT MEAN SQUARE-------------------------------------###
if rms_flag:
    file_path = os.path.join(folder, 'fluctuations.h5')
    with h5py.File(file_path, "a") as f:
        if "rms/u" in f:
            del f["rms/u"]
        if "rms/v" in f:
            del f["rms/v"]
        if "rms/w" in f:
            del f["rms/w"]
    print(f"Deleted RMS from {file_path}")
###------------TEMPORAL AVERAGES------------------------------------###
if compute_temporal_averages_flag:
    folder_contour = f"contour temporal averages"

    with h5py.File(bin_path, "a") as f:
        if folder_contour in f:
            del f[folder_contour]
    print(f"Deleted temporal averages from {bin_path}")
###------------PLUME CONTOURS---------------------------------------###
if contour_flag:
    contours = np.array([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05])
    for contour in contours:

        with h5py.File(bin_path, "a") as f:
            key = f"r given contour/contour = {contour}"
            if key in f:
                del f[key]

    print(f"Deleted contours from {bin_path}")
###------------MASS CALCULATIONS------------------------------------###
if mass_flag:
    with h5py.File(bin_path, "a") as f:
        if "mass/S" in f:
            del f["mass/S"]
        if "mass/dmdt" in f:
            del f["mass/dmdt"]
        if "S mass" in f:
            del f["S mass"]
        if "time gradient of S mass" in f:
            del f["time gradient of S mass"]
    print(f"Deleted mass calculations from {bin_path}")