import os
import numpy as np
import h5py
import math
import scipy.signal as signal
import dask.array as da

from reader import OceananigansData
from diagnostics import compute_temporal_averages, compute_rms, binning_oc
from interpolation import vertical_line, velocities_to_center, point

# ==========================================================
# FLAGS
# ==========================================================
binning_flag = False # creates binning of S, T, u, w in r-z space with the S and w contour values
outer_length_flag = False # creates binning of S, T, u, w in r-z space with the S and w contour values
centerline_flag = False # creates vertical line of S, T, u, w at x = 0, y = 0 for all time steps
planelsice_flag = False # creates plane slices of S, T, u, v, w at x = 0 for all time steps
buoyancy_flag = False
fluc_flag = True # calculates turbulent statistics from binning information
rms_flag = True # calculates RMS from 3D fields
compute_temporal_averages_flag = False # computes temporal averages of S and w at the default contour value and writes to file
contour_flag = False # calculates radius of contour at each depth and time that is not in the default
mass_flag = False
negative_tracer_flag = False # calculates the number of negative tracer values in the domain and the average of those values

binning_only = False # certain calculations will be done with binning data

# model options
with_halos = False
salinity = True

# update flags if salinity is False
if not salinity:
    compute_temporal_averages_flag = False
    contour_flag = False
    mass_flag = False

# ==========================================================
# READER
# ==========================================================
folder = '/glade/derecho/scratch/apauls/outputs/version109/square-inlet/open-bottom-BC/AR1/dxi0125/longer/condensed_files'


print(f"Reading data from {folder}")
bin_path = os.path.join(folder, 'binning_rtz.h5')

reader = OceananigansData(folder, salinity = salinity, with_halos=with_halos, Sval=0.1)

# ==========================================================
# PARAMETERS
# ==========================================================
g = 9.80665
T0 = 25.0
rho0 = 1026 # kg/m^3
mld = -60
w0 = -0.001
rp = 4.0
Sval = reader.Sval
# ==========================================================
# MODEL INFORMATION
# ==========================================================
# grid info
nx = reader.nx
nt = reader.nt
dx = reader.dx
lx = reader.lx
time = reader.t

dx_scale = max(dx[:-1]) # not including dz
r = np.arange(dx[0]/2, lx[0]/2, dx_scale)
x, y, z = reader.x, reader.y, reader.z
ncirc = min(nx[0], nx[1])//2 # full circular shells

# ==========================================================
# ANALYSIS
# ==========================================================
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
        if "ccc/rosadfasdfasdfadvadzfkjl;adfjgl;dfakgjadl;fkgjl'rkfi velocity" in f:
            del f["ccc/rotational velocity"]
        if "ccc/w" in f:
            del f["ccc/w"]
        f.create_dataset("ccc/dimensions/r_bin", data = r)
        f.create_dataset("ccc/dimensions/z", data=z)
        f.create_dataset("ccc/dimensions/time", data=time)
    if reader.salinity:
        S_rz = binning_oc('S', reader).transpose(2, 0, 1) # shape: (Nt, Nr, Nz)
        with h5py.File(bin_path, "a") as f:
            if "ccc/S" in f:
                del f["ccc/S"]
            f.create_dataset("ccc/S", data=S_rz)
    T_rz= binning_oc('T', reader).transpose(2, 0, 1) # shape: (Nt, Nr, Nz)
    with h5py.File(bin_path, "a") as f:
        f.create_dataset("ccc/T", data=T_rz)
    del T_rz
    ur_rz = binning_oc('ur', reader).transpose(2, 0, 1) # shape: (Nt, Nr, Nz)
    with h5py.File(bin_path, "a") as f:
        f.create_dataset("ccc/horizontal velocity", data=ur_rz)
    del ur_rz
    utheta_rz = binning_oc('utheta', reader).transpose(2, 0, 1) # shape: (Nt, Nr, Nz)
    with h5py.File(bin_path, "a") as f:
        if "ccc/rotational velocity" in f:
            del f["ccc/rotational velocity"]
        f.create_dataset("ccc/rotational velocity", data=utheta_rz)
    del utheta_rz
    w_rz = binning_oc('w', reader).transpose(2, 0, 1) # shape: (Nt, Nr, Nz)
    with h5py.File(bin_path, "a") as f:
        if "ccc/w" in f:
            del f["ccc/w"]
        f.create_dataset("ccc/w", data=w_rz)
    del w_rz
    print(f"Saved binning to {bin_path}")
    reader.binning = True
    reader.bin_file = 'binning_rtz.h5'
    
###------------APPLYING AZIMUTHAL AVERAGING TO DATA-----------------###
if outer_length_flag:
    w_rz = reader.load_binning_var('w')
    T_rz = reader.load_binning_var('T')
    S_rz = reader.load_binning_var('S')
    b_rz = g * reader.alpha * (T_rz - T0) - g * reader.beta * S_rz
    ur_rz = reader.load_binning_var('horizontal velocity')
    del T_rz, S_rz
    area = reader.dx[0]*reader.dx[1]
    Q = area*np.mean(w_rz, axis=-3) # [m^3/s]
    M = area*np.mean(w_rz**2, axis=-3) # [m^4/s^2]
    B = area*np.mean(b_rz*w_rz, axis=-3) # [m^4/s^3]
    above_mld = np.where(reader.z>=mld+1.0)[0]
    # ignoring first few time steps and information below the MLD
    if reader.nt < 10:
        it_range = np.arange(6, reader.nt)
    else:
        it_range = np.arange(6, 10)

    IT, J, K = np.meshgrid(it_range, np.arange(len(r)), above_mld, indexing='ij')

    # morton et al. 1956 scaling analysis
    #Q = Q[above_mld[None, :], it_range]
    #M = M[above_mld[None, :], it_range]
    #B = B[above_mld[None, :], it_range]
    #w_s = M / Q # outer velocity scale, shape (z, t)
    #Q0 = (2*rp)**2*w0
    #M0 = (2*rp)**2*w0**2
    B0 = -g * reader.beta * Sval * w0 # [m^2/s^3]
    #Ln = M**(3/4)/B**(1/2) # negative sqrt is nan and 
    # scaling analysis from Peter's textbook
    delta = np.empty((len(it_range), len(above_mld), ))
    w_c = -w_rz[IT[:, 0, :], 0, K[:, 0, :]]
    w_rz_filtered = np.empty((len(it_range), len(r), len(above_mld), ))
    for it in range(len(it_range)):
        for k in range(len(above_mld)):
            var = np.pad(w_rz[it_range[it], :, above_mld[k]].squeeze(), (nx[0]//2, 0), mode='symmetric')
            var = signal.fftconvolve(var, np.ones(int(rp//dx[0]))*dx[0]/rp, mode='same')
            var = var[len(var)//2:]
            w_rz_filtered[it, :, k] = var
            var = w_rz[it_range[it], :, above_mld[k]].squeeze()
            r_opt = point(var, r, f0 = -w_c[it, k]*10**-5)#0.1*w0)#
            if np.size(r_opt) > 1:
                delta[it, k] = np.max(r_opt)
            elif np.size(r_opt) == 0:
                delta[it, k] = np.nan
            else:
                delta[it, k] = r_opt

    # calculate c_delta and c_w
    c_delta = np.abs(delta/z[None, above_mld])
    c_w = w_c*np.abs(z[None, above_mld])**(1/3)*(B0*(2*rp)**2)**(-1/3)
    # caclulate ND 
    eta = r[None, :, None]/delta[:, None, :]
    F = -w_rz_filtered/w_c[:, None, :]
    #alpha = -np.log(F)/(eta**2)
    F_transverse = ur_rz[IT, J, K]/w_c[:, None, :]
    # remove information that is outside the delta bounds
    for it in range(len(it_range)):
        for k in range(len(above_mld)):
            F[it, :, k][r > delta[it, k]] = np.nan
            F_transverse[it, :, k][r > delta[it, k]] = np.nan

    # write to file 
    with h5py.File(bin_path, "a") as f:
        #if "scaling analysis/momentum buoyancy analysis/Ln" in f:
        #    del f["scaling analysis/momentum buoyancy analysis/Ln"]
        #f.create_dataset("scaling analysis/momentum buoyancy analysis/Ln", data=Ln)
        #if "scaling analysis/momentum buoyancy analysis/velocity scale" in f:
        #    del f["scaling analysis/momentum buoyancy analysis/velocity scale"]
        #f.create_dataset("scaling analysis/momentum buoyancy analysis/velocity scale", data=w_s)
        delta_opt = 'fft convolve w_c*10**-5' #'fft convolve w0*0.1'#w_c*10**-3
        if "scaling analysis/outer length scale/"+delta_opt+"/delta" in f:
            del f["scaling analysis/outer length scale/"+delta_opt+"/delta"]
        f.create_dataset("scaling analysis/outer length scale/"+delta_opt+"/delta", data=delta)
        if "scaling analysis/outer length scale/w filtered" in f:
            del f["scaling analysis/outer length scale/w filtered"]
        f.create_dataset("scaling analysis/outer length scale/w filtered", data=w_rz_filtered)
        if "scaling analysis/outer length scale/"+delta_opt+"/outer velocity scale" in f:
            del f["scaling analysis/outer length scale/"+delta_opt+"/outer velocity scale"]
        f.create_dataset("scaling analysis/outer length scale/"+delta_opt+"/outer velocity scale", data=w_c)
        if "scaling analysis/outer length scale/"+delta_opt+"/F" in f:
            del f["scaling analysis/outer length scale/"+delta_opt+"/F"]
        f.create_dataset("scaling analysis/outer length scale/"+delta_opt+"/F", data=F)
        if "scaling analysis/outer length scale/"+delta_opt+"/F_transverse" in f:
            del f["scaling analysis/outer length scale/"+delta_opt+"/F_transverse"]
        f.create_dataset("scaling analysis/outer length scale/"+delta_opt+"/F_transverse", data=F_transverse)
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
    xy = False
    yz = True
    xz = False
    internal_gravity_waves = False
    file_path = os.path.join(folder, 'plane_slice.h5')
    if xy:
        z_locs = [0.0, -60.0]#[-reader.dx[-1]/2, 0.0]#
        for z_loc in z_locs:
            if any([z_loc < 0, with_halos]):
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
            else:
                w = reader.field_slice('w', plane = 'XY', loc = z_loc)
                with h5py.File(file_path, "a") as f:
                    if f"XY/z = {z_loc}/w" in f:
                        del f[f"XY/z = {z_loc}/w"]
                    f.create_dataset(f"XY/z = {z_loc}/w", data=w)

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

    if internal_gravity_waves:
        N = reader.nx[0] - 1
        bound = reader.x[N] # center of last grid cell in x direction
        with h5py.File(file_path, "a") as f:
            if f"YZ/x = {bound}/T'" in f:
                del f[f"YZ/x = {bound}/T'"]
            T = reader.field_slice('T', plane = 'YZ', N = N)
            T_avg = reader.load_averages('T')[::100, :]
            T_fluc = T - T_avg[:, None, :]
            f.create_dataset(f"YZ/x = {bound}/T'", data=T_fluc)
            del T, T_fluc
            if f"YZ/x = {bound}/u'" in f:
                del f[f"YZ/x = {bound}/u'"]
            u = reader.field_slice('u', plane = 'YZ', N = N)
            u_avg = reader.load_averages('u')[::100, :]
            u_fluc = u - u_avg[:, None, :]
            f.create_dataset(f"YZ/x = {bound}/u'", data=u_fluc)
            del u, u_fluc
            if f"YZ/x = {bound}/v'" in f:
                del f[f"YZ/x = {bound}/v'"]
            v = reader.field_slice('v', plane = 'YZ', loc = bound)
            v_avg = reader.load_averages('v')[::100, :]
            v_fluc = v - v_avg[:, None, :]
            f.create_dataset(f"YZ/x = {bound}/v'", data=v_fluc)
            del v, v_fluc
            if f"YZ/x = {bound}/w'" in f:
                del f[f"YZ/x = {bound}/w'"]
            w = velocities_to_center(reader.field_slice('w', plane = 'YZ', N = N), -1)
            w_avg = reader.load_averages('w')[::100, :]
            w_fluc = w - w_avg[:, None, :]
            f.create_dataset(f"YZ/x = {bound}/w'", data=w_fluc)
            del w, w_fluc
            if reader.salinity:
                S = reader.field_slice('S', plane = 'YZ', N = N)
                S_avg = reader.load_averages('S')[::100, :]
                S_fluc = S - S_avg[:, None, :]
                if f"YZ/x = {bound}/S'" in f:
                    del f[f"YZ/x = {bound}/S'"]
                f.create_dataset(f"YZ/x = {bound}/S'", data = S_fluc)
                del S, S_fluc

        N = reader.nx[1] -1
        bound = reader.y[N] # center of last grid cell in y direction
        with h5py.File(file_path, "a") as f:
            if f"XZ/y = {bound}/T'" in f:
                del f[f"XZ/y = {bound}/T'"]
            T = reader.field_slice('T', plane = 'XZ', N = N)
            T_avg = reader.load_averages('T')[::100, :]
            T_fluc = T - T_avg[:, None, :]
            f.create_dataset(f"XZ/y = {bound}/T'", data=T_fluc)
            del T, T_fluc
            if f"XZ/y = {bound}/u'" in f:
                del f[f"XZ/y = {bound}/u'"]
            u = reader.field_slice('u', plane = 'XZ', loc = bound)
            u_avg = reader.load_averages('u')[::100, :]
            u_fluc = u - u_avg[:, None, :]
            f.create_dataset(f"XZ/y = {bound}/u'", data=u_fluc)
            del u, u_fluc
            if f"XZ/y = {bound}/v'" in f:
                del f[f"XZ/y = {bound}/v'"]
            v = reader.field_slice('v', plane = 'XZ', N = N)
            v_avg = reader.load_averages('v')[::100, :]
            v_fluc = v - v_avg[:, None, :]
            f.create_dataset(f"XZ/y = {bound}/v'", data=v_fluc)
            del v, v_fluc
            if f"XZ/y = {bound}/w'" in f:
                del f[f"XZ/y = {bound}/w'"]
            w = velocities_to_center(reader.field_slice('w', plane = 'XZ', N = N), -1)
            w_avg = reader.load_averages('w')[::100, :]
            w_fluc = w - w_avg[:, None, :]
            f.create_dataset(f"XZ/y = {bound}/w'", data=w_fluc)
            del w, w_fluc
            if reader.salinity:
                S = reader.field_slice('S', plane = 'XZ', N = N)
                S_avg = reader.load_averages('S')[::100, :]
                S_fluc = S - S_avg[:, None, :]
                if f"XZ/y = {bound}/S'" in f:
                    del f[f"XZ/y = {bound}/S'"]
                f.create_dataset(f"XZ/y = {bound}/S'", data = S_fluc)
                del S, S_fluc

        bound = float(mld) # center of last grid cell in x direction
        with h5py.File(file_path, "a") as f:
            if f"XY/z = {bound}/T'" in f:
                del f[f"XY/z = {bound}/T'"]
            T = reader.field_slice('T', plane = 'XY', loc = bound)
            T_avg = point(reader.load_averages('T')[::100, :], reader.z, z0 = bound)
            T_fluc = T - T_avg[:, None, None]
            f.create_dataset(f"XY/z = {bound}/T'", data=T_fluc)
            del T, T_fluc
            if f"XY/z = {bound}/w'" in f:
                del f[f"XY/z = {bound}/w'"]
            w = reader.field_slice('w', plane = 'XY', loc = bound)
            w_avg = point(reader.load_averages('w')[::100, :], reader.z, z0 = bound)
            w_fluc = w - w_avg[:, None, None]
            f.create_dataset(f"XY/z = {bound}/w'", data=w_fluc)
            del w, w_fluc
            if reader.salinity:
                S = reader.field_slice('S', plane = 'XY', loc = bound)
                S_avg = point(reader.load_averages('S')[::100, :], reader.z, z0 = bound)
                S_fluc = S - S_avg[:, None, None]
                if f"XY/z = {bound}/S'" in f:
                    del f[f"XY/z = {bound}/S'"]
                f.create_dataset(f"XY/z = {bound}/S'", data = S_fluc)
                del S, S_fluc
        print(f"Saved plane slices for checking internal gravity waves to {file_path}")
###------------BUOYANCY CALCULATIONS--------------------------------###
if buoyancy_flag:
    buoyancy_file = os.path.join(folder, 'buoyancy_profile.h5')
    alpha = reader.alpha
    T = reader.lazy_field('T').compute()
    b = g * alpha * (T - T0)
    if reader.salinity:
        beta = reader.beta
        S = reader.lazy_field('S').compute()
        b += - g * beta * S
        del S
    del T
    b_avg = np.mean(b, axis=(-3, -2))
    b_fluc = b - b_avg[:, None, None, :]
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
        b_centerline = vertical_line(b, x = reader.x, y = reader.y)
        b_fluc_centerline = vertical_line(b_fluc, x = reader.x, y = reader.y)
    with h5py.File(buoyancy_file, "a") as f:
        if "z" in f:
            del f["z"]
        if "b_rms" in f:
            del f["b_rms"]
        if "b_avg" in f:
            del f["b_avg"]
        if "field data" in f:
            del f["field data"]
        f.create_dataset("z", data = z)
        f.create_dataset("b_rms", data = b_rms)
        f.create_dataset("b_avg", data = b_avg)
        f.create_dataset("field data/b", data = b)
        f.create_dataset("field data/b_fluc", data = b_fluc)
        if not reader.centerline:
            f.create_dataset("centerline/b", data = b_centerline)
            f.create_dataset("centerline/b_fluc", data = b_fluc_centerline)
    del b
    print(f"Saved buoyancy information to {buoyancy_file}")
###------------FLUCTUATION AVERAGES---------------------------------###
if fluc_flag:
    if binning_only:
        ur_rz = reader.load_binning_var('horizontal velocity')
        w_rz = reader.load_binning_var('w')
        T_rz = reader.load_binning_var('T')
        S_rz = reader.load_binning_var('S')
        b_rz = g * reader.alpha * (T_rz - T0) - g * reader.beta * S_rz
        bw_rz = b_rz * w_rz
        bur_rz = b_rz * ur_rz
        del T_rz, S_rz

        # calculate averages
        ur_avg = np.mean(ur_rz, axis=-3)
        w_avg = np.mean(w_rz, axis=-3)
        b_avg = np.mean(b_rz, axis=-3)
        bw_avg = np.mean(bw_rz, axis=-3)
        bur_avg = np.mean(bur_rz, axis=-3)

        # calculate fluctuations
        ur_fluc = ur_rz - ur_avg[None, :, :]
        del ur_rz
        w_fluc = w_rz - w_avg[None, :, :]
        b_fluc = b_rz - b_avg[None, :, :]
        b_fluc_avg = np.mean(b_fluc, axis=-3)
        b_fluc_w_avg = np.mean(b_fluc * w_rz, axis=-3)
        del b_rz
        b_fluc_w_fluc = b_fluc * w_rz - b_fluc_w_avg[None, :, :]
        bw_fluc = bw_rz - bw_avg[None, :, :]
        del bw_rz, w_rz
        bur_fluc = bur_rz - bur_avg[None, :, :]
        del bur_rz

        with h5py.File(bin_path, "a") as f:
            if "averages/horizontal velocity" in f:
                del f["averages/horizontal velocity"]
            f.create_dataset("averages/horizontal velocity", data=ur_avg)
            if "averages/w" in f:
                del f["averages/w"]
            f.create_dataset("averages/w", data=w_avg)
            if "averages/b" in f:
                del f["averages/b"]
            f.create_dataset("averages/b", data=b_avg)
            if "averages/b_fluc" in f:
                del f["averages/b_fluc"]
            f.create_dataset("averages/b_fluc", data=b_fluc_avg)
            if "averages/b_fluc_w" in f:
                del f["averages/b_fluc_w"]
            f.create_dataset("averages/b_fluc_w", data=b_fluc_w_avg)
            if "averages/bw" in f:
                del f["averages/bw"]
            f.create_dataset("averages/bw", data=bw_avg)
            if "averages/bur" in f:
                del f["averages/bur"]
            f.create_dataset("averages/bur", data=bur_avg)
            del ur_avg, w_avg, b_avg, bw_avg, bur_avg

            if "fluctuations/ur" in f:
                del f["fluctuations/ur"]
            f.create_dataset("fluctuations/ur", data=ur_fluc)
            if "fluctuations/w" in f:
                del f["fluctuations/w"]
            f.create_dataset("fluctuations/w", data=w_fluc)
            if "fluctuations/b" in f:
                del f["fluctuations/b"]
            f.create_dataset("fluctuations/b", data=b_fluc)
            if "fluctuations/bur" in f:
                del f["fluctuations/bur"]
            f.create_dataset("fluctuations/bur", data=bur_fluc)
            if "fluctuations/bw" in f:
                del f["fluctuations/bw"]
            f.create_dataset("fluctuations/bw", data=bw_fluc)
            if "fluctuations/b_fluc_w" in f:
                del f["fluctuations/b_fluc_w"]
            f.create_dataset("fluctuations/b_fluc_w", data=b_fluc_w_fluc)
            del b_fluc, bur_fluc, bw_fluc
        print(f"Saved fluctuations to {bin_path}")
    else:
        file_path = os.path.join(folder, 'fluctuations.h5')

        # Calcualting buoyancy
        dims = (-3, -2)
        T = reader.lazy_field('T').compute()
        if reader.salinity:
            S = reader.lazy_field('S').compute()
            # Buoyancy (still lazy)
            beta  = reader.beta
            b = g * alpha * (T - T0) - (g * beta * S)
            del S
        else:
            b = g * alpha * (T - T0)

        # Temperature fluctuations
        T_xy = da.mean(T, axis=dims)

        T_fluc = T - T_xy[:, np.newaxis, np.newaxis, :]

        with h5py.File(file_path, "a") as f:
            if "fluctuations/T_fluc" in f:
                del f["fluctuations/T_fluc"]
            f.create_dataset("fluctuations/T_fluc", data=da.mean(T_fluc, axis=dims))

        del T, T_fluc

        b_xy = da.mean(b, axis=dims)

        b_fluc = b - b_xy[:, np.newaxis, np.newaxis, :]
        w = reader.lazy_field('w').compute()
        
        # Center velocities (still lazy)
        w = velocities_to_center(w, axis=-1)

        with h5py.File(file_path, "a") as f:
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
            f.create_dataset("fluctuations/b_fluc", data=da.mean(b_fluc, axis=dims))
            f.create_dataset("fluctuations/bw_fluc", data=da.mean(b_fluc * w, axis=dims))
        del b, b_fluc, w
        print(f"Saved fluctuations to {file_path}")
###------------ROOT MEAN SQUARE-------------------------------------###
if rms_flag:
    if binning_only:
        ur_fluc = reader.load_fluc('ur', file = 'binning_rtz.h5')
        ur_rms = np.mean(ur_fluc**2, axis=-3)**0.5
        del ur_fluc
        w_fluc = reader.load_fluc('w', file = 'binning_rtz.h5')
        w_rms = np.mean(w_fluc**2, axis=-3)**0.5
        del w_fluc
        b_fluc = reader.load_fluc('b', file = 'binning_rtz.h5')
        b_rms = np.mean(b_fluc**2, axis=-3)**0.5
        del b_fluc
        bw_fluc = reader.load_fluc('bw', file = 'binning_rtz.h5')
        bw_rms = np.mean(bw_fluc**2, axis=-3)**0.5
        del bw_fluc
        bur_fluc = reader.load_fluc('bur', file = 'binning_rtz.h5')
        bur_rms = np.mean(bur_fluc**2, axis=-3)**0.5
        del bur_fluc

        with h5py.File(bin_path, "a") as f:
            if "rms/horizontal velocity" in f:
                del f["rms/horizontal velocity"]
            f.create_dataset("rms/horizontal velocity", data=ur_rms)
            del ur_rms
            if "rms/w" in f:
                del f["rms/w"]
            f.create_dataset("rms/w", data=w_rms)
            del w_rms
            if "rms/b" in f:
                del f["rms/b"]
            f.create_dataset("rms/b", data=b_rms)
            del b_rms
            if "rms/bw" in f:
                del f["rms/bw"]
            f.create_dataset("rms/bw", data=bw_rms)
            del bw_rms
            if "rms/bur" in f:
                del f["rms/bur"]
            f.create_dataset("rms/bur", data=bur_rms)
            del bur_rms
        print(f"Saved RMS to {bin_path}")
    else:
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
    S = reader.lazy_field('S').compute()
    vol = math.prod(reader.lx)
    dims = (1, 2, 3)
    # volume integral of S value in domain
    S_mass = np.mean(S, axis = dims)*vol*rho0
    del S
    dmdt = np.gradient(S_mass, time)
    with h5py.File(bin_path, "a") as f:
        if "S mass" in f:
            del f["S mass"]
        if "time gradient of S mass" in f:
            del f["time gradient of S mass"]
        f.create_dataset("S mass", data=S_mass)
        f.create_dataset("time gradient of S mass", data=dmdt)
    print(f"Saved mass calculations to {bin_path}")
###------------NEGATIVE MASS CALCULATIONS---------------------------###
if negative_tracer_flag:
    S = reader.lazy_field('S').compute()
    vol = math.prod(reader.lx)
    dims = (1, 2, 3)
    Smin = np.min(S, axis = dims)
    Smax = np.max(S, axis = dims)
    S_neg_count = np.sum(S<0, axis = dims)

    S[S>=0] = None
    S_neg_avg = np.nanmean(S, axis = dims)

    del S
    with h5py.File(bin_path, "a") as f:
        if "max of S" in f:
            del f["max of S"]
        if "min of S" in f:
            del f["min of S"]
        if "negative S count" in f:
            del f["negative S count"]
        if "negative S average" in f:
            del f["negative S average"]
        f.create_dataset("max of S", data=Smax)
        f.create_dataset("min of S", data=Smin)
        f.create_dataset("negative S count", data=S_neg_count)
        f.create_dataset("negative S average", data=S_neg_avg)
    print(f"Saved negative tracer calculations to {bin_path}")