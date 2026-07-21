import os
import numpy as np

from reader import OceananigansData
from diagnostics import comparison_info
from interpolation import point
from physics import buoyancy
from plotting_general import plot_format, plot_ranges, create_video, comparison_plot_opt
from plotting_lines import plot_turb_stats_bin, temporal_avg, plot_plume_depths, plot_plume_vertical_spatial
from plotting_planes import plot_variable_vert_slice, plot_variable_xy_slice

# flags for what to plot
plot_variables_vert = False
plot_variables_hor = False
plot_var_bin = False
plot_turb_stats = False
plot_temporal_avg = False
plot_depths = True
plot_plume_z = True
video = True

# flags for how to read data
with_halos = False
closure = False
stokes = False

contour = 0.05
name_uni = f'contour-{contour:.4f}'
universal_folder ='/Users/annapauls/Documents/TESLa /Simulations/Oceananigans/dense plume/salinity and temperature/version109/res testing/square inlet/open BC /'
#'/glade/derecho/scratch/apauls/outputs/'

# selecting cases to compare
variations = 'horizontal resolution' #'MLD', 'flux', 'strat', 'all', 'vertical length', 'Lz160m','WENO', 'vertical resolution', 'horizontal resolution', 'AR=1', 'else'
if variations != 'else':
    cases_info = comparison_info(variations, universal_folder = universal_folder)
    dTdz = cases_info['dTdz']
    case_names = cases_info['case_names']
    num_cases = cases_info['num_cases']
    folder_names = cases_info['folder_names']
    fig_folder = cases_info['fig_folder']
    mld = cases_info['mld']
    F_s = cases_info['F_s']
else:
    folder_names = ['proposed resolution/S0 = 0.1 dTdz = 0.01 MLD = 60', 'Lz = 160m/S0 = 0.1 dTdz = 0.01 MLD = 60']
    num_cases = len(folder_names)
    fig_folder = os.path.join(universal_folder, 'comparison figures', '96m vs 160m' + ' comparison figures', 'default case')
    case_names =[r'L$_z = 96$m', r'L$_z = 160$m']
    dTdz = 0.01*np.ones(num_cases)
    mld = np.array([60, 60])

readers = []
salinity = []
for n, folder in enumerate(folder_names):
    if F_s[n] == 0.0:
        salinity.append(False)
    else:
        salinity.append(True)
    folder = os.path.join(universal_folder, folder)
    readers.append(OceananigansData(folder, salinity = salinity[n], Sval = 0.1))

# collecting model information for all cases
if plot_var_bin:
    r_bin = []
x = []
y = []
z = []
time  = []
nx = np.empty((3, num_cases), dtype=object)
lx = np.empty((3, num_cases), dtype=object)
grid_specs = False*np.ones(num_cases)
for i, reader in enumerate(readers):
    reader.load_equation_of_state()
    if plot_variables_hor:
        x.append(reader.x)
    y.append(reader.y)
    z.append(reader.z)
    time.append(reader.t)
    nx[:, i] = reader.nx
    lx[:, i] = reader.lx
    if i == 0:
        nt_min = reader.nt
    else:
        nt_min = np.min([nt_min, reader.nt])
    if plot_var_bin:
        r_bin.append(reader.loading_bin_contours(contour = contour))
    if salinity[n] and plot_turb_stats:
        S_value = reader.load_S_temporal_avg()

# physical parameters
x0 = 0.0
y0 = 0.0
rj = 5 # m, radius of salinity flux circle at the surface
g = 9.80665  # gravity in m/s^2
T0 = 25

# collecting variables for plotting
if any([plot_variables_vert, plot_variables_hor, plot_var_bin, plot_turb_stats, plot_depths, plot_plume_z]):
    plot_format()
    if all(salinity):
        S_hor = []
        S_vert_plane = []
    if plot_turb_stats:
        b_avg = []
        w_rms = []
        ur_rms = []
        urw_avg = []
        bur_fluc_avg = []
        bw_fluc_avg = []
        Tur_avg = []
        Tw_avg = []
        if all(salinity):
            Sur_avg = []
            Sw_avg = []

    if plot_variables_vert:
        T_vert_plane = []
        u_vert_plane = []
        v_vert_plane = []
        w_vert_plane = []
        bw_plane = []
    if plot_variables_hor:
        T_hor_plane = []
        u_hor_plane = []
        v_hor_plane = []
        w_hor_plane = []
        S_hor_plane = []
    if plot_var_bin:
        T_bin = []
        S_bin = []
        ur_bin = []
        utheta_bin = []
        w_bin = []
        b_bin = []


    if plot_depths:
        bS = []
        zp = []
        zneutral = []
        zc = []
        r_bin = []
    if plot_plume_z:
        b_avg = []
        u_rms = []
        v_rms = []
        w_rms = []
        b_center = []
        bur_fluc_avg = []
        bw_fluc_avg = []
        T_avg = []
        S_avg = []
        T_fluc_center = []
        S_center = []
        r_bin = []

    for i, reader in enumerate(readers):
        # Load data from files [nt, nx, ny, nz]
        if plot_variables_vert:
            T_vert_plane.append(reader.load_plane_var('T'))
            u_vert_plane.append(reader.load_plane_var('u'))
            v_vert_plane.append(reader.load_plane_var('v'))
            w_vert_plane.append(reader.load_plane_var('w'))
            if all(salinity):
                S_vert_plane.append(reader.load_plane_var('S'))
        if plot_variables_hor:
            z_loc =-reader.lx[-1] + reader.dx[-1]/2#-reader.dx[-1]/2# -reader.dx[-1]/2#0.0#
            T_hor_plane.append(reader.load_plane_var('T', loc=z_loc, plane = 'XY'))
            u_hor_plane.append(reader.load_plane_var('u', loc=z_loc, plane = 'XY'))
            v_hor_plane.append(reader.load_plane_var('v', loc=z_loc, plane = 'XY'))
            w_hor_plane.append(reader.load_plane_var('w', loc=z_loc, plane = 'XY'))
            if all(salinity):
                S_hor_plane.append(reader.load_plane_var('S', loc=z_loc, plane = 'XY'))
        # Load binning from files
        if plot_var_bin or plot_turb_stats:
            ur_rz = reader.load_binning_var('horizontal velocity')
            utheta_rz = reader.load_binning_var('rotation velocity')
            w_rz = reader.load_binning_var('w')
            T_rz = reader.load_binning_var('T')
            if all(salinity):
                S_rz = reader.load_binning_var('S')
            b_rz = buoyancy(reader, type = 'bin')
            b_xy = np.mean(b_rz, axis=0)
        if plot_var_bin:
            T_bin.append(T_rz)
            ur_bin.append(ur_rz)
            utheta_bin.append(utheta_rz)
            w_bin.append(w_rz)
            if all(salinity):
                S_bin.append(S_rz)
            b_bin.append(b_rz)
        if plot_turb_stats:
            urw_avg.append(np.mean(ur_rz * w_rz, axis=0))
            # rms fluctuations
            ur_avg = np.mean(ur_rz, axis=0)
            ur_rms.append(np.sqrt(np.mean((ur_rz - ur_avg)**2, axis=0)))
            w_avg = np.mean(w_rz, axis=0)
            w_rms.append(np.sqrt(np.mean((w_rz - w_avg)**2, axis=0)))
            bur_avg = np.mean(b_rz * ur_rz, axis=0)
            bw_avg = np.mean(b_rz * w_rz, axis=0)
            bur_fluc_avg.append(bur_avg)
            bw_fluc_avg.append(bw_avg)
            # calculate means
            b_avg.append(b_xy)
            T_avg = np.mean(T_rz, axis=0)
            Tur_avg.append(np.mean((T_rz-T_avg) * ur_rz, axis=0))
            Tw_avg.append(np.mean((T_rz-T_avg) * w_rz, axis=0))
            if all(salinity):
                Sur_avg.append(np.mean(S_rz * ur_rz, axis=0))
                Sw_avg.append(np.mean(S_rz * w_rz, axis=0))
        if plot_depths or plot_plume_z:
            r_bin.append(reader.loading_bin_contours(contour = contour))
            if salinity and plot_depths:
                S_value = reader.load_S_temporal_avg()
                reader.load_equation_of_state()
                bS.append(-g*reader.beta*S_value)
                T_rz = reader.load_binning_var('T')
                S_rz = reader.load_binning_var('S')
                # calculate buoyancy differences
                S_value = reader.load_S_temporal_avg()
                reader.load_equation_of_state()
                bT = g*reader.alpha*np.mean(T_rz, axis = 0) - g*reader.alpha*reader.T0
                # calculate where w = 0 on the centerline
                w_rz = reader.load_binning_var('w')

                b_it = np.empty(reader.nt)
                w_centerline_it = np.empty(reader.nt)
                S_it = np.empty(reader.nt)
                for it in range(reader.nt):
                    b_it[it] = point(bT[:, it] - bS[i], z[i], f0 = 0.0)
                    w_centerline_it[it] = point(w_rz[0, :, it], z[i], f0 = 0.0)
                    S_it[it] = point(S_rz[0, :, it], z[i], f0 = S_value*contour)
                zneutral.append(b_it)
                zp.append(w_centerline_it)
                zc.append(S_it)

            if plot_plume_z:
                u_rms.append(reader.load_rms('u'))
                v_rms.append(reader.load_rms('v'))
                w_rms.append(reader.load_rms('w'))
                b_avg_loc, b_rms_loc, b_centerline_loc, b_fluc_centerline_loc = reader.load_buoyancy()
                b_avg.append(b_avg_loc)
                b_center.append(b_centerline_loc)
                del b_avg_loc, b_rms_loc, b_centerline_loc, b_fluc_centerline_loc
                bur_fluc_avg.append(reader.load_fluc('bur'))
                bw_fluc_avg.append(reader.load_fluc('bw'))
                T_avg.append(reader.load_averages('T', steps=reader.t_save))
                S_avg.append(reader.load_averages('S', steps=reader.t_save))

                T_center = reader.field_centerline('T')[::100, :]
                T_fluc_center.append(T_center-T_avg[i])
                S_center.append(reader.field_centerline('S')[::100, :])
                del T_center


if plot_temporal_avg:
    t_range = np.array([0.4, 0.5])*24*3600 # seconds
    w_rms_t_avg = []
    b_rms_t_avg = []
    w_centerline_t_avg = []
    b_fluc_centerline_t_avg = []
    S_centerline_t_avg = []
    for i, reader in enumerate(readers):
        t_idx = np.where((reader.t >= t_range[0]) & (reader.t <= t_range[1]))
        t_idx = [np.min(t_idx), np.max(t_idx)]
        # check if higher frequency output files exist
        if reader.centerline:
            t_idx1 = (t_idx-1)*100+1 # the centerline file is output every 100 time steps, so need to adjust indices
            t_save = None
        else:
            t_idx1 = t_idx
            t_save = reader.t_save[t_idx1[0]:t_idx1[1]]
        # load in information
        w_center_temp = reader.field_centerline('w', steps = t_save)
        w_centerline_t_avg.append(np.mean(w_center_temp, axis=0))
        _, b_rms_temp, _, b_fluc_centerline_temp = reader.load_buoyancy()
        b_fluc_centerline_t_avg.append(np.mean(b_fluc_centerline_temp[t_idx1[0]:t_idx1[1], :], axis=0))

        b_rms_t_avg.append(np.mean(b_rms_temp[t_idx[0]:t_idx[1], :], axis=0))
        w_rms_temp = reader.load_rms('w')
        w_rms_t_avg.append(np.mean(w_rms_temp[t_idx[0]:t_idx[1], :], axis=0))
        if reader.salinity:
            S_center_temp = reader.field_centerline('S', steps = t_save)
            S_centerline_t_avg.append(np.mean(S_center_temp, axis=0))
        print("Finished temporal averaging for case ", case_names[i])
    t_range = t_range/(24*3600) # days
############ PLOTTING ############
# plotting prep
if plot_variables_vert or plot_variables_hor:
    if all(salinity):
        range_names = ['w']#['T',]#['log w']#['w']#
        var_names = range_names#['Temperature',]#range_names
        range_names = ['Tracer', 'T', 'u', 'v', 'w']
        var_names = ['Tracer', 'Temperature', 'u', 'v', 'w']
    else:
        range_names = ['T', 'u', 'v', 'w']
        var_names = ['Temperature', 'u', 'v', 'w']
    variable_dir = {}
    variable_dir_hor = {}
if plot_var_bin:
    if all(salinity):
        bin_var_names = ['Tracer', 'Temperature', r'u$_r$', r'u$_{\theta}$', 'w']
        bin_range_names = ['Tracer', 'T', 'u', 'v', 'w']
    else:
        bin_var_names = ['Temperature', 'u', 'v', 'w']
        bin_range_names = ['T', 'u', 'v', 'w']
    bin_dir = {}
if plot_turb_stats or plot_temporal_avg or plot_plume_z:
    color_opt, line_opt = comparison_plot_opt(num_cases)
S_tol = 10**(-6)
ranges = plot_ranges(lz = 96, mld = np.max(mld), T0 = T0, dTdz = np.max(dTdz), C_tol = S_tol)
ranges['Tracer'] =[S_tol, 0.15]
ranges['Tracer negative'] = [-0.15, 0.15]
ranges['Tracer_fluc'] = [-0.2, 0.2]
ranges['Tracer_avg'] = [0, 1.2*10**(-3)]
ranges['T'] = [T0-0.7, T0 + 0.05]
ranges['w'] = [-1*10**(-1), 1*10**(-1)]
ranges['u'] = [-1*10**(-2), 1*10**(-2)]
ranges['v'] = [-2*10**(-2), 2*10**(-2)]
ranges['vel_rms'] = [0, 3*10**(-2)]
ranges['b_rms'] = [0, 1.5*10**(-5)]
ranges['S'] = [0.0, 0.05]
ranges['b_fluc'] = [-7*10**(-4), 7*10**(-4)]
ranges['bw_fluc'] = [-8*10**(-7), 8*10**(-7)]
if plot_turb_stats:
    ranges['restress'] = [-2*10**(-5), 2*10**(-5)]
    ranges['Tw_fluc'] = [-5*10**(-4), 5*10**(-4)]
    ranges['Cw'] = [-5*10**(-5), 5*10**(-5)]

# plotting with flags that have the possibility of being videos
if any([plot_variables_vert, plot_variables_hor, plot_var_bin, plot_turb_stats]):
    time_min = min(time, key=len)
    for it in range(nt_min):
        if plot_variables_vert or plot_variables_hor:
            if all(salinity): #'Tracer', 'T', 'u', 'v', 'w'
                colorbar_labels = [r"g/kg", r"$^\circ$C", r"m/s", r"m/s", r"m/s"]
                cmaps = ['Blues', 'viridis', 'RdBu_r', 'RdBu_r', 'RdBu_r']
                #colorbar_labels = [r"m/s"]#[r"$^\circ$C", ]#
                #cmaps = ['RdBu_r',]#['viridis',]#['Blues']#
            else: #'T', 'u', 'v', 'w'
                colorbar_labels = [r"$^\circ$C", r"m/s", r"m/s", r"m/s"]
                cmaps = ['viridis', 'RdBu_r', 'RdBu_r', 'RdBu_r']
            if plot_variables_vert:
                if all(salinity):
                    variables = [[S_vert_plane[n][it, :, :].T for n in range(num_cases)], [T_vert_plane[n][it, :, :].T for n in range(num_cases)], [u_vert_plane[n][it, :, :].T for n in range(num_cases)], [v_vert_plane[n][it, :, :].T for n in range(num_cases)], [w_vert_plane[n][it, :, :].T for n in range(num_cases)]]
                else:
                    variables = [[T_vert_plane[n][it, :, :].T for n in range(num_cases)], [u_vert_plane[n][it, :, :].T for n in range(num_cases)], [v_vert_plane[n][it, :, :].T for n in range(num_cases)], [w_vert_plane[n][it, :, :].T for n in range(num_cases)]]
                for dir, var in enumerate(variables):
                    variable_dir[var_names[dir]] = plot_variable_vert_slice(time_min[it], it, ranges, fig_folder, lx, y, z, var, case_names, var_names[dir], range_names[dir], colorbar_label = colorbar_labels[dir], cmap = cmaps[dir], plane='YZ')
            if plot_variables_hor:
                hor_ranges = ranges.copy()
                hor_ranges['T'] = [T0-0.68, T0-0.62]
                hor_ranges['u'] = [-2*10**(-5), 2*10**(-5)]
                hor_ranges['v'] = hor_ranges['u']
                hor_ranges['w'] = [-8*10**(-2), 8*10**(-2)]
                hor_ranges['log w'] = [10**-7, hor_ranges['w'][-1]]

                if all(salinity):
                    #variables = [[w_hor_plane[n][it, :, :].T for n in range(num_cases)],]#[[T_hor_plane[n][it, :, :].T for n in range(num_cases)],]#[np.abs([w_hor_plane[n][it, :, :].T for n in range(num_cases)]),]#
                    variables = [[S_hor_plane[n][it, :, :].T for n in range(num_cases)], [T_hor_plane[n][it, :, :].T for n in range(num_cases)], [u_hor_plane[n][it, :, :].T for n in range(num_cases)], [v_hor_plane[n][it, :, :].T for n in range(num_cases)], [w_hor_plane[n][it, :, :].T for n in range(num_cases)]]
                else:
                    variables = [[T_hor_plane[n][it, :, :].T for n in range(num_cases)], [u_hor_plane[n][it, :, :].T for n in range(num_cases)], [v_hor_plane[n][it, :, :].T for n in range(num_cases)], [w_hor_plane[n][it, :, :].T for n in range(num_cases)]]
                for dir, var in enumerate(variables):
                    variable_dir_hor[var_names[dir]] = plot_variable_xy_slice(time_min[it], it, hor_ranges, fig_folder, lx, x, y, var, case_names, var_names[dir], range_names[dir], colorbar_label = colorbar_labels[dir], cmap = cmaps[dir], loc = z_loc)
        if plot_var_bin:
            if all(salinity): #'Tracer', 'T', 'u', 'v', 'w'
                variables = [[S_bin[i][:, :, it].T for i in range(num_cases)], [T_bin[i][:, :, it].T for i in range(num_cases)], [ur_bin[i][:, :, it].T for i in range(num_cases)], [utheta_bin[i][:, :, it].T for i in range(num_cases)], [w_bin[i][:, :, it].T for i in range(num_cases)]]
                colorbar_labels = [r"g/kg", r"$^\circ$C", r"m/s", r"m/s", r"m/s"]
                cmaps = ['Blues', 'viridis', 'RdBu_r', 'RdBu_r', 'RdBu_r']
            else: #'T', 'u', 'v', 'w'
                variables = [[T_bin[i][:, :, it].T for i in range(num_cases)], [ur_bin[i][:, :, it].T for i in range(num_cases)], [utheta_bin[i][:, :, it].T for i in range(num_cases)], [w_bin[i][:, :, it].T for i in range(num_cases)]]
                colorbar_labels = [r"$^\circ$C", r"m/s", r"m/s", r"m/s"]
                cmaps = ['viridis', 'RdBu_r', 'RdBu_r', 'RdBu_r']

            for dir, var in enumerate(variables):
                l_bin = [lx[0]/2, lx[-1]]
                bin_dir[bin_var_names[dir]] = plot_variable_vert_slice(time_min[it], it, ranges, fig_folder, lx, r_bin, z, var, case_names, bin_var_names[dir], bin_range_names[dir], colorbar_label = colorbar_labels[dir], cmap = cmaps[dir], plane='binning')
        if plot_turb_stats:
            ur_rms_it = [ur_rms[i][:, it] for i in range(num_cases)]
            w_rms_it = [w_rms[i][:, it] for i in range(num_cases)]
            uw_avg_it = [urw_avg[i][:, it] for i in range(num_cases)]
            b_avg_it = [b_avg[i][:, it] for i in range(num_cases)]
            bur_fluc_avg_it = [bur_fluc_avg[i][:, it] for i in range(num_cases)]
            bw_fluc_avg_it = [bw_fluc_avg[i][:, it] for i in range(num_cases)]
            Tur_avg_it = [Tur_avg[i][:, it] for i in range(num_cases)]
            Tw_avg_it = [Tw_avg[i][:, it] for i in range(num_cases)]
            Sur_avg_it = [Sur_avg[i][:, it] for i in range(num_cases)]
            Sw_avg_it = [Sw_avg[i][:, it] for i in range(num_cases)]
            buoyancy_dir_z = plot_turb_stats_bin(time_min[it], it, ranges, color_opt, fig_folder, case_names, z, ur_rms_it, w_rms_it, uw_avg_it, b_avg_it, bur_fluc_avg_it, bw_fluc_avg_it, Tur_avg_it, Tw_avg_it, Sur_avg_it, Sw_avg_it)
if plot_plume_z:
    buoyancy_dir_z = plot_plume_vertical_spatial(min(time, key=len), ranges, color_opt, fig_folder, case_names, name_uni, lx, z, S_avg, u_rms, v_rms, w_rms, b_avg, b_center, r_bin, bur_fluc_avg, bw_fluc_avg, T_avg, T_fluc_center, S_center)

# plotting with flags that don't have the possibility of being videos
if plot_depths:
    depth_dir = plot_plume_depths(time, color_opt, fig_folder, case_names, name_uni, lx, zp, zneutral, zc, contour, trend = False)
if plot_temporal_avg:
    plot_format(fontsize = 10)
    temporal_avg(t_range, ranges, color_opt, fig_folder, case_names, lx, z, w_centerline_t_avg, S_centerline_t_avg, b_fluc_centerline_t_avg, w_rms_t_avg, b_rms_t_avg, h_ml = mld)
    print("Finished plotting temporal average for all cases.")
# creating videos
if video:
    if plot_var_bin:
        for n, name in enumerate(bin_var_names):
            create_video(bin_dir[bin_var_names[n]], fig_folder, 'binning', name)
    if plot_variables_vert:
        for dir, name in enumerate(var_names):
            create_video(variable_dir[var_names[dir]], fig_folder, 'vertical', name)
    if plot_variables_hor:
        for dir, name in enumerate(var_names):
            create_video(variable_dir_hor[var_names[dir]], fig_folder, f'z = {z_loc:.2f} m', name)
    if plot_turb_stats:
        create_video(buoyancy_dir_z, fig_folder, 'binning', 'turb_stats')
    if plot_plume_z:
        create_video(buoyancy_dir_z, fig_folder, 'binning', 'plume')
