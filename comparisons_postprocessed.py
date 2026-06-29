import os
import numpy as np

from reader import OceananigansData
from diagnostics import comparison_info
from physics import buoyancy
from plotting_general import plot_format, plot_ranges, create_video, comparison_plot_opt
from plotting_lines import plot_turb_stats_bin, temporal_avg
from plotting_planes import plot_variable_vert_slice

# flags for what to plot
plot_variables = True
plot_var_bin = False
plot_turb_stats = False
plot_temporal_avg = False
video = True

# flags for how to read data
with_halos = False
closure = False
stokes = False

contour_bound = 0.001
name_uni = f'contour-{contour_bound:.4f}'
universal_folder ='/Users/annapauls/Documents/TESLa /Simulations/Oceananigans/dense plume/salinity and temperature/no noise circle inlet/version109/res testing/square inlet/'
#'/Users/annapauls/Documents/TESLa /Simulations/Oceananigans/dense plume/salinity and temperature/no noise circle inlet/version109/res testing/square inlet/'
#'/glade/derecho/scratch/apauls/outputs/'
#harddrive: '/Volumes/Anna External/Oceananigans/dense plume with stratification/salinity and temperature /no noise circle inlet/resolution testing'#

# selecting cases to compare
variations = 'erf' # 'MLD', 'flux', 'strat', 'all', 'vertical length', 'Lz160m','WENO', 'vertical resolution', 'horizontal resolution', 'AR=1', 'else'
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
    readers.append(OceananigansData(folder, salinity = salinity[n]))

# collecting model information for all cases
if plot_var_bin:
    r_bin = []
y = []
z = []
nx = np.empty((3, num_cases), dtype=object)
lx = np.empty((3, num_cases), dtype=object)
grid_specs = False*np.ones(num_cases)
for i, reader in enumerate(readers):
    reader.load_equation_of_state()
    y.append(reader.y)
    z.append(reader.z)
    nx[:, i] = reader.nx
    lx[:, i] = reader.lx
    if i == 0:
        nt = reader.nt
    else:
        nt = np.min([nt, reader.nt])
    if plot_var_bin:
        r_bin.append(reader.loading_bin_radius())
    if salinity[n] and plot_turb_stats:
        S_value = reader.load_S_temporal_avg('binning_rtz.h5')

# physical parameters
x0 = 0.0
y0 = 0.0
rj = 5 # m, radius of salinity flux circle at the surface
g = 9.80665  # gravity in m/s^2
T0 = 25

# video or not setup
if video or plot_temporal_avg:
    time = readers[0].t
else:
    time = readers[0].t[-1]

# collecting variables for plotting
if any([plot_variables, plot_var_bin, plot_turb_stats]):
    plot_format()
    if all(salinity):
        S_avg = []
        S_fluc_center = []
        S_hor = []
        S_plane = []
    if plot_turb_stats:
        T_avg = []
        b_avg = []
        u_rms = []
        v_rms = []
        w_rms = []
        u_fluc_avg = []
        v_fluc_avg = []
        w_fluc_avg = []
        bu_fluc_avg = []
        bv_fluc_avg = []
        bw_fluc_avg = []
        r_profile = []
        b_center = []
        T_fluc_center = []

    if plot_variables:
        T_plane = []
        u_plane = []
        v_plane = []
        w_plane = []
        bw_plane = []
    if plot_var_bin:
        T_bin = []
        S_bin = []
        ur_bin = []
        utheta_bin = []
        w_bin = []
        b_bin = []

    for i, reader in enumerate(readers):
        # Load data from files [nt, nx, ny, nz]
        if plot_variables:
            T_plane.append(reader.load_plane_var('T'))
            u_plane.append(reader.load_plane_var('u'))
            v_plane.append(reader.load_plane_var('v'))
            w_plane.append(reader.load_plane_var('w'))
            if all(salinity):
                S_plane.append(reader.load_plane_var('S'))
            b_plane = buoyancy(reader, type = 'plane')
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
            # rms fluctuations
            u_rms.append(reader.load_rms('u'))
            v_rms.append(reader.load_rms('v'))
            w_rms.append(reader.load_rms('w'))
            bu_avg = np.mean(b_rz * ur_rz, axis=0)
            bv_avg = np.mean(b_rz * utheta_rz, axis=0)
            bw_avg = np.mean(b_rz * w_rz, axis=0)
            bu_fluc_avg.append(bu_avg)
            bv_fluc_avg.append(bv_avg)
            bw_fluc_avg.append(bw_avg)
            # calculate means
            b_avg.append(b_xy)
            T_avg.append(np.mean(T_rz, axis=0))
            # dense plume analysis
            if all(salinity):
                S_avg.append(np.mean(S_rz, axis=0))
                r_profile.append(reader.loading_bin_contours())
                b_center.append(b_rz[0, :, :])
                T_fluc_center.append(T_rz[0, :, :])
                S_fluc_center.append(S_rz[0, :, :])
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
        _, b_rms_temp, _, b_fluc_centerline_temp = reader.load_buoyancy_small()
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
if plot_variables:
    if all(salinity):
        var_names = ['Tracer', 'Temperature', 'u', 'v', 'w']
        range_names = ['Tracer', 'T', 'u', 'v', 'w']
    else:
        var_names = ['Temperature', 'u', 'v', 'w']
        range_names = ['T', 'u', 'v', 'w']
    variable_dir = {}
if plot_var_bin:
    if all(salinity):
        bin_var_names = ['Tracer', 'Temperature', r'u$_r$', r'u$_{\theta}$', 'w']
        bin_range_names = ['Tracer', 'T', 'u', 'v', 'w']
    else:
        bin_var_names = ['Temperature', 'u', 'v', 'w']
        bin_range_names = ['T', 'u', 'v', 'w']
    bin_dir = {}
if plot_turb_stats or plot_temporal_avg:
    color_opt, line_opt = comparison_plot_opt(num_cases)
S_tol = 10**(-6)
ranges = plot_ranges(lz = 96, mld = np.max(mld), T0 = T0, dTdz = np.max(dTdz), C_tol = S_tol)
ranges['Tracer'] =[S_tol, 0.15]
ranges['Tracer negative'] = [-0.15, 0.15]
ranges['Tracer_fluc'] = [-0.2, 0.2]
ranges['Tracer_avg'] = [0, 1.2*10**(-3)]
ranges['T'] = [T0-0.7, T0 + 0.01]
ranges['w'] = [-1.5*10**(-1), 1.5*10**(-1)]
ranges['u'] = [-1.2*10**(-2), 1.2*10**(-2)]
ranges['v'] = [-2*10**(-2), 2*10**(-2)]
ranges['vel_rms'] = [0, 4*10**-3]
ranges['b_rms'] = [0, 1.5*10**(-5)]
ranges['S'] = [0.0, 0.05]
ranges['b_fluc'] = [-7*10**(-4), 7*10**(-4)]
ranges['bw_fluc'] = [-5*10**(-9), 5*10**(-9)]

# plotting with flags that have the possibility of being videos
if any([plot_variables, plot_var_bin, plot_turb_stats]):
    for it in range(nt):
        if plot_variables:
            if all(salinity): #'Tracer', 'T', 'u', 'v', 'w'
                variables = [[S_plane[i][it, :, :].T for i in range(num_cases)], [T_plane[i][it, :, :].T for i in range(num_cases)], [u_plane[i][it, :, :].T for i in range(num_cases)], [v_plane[i][it, :, :].T for i in range(num_cases)], [w_plane[i][it, :, :].T for i in range(num_cases)]]
                colorbar_labels = [r"g/kg", r"$^\circ$C", r"m/s", r"m/s", r"m/s"]
                cmaps = ['Blues', 'viridis', 'RdBu_r', 'RdBu_r', 'RdBu_r']
            else: #'T', 'u', 'v', 'w'
                variables = [[T_plane[i][it, :, :].T for i in range(num_cases)], [u_plane[i][it, :, :].T for i in range(num_cases)], [v_plane[i][it, :, :].T for i in range(num_cases)], [w_plane[i][it, :, :].T for i in range(num_cases)]]
                colorbar_labels = [r"$^\circ$C", r"m/s", r"m/s", r"m/s"]
                cmaps = ['viridis', 'RdBu_r', 'RdBu_r', 'RdBu_r']

            for dir, var in enumerate(variables):
                variable_dir[var_names[dir]] = plot_variable_vert_slice(time[it], it, ranges, fig_folder, lx, y, z, var, case_names, var_names[dir], range_names[dir], colorbar_label = colorbar_labels[dir], cmap = cmaps[dir], plane='YZ')
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
                bin_dir[bin_var_names[dir]] = plot_variable_vert_slice(time[it], it, ranges, fig_folder, lx, r_bin, z, var, case_names, bin_var_names[dir], bin_range_names[dir], colorbar_label = colorbar_labels[dir], cmap = cmaps[dir], plane='binning')
        if plot_turb_stats:
            S_avg_it = [S_avg[i][:, it] for i in range(num_cases)]
            u_rms_it = [u_rms[i][it, :] for i in range(num_cases)]
            v_rms_it = [v_rms[i][it, :] for i in range(num_cases)]
            w_rms_it = [w_rms[i][it, :] for i in range(num_cases)]
            b_avg_it = [b_avg[i][:, it] for i in range(num_cases)]
            b_center_it = [b_center[i][:, it] for i in range(num_cases)]
            r_profile_it = [r_profile[i][:, it] for i in range(num_cases)]
            bu_fluc_avg_it = [bu_fluc_avg[i][:, it] for i in range(num_cases)]
            bv_fluc_avg_it = [bv_fluc_avg[i][:, it] for i in range(num_cases)]
            bw_fluc_avg_it = [bw_fluc_avg[i][:, it] for i in range(num_cases)]
            T_avg_it = [T_avg[i][:, it] for i in range(num_cases)]
            T_fluc_center_it = [T_fluc_center[i][:, it] for i in range(num_cases)]
            S_fluc_center_it = [S_fluc_center[i][:, it] for i in range(num_cases)]
            buoyancy_dir_z = plot_turb_stats_bin(time[it], it, ranges, color_opt, fig_folder, case_names, name_uni, lx, z, S_avg_it, u_rms_it, v_rms_it, w_rms_it, b_avg_it, b_center_it, r_profile_it, bu_fluc_avg_it, bv_fluc_avg_it, bw_fluc_avg_it, T_avg_it, T_fluc_center_it, S_fluc_center_it)
# plotting with flags that don't have the possibility of being videos
if plot_temporal_avg:
    plot_format(fontsize = 10)
    temporal_avg(t_range, ranges, color_opt, fig_folder, case_names, lx, z, w_centerline_t_avg, S_centerline_t_avg, b_fluc_centerline_t_avg, w_rms_t_avg, b_rms_t_avg, h_ml = mld)
    print("Finished plotting temporal average for all cases.")
# creating videos
if video:
    if plot_var_bin:
        for n, name in enumerate(bin_var_names):
            create_video(bin_dir[bin_var_names[n]], fig_folder, 'binning', name)
    if plot_variables:
        for dir, name in enumerate(var_names):
            create_video(variable_dir[var_names[dir]], fig_folder, '', name)
    if plot_turb_stats:
        create_video(buoyancy_dir_z, fig_folder, 'binning', 'turb_stats')
