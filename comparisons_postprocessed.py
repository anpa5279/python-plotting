import os
import numpy as np
import matplotlib.pyplot as plt

from reader import OceananigansData
from diagnostics import comparison_info
from physics import buoyancy
from plotting_general import plot_format, plot_ranges, create_video, comparison_plot_opt
from plotting_lines import plot_plume_vertical_spatial #, plot_plume_horizontal_spatial
from plotting_planes import plot_variable_vert_slice
#from interpolation import vertical_line, horizontal_line

# flags for what to plot
plot_variables = True
plot_var_bin = True
plot_turb_stats = True
video = True

# flags for how to read data
with_halos = False
closure = False
salinity = True
stokes = False

contour_bound = 0.001
name_uni = f'contour-{contour_bound:.4f}'
universal_folder = '/Users/annapauls/Documents/TESLa /Simulations/Oceananigans/dense plume/salinity and temperature/no noise circle inlet/'
#'/glade/derecho/scratch/apauls/outputs/'
#harddrive: '/Volumes/Anna External/Oceananigans/dense plume with stratification/salinity and temperature /no noise circle inlet/resolution testing'#

# selecting cases to compare
variations = 'horizontal resolution' # 'MLD', 'flux', 'strat', 'all', 'vertical length', 'WENO', 'vertical resolution', 'horizontal resolution', 'else'
if variations != 'else':
    cases_info = comparison_info(variations, universal_folder = universal_folder)
    dTdz = cases_info['dTdz']
    case_names = cases_info['case_names']
    num_cases = cases_info['num_cases']
    folder_names = cases_info['folder_names']
    fig_folder = cases_info['fig_folder']
    mld = cases_info['mld']
else:
    folder_names = ['proposed resolution/S0 = 0.1 dTdz = 0.01 MLD = 60', 'Lz = 160m/S0 = 0.1 dTdz = 0.01 MLD = 60']
    num_cases = len(folder_names)
    fig_folder = os.path.join(universal_folder, 'comparison figures', '96m vs 160m' + ' comparison figures', 'interpolated', 'default case')
    case_names =[r'L$_z = 96$m', r'L$_z = 160$m']#r'$\Delta z = 0.5$m', r'$\Delta z = 0.375$m'#[r'F$_{\text{C}} = -1.0\cdot 10^{-4}$, MLD = 60m, dTdz = 0.01', r'F$_{\text{C}} = -1.0\cdot 10^{-4}$, MLD = 70m, dTdz = 0.01', r'F$_{\text{C}} = - 2.0\cdot 10^{-4}$, MLD = 60m, dTdz = 0.01']
    dTdz = 0.01*np.ones(num_cases)
    mld = np.array([60, 60])

readers = []
for folder in folder_names:
    folder = os.path.join(universal_folder, folder)
    readers.append(OceananigansData(folder, salinity = salinity))

# collecting model information for all cases
mld_idx = []
x = []
y = []
z = []
nx = []
lx = []

for i, reader in enumerate(readers):
    reader.load_time()
    reader.load_grid()
    x.append(reader.x)
    y.append(reader.y)
    z.append(reader.z)
    nx.append(reader.nx)
    lx.append(reader.lx)
    if i == 0:
        nt = reader.nt
        nz = reader.nx[2]
        ny = reader.nx[1]
    else:
        nt = np.min([nt, reader.nt])
        nz = np.max([nz, reader.nx[2]])
        ny = np.max([ny, reader.nx[1]])
    if salinity and plot_turb_stats:
        S_value, w_value = reader.load_contour_temporal_averages('binning_rtz.h5')

# physical parameters
x0 = 0.0
y0 = 0.0
rj = 5 # m, radius of salinity flux circle at the surface
g = 9.80665  # gravity in m/s^2
T0 = 25

# video or not setup
if video:
    time = readers[0].time
else:
    time = readers[0].time[-1]

# plotting prep
plot_format()
if plot_variables:
    if salinity:
        var_names = ['Tracer', 'Temperature', 'u', 'v', 'w']
        range_names = ['Tracer', 'T', 'u', 'v', 'w']
    else:
        var_names = ['Temperature', 'u', 'v', 'w']
        range_names = ['T', 'u', 'v', 'w']
    variable_dir = {}

S_tol = 10**(-6)
ranges = plot_ranges(lz = 96, mld = np.max(mld), T0 = T0, dTdz = np.max(dTdz), C_tol = S_tol)
ranges['Tracer'] =[S_tol, 0.15]
ranges['Tracer_fluc'] = [-0.2, 0.2]
ranges['Tracer_avg'] = [0, 1.2*10**(-3)]
ranges['T'] = [T0-1.0, T0 + 0.01]
ranges['w'] = [-1.5*10**(-1), 1.5*10**(-1)]
ranges['u'] = [-1.2*10**(-2), 1.2*10**(-2)]
ranges['v'] = [-2*10**(-2), 2*10**(-2)]
ranges['vel_rms'] = [0, 4*10**-3]
ranges['bw_fluc'] = [-5*10**(-9), 5*10**(-9)]
if plot_turb_stats:
    color_opt, line_opt = comparison_plot_opt(num_cases)

if salinity:
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

for i, reader in enumerate(readers):
    # Load data from files [nt, nx, ny, nz]
    print(i)
    if plot_variables:
        print('loading plane variables')
        T_plane.append(reader.load_plane_var('T'))
        u_plane.append(reader.load_plane_var('u'))
        v_plane.append(reader.load_plane_var('v'))
        w_plane.append(reader.load_plane_var('w'))
        if salinity:
            S_plane.append(reader.load_plane_var('S'))
        b_plane = buoyancy(reader, type = 'plane')
    # Load binning from files
    if plot_var_bin or plot_turb_stats:
        print('loading binning variables')
        ur_rz = reader.load_binning_var('horizontal velocity')
        utheta_rz = reader.load_binning_var('rotation velocity')
        w_rz = reader.load_binning_var('w')
        T_rz = reader.load_binning_var('T')
        print(T_rz)
        S_rz = reader.load_binning_var('S')
        r = reader.loading_bin_radius()
        b_rz = buoyancy(reader, type = 'bin')
        b_xy = np.mean(b_rz, axis=0)
    if plot_turb_stats:
        # rms fluctuations
        u_rms.append(reader.load_vel_rms('u'))
        v_rms.append(reader.load_vel_rms('v'))
        w_rms.append(reader.load_vel_rms('w'))
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
        if salinity:
            S_avg.append(np.mean(S_rz, axis=0))
            r_profile.append(reader.loading_bin_radius())
            b_center.append(b_rz[0, :, :])
            T_fluc_center.append(T_rz[0, :, :])
            S_fluc_center.append(S_rz[0, :, :])

############ PLOTTING ############
for it in nt:
    if plot_variables:
        if salinity: #'Tracer', 'T', 'u', 'v', 'w'
            variables = [S_plane[it], T_plane[it], u_plane[it], v_plane[it], w_plane[it]]
            colorbar_labels = [r"g/kg", r"$^\circ$C", r"m/s", r"m/s", r"m/s"]
            cmaps = ['viridis', 'viridis', 'RdBu_r', 'RdBu_r', 'RdBu_r']
        else: #'T', 'u', 'v', 'w'
            variables = [T_plane[it], u_plane[it], v_plane[it], w_plane[it]]
            colorbar_labels = [r"$^\circ$C", r"m/s", r"m/s", r"m/s"]
            cmaps = ['viridis', 'viridis', 'RdBu_r', 'RdBu_r', 'RdBu_r']

        for dir, var in enumerate(variables):
            variable_dir[var_names[dir]] = plot_variable_vert_slice(time[it], it, ranges, fig_folder, lx[-1], reader.y, z, var, case_names, var_names[dir], range_names[dir], colorbar_label = colorbar_labels[dir], cmap = cmaps[dir], plane='YZ')
    if plot_turb_stats:
        buoyancy_dir_z = plot_plume_vertical_spatial(time[it], it, ranges, color_opt, fig_folder, case_names, name_uni, lx[-1], z, S_avg[it], u_rms[it], v_rms[it], w_rms[it], b_avg[it], b_center[it], r_profile[it], bu_fluc_avg[it], bv_fluc_avg[it], bw_fluc_avg[it], T_avg[it], T_fluc_center[it], S_fluc_center[it])


# creating videos
if video:
    if plot_var_bin:
        for n, name in enumerate(var_names):
            create_video(variable_dir[var_names[n]], fig_folder, 'binning', name)
    if plot_variables:
        for dir, name in enumerate(var_names):
            create_video(variable_dir[var_names[dir]], fig_folder, name_uni, name)
    if plot_turb_stats:
        create_video(turb_plot, fig_folder, 'binning', 'turb_stats')
