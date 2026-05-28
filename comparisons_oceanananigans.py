import os
import numpy as np
import matplotlib.pyplot as plt

from reader import OceananigansData
from dense_plume import PlumeAnalysis
from diagnostics import comparison_info
from physics import rms, buoyancy, buoyancy_flux_avg_1d, buoyancy_flux_line
from plotting_general import plot_format, plot_ranges, create_video, comparison_plot_opt
from plotting_lines import plot_plume_vertical_spatial, plot_plume_horizontal_spatial
from plotting_planes import plot_variable_vert_slice, plot_variable_xy_slice
from interpolation import vertical_line, horizontal_line

# flags for what to plot
plot_variables = False
plot_1d_z = True
plot_1d_y = False
temporal_averages_flag = False
video = True

# flags for how to read data
with_halos = False
closure = False
salinity = True
stokes = False

contour_bound = 0.001
name_uni = f'contour-{contour_bound:.4f}'
universal_folder = '/glade/derecho/scratch/apauls/outputs/'
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
z = []
nx = []
lx = []
if salinity:
    dense_plume = []

for i, reader in enumerate(readers):
    reader.load_time()
    reader.load_grid()
    z.append(reader.z)
    nx.append(reader.nx)
    lx.append(reader.lx)
    if i == 0:
        nt = reader.nt
        nz = reader.nx[2]
    else:
        nt = np.min([nt, reader.nt])
        nz = np.max([nz, reader.nx[2]])
    if salinity and plot_1d_z:
        S_value, w_value = reader.load_contour_temporal_averages('interp_temporal_averages.h5')
        dense_plume.append(PlumeAnalysis(S_value*contour_bound))

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
    planeslice = 'vertical' # 'vertical' or 'horizontal'
    variable_dir = {}
    if planeslice == 'horizontal':
        name_uni += '_horizontal_slice'
        loc = 'z' # 'cell' or 'z'
        if loc == 'z':
            loc_z = -mld
            name_uni += '_at_mld'
        else:
            n = 254
            loc_z = z[:, n]

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
if plot_1d_z or plot_1d_y:
    color_opt, line_opt = comparison_plot_opt(num_cases)

if plot_1d_y:
    ranges_hor = ranges.copy()
    ranges_hor['Tracer'] = [S_tol, 3*10**(-2)]
    ranges_hor['vel_rms'] = [0, 4*10**-3]
    ranges_hor['bw_fluc'] = [-2*10**(-5), 2*10**(-5)]
    ranges_hor['b_flux'] = [-4*10**(-6), 4*10**(-6)]
    ranges_hor['b_fluc'] = [-2*10**(-4), 2*10**(-4)]
    ranges_hor['w'] = [-0.15, 0.15]
    ranges_hor['T'] = [T0 - 0.2, T0 + 0.2]
    loc_z = -mld
    hor_str = ' '.join([f"{depth} m" for depth in loc_z])
    name_xy = name_uni + f"at z = {hor_str}"

if salinity:
    S_avg = []
    S_fluc_center = []
    S_hor = []
    S_plane = []
if plot_1d_z:
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
if plot_1d_y:
    y0 = 0.0
    u_hor = []
    v_hor = []
    w_hor = []
    b_fluc_hor = []
    bu_fluc_hor = []
    bv_fluc_hor = []
    bw_fluc_hor = []
    T_hor = []
if plot_variables:
    T_plane = []
    u_plane = []
    v_plane = []
    w_plane = []
    bw_plane = []

for i, reader in enumerate(readers):
    # Load data from files [nt, nx, ny, nz]
    if plot_variables:
        if planeslice == 'vertical': # load only vertical plane slices
            u = reader.field_slice('u')
            v = reader.field_slice('v')
            w = reader.field_slice('w')
            T = reader.field_slice('T')
        elif planeslice == 'horizontal': # load only horizontal plane slices
            u = reader.field_slice('u', slice='XY', loc=loc_z[i])
            v = reader.field_slice('v', slice='XY', loc=loc_z[i])
            w = reader.field_slice('w', slice='XY', loc=loc_z[i])
            T = reader.field_slice('T', slice='XY', loc=loc_z[i])
        T_plane.append(T)
        u_plane.append(u)
        v_plane.append(v)
        w_plane.append(w)
        if salinity:
            S = reader.field_slice('S')
            S_plane.append(S)
    if plot_1d_z or plot_1d_y:
        b = buoyancy(reader, T0 = T0)
    if plot_1d_z:
        # rms fluctuations
        u_rms.append(rms(reader, 'u'))
        v_rms.append(rms(reader, 'v'))
        w_rms.append(rms(reader, 'w'))
        bu_avg, bv_avg, bw_avg = buoyancy_flux_avg_1d(reader)
        bu_fluc_avg.append(bu_avg, axis=(-3, -2))
        bv_fluc_avg.append(bv_avg, axis=(-3, -2))
        bw_fluc_avg.append(bw_avg)
        # calculate means
        b_avg.append(b_xy)
        T_avg.append(reader.xy_avg_1d('T'))
        # dense plume analysis
        if salinity:
            S_avg.append(reader.xy_avg_1d('S'))
            dense_plume[i].input_info(S, b_tracer = b['b_C'], b_background = b['b_T'], bw_fluc = bw_fluc)
            r_profile.append(reader.loading_bin_radius())
            b_center.append(vertical_line(b, reader.x, reader.y, x0, y0))
            T_fluc_center.append(reader.field_line('T', x0 = x0, y0 = y0) - T_avg[i])
            S_fluc_center.append(reader.field_line('S', x0 = x0, y0 = y0) - S_avg[i])
    # horizontal lines to save for plotting
    if plot_1d_y:
        u_hor.append(reader.field_line('u', y0 = y0, z0 = loc_z[i]))
        v_hor.append(reader.field_line('v', y0 = y0, z0 = loc_z[i]))
        w_hor.append(reader.field_line('w', y0 = y0, z0 = loc_z[i]))
        T_hor.append(reader.field_line('T', y0 = y0, z0 = loc_z[i]))
        S_hor.append(reader.field_line('S', y0 = y0, z0 = loc_z[i]))
        b_fluc_hor.append(horizontal_line(b - b_avg, reader.x, reader.y, x0, y0, loc_z[i]))

        bu_fluc, bv_fluc, bw_fluc = buoyancy_flux_line(reader, loc_z[i], y0 = y0)
        bu_fluc_hor.append(reader.field_line('bu_fluc', y0 = y0, z0 = loc_z[i]))
        bv_fluc_hor.append(reader.field_line('bv_fluc', y0 = y0, z0 = loc_z[i]))
        bw_fluc_hor.append(reader.field_line('bw_fluc', y0 = y0, z0 = loc_z[i]))

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
        if planeslice == 'vertical':
            for dir, var in enumerate(variables):
                variable_dir[var_names[dir]] = plot_variable_vert_slice(time[it], it, ranges, fig_folder, lx[-1], reader.y, z, var, case_names, var_names[dir], range_names[dir], colorbar_label = colorbar_labels[dir], cmap = cmaps[dir], plane='YZ')
        elif planeslice == 'horizontal':
            for dir, var in enumerate(variables):
                variable_dir[var_names[dir]] = plot_variable_xy_slice(time[it], it, ranges, fig_folder, lx[-1], reader.x, reader.y, var, case_names, var_names[dir], range_names[dir], colorbar_label = colorbar_labels[dir], cmap = cmaps[dir])
    if plot_1d_z:
        buoyancy_dir_z = plot_plume_vertical_spatial(time[it], it, ranges, color_opt, fig_folder, case_names, name_uni, lx[-1], z, S_avg[it], u_rms[it], v_rms[it], w_rms[it], b_avg[it], b_center[it], r_profile[it], bu_fluc_avg[it], bv_fluc_avg[it], bw_fluc_avg[it], T_avg[it], T_fluc_center[it], S_fluc_center[it])
    if plot_1d_y:
        buoyancy_dir_y = plot_plume_horizontal_spatial(time[it], it, ranges_hor, color_opt, fig_folder, case_names, name_xy, lx[-1], reader.y, u_hor[it], v_hor[it], w_hor[it], b_fluc_hor[it], bu_fluc_hor[it], bv_fluc_hor[it], bw_fluc_hor[it], T_hor[it], S_hor[it])

print("All frames created.")
# creating videos
if video:
    if plot_variables:
        for dir, name in enumerate(var_names):
            create_video(variable_dir[var_names[dir]], fig_folder, name_uni, name)
    if plot_1d_z:
        create_video(buoyancy_dir_z, fig_folder, name_uni, 'vertical profile')
    if plot_1d_y:
        create_video(buoyancy_dir_y, fig_folder, name_uni, 'horizontal profile')
