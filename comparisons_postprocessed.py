import os
import numpy as np

from reader import OceananigansData
from diagnostics import comparison_info
from interpolation import point, horizontal_line
from physics import buoyancy
from plotting_general import plot_format, plot_ranges, create_video, comparison_plot_opt
from plotting_lines import plot_turb_stats_bin, plot_plume_depths, plot_plume_vertical_spatial, plot_lines, plot_lines_h, plot_scaling_analysis
from plotting_planes import plot_var_planeslice

# ==========================================================
# FLAGS
# ==========================================================
# plotting flags
plot_plane_vert = False
plot_plane_hor = False
plot_plane_bin_vert = False
plot_turb_stats = False
plot_depths = False
plot_plume_z = False
plot_vert_profiles = False
plot_horiz_profiles = False
plot_hor_bin_profiles = True
plot_scaling_analysis_text = False

video = True

# how to process the data
binning_only = True
temporal_avg = True
plot_difference = False

if plot_vert_profiles or plot_horiz_profiles or plot_plume_z or plot_turb_stats:
    buoyancy_calc = True
else:
    buoyancy_calc = False
if binning_only:
    if plot_horiz_profiles:
        plot_horiz_profiles = False
        plot_hor_bin_profiles = True
    fluc_file = 'binning_rtz.h5'
else:
    fluc_file = 'fluctuations.h5'
if temporal_avg:
    nsteps = 5 # number of time steps to average over for plotting, also trying 5

# flags for how to read the model information
with_halos = False
closure = False
stokes = False

# ==========================================================
# COMPARISON CASES
# ==========================================================
contour = 0.01
name_uni = f'contour-{contour:.4f}'
universal_folder = '/Users/annapauls/Documents/Github repositories/3d_langmuir_gpu/localoutputs/scheme-tests/longer'
#'/Users/annapauls/Documents/Github repositories/3d_langmuir_gpu/localoutputs/scaled S double to match both gauss/WENO5'

variations = 'else'
if variations != 'else':
    cases_info = comparison_info(variations, universal_folder = universal_folder)
    case_names = cases_info['case_names']
    num_cases = cases_info['num_cases']
    folder_names = cases_info['folder_names']
    fig_folder = cases_info['fig_folder']
else:
    dx_res = 0.25
    folder_names = [f'WENO9/dx{dx_res}', f'WENO9/dx{dx_res*0.5}', f'WENO5/dx{dx_res}', f'WENO5/dx{dx_res*0.5}', f'WENO5/dx{dx_res*0.5**2}'] #
    #['dx2.0', 'dx1.0', 'dx0.5', 'dx0.25', 'dx0.125']#, 'dx0.0625']#['dx2', 'dx1', 'dx05', 'dx025', 'dx0125', 'dx00625']#

    case_names = [rf'WENO9, $\Delta x = {dx_res}$', rf'WENO9, $\Delta x = {dx_res*0.5}$', rf'WENO5, $\Delta x = {dx_res}$', rf'WENO5, $\Delta x = {dx_res*0.5}$', rf'WENO5, $\Delta x = {dx_res*0.5**2}$'] #
    #[r'$\Delta x = 2.0$', r'$\Delta x = 1.0$', r'$\Delta x = 0.5$', r'$\Delta x = 0.25$', r'$\Delta x = 0.125$', r'$\Delta x = 0.0625$']#, r'$\Delta x = 0.25$']#[r'$\Delta x = \Delta y = \Delta z = 2.0$', r'$\Delta x = \Delta y = 1.0$ $ \Delta z = 2.0$', r'$\Delta x = \Delta y = 0.5$ $ \Delta z = 2.0$']#[r'$\Delta x = \Delta y = \Delta z = 2.0$', r'$\Delta x = \Delta y = 2.0$ $ \Delta z = 1.0$', r'$\Delta x = \Delta y = 2.0$ $ \Delta z = 0.5$']#

    num_cases = len(folder_names)
    case_opt = [2, 3] #[1, ]
    fig_folder = os.path.join(universal_folder, 'comparison figures', 'comparing cases to WENO9 high res', 'coloring test')#f'dx{dx_res}')
    dTdz = 0.01*np.ones(num_cases)
    mld = 60**np.ones(num_cases)
    F_s = 0.1*np.ones(num_cases)

# ==========================================================
# READERS
# ==========================================================
readers = []
salinity = []
for n, folder in enumerate(folder_names):
    if F_s[n] == 0.0:
        salinity.append(False)
    else:
        salinity.append(True)
    folder = os.path.join(universal_folder, folder)
    readers.append(OceananigansData(folder, salinity = salinity[n], Sval = 0.1))

# ==========================================================
# MODEL INFORMATION
# ==========================================================
if plot_plane_bin_vert or plot_hor_bin_profiles or plot_scaling_analysis_text:
    r = []
x = []
y = []
z = []
time  = []
nx = np.empty((3, num_cases), dtype=object)
lx = np.empty((3, num_cases), dtype=object)
grid_specs = False*np.ones(num_cases)
nt_avg = np.inf
nt_min = np.inf
for n, reader in enumerate(readers):
    if plot_plane_hor:
        x.append(reader.x)
    y.append(reader.y)
    z.append(reader.z)
    time.append(reader.t)
    nx[:, n] = reader.nx
    lx[:, n] = reader.lx
    nt_min = int(np.min([nt_min, reader.nt]))

    if plot_plane_bin_vert or plot_hor_bin_profiles or plot_scaling_analysis_text:
        r.append(reader.r)

    if salinity[n] and plot_turb_stats:
        S_value = reader.load_S_temporal_avg()

    if plot_vert_profiles:
        nt_avg_loc = len(reader.time_avg)
        if nt_avg_loc < nt_avg:
            nt_avg = nt_avg_loc
            min_time_avg = reader.time_avg

lx[-1, :] = -lx[-1, :]

# ==========================================================
# PARAMETERS
# ==========================================================
x0 = 0.0
y0 = 0.0
rj = 5 # m, radius of salinity flux circle at the surface
g = 9.80665  # gravity in m/s^2
T0 = 25
S_tol = 10**(-6)
w0 = -0.001
Sval = 0.1

# collecting variables for plotting
if plot_horiz_profiles or plot_hor_bin_profiles:
    loc_z = [-10,]#[-70, -mld[0], -50, -40, -30, -20, -10]#
    factor = 0.7
    hor_str = ' '.join([f"{depth} m" for depth in loc_z])
    name_xy = name_uni + f"at z = {hor_str}"
if plot_hor_bin_profiles:
    S_range = [np.inf, -np.inf]
    b_range = [np.inf, -np.inf]
    w_range = [np.inf, -np.inf]



# ==========================================================
# ANALYSIS
# ==========================================================
# plane slices
S_vert_plane = []
T_vert_plane = []
u_vert_plane = []
v_vert_plane = []
w_vert_plane = []
S_hor_plane = []
T_hor_plane = []
u_hor_plane = []
v_hor_plane = []
w_hor_plane = []
bw_plane = []

# vertical averages
T_avg = []
S_avg = []
b_avg = []

ur_avg = []

Sur_avg = []
Sw_avg = []
bur_fluc_avg = []
bw_fluc_avg = []
urw_avg = []
Tur_avg = []
Tw_avg = []

# RMS
u_rms = []
v_rms = []
w_rms = []
ur_rms = []
b_rms = []

# binning
r_bin = []
S_bin = []
T_bin = []
ur_bin = []
w_bin = []
b_bin = []

# centerlines / lines
b_center = []
b_fluc_center = []
T_fluc_center = []
S_center = []
T_center = []

w_hor = []
S_hor = []

w_bin_hor = []
b_bin_hor = []
S_bin_hor = []

# scaling analysis
F = []
F_transverse = []
c_delta = []
c_w = []
eta = []
z_mld = []

# plume depths
zp = []
zneutral = []
zc = []

# time
time_output = []

for n, reader in enumerate(readers):
    # load buoyancy information
    if buoyancy_calc:
        b_avg_loc, b_rms_loc, b_centerline_loc, b_fluc_centerline_loc = reader.load_buoyancy(file = fluc_file)
    # Load data from files [nt, nx, ny, nz]
    if plot_plane_vert or plot_horiz_profiles:
        T_vert_plane.append(reader.load_plane_var('T'))
        u_vert_plane.append(reader.load_plane_var('u'))
        v_vert_plane.append(reader.load_plane_var('v'))
        w_vert_plane.append(reader.load_plane_var('w'))
        if all(salinity):
            S_vert_plane.append(reader.load_plane_var('S'))
            #S_vert_plane[n][S_vert_plane[n]<S_tol] = S_tol # set values below threshold to threshold for log plotting
    if plot_plane_hor:
        z_locs = [-0.75, -1.0, -1.5, -60.0]#-3*reader.dx[-1]/2#0.0#-reader.lx[-1] + reader.dx[-1]/2#-reader.dx[-1]/2# 
        T_hor_loc = np.empty((len(z_locs), reader.nt, reader.nx[0], reader.nx[1]))
        u_hor_loc = np.empty((len(z_locs), reader.nt, reader.nx[0], reader.nx[1]))
        v_hor_loc = np.empty((len(z_locs), reader.nt, reader.nx[0], reader.nx[1]))
        w_hor_loc = np.empty((len(z_locs), reader.nt, reader.nx[0], reader.nx[1]))
        if all(salinity):
            S_hor_loc = np.empty((len(z_locs), reader.nt, reader.nx[0], reader.nx[1]))
        for k, z_loc in enumerate(z_locs):
            T_hor_loc[k, :, :, :] = reader.load_plane_var('T', loc=z_loc, plane = 'XY')
            u_hor_loc[k, :, :, :] = reader.load_plane_var('u', loc=z_loc, plane = 'XY')
            v_hor_loc[k, :, :, :] = reader.load_plane_var('v', loc=z_loc, plane = 'XY')
            w_hor_loc[k, :, :, :] = reader.load_plane_var('w', loc=z_loc, plane = 'XY')
            if all(salinity):
                S_hor_loc[k, :, :, :] = reader.load_plane_var('S', loc=z_loc, plane = 'XY')
        T_hor_plane.append(T_hor_loc)
        u_hor_plane.append(u_hor_loc)
        v_hor_plane.append(v_hor_loc)
        w_hor_plane.append(w_hor_loc)
        if all(salinity):
            S_hor_plane.append(S_hor_loc)
    # load averages
    if plot_vert_profiles or plot_plume_z or plot_turb_stats:
        if binning_only:
            ur_rms.append(reader.load_rms('ur', file = fluc_file))
            ur_avg.append(reader.load_averages('ur', binning = binning_only))
        else:
            u_rms.append(reader.load_rms('u', file = fluc_file))
            v_rms.append(reader.load_rms('v', file = fluc_file))
        w_rms.append(reader.load_rms('w', file = fluc_file))

        S_avg.append(reader.load_averages('S', binning = binning_only))

        b_avg.append(b_avg_loc)
        b_center.append(b_centerline_loc)
        b_fluc_center.append(b_fluc_centerline_loc)
        bw_fluc_avg.append(reader.load_averages('b_fluc_w', binning = binning_only))
        b_rms.append(b_rms_loc)

        T_avg.append(reader.load_averages('T', binning = binning_only))

        T_center.append(reader.field_centerline('T', binning = binning_only))
        T_fluc_center.append(T_center[n]-T_avg[n])
        S_center.append(reader.field_centerline('S', binning = binning_only))

    # load horizontal profile information            
    if plot_horiz_profiles:
        w_line_loc = np.empty((reader.nt, reader.nx[1], len(loc_z)))
        S_line_loc = np.empty((reader.nt, reader.nx[1], len(loc_z)))
        for k, loc in enumerate(loc_z):
            w_line_loc[:, :, k] = horizontal_line(w_vert_plane[n], z = reader.zf, z0 = loc, axis=-1)
            S_line_loc[:, :, k] = horizontal_line(S_vert_plane[n], z = reader.z, z0 = loc, axis=-1)

        w_hor.append(w_line_loc)
        S_hor.append(S_line_loc)

    # Load binning from files
    if plot_plane_bin_vert or plot_turb_stats or plot_hor_bin_profiles:
        if all(salinity):
            S_rz = reader.load_binning_var('S')
            S_bin.append(S_rz)
        w_rz = reader.load_binning_var('w')
        b_rz = buoyancy(reader, type = 'bin')
        w_bin.append(w_rz)
        b_bin.append(b_rz)
    if plot_plane_bin_vert or plot_turb_stats:
        ur_rz = reader.load_binning_var('horizontal velocity')
        T_rz = reader.load_binning_var('T')
        T_bin.append(T_rz)
        ur_bin.append(ur_rz)
    if plot_hor_bin_profiles:
        w_bin_line_loc = np.empty((reader.nt, len(reader.r), len(loc_z)))
        S_bin_line_loc = np.empty((reader.nt, len(reader.r), len(loc_z)))
        b_bin_line_loc = np.empty((reader.nt, len(reader.r), len(loc_z)))
        for k, loc in enumerate(loc_z):
            w_bin_line_loc[:, :, k] = horizontal_line(w_bin[n], z = reader.zf, z0 = loc, axis=-1)
            S_bin_line_loc[:, :, k] = horizontal_line(S_bin[n], z = reader.z, z0 = loc, axis=-1)
            b_bin_line_loc[:, :, k] = horizontal_line(b_bin[n], z = reader.z, z0 = loc, axis=-1)
        w_range[0] = min(w_range[0], np.min(w_bin_line_loc))#*factor
        w_range[1] = max(w_range[1], np.max(w_bin_line_loc))
        S_range[0] = min(S_range[0], np.min(S_bin_line_loc))
        S_range[1] = max(S_range[1], np.max(S_bin_line_loc))*factor
        b_range[0] = min(b_range[0], np.min(b_bin_line_loc))
        b_range[1] = max(b_range[1], np.max(b_bin_line_loc))*factor
        w_bin_hor.append(w_bin_line_loc)
        S_bin_hor.append(S_bin_line_loc)
        b_bin_hor.append(b_bin_line_loc)
    if plot_turb_stats:
        urw_avg.append(np.mean(ur_rz * w_rz, axis=0))
        # rms fluctuations
        ur_avg = np.mean(ur_rz, axis=0)
        ur_rms.append(np.sqrt(np.mean((ur_rz - ur_avg)**2, axis=0)))
        bur_avg = np.mean(b_rz * ur_rz, axis=0)
        bw_avg = np.mean(b_rz * w_rz, axis=0)
        bur_fluc_avg.append(bur_avg)
        bw_fluc_avg.append(bw_avg)
        # calculate means
        Tur_avg.append(np.mean((T_rz-T_avg) * ur_rz, axis=0))
        Tw_avg.append(np.mean((T_rz-T_avg) * w_rz, axis=0))
        if all(salinity):
            Sur_avg.append(np.mean(S_rz * ur_rz, axis=0))
            Sw_avg.append(np.mean(S_rz * w_rz, axis=0))
    if plot_depths:
        r_bin.append(reader.loading_bin_contours(contour = contour))
        time_output.append(reader.time_avg)
        S_value = reader.load_S_temporal_avg()
        bS = -g*reader.beta*S_value
        bT_avg = g*reader.alpha*(reader.load_averages('T') - reader.T0)
        # calculate where w = 0 on the centerline
        w_reader_center = reader.field_centerline('w')

        nt_output = len(reader.time_avg)
        neutral_it = np.zeros(nt_output)
        w_centerline_it = np.zeros(nt_output)
        S_it = np.zeros(nt_output)
        for it in range(0, nt_output):
            # based on buoyancy differences
            b_diff = bT_avg[it,:] - bS
            z_it = point(b_diff, z[n], f0 = 0.0)
            neutral_it[it] = z_it
            # based on w = 0
            z_it = point(w_reader_center[it, :], z[n], f0 = 0.0)
            if np.size(z_it) == 1:
                w_centerline_it[it] = z_it
            elif np.size(z_it) == 0:
                w_centerline_it[it] = 0.0
            else: # there are multiple points where w = 0
                # check if w_reader_center[it, :] goes from + to -
                w_centerline_it[it] = z_it[-1] # take the shallowest point
            # based on contour 
            z_it = point(S_center[n][it, :], z[n], f0 = S_value*contour)
            if np.size(z_it) == 1:
                S_it[it] = z_it
            elif np.size(z_it) == 0:
                S_it[it] = 0.0
            else: # there are multiple points where S = contour
                # get the deepest
                S_it[it] = z_it[0] # take the shallowest point
        zneutral.append(neutral_it)
        zp.append(w_centerline_it)
        zc.append(S_it)
    if plot_plume_z:
        bur_fluc_avg.append(reader.load_fluc('bur'))
        bw_fluc_avg.append(reader.load_fluc('bw'))
    if plot_scaling_analysis_text:
        above_mld = np.where(reader.z>=-mld[n]+10.0)[0]
        F.append(reader.load_scaling_analysis('F'))
        F_transverse.append(reader.load_scaling_analysis('F_transverse'))
        c_delta.append(reader.load_scaling_analysis('c_delta'))
        c_w.append(reader.load_scaling_analysis('c_w'))
        eta.append(reader.load_scaling_analysis('eta'))
        z_mld.append(z[n][above_mld])
if temporal_avg:
    n_start = (nsteps-1)//2
    n_end = nt_min - (nsteps-1)//2
    for n, reader in enumerate(readers):
        if plot_hor_bin_profiles:
            w_bin_line_loc = np.empty((n_end - n_start, len(reader.r), len(loc_z)))
            S_bin_line_loc = np.empty((n_end - n_start, len(reader.r), len(loc_z)))
            #b_bin_line_loc = np.empty((n_end - n_start, len(reader.r), len(loc_z)))
            for it, it_true in enumerate(np.arange(n_start, n_end)):
                for k, loc in enumerate(loc_z):
                    w_bin_line_loc[it, :, k] = np.mean(w_bin_hor[n][it_true-n_start:it_true-n_start+nsteps, :, k], axis=0)
                    S_bin_line_loc[it, :, k] = np.mean(S_bin_hor[n][it_true-n_start:it_true-n_start+nsteps, :, k], axis=0)
                    #b_bin_line_loc[it-n_start, :, k] = np.mean(b_bin_hor[n][it-n_start:it-n_start+2, :, k], axis=0)
            w_bin_hor[n] = w_bin_line_loc
            S_bin_hor[n] = S_bin_line_loc
            #b_bin_hor[n] = b_bin_line_loc
        if plot_vert_profiles: 
            ur_rms_line_loc = np.empty((n_end - n_start, nx[-1, n]))
            w_rms_line_loc = np.empty((n_end - n_start, nx[-1, n]))
            b_rms_line_loc = np.empty((n_end - n_start, nx[-1, n]))
            S_bin_line_loc = np.empty((n_end - n_start, nx[-1, n]))
            T_fluc_center_loc = np.empty((n_end - n_start, nx[-1, n]))
            ur_avg_line_loc = np.empty((n_end - n_start, nx[-1, n]))
            for it, it_true in enumerate(np.arange(n_start, n_end)):
                ur_rms_line_loc[it, :] = np.mean(ur_rms[n][it_true-n_start:it_true-n_start+nsteps, :], axis=0)
                w_rms_line_loc[it, :] = np.mean(w_rms[n][it_true-n_start:it_true-n_start+nsteps, :], axis=0)
                b_rms_line_loc[it, :] = np.mean(b_rms[n][it_true-n_start:it_true-n_start+nsteps, :], axis=0)
                S_bin_line_loc[it, :] = np.mean(S_avg[n][it_true-n_start:it_true-n_start+nsteps, :], axis=0)
                T_fluc_center_loc[it, :] = np.mean(T_fluc_center[n][it_true-n_start:it_true-n_start+nsteps, :], axis=0)
                ur_avg_line_loc[it, :] = np.mean(ur_avg[n][it_true-n_start:it_true-n_start+nsteps, :], axis=0)
            ur_rms[n] = ur_rms_line_loc
            w_rms[n] = w_rms_line_loc
            b_rms[n] = b_rms_line_loc
            S_avg[n] = S_bin_line_loc
            T_fluc_center[n] = T_fluc_center_loc
            ur_avg[n] = ur_avg_line_loc
            #b_bin_hor[n] = b_bin_line_loc

# ==========================================================
# PLOTTING
# ==========================================================
# plotting prep
plot_format(fontsize = 20)
if plot_plane_vert or plot_plane_hor or plot_plane_bin_vert:
    variable_dir = {}
    variable_dir_hor = {}
    bin_dir = {}
if plot_turb_stats or plot_plume_z or plot_depths or plot_vert_profiles or plot_horiz_profiles or plot_hor_bin_profiles or plot_scaling_analysis_text:
    case_opt = comparison_plot_opt(case_names, distinct_groups = case_opt)

ranges = plot_ranges(lz = 96, mld = np.max(mld), T0 = T0, dTdz = np.max(dTdz), C_tol = S_tol)
ranges['log Tracer'] =[S_tol, 0.15]
ranges['Tracer negative'] = [-0.15, 0.15]
ranges['Tracer_fluc'] = [-0.2, 0.2]
ranges['Tracer_avg'] = [0, 8*10**(-4)]
ranges['T'] = [T0-0.7, T0 + 0.05]
ranges['w'] = [-8*10**(-2), 8*10**(-2)]
ranges['u'] = [-5*10**(-3), 5*10**(-3)]
ranges['ur'] = [-1*10**(-2), 1*10**(-2)]
ranges['v'] = [-2*10**(-2), 2*10**(-2)]
ranges['vel_rms'] = [0, 1*10**(-2)]
ranges['b_rms'] = [0, 1*10**(-4)]
ranges['S'] = [0.0, 0.08]
ranges['b_fluc_center'] = [-6*10**(-4), 6*10**(-4)]
ranges['bw_fluc'] = [-1*10**(-6), 1*10**(-6)]
ranges['S_avg'] = [0, 6.0*10**(-3)]
ranges['log w'] = [-0.1, 0.1]
ranges['T_fluc_center'] = [-2*10**(-1), 2*10**(-1)]
if plot_turb_stats:
    ranges['restress'] = [-2*10**(-5), 2*10**(-5)]
    ranges['Tw_fluc'] = [-5*10**(-4), 5*10**(-4)]
    ranges['Cw'] = [-5*10**(-5), 5*10**(-5)]
if plot_plane_hor or plot_horiz_profiles:
    ranges_horiz = ranges.copy()
    ranges_horiz['T'] = [T0-0.05, T0+0.05]
    ranges_horiz['u'] = [-2*10**(-2), 2*10**(-2)]
    ranges_horiz['v'] = ranges_horiz['u']
    ranges_horiz['w'] = [-2*10**(-1), 3*10**(-2)]
    ranges_horiz['log w'] = [-6, -3]
    ranges_horiz['w scaled'] = [-1.05, 0]
    ranges_horiz['log neg S'] = [-0.08, 0.08]
    ranges_horiz['Tracer'] = [0, 3*10**(-2)]
    ranges_horiz['vel_rms'] = [0, 4*10**-3]
    ranges_horiz['vel_rms'] = [0, 4*10**-3]
    ranges_horiz['bw_fluc'] = [-2*10**(-5), 2*10**(-5)]
    ranges_horiz['b_flux'] = [-4*10**(-6), 4*10**(-6)]
    ranges_horiz['b_fluc'] = [-2*10**(-4), 2*10**(-4)]
    ranges_horiz['S'] = [0, 0.12]

# plotting with flags that have the possibility of being videos
if any([plot_plane_vert, plot_plane_hor, plot_plane_bin_vert, plot_turb_stats]):
    time_min = min(time, key=len)
    var_names = ['T', 'u', 'v', 'w', 'Tracer negative']
    bin_var_names = ['T', 'ur', 'log w', 'Tracer negative']
    for it in range(nt_min):
        if plot_plane_vert or plot_plane_hor:
            if plot_plane_vert:
                variables = {}
                if all(salinity):
                    variables[var_names[-1]] = {'var':[S_vert_plane[n][it, :, :].T for n in range(num_cases)], 'title':'Tracer', 'label':r"g/kg", 'cmap':'RdBu', 'range':ranges[var_names[-1]], 'range_name':var_names[-1]}
  
                variables[var_names[0]] = {'var':[T_vert_plane[n][it, :, :].T for n in range(num_cases)], 'title':'Temperature', 'label':r"$^\circ$C", 'cmap':'viridis', 'range':ranges[var_names[0]], 'range_name':var_names[0]}
                variables[var_names[1]] = {'var':[u_vert_plane[n][it, :, :].T for n in range(num_cases)], 'title':'u', 'label':r"m/s", 'cmap':'RdBu_r', 'range':ranges[var_names[1]], 'range_name':var_names[1]}
                variables[var_names[2]] = {'var':[v_vert_plane[n][it, :, :].T for n in range(num_cases)], 'title':'v', 'label':r"m/s", 'cmap':'RdBu_r', 'range':ranges[var_names[2]], 'range_name':var_names[2]}
                variables[var_names[3]] = {'var':[w_vert_plane[n][it, :, :].T for n in range(num_cases)], 'title':'w', 'label':r"m/s", 'cmap':'RdBu_r', 'range':ranges[var_names[3]], 'range_name':var_names[3]}
                for dir, var in enumerate(var_names):
                    variable_dir[var_names[dir]] = plot_var_planeslice(time_min[it], it, fig_folder, lx, y, z, variables[var], case_names, plane='YZ')
            if plot_plane_hor:
                for k, z_loc in enumerate(z_locs):
                    variables = {}
                    if all(salinity):
                        variables[var_names[-1]] = {'var':[S_hor_plane[n][k, it, :, :].T for n in range(num_cases)], 'title':'Tracer', 'label':r"g/kg", 'cmap':'RdBu', 'range':ranges_horiz[var_names[-1]], 'range_name':var_names[-1]}
                    variables[var_names[0]] = {'var':[T_hor_plane[n][k, it, :, :].T for n in range(num_cases)], 'title':'Temperature', 'label':r"$^\circ$C", 'cmap':'viridis', 'range':ranges_horiz[var_names[0]], 'range_name':var_names[0]}
                    variables[var_names[1]] = {'var':[u_hor_plane[n][k, it, :, :].T for n in range(num_cases)], 'title':'u', 'label':r"m/s", 'cmap':'RdBu_r', 'range':ranges_horiz[var_names[1]], 'range_name':var_names[1]}
                    variables[var_names[2]] = {'var':[v_hor_plane[n][k, it, :, :].T for n in range(num_cases)], 'title':'v', 'label':r"m/s", 'cmap':'RdBu_r', 'range':ranges_horiz[var_names[2]], 'range_name':var_names[2]}
                    variables[var_names[3]] = {'var':[w_hor_plane[n][k, it, :, :].T for n in range(num_cases)], 'title':'w', 'label':r"m/s", 'cmap':'RdBu_r', 'range':ranges_horiz[var_names[3]], 'range_name':var_names[3]}
                    for dir, var in enumerate(var_names):
                        variable_dir_hor.setdefault(var_names[dir], {})[z_loc] = plot_var_planeslice(time_min[it], it, fig_folder, lx, x, y, variables[var], case_names, loc = z_loc, plane='XY')

        if plot_plane_bin_vert:
            variables = {}
           #if all(salinity): #'Tracer', 'T', 'u', 'v', 'w'
           #    variables[bin_var_names[-1]] = {'var':[S_bin[n][it, :, :].T for n in range(num_cases)], 'title':'Tracer', 'label':r"g/kg", 'cmap':'RdBu', 'range':ranges[bin_var_names[-1]], 'range_name':bin_var_names[-1]}
            variables[bin_var_names[0]] = {'var':[T_bin[n][it, :, :].T for n in range(num_cases)], 'title':'Temperature', 'label':r"$^\circ$C", 'cmap':'viridis', 'range':ranges[bin_var_names[0]], 'range_name':bin_var_names[0]}
            variables[bin_var_names[1]] = {'var':[ur_bin[n][it, :, :].T for n in range(num_cases)], 'title':r'u$_r$', 'label':r"m/s", 'cmap':'RdBu_r', 'range':ranges[bin_var_names[1]], 'range_name':bin_var_names[1]}
            variables[bin_var_names[2]] = {'var':[w_bin[n][it, :, :].T for n in range(num_cases)], 'title':'w', 'label':r"m/s", 'cmap':'RdBu_r', 'range':ranges[bin_var_names[2]], 'range_name':bin_var_names[2]}

            for dir, var in enumerate(bin_var_names[:-1]):
                l_bin = [lx[0]/2, lx[-1]]
                bin_dir[bin_var_names[dir]] = plot_var_planeslice(time_min[it], it, fig_folder, lx, r, z, variables[var], case_names, plane='binning')

        if plot_turb_stats:
            ur_rms_it = [ur_rms[n][:, it] for n in range(num_cases)]
            w_rms_it = [w_rms[n][:, it] for n in range(num_cases)]
            uw_avg_it = [urw_avg[n][:, it] for n in range(num_cases)]
            b_avg_it = [b_avg[n][:, it] for n in range(num_cases)]
            bur_fluc_avg_it = [bur_fluc_avg[n][:, it] for n in range(num_cases)]
            bw_fluc_avg_it = [bw_fluc_avg[n][:, it] for n in range(num_cases)]
            Tur_avg_it = [Tur_avg[n][:, it] for n in range(num_cases)]
            Tw_avg_it = [Tw_avg[n][:, it] for n in range(num_cases)]
            Sur_avg_it = [Sur_avg[n][:, it] for n in range(num_cases)]
            Sw_avg_it = [Sw_avg[n][:, it] for n in range(num_cases)]
            buoyancy_dir_z = plot_turb_stats_bin(time_min[it], it, ranges, case_opt, fig_folder, case_names, z, ur_rms_it, w_rms_it, uw_avg_it, b_avg_it, bur_fluc_avg_it, bw_fluc_avg_it, Tur_avg_it, Tw_avg_it, Sur_avg_it, Sw_avg_it)
if plot_plume_z:
    buoyancy_dir_z = plot_plume_vertical_spatial(min(time, key=len), ranges, case_opt, fig_folder, case_names, name_uni, lx, z, S_avg, u_rms, v_rms, w_rms, b_avg, b_center, r_bin, bur_fluc_avg, bw_fluc_avg, T_avg, T_fluc_center, S_center)
if plot_vert_profiles:
    if temporal_avg:
        for it, it_true in enumerate(np.arange(n_start, n_end)):
            lines_var_it = {}
            title = f'Temporal average from {reader.t[it_true-n_start]/3600:.2f} hours to {reader.t[it_true+n_start]/3600:.2f} hours '
            if binning_only:
                lines_var_it['ur_rms'] = {'var':[ur_rms[n][it, :] for n in range(num_cases)], 'title': r'u$_{r,\text{rms}}$', 'label': '[m/s]', 'range': [0, 0.4*ranges['vel_rms'][1]]}
            else:
                lines_var_it['u_rms'] = {'var':[u_rms[n][it, :] for n in range(num_cases)], 'title': r'u$_{\text{rms}}$', 'label': '[m/s]', 'range': ranges['vel_rms']}
                lines_var_it['v_rms'] = {'var':[v_rms[n][it, :] for n in range(num_cases)], 'title': r'v$_{\text{rms}}$', 'label': '[m/s]', 'range': ranges['vel_rms']}
            lines_var_it['w_rms'] = {'var':[w_rms[n][it, :] for n in range(num_cases)], 'title': r'w$_{\text{rms}}$', 'label': '[m/s]', 'range': [0, 2*ranges['vel_rms'][1]]}
            lines_var_it['b_rms'] = {'var':[b_rms[n][it, :] for n in range(num_cases)], 'title': r'b$_{\text{rms}}$', 'label': r"[m/s$^2$]", 'range': ranges['b_rms']}
            if binning_only:
                lines_var_it['ur_avg'] = {'var':[ur_avg[n][it, :] for n in range(num_cases)], 'title': r'$\langle\text{u}_r\rangle_{\text{r}}$', 'label': '[m/s]', 'range': ranges['u']}
                lines_var_it['S'] = {'var':[S_avg[n][it, :] for n in range(num_cases)], 'title': r'$\langle\text{S}\rangle_{\text{r}}$', 'label': '[g/kg]', 'range': ranges['S_avg'] }
                lines_var_it['T_fluc_center'] = {'var':[T_fluc_center[n][it, :] for n in range(num_cases)], 'title': r"T'(0, z)", 'label': r"[$^\circ$C]", 'range': ranges['T_fluc_center']}
            vert_dir_frames = plot_lines(title, it, case_opt, fig_folder, case_names, z, lines_var_it, save_folder = f'profiles temporal average nsteps= {nsteps}')
        del lines_var_it
    else:
        time_min_opt = np.arange(0, len(min_time_avg), 100) # can change based on data plotted
        for it, it_opt in enumerate(time_min_opt):
            lines_var_it = {}
            if binning_only:
                lines_var_it['ur_rms'] = {'var':[ur_rms[n][it, :] for n in range(num_cases)], 'title': r'u$_{r,\text{rms}}$', 'label': '[m/s]', 'range': [0, 0.4*ranges['vel_rms'][1]]}
            else:
                lines_var_it['u_rms'] = {'var':[u_rms[n][it, :] for n in range(num_cases)], 'title': r'u$_{\text{rms}}$', 'label': '[m/s]', 'range': ranges['vel_rms']}
                lines_var_it['v_rms'] = {'var':[v_rms[n][it, :] for n in range(num_cases)], 'title': r'v$_{\text{rms}}$', 'label': '[m/s]', 'range': ranges['vel_rms']}
            lines_var_it['w_rms'] = {'var':[w_rms[n][it, :] for n in range(num_cases)], 'title': r'w$_{\text{rms}}$', 'label': '[m/s]', 'range': [0, 2*ranges['vel_rms'][1]]}
            lines_var_it['b_rms'] = {'var':[b_rms[n][it, :] for n in range(num_cases)], 'title': r'b$_{\text{rms}}$', 'label': r"[m/s$^2$]", 'range': ranges['b_rms']}
            if binning_only:
                lines_var_it['ur_avg'] = {'var':[ur_avg[n][it, :] for n in range(num_cases)], 'title': r'$\langle\text{u}_r\rangle_{\text{r}}$', 'label': '[m/s]', 'range': ranges['u']}
                lines_var_it['S'] = {'var':[S_avg[n][it, :] for n in range(num_cases)], 'title': r'$\langle\text{S}\rangle_{\text{r}}$', 'label': '[g/kg]', 'range': ranges['S_avg'] }
                lines_var_it['T_fluc_center'] = {'var':[T_fluc_center[n][it, :] for n in range(num_cases)], 'title': r"T'(0, z)", 'label': r"[$^\circ$C]", 'range': ranges['T_fluc_center']}
                title = f'{reader.t[it]/3600:.2f} hours'
            else:
                lines_var_it['S'] = {'var':[S_avg[n][it_opt, :] for n in range(num_cases)], 'title': r'$\langle\text{S}\rangle_{\text{xy}}$', 'label': '[g/kg]', 'range': ranges['S_avg'] }
                lines_var_it['T_fluc_center'] = {'var':[T_fluc_center[n][it_opt, :] for n in range(num_cases)], 'title': r"T'(0, 0, z)", 'label': r"[$^\circ$C]", 'range': ranges['T_fluc_center']}
                #lines_var_it['b_fluc_center'] = {'var':[b_fluc_center[n][it_opt, :] for n in range(num_cases)], 'title': r"b'(0, 0, z)", 'label': r"[m/s$^2$]", 'range': ranges['b_fluc_center']}
                title = f'{min_time_avg[it_opt]/3600:.2f} hours'
            vert_dir_frames = plot_lines(title, it, case_opt, fig_folder, case_names, z, lines_var_it)
        del lines_var_it
if plot_horiz_profiles:
    hor_dir_frames = {}
    for k, loc in enumerate(loc_z):
        for it in range(nt_min):
            lines_var_it = {}
            lines_var_it['w'] = {'var':[w_hor[n][it, :, k] for n in range(num_cases)], 'title': f'w(0.0, y, {loc})', 'label': '[m/s]', 'range': ranges_horiz['w']}
            lines_var_it['S'] = {'var':[S_hor[n][it, :, k] for n in range(num_cases)], 'title': f'S(0.0, y, {loc})', 'label': '[g/kg]', 'range': ranges_horiz['S']}
            hor_dir_frames[loc] = plot_lines_h(f'{reader.t[it]/3600:.2f} hours', it, case_opt, fig_folder, case_names, y, lines_var_it, f'z = {loc:.2f} m', xlabel = 'y [m]', error_norm = {'w': w0*100, 'S': Sval, 'b': -Sval*reader.beta*g})
if plot_hor_bin_profiles:
    bin_dir_frames = {}
    save_folder = 'bin outputs'
    if plot_difference:
        error_norm = {'w': w0*100, 'S': Sval, 'b': -Sval*reader.beta*g}
    else:
        error_norm = None
    for k, loc in enumerate(loc_z):
        # time stepping instantaneous values for plotting
        if temporal_avg:
            for it, it_true in enumerate(np.arange(n_start, n_end)):
                title = f'Temporal average from {reader.t[it_true-n_start]/3600:.2f} hours to {reader.t[it_true+n_start]/3600:.2f} hours '
                lines_var_it = {}
                lines_var_it['w'] = {'var':[w_bin_hor[n][it, :, k] for n in range(num_cases)], 'title': f'w(r, {loc})', 'label': '[m/s]', 'range': w_range}
                lines_var_it['S'] = {'var':[S_bin_hor[n][it, :, k] for n in range(num_cases)], 'title': f'S(r, {loc})', 'label': '[g/kg]', 'range': S_range}
                #lines_var_it['b'] = {'var':[b_bin_hor[n][it, :, k] for n in range(num_cases)], 'title': f'b(r, {loc})', 'label': r"[m/s$^2$]", 'range': b_range}
                bin_dir_frames[loc] = plot_lines_h(title, it_true, case_opt, fig_folder, case_names, r, lines_var_it, f'z = {loc:.2f} m', xlabel = 'r [m]', error_norm = error_norm, save_folder = save_folder + f' temporal average nsteps= {nsteps}') 
        else:
            for it in range(nt_min):
                title = f'{reader.t[it]/3600:.2f} hours'
                lines_var_it = {}
                lines_var_it['w'] = {'var':[w_bin_hor[n][it, :, k] for n in range(num_cases)], 'title': f'w(r, {loc})', 'label': '[m/s]', 'range': w_range}
                lines_var_it['S'] = {'var':[S_bin_hor[n][it, :, k] for n in range(num_cases)], 'title': f'S(r, {loc})', 'label': '[g/kg]', 'range': S_range}
                #lines_var_it['b'] = {'var':[b_bin_hor[n][it, :, k] for n in range(num_cases)], 'title': f'b(r, {loc})', 'label': r"[m/s$^2$]", 'range': b_range}
                bin_dir_frames[loc] = plot_lines_h(title, it, case_opt, fig_folder, case_names, r, lines_var_it, f'z = {loc:.2f} m', xlabel = 'r [m]', error_norm = error_norm, save_folder = save_folder) 
        # temporal averaging of the last two time steps for plotting
        lines_var_it = {}
        lines_var_it['w'] = {'var':[np.mean(w_bin_hor[n][it-1:it+1, :, k], axis=0) for n in range(num_cases)], 'title': f'w(r, {loc})', 'label': '[m/s]', 'range': w_range}
        lines_var_it['S'] = {'var':[np.mean(S_bin_hor[n][it-1:it+1, :, k], axis=0) for n in range(num_cases)], 'title': f'S(r, {loc})', 'label': '[g/kg]', 'range': S_range}
        #lines_var_it['b'] = {'var':[np.mean(b_bin_hor[n][it-1:it+1, :, k], axis=0) for n in range(num_cases)], 'title': f'b(r, {loc})', 'label': r"[m/s$^2$]", 'range': b_range}
        plot_lines_h('Last two time steps -Temporal Average', int(0), case_opt, os.path.join(fig_folder, 'temporal average'), case_names, r, lines_var_it, f'z = {loc:.2f} m', xlabel = 'r [m]', error_norm = error_norm, save_folder = save_folder)
if plot_scaling_analysis_text:
    #print(c_delta)
    #test = word
    plot_scaling_analysis(time, fig_folder, case_names, z_mld, r, F, F_transverse, c_delta, c_w, eta)
# plotting with flags that don't have the possibility of being videos
if plot_depths:
    plot_plume_depths(time_output, case_opt, fig_folder, case_names, lx, zp, zneutral, zc, contour, trend = False)

# ==========================================================
# VIDEOS
# ==========================================================
if video:
    if temporal_avg:
        if plot_hor_bin_profiles:
            for loc in loc_z:
                create_video(bin_dir_frames[loc], fig_folder, f'z_{loc:.2f}_bin_profiles', f' temporal average nsteps= {nsteps}')
        if plot_vert_profiles:
            create_video(vert_dir_frames, fig_folder, 'vert_profiles', f' temporal average nsteps= {nsteps}')
    else:
        if plot_plane_vert:
            for dir, name in enumerate(var_names):
                create_video(variable_dir[var_names[dir]], fig_folder, 'vertical', name)
        if plot_plane_hor:
            for name in var_names:
                for z_loc in z_locs:
                    create_video(variable_dir_hor[name][z_loc], fig_folder, f'z = {z_loc:.2f} m', name)
        if plot_plane_bin_vert:
            for n, name in enumerate(bin_var_names[:-1]):
                create_video(bin_dir[bin_var_names[n]], fig_folder, 'binning', name)
        if plot_turb_stats:
            create_video(buoyancy_dir_z, fig_folder, 'binning', 'turb_stats')
        if plot_plume_z:
            create_video(buoyancy_dir_z, fig_folder, 'binning', f'plume-contour-{contour}')
        if plot_horiz_profiles:
            for loc in loc_z:
                create_video(hor_dir_frames[loc], fig_folder, f'z_{loc:.2f}_horiz_profiles', '')
