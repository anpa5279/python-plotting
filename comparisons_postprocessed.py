import os
import numpy as np

from reader import OceananigansData
from diagnostics import comparison_info
from interpolation import point
from physics import buoyancy
from plotting_general import plot_format, plot_ranges, create_video, comparison_plot_opt
from plotting_lines import plot_turb_stats_bin, temporal_avg, plot_plume_depths, plot_plume_vertical_spatial
from plotting_planes import plot_variable_vert_slice, plot_variable_xy_slice

# ==========================================================
# FLAGS
# ==========================================================
plot_variables_vert = False
plot_variables_hor = False
plot_difference_vert = True
plot_var_bin = False
plot_turb_stats = False
plot_temporal_avg = False
plot_depths = False
plot_plume_z = False
video = True

# flags for how to read data
with_halos = False
closure = False
stokes = False

# ==========================================================
# COMPARISON CASES
# ==========================================================
contour = 0.01
name_uni = f'contour-{contour:.4f}'
universal_folder = '/Users/annapauls/Documents/TESLa /Simulations/Oceananigans/dense plume/salinity and temperature/version109/res testing/square inlet/open BC/no SGS/default WENO/'

variations = 'else'
if variations != 'else':
    cases_info = comparison_info(variations, universal_folder = universal_folder)
    case_names = cases_info['case_names']
    num_cases = cases_info['num_cases']
    folder_names = cases_info['folder_names']
    fig_folder = cases_info['fig_folder']
else:
    fig_folder = os.path.join(universal_folder, 'bottom PA', 'horizontal resolution', 'comparison figures')

    folder_names = ['dx2', 'horizontal resolution/dx1', 'horizontal resolution/dx05', 'bottom PA/dx2', 'bottom PA/horizontal resolution/dx1', 'bottom PA/horizontal resolution/dx05']#
    #['dx2', 'horizontal resolution/dx1', 'horizontal resolution/dx05']#['dx2', 'dx1']#, 'dx05']#, 'dx025']#
    #['bottom PA/vertical resolution/dz2', 'bottom PA/vertical resolution/dz1', 'bottom PA BC/bottom PA/vertical resolution/dz2', 'bottom PA BC/bottom PA/vertical resolution/dz1']
    
    case_names = [r'No flux BC $\Delta x = 2.0$', r'No flux BC $ \Delta x = 1.0$', r'No flux BC $ \Delta x = 0.5$', r'PA BC $\Delta x = 2.0$', r'PA BC$\Delta x = 1.0$', r'PA BC$\Delta x = 0.5$']#[r'No flux BC $\Delta z = 2.0$', r'No flux BC $ \Delta z = 1.0$', r'PA BC $\Delta z = 2.0$', r'PA BC$\Delta z = 1.0$']
    #[r'$\Delta x = \Delta y = \Delta z = 2.0$', r'$\Delta x = \Delta y = 1.0$ $ \Delta z = 2.0$', r'$\Delta x = \Delta y = 0.5$ $ \Delta z = 2.0$']#[r'$\Delta x = \Delta y = \Delta z = 2.0$', r'$\Delta x = \Delta y = 2.0$ $ \Delta z = 1.0$', r'$\Delta x = \Delta y = 2.0$ $ \Delta z = 0.5$']#[r'$\Delta x = 2.0$', r'$\Delta x = 1.0$', r'$\Delta x = 0.5$']#, r'$\Delta x = 0.25$']#


    num_cases = len(folder_names)
    dTdz = 0.01*np.ones(num_cases)
    mld = 60**np.ones(num_cases)
    F_s = 0.1*np.ones(num_cases)

if plot_difference_vert:
    fig_folder_diff = os.path.join(fig_folder, 'percent differences')

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
if plot_var_bin:
    r = []
x = []
y = []
z = []
time  = []
nx = np.empty((3, num_cases), dtype=object)
lx = np.empty((3, num_cases), dtype=object)
grid_specs = False*np.ones(num_cases)
for n, reader in enumerate(readers):
    
    if plot_variables_hor:
        x.append(reader.x)
    y.append(reader.y)
    z.append(reader.z)
    time.append(reader.t)
    nx[:, n] = reader.nx
    lx[:, n] = reader.lx
    if n == 0:
        nt_min = reader.nt
    else:
        nt_min = np.min([nt_min, reader.nt])
    if plot_var_bin:
        r.append(reader.r)
    if salinity[n] and plot_turb_stats:
        S_value = reader.load_S_temporal_avg()


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
if any([plot_variables_vert, plot_variables_hor, plot_var_bin, plot_turb_stats, plot_depths, plot_plume_z, plot_difference_vert]):
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

    if plot_variables_vert or plot_difference_vert:
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
        time_output = []
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

    for n, reader in enumerate(readers):
        # Load data from files [nt, nx, ny, nz]
        if plot_variables_vert or plot_difference_vert:
            T_vert_plane.append(reader.load_plane_var('T'))
            u_vert_plane.append(reader.load_plane_var('u'))
            v_vert_plane.append(reader.load_plane_var('v'))
            w_vert_plane.append(reader.load_plane_var('w'))
            if all(salinity):
                S_vert_plane.append(reader.load_plane_var('S'))
                S_vert_plane[n][S_vert_plane[n]<S_tol] = S_tol # set values below threshold to threshold for log plotting
        if plot_variables_hor:
            z_loc = -3*reader.dx[-1]/2#0.0#-reader.lx[-1] + reader.dx[-1]/2#-reader.dx[-1]/2# 
            #T_hor_plane.append(reader.load_plane_var('T', loc=z_loc, plane = 'XY'))
            #u_hor_plane.append(reader.load_plane_var('u', loc=z_loc, plane = 'XY'))
            #v_hor_plane.append(reader.load_plane_var('v', loc=z_loc, plane = 'XY'))
            #w_hor_plane.append(reader.load_plane_var('w', loc=z_loc, plane = 'XY'))
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
            S_reader_center = reader.field_centerline('S')
            r_bin.append(reader.loading_bin_contours(contour = contour))
            if salinity and plot_depths:
                time_output.append(reader.time_avg)
                S_value = reader.load_S_temporal_avg()
                
                bS.append(-g*reader.beta*S_value)
                # calculate buoyancy differences
                S_value = reader.load_S_temporal_avg()
                
                bT_avg = g*reader.alpha*(reader.load_averages('T') - reader.T0)
                # calculate where w = 0 on the centerline
                w_reader_center = reader.field_centerline('w')

                nt_output = len(reader.time_avg)
                neutral_it = np.zeros(nt_output)
                w_centerline_it = np.zeros(nt_output)
                S_it = np.zeros(nt_output)
                for it in range(0, nt_output):
                    # based on buoyancy differences
                    b_diff = bT_avg[it,:] - bS[n]
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
                    z_it = point(S_reader_center[it, :], z[n], f0 = S_value*contour)
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
                T_fluc_center.append(T_center-T_avg[n])
                S_center.append(S_reader_center[::100, :])
                del T_center


if plot_temporal_avg:
    t_range = np.array([0.4, 0.5])*24*3600 # seconds
    w_rms_t_avg = []
    b_rms_t_avg = []
    w_centerline_t_avg = []
    b_fluc_centerline_t_avg = []
    S_centerline_t_avg = []
    for n, reader in enumerate(readers):
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
        print("Finished temporal averaging for case ", case_names[n])
    t_range = t_range/(24*3600) # days

# ==========================================================
# PLOTTING
# ==========================================================

# plotting prep
if plot_variables_vert or plot_variables_hor or plot_var_bin or plot_difference_vert:
    if all(salinity):
        range_names =['Tracer', 'T', 'u', 'v', 'w'] #['log w', 'w scaled', 'w']# 
        var_names =['Tracer', 'Temperature', 'u', 'v', 'w']#['Order of Magnitude of w', r"(w-w$_0$)/w$_0$", 'w']# ['Tracer', 'Temperature', 'u', 'v', 'w']
    else:
        range_names = ['T', 'u', 'v', 'w']
        var_names = ['Temperature', 'u', 'v', 'w']
    variable_dir = {}
    variable_dir_hor = {}
    variable_dir_diff = {}
if plot_var_bin:
    if all(salinity):
        bin_var_names = ['Tracer', 'Temperature', r'u$_r$', r'u$_{\theta}$', 'w']
        bin_range_names = ['Tracer', 'T', 'u', 'v', 'w']
    else:
        bin_var_names = ['Temperature', 'u', 'v', 'w']
        bin_range_names = ['T', 'u', 'v', 'w']
    bin_dir = {}
if plot_turb_stats or plot_temporal_avg or plot_plume_z or plot_depths:
    color_opt, line_opt = comparison_plot_opt(num_cases)

ranges = plot_ranges(lz = 96, mld = np.max(mld), T0 = T0, dTdz = np.max(dTdz), C_tol = S_tol)
ranges['Tracer'] =[S_tol, 0.15]
ranges['Tracer negative'] = [-0.15, 0.15]
ranges['Tracer_fluc'] = [-0.2, 0.2]
ranges['Tracer_avg'] = [0, 8*10**(-4)]
ranges['T'] = [T0-0.7, T0 + 0.05]
ranges['w'] = [-1*10**(-1), 1*10**(-1)]
ranges['u'] = [-1*10**(-2), 1*10**(-2)]
ranges['v'] = [-2*10**(-2), 2*10**(-2)]
ranges['vel_rms'] = [0, 1*10**(-2)]
ranges['b_rms'] = [0, 1.5*10**(-5)]
ranges['S'] = [S_tol, 0.05]
ranges['b_fluc'] = [-7*10**(-4), 7*10**(-4)]
ranges['bw_fluc'] = [-1*10**(-7), 1*10**(-7)]
if plot_turb_stats:
    ranges['restress'] = [-2*10**(-5), 2*10**(-5)]
    ranges['Tw_fluc'] = [-5*10**(-4), 5*10**(-4)]
    ranges['Cw'] = [-5*10**(-5), 5*10**(-5)]

# plotting with flags that have the possibility of being videos
if any([plot_variables_vert, plot_variables_hor, plot_var_bin, plot_turb_stats, plot_difference_vert]):
    time_min = min(time, key=len)
    for it in range(nt_min):
        if plot_variables_vert or plot_variables_hor:
            if all(salinity): #'Tracer', 'T', 'u', 'v', 'w'
                colorbar_labels = [r"g/kg", r"$^\circ$C", r"m/s", r"m/s", r"m/s"]
                cmaps = ['Blues', 'viridis', 'RdBu_r', 'RdBu_r', 'RdBu_r', 'RdBu_r', ]#
                #colorbar_labels = [r"Order of Magnitude", r"(w-w$_0$)/w$_0$", 'm/s']#[r"$^\circ$C", ]#
                #cmaps = ['Blues_r', 'Blues_r', 'RdBu_r']#['viridis',]#['Blues']#
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
                hor_ranges['T'] = [T0-0.05, T0+0.05]
                hor_ranges['u'] = [-2*10**(-2), 2*10**(-2)]
                hor_ranges['v'] = hor_ranges['u']
                hor_ranges['w'] = [-2*10**(-3), 2*10**(-3)]
                hor_ranges['Sw'] = [-2*10**(-4), 2*10**(-4)]
                hor_ranges['log w'] = [-6, -3]
                hor_ranges['w scaled'] = [-1.05, 0]

                if all(salinity):
                    #w = [w_hor_plane[n][it, :, :].T for n in range(num_cases)]
                    #variables = [w, ]#[[np.floor(np.log10(np.abs(w[n]))) for n in range(num_cases)], [(w[n]-w0)/w0 for n in range(num_cases)], w]
                    variables = [[S_hor_plane[n][it, :, :].T for n in range(num_cases)], [T_hor_plane[n][it, :, :].T for n in range(num_cases)], [u_hor_plane[n][it, :, :].T for n in range(num_cases)], [v_hor_plane[n][it, :, :].T for n in range(num_cases)], [w_hor_plane[n][it, :, :].T for n in range(num_cases)]]
                else:
                    variables = [[T_hor_plane[n][it, :, :].T for n in range(num_cases)], [u_hor_plane[n][it, :, :].T for n in range(num_cases)], [v_hor_plane[n][it, :, :].T for n in range(num_cases)], [w_hor_plane[n][it, :, :].T for n in range(num_cases)]]

                for dir, var in enumerate(variables):
                    variable_dir_hor[var_names[dir]] = plot_variable_xy_slice(time_min[it], it, hor_ranges, fig_folder, lx, x, y, var, case_names, var_names[dir], range_names[dir], colorbar_label = colorbar_labels[dir], cmap = cmaps[dir], loc = z_loc)
        if plot_var_bin:
            if all(salinity): #'Tracer', 'T', 'u', 'v', 'w'
                variables = [[S_bin[n][:, :, it].T for i in range(num_cases)], [T_bin[n][:, :, it].T for i in range(num_cases)], [ur_bin[n][:, :, it].T for i in range(num_cases)], [utheta_bin[n][:, :, it].T for i in range(num_cases)], [w_bin[n][:, :, it].T for i in range(num_cases)]]
                colorbar_labels = [r"g/kg", r"$^\circ$C", r"m/s", r"m/s", r"m/s"]
                cmaps = ['Blues', 'viridis', 'RdBu_r', 'RdBu_r', 'RdBu_r']
            else: #'T', 'u', 'v', 'w'
                variables = [[T_bin[n][:, :, it].T for i in range(num_cases)], [ur_bin[n][:, :, it].T for i in range(num_cases)], [utheta_bin[n][:, :, it].T for i in range(num_cases)], [w_bin[n][:, :, it].T for i in range(num_cases)]]
                colorbar_labels = [r"$^\circ$C", r"m/s", r"m/s", r"m/s"]
                cmaps = ['viridis', 'RdBu_r', 'RdBu_r', 'RdBu_r']

            for dir, var in enumerate(variables):
                l_bin = [lx[0]/2, lx[-1]]
                bin_dir[bin_var_names[dir]] = plot_variable_vert_slice(time_min[it], it, ranges, fig_folder, lx, r, z, var, case_names, bin_var_names[dir], bin_range_names[dir], colorbar_label = colorbar_labels[dir], cmap = cmaps[dir], plane='binning')

        if plot_difference_vert:

            range_names = ['Tracer diff', 'T diff', 'w diff', 'u diff', 'v diff']
            ranges['Tracer diff'] = [-5, 5]
            ranges['T diff'] = [-0.5, 0.5]
            ranges['w diff'] = [-20, 20]
            ranges['u diff'] = ranges['T diff']
            ranges['v diff'] = [-10, 10]
            var_names = ['Tracer', 'T', 'w', 'u', 'v']
            colorbar_labels = [r"(S$_{PA}$ - S$_{Default}$)/S$_{0}$ [%]", r"(T$_{PA}$ - T$_{Default}$)/T$_{0}$ [%]", r"(w$_{PA}$ - w$_{Default}$)/(100$\cdot$w$_{0})$ [%]", r"(u$_{PA}$ - u$_{Default}$)/(100$\cdot$w$_{0})$ [%]", r"(v$_{PA}$ - v$_{Default}$)/(100$\cdot$w$_{0})$ [%]"]
            cmaps = ['RdBu_r', 'RdBu_r', 'RdBu_r', 'RdBu_r', 'RdBu_r']
            num_compare = num_cases//2
            variables = [
                         [100*(S_vert_plane[n][it, :, :].T - S_vert_plane[n-num_compare][it, :, :].T)/Sval for n in np.arange(num_compare, num_cases)], 
                         [100*(T_vert_plane[n][it, :, :].T - T_vert_plane[n-num_compare][it, :, :].T)/T0 for n in np.arange(num_compare, num_cases)], 
                         [100*(w_vert_plane[n][it, :, :].T - w_vert_plane[n-num_compare][it, :, :].T)/(100*w0) for n in np.arange(num_compare, num_cases)], 
                         [100*(u_vert_plane[n][it, :, :].T - u_vert_plane[n-num_compare][it, :, :].T)/(100*w0) for n in np.arange(num_compare, num_cases)], 
                         [100*(v_vert_plane[n][it, :, :].T - v_vert_plane[n-num_compare][it, :, :].T)/(100*w0) for n in np.arange(num_compare, num_cases)]
                         ]
            for dir, var in enumerate(variables):
                variable_dir_diff[var_names[dir]] = plot_variable_vert_slice(time_min[it], it, ranges, fig_folder_diff, lx, y, z, var, case_names[2:], var_names[dir], range_names[dir], colorbar_label = colorbar_labels[dir], cmap = cmaps[dir], plane='YZ')
                    
        if plot_turb_stats:
            ur_rms_it = [ur_rms[n][:, it] for i in range(num_cases)]
            w_rms_it = [w_rms[n][:, it] for i in range(num_cases)]
            uw_avg_it = [urw_avg[n][:, it] for i in range(num_cases)]
            b_avg_it = [b_avg[n][:, it] for i in range(num_cases)]
            bur_fluc_avg_it = [bur_fluc_avg[n][:, it] for i in range(num_cases)]
            bw_fluc_avg_it = [bw_fluc_avg[n][:, it] for i in range(num_cases)]
            Tur_avg_it = [Tur_avg[n][:, it] for i in range(num_cases)]
            Tw_avg_it = [Tw_avg[n][:, it] for i in range(num_cases)]
            Sur_avg_it = [Sur_avg[n][:, it] for i in range(num_cases)]
            Sw_avg_it = [Sw_avg[n][:, it] for i in range(num_cases)]
            buoyancy_dir_z = plot_turb_stats_bin(time_min[it], it, ranges, color_opt, fig_folder, case_names, z, ur_rms_it, w_rms_it, uw_avg_it, b_avg_it, bur_fluc_avg_it, bw_fluc_avg_it, Tur_avg_it, Tw_avg_it, Sur_avg_it, Sw_avg_it)
if plot_plume_z:
    buoyancy_dir_z = plot_plume_vertical_spatial(min(time, key=len), ranges, color_opt, fig_folder, case_names, name_uni, lx, z, S_avg, u_rms, v_rms, w_rms, b_avg, b_center, r_bin, bur_fluc_avg, bw_fluc_avg, T_avg, T_fluc_center, S_center)

# plotting with flags that don't have the possibility of being videos
if plot_depths:
    plot_format()
    plot_plume_depths(time_output, color_opt, fig_folder, case_names, lx, zp, zneutral, zc, contour, trend = False)
if plot_temporal_avg:
    plot_format()
    temporal_avg(t_range, ranges, color_opt, fig_folder, case_names, lx, z, w_centerline_t_avg, S_centerline_t_avg, b_fluc_centerline_t_avg, w_rms_t_avg, b_rms_t_avg, h_ml = mld)
    print("Finished plotting temporal average for all cases.")

# ==========================================================
# VIDEOS
# ==========================================================
if video:
    if plot_var_bin:
        for n, name in enumerate(bin_var_names):
            create_video(bin_dir[bin_var_names[n]], fig_folder, 'binning', name)
    if plot_variables_vert:
        for dir, name in enumerate(var_names):
            create_video(variable_dir[var_names[dir]], fig_folder, 'vertical', name)
    if plot_difference_vert:
        for dir, name in enumerate(var_names):
            create_video(variable_dir_diff[var_names[dir]], fig_folder, 'diff-vertical', name)
    if plot_variables_hor:
        for dir, name in enumerate(range_names):
            create_video(variable_dir_hor[var_names[dir]], fig_folder, f'z = {z_loc:.2f} m', name)
    if plot_turb_stats:
        create_video(buoyancy_dir_z, fig_folder, 'binning', 'turb_stats')
    if plot_plume_z:
        create_video(buoyancy_dir_z, fig_folder, 'binning', f'plume-contour-{contour}')
