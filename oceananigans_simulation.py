import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from matplotlib.lines import Line2D

from reader import OceananigansData
from plotting_general import plot_format, create_video, comparison_plot_opt, plot_ranges
from plotting_planes import plot_var_planeslice
from interpolation import point, vertical_line, interp1d_axis, velocities_to_center
"""
    what is the best way to find the maximum penetration depth of a plume via momentum?
    ways to consider w:
        1. w_avg = 1D array of shape (nt, nz) --> average w at each depth and time step
        2. w centerline = 1D array of shape (nt, nz) --> w at (0.0, 0.0) and time step
        3. w_rms = 1D array of shape (nt, nz) --> average w' at each depth and time step
        4. w_fluc_centerline = 1D array of shape (nt, nz) --> w' at (0.0, 0.0) and time step

    ways to consider buoyancy, b:
        1. b_rms = 1D array of shape (nt, nz) --> average b' at each depth and time step
        2. b_fluc_centerline = 1D array of shape (nt, nz) --> b' at (0.0, 0.0) and time step

    ways to percieve w and/or b:
        1. sign changes
        2. order of magnitude changes
        3. gradient changes
"""
# ==========================================================
# FLAGS
# ==========================================================
plot_xt = False
plot_yt = False
plot_zt = False
plot_rt = False
plot_xy_plane = False
plot_bin_plane = True
plot_raw_centerline = False
plot_1dz_stats = False
verify_outputs = False

temporal_avg = True
video = True

if temporal_avg:
    nsteps = 5 # number of time steps to average over for plotting, also trying 5


# ==========================================================
# MODEL INFORMATION
# ==========================================================
folder = '/Users/annapauls/Documents/Github repositories/3d_langmuir_gpu/localoutputs/scheme-tests/longer/WENO9/dx0.5'

outdir = os.path.join(folder, 'figures')
reader = OceananigansData(folder, salinity = True, with_halos=True)
if plot_1dz_stats:
    fig_var_folder = os.path.join(outdir, 'max penetration variables')
    os.makedirs(fig_var_folder, exist_ok=True)

# collecting model information for all cases
x = reader.x
y = reader.y
z = reader.z
nx = reader.nx
lx = reader.lx
nt = reader.nt
dx = reader.dx
dx_scale = max(dx[:-1]) # not including dz
r = np.arange(dx[0]/2, lx[0]/2, dx_scale)
# video or not setup
if video or plot_zt or plot_raw_centerline or plot_yt or plot_xt:
    time = reader.t
    if reader.centerline:
        time1 = reader.time_center
else:
    time = reader.t[-1]

# ==========================================================
# PARAMETERS
# ==========================================================
S_tol = 10**(-6)
mld = 60
g = 9.80665
T0 = reader.T0

# ==========================================================
# ANALYSIS
# ==========================================================
if plot_yt or plot_xt:
    z_loc = -1.0*np.array([mld, mld+1, mld+2, mld+5, mld+10])#, mld+20]) #-[mld, ]
# load in information
if plot_zt or verify_outputs:
    w_centerline = reader.field_centerline('w')
    b_avg, b_rms, b_centerline, b_fluc_centerline = reader.load_buoyancy()
    if reader.salinity:
        S_centerline = reader.field_centerline('S')
    if verify_outputs:
        w_centerline = w_centerline[::100, :]
        b_centerline = b_centerline[::100, :]
        reader.centerline = False
        w_slice = reader.load_plane_var('w')
        w_slice = velocities_to_center(w_slice, axis = -1)
        w_coarse_centerline = vertical_line(w_slice, y = reader.y, y0 = 0.0)
        T_slice = reader.load_plane_var('T')
        T_coarse_centerline = vertical_line(T_slice, y = reader.y, y0 = 0.0)
        b_coarse_centerline = g * reader.alpha * (T_coarse_centerline - reader.T0)
        del T_coarse_centerline
        if reader.salinity:
            S_centerline = S_centerline[::100, :]
            S_slice = reader.load_plane_var('S')
            S_coarse_centerline = vertical_line(S_slice, y = reader.y, y0 = 0.0)
            b_coarse_centerline += - g * reader.beta * S_coarse_centerline
if plot_1dz_stats or (plot_zt or verify_outputs):
    w_rms = reader.load_rms('w')
    u_r = reader.load_binning_var('horizontal velocity')
    u_r_avg = np.mean(u_r, axis = -3)
    u_r_rms = np.mean((u_r - u_r_avg[None, :, :])**2, axis = -3)**0.5
if plot_yt:
    buoyancy_file = os.path.join(reader.folder, 'buoyancy_profile.h5')
    with h5py.File(buoyancy_file, 'r') as f:
        b_avg = f['b_avg'][()]
        b_avg = b_avg[::100, :]
    w_plane = reader.load_plane_var('w')
    S_plane = reader.load_plane_var('S')
    T_plane = reader.load_plane_var('T')
    w_yt = np.empty((nt, len(y), len(z_loc)))
    S_yt = np.empty((nt, len(y), len(z_loc)))
    T_yt = np.empty((nt, len(y), len(z_loc)))
    b_avg_yt = np.empty((nt, len(z_loc)))
    for j, z_opt in enumerate(z_loc):
        w_yt[:, :, j] = interp1d_axis(w_plane, z, coord_new = z_opt, axis = -1)
        T_yt[:, :, j] = interp1d_axis(T_plane, z, coord_new = z_opt, axis = -1)
        b_avg_yt[:, j] = point(b_avg, z, z0 = z_opt)
        if reader.salinity:
            S_yt[:, :, j] = interp1d_axis(S_plane, z, coord_new = z_opt, axis = -1)
    b_yt = g * reader.alpha * (T_yt - reader.T0)
    if reader.salinity:
        b_yt += - g * reader.beta * S_yt
    b_fluc_yt = b_yt - b_avg_yt[:, None, :]
    del w_plane, S_plane, T_plane, b_yt
if plot_xt:
    buoyancy_file = os.path.join(reader.folder, 'buoyancy_profile.h5')
    with h5py.File(buoyancy_file, 'r') as f:
        b_avg = f['b_avg'][()]
    w_plane = reader.field_slice('w', plane='XZ')
    S_plane = reader.field_slice('S', plane='XZ')
    T_plane = reader.field_slice('T', plane='XZ')
    w_xt = np.empty((nt, len(x), len(z_loc)))
    S_xt = np.empty((nt, len(x), len(z_loc)))
    T_xt = np.empty((nt, len(x), len(z_loc)))
    b_avg_xt = np.empty((nt, len(z_loc)))
    for j, z_opt in enumerate(z_loc):
        w_xt[:, :, j] = interp1d_axis(w_plane, z, coord_new = z_opt, axis = -1)
        T_xt[:, :, j] = interp1d_axis(T_plane, z, coord_new = z_opt, axis = -1)
        b_avg_xt[:, j] = point(b_avg, z, z0 = z_opt)
        if reader.salinity:
            S_xt[:, :, j] = interp1d_axis(S_plane, z, coord_new = z_opt, axis = -1)
    b_xt = g * reader.alpha * (T_xt - reader.T0)
    if reader.salinity:
        b_xt += - g * reader.beta * S_xt
    b_fluc_xt = b_xt - b_avg_xt[:, None, :]
    del w_plane, S_plane, T_plane, b_xt
if plot_xy_plane:

    # choose depth of horizontal slice
    z_plot = 0.0

    w_hor = reader.field_slice('w', plane='XY')

    if reader.salinity:
        S_hor = reader.field_slice('S', plane='XY')
if plot_bin_plane:
    range_names = ['T', 'ur', 'w', 'S', 'log w', 'log Tracer'] 
    var_names = ['T', 'ur', 'w', 'S', 'w', 'S'] 
    var_units = [r'[$^\circ$C]', '[m/s]', '[m/s]', '[g/kg]', '[m/s]', '[g/kg]']
    cmap_opt = ['Blues_r', 'RdBu_r', 'RdBu_r', 'Blues', 'RdBu_r', 'Blues']

    variables = {}
    for n in range(len(range_names)):
        variables[var_names[n]] = {'var':reader.load_binning_var(var_names[n]), 'title': var_names[n], 'label':var_units[n], 'cmap': cmap_opt[n], 'range_name':range_names[n]}

# finding centerlines
if plot_raw_centerline:
    steps = reader.t_save_center
    w_output = np.empty(((nt-1)*100 + 1, 2, 2, nx[2]+1))
    T_output = np.empty(((nt-1)*100 + 1, 2, 2, nx[2]))
    S_output = np.empty(((nt-1)*100 + 1, 2, 2, nx[2]))
    with h5py.File(os.path.join(reader.folder, reader.centerline_output), 'r') as f:
        for it, t in enumerate(steps):
            w_data = f[f'timeseries/w/{int(t)}']
            T_data = f[f'timeseries/T/{int(t)}']
            S_data = f[f'timeseries/S/{int(t)}']
            w_data = w_data[reader.hx[2]:-reader.hx[2], :, :] # (z, y, x_local)
            T_data = T_data[reader.hx[2]:-reader.hx[2], :, :] 
            S_data = S_data[reader.hx[2]:-reader.hx[2], :, :] 
            w_output[it, :, :, :] = w_data.transpose(2, 1, 0) # (x_local, y, z)
            T_output[it, :, :, :] = T_data.transpose(2, 1, 0) 
            S_output[it, :, :, :] = S_data.transpose(2, 1, 0) 
if plot_1dz_stats:
    # calculating gradients
    dwdz_centerline = np.gradient(w_centerline, z, axis = -1)
    dwrmsdz = np.gradient(w_rms, z, axis = -1)

    dbrmsdz = np.gradient(b_rms, z, axis = -1)
    dbflucdz = np.gradient(b_fluc_centerline, z, axis = -1)
if temporal_avg:
    n_start = (nsteps-1)//2
    n_end = nt - (nsteps-1)//2
    for it, it_true in enumerate(np.arange(n_start, n_end)):
        if plot_bin_plane:
            for n in range(len(range_names)):
                variables[var_names[n]]['var'][it, :, :] = np.mean(variables[var_names[n]]['var'][it_true-n_start:it_true+n_start+1, :, :], axis = 0)
print("finished loading data")
# ==========================================================
# PLOTTING
# ==========================================================

mld_var = np.arange(-10**2, 10**2)
mld_array = -mld * np.ones(len(mld_var))
ranges = plot_ranges()
ranges['w'] = [-1*10**-1, 1*10**-1]
ranges['b_fluc'] = [-7*10**(-4), 7*10**(-4)]
ranges['gradw'] = [-0.05, 0.05]
ranges['gradb'] = [-0.0008, 0.0008]
ranges['b_rms'] = [0, 1.5*10**(-5)]
ranges['vel_rms'] = [0, 4*10**-3]
ranges['S'] = [0.0, 0.1]
ranges['log Tracer'] =[S_tol, 0.1]
ranges['log w'] = [-0.1, 0.1]
ranges['T'] =[T0 - 0.7, T0 + 0.05]
ranges['u'] = [-1*10**(-2), 1*10**(-2)]
factor = 10**(-2)
nvars = 7
plot_format(fontsize = 20)
os.makedirs(outdir, exist_ok=True)
if plot_raw_centerline:
    ranges['T'] = [reader.T0 - 0.7, reader.T0 + 0.05]
    gridspec_kw={'height_ratios': [0.8, 0.8, 0.8, 1.15]}
    if reader.salinity:
        range_opt = [ranges['w'], ranges['S'], ranges['T']]
        titles = [r"w", r"S", r"T"]
        colors = ['RdBu', 'Blues', 'viridis']
        labels = ['[m/s]', '[g/kg]', r'[$^\circ$C]']
        vars = [w_output, S_output, T_output]
        fig, axes = plt.subplots(4, 3, figsize=(12, 17), gridspec_kw=gridspec_kw, sharex = True, sharey = True)
        file = 'wST_outputs_zt.svg'
    else:
        range_opt = [ranges['w'], ranges['T']]
        titles = [r"w", r"T"]
        colors = ['RdBu', 'viridis']
        labels = ['[m/s]', '[g/kg]', r'[$^\circ$C]']
        vars = [w_output, T_output]
        fig, axes = plt.subplots(4, 2, figsize=(7, 17), gridspec_kw=gridspec_kw, sharex = True, sharey = True)
        file = 'wT_outputs_zt.svg'
    case_opt = comparison_plot_opt(labels)
    last_t = time.max()/(3600*24)
    ratio = (last_t)/lx[2]
    axes = axes.ravel()
    plt.subplots_adjust(bottom = 0.1, top = 0.95)
    count = 0
    for ix, i in enumerate([nx[0]//2, nx[0]//2+1]):
        for jy, j in enumerate([nx[1]//2, nx[1]//2+1]):
            for n, var in enumerate(vars):
                im = axes[count].imshow(var[:, ix, jy, :].T, extent=[time.min()/(3600*24), last_t, z.min(), z.max()], interpolation ='none', cmap=colors[n], vmin=range_opt[n][0], vmax=range_opt[n][1])
                axes[count].plot(time, -mld*np.ones_like(time), color = 'k', label=r"$\text{h}_{ML}$", linewidth = 0.9, linestyle = case_opt['linestyle'][1])
                axes[count].legend(loc='lower left')
                axes[count].set_xlim(time.min()/(3600*24), last_t)
                axes[count].set_ylim(z.min(), z.max())
                axes[count].set_aspect(ratio)
                if count <= 3: # only include title in first row
                    axes[count].set_title(titles[n])
                if count == 0 or count % 3 == 0: # only include y label in first column
                    axes[count].set_ylabel(f"[{i}, {j}, 1:N$_z$]\nz [m]")
                if count >= 9: # only add colorbar to last row 
                    axes[count].set_xlabel("time [days]")
                    cbar = fig.colorbar(im, ax = axes[count], anchor = (0.5, 0.9), orientation='horizontal', label=labels[n], shrink=0.75, aspect=30)
                    cbar.formatter.set_useOffset(False)
                    cbar.formatter.set_powerlimits((-3, 5))
                    cbar.update_ticks() 
                count += 1
    frame_path = os.path.join(outdir, file)
    plt.savefig(frame_path)
    plt.close(fig)
if plot_zt:
    if reader.salinity:
        range_opt = [ranges['w'], ranges['S'], ranges['b_fluc'], ranges['vel_rms'], ranges['b_rms'], ranges['vel_rms']]
        titles = [r"w(0, 0)", r"S(0, 0)", r"b'(0, 0)", r"w$_{rms}$", r"b$_{rms}$", r"u$_{\text{r},rms}$"]
        colors = ['RdBu', 'Blues', 'RdBu', 'Blues', 'Blues', 'Blues']
        labels = ['[m/s]', '[g/kg]', r'[m/s$^2$]', '[m/s]', r'[m/s$^2$]', '[m/s]']
        stop = 5
        vars = [w_centerline[:stop*100, :], S_centerline[:stop*100, :], b_fluc_centerline[:stop*100, :], w_rms[:stop, :], b_rms[:stop, :], u_r_rms.T[:stop, :]]
        fig, axes = plt.subplots(1, 6, figsize=(12, 4))
        file = 'wSbur_rms_zt-smallert.svg'
    else:
        range_opt = [ranges['w'], ranges['b_fluc'], ranges['vel_rms'], ranges['b_rms']]
        titles = [r"w(0, 0)", r"b'(0, 0)", r"w$_{rms}$", r"b$_{rms}$"]
        colors = ['RdBu', 'RdBu', 'Blues', 'Blues']
        labels = ['[m/s]', r'[m/s$^2$]', '[m/s]', r'[m/s$^2$]']
        stop = 5
        vars = [w_centerline[:stop*100, :], b_fluc_centerline[:stop*100, :], w_rms[:stop, :], b_rms[:stop, :]]
        fig, axes = plt.subplots(1, 4, figsize=(12, 4))
        file = 'wb_rms_zt.svg'
    case_opt = comparison_plot_opt(labels)
    last_t = time[stop]/(3600*24)#time.max()/(3600*24)
    ratio = (last_t)/lx[2]
    axes = axes.ravel()
    plt.subplots_adjust(bottom = 0.1, top = 0.95)
    for n, var in enumerate(vars):
        #np.flipud(var.T), var.T
        im = axes[n].imshow(np.flipud(var.T), extent=[time.min()/(3600*24), last_t, z.min(), z.max()], interpolation ='none', cmap=colors[n], vmin=range_opt[n][0], vmax=range_opt[n][1])
        axes[n].plot(time, -mld*np.ones_like(time), color = 'k', label=r"$\text{h}_{ML}$", linewidth = 0.9, linestyle = case_opt['linestyle'][1])
        axes[n].legend(loc='lower left')
        axes[n].set_xlim(time.min()/(3600*24), last_t)
        axes[n].set_ylim(z.min(), z.max())
        axes[n].set_xlabel("time [days]")
        axes[n].set_ylabel("z [m]")
        axes[n].set_title(titles[n])
        axes[n].set_aspect(ratio)
        cbar = fig.colorbar(im, ax = axes[n], anchor = (0.5, 0.9), orientation='horizontal', label=labels[n], shrink=0.8, aspect=30)
        cbar.formatter.set_useOffset(False)
        cbar.formatter.set_powerlimits((-3, 5))
        cbar.update_ticks() 
    frame_path = os.path.join(outdir, file)
    plt.savefig(frame_path)
    plt.close(fig)
if plot_yt:
    w_max = np.max(np.abs(w_yt))
    b_fluc_max = np.max(np.abs(b_fluc_yt))
    if reader.salinity:
        range_opt = [(-w_max, w_max), (0.0, np.max(S_yt)), (np.min(T_yt), np.max(T_yt)), (-b_fluc_max, b_fluc_max)]
        titles = [r"w", r"S", r"T", r"b'", r"b"]
        colors = ['RdBu', 'Blues', 'viridis', 'RdBu']
        labels = ['[m/s]', '[g/kg]', r'[$^\circ$C]', r'[m/s$^2$]']
        vars = [w_yt, S_yt, T_yt, b_fluc_yt]
        file = 'wSTb_yt.svg'
    else:
        range_opt = [(-w_max, w_max), (np.min(T_yt), np.max(T_yt)), (-b_fluc_max, b_fluc_max)]
        titles = [r"w", r"T", r"b'", r"b"]
        colors = ['RdBu', 'viridis', 'RdBu']
        labels = ['[m/s]', r'[$^\circ$C]', r'[m/s$^2$]']
        vars = [w_yt, T_yt, b_fluc_yt]
        file = 'wTb_yt.svg'
    case_opt = comparison_plot_opt(labels)
    if len(vars) <= 4:
        ncols = len(vars)
        nrows = 1
    else:
        ncols = 4
        nrows = int(np.ceil(len(vars)/ncols))
    for j, z_opt in enumerate(z_loc):
    
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4*nrows+1), sharex = True, sharey = True)

        ratio = ((time.max()/(3600*24))/lx[1])**-1
        axes = axes.ravel()
        fig.suptitle(f"z = {z_opt} m")
        plt.subplots_adjust(bottom = 0.1, top = 0.9)
        axes[0].set_ylabel("time [days]")
        for n, var in enumerate(vars):
            im = axes[n].imshow(var[:, :, j], extent=[y.min(), y.max(), time.min()/(3600*24), time.max()/(3600*24)], interpolation ='none', cmap=colors[n], vmin=range_opt[n][0], vmax=range_opt[n][1])
            axes[n].set_xlim(y.min(), y.max())
            axes[n].set_ylim(time.min()/(3600*24), time.max()/(3600*24))
            axes[n].set_xlabel("y [m]")
            axes[n].set_title(titles[n]+rf'(0, y, {z_opt})')
            axes[n].set_aspect(ratio)
            cbar = fig.colorbar(im, ax = axes[n], anchor = (0.5, 0.9), orientation='horizontal', label=labels[n], shrink=0.8, aspect=30)
            cbar.formatter.set_useOffset(False)
            cbar.formatter.set_powerlimits((-3, 5))
            cbar.update_ticks() 
        frame_path = os.path.join(outdir, rf'z{z_loc[j]}_{file}')
        plt.savefig(frame_path)
        plt.close(fig)
if plot_xt:
    w_max = np.max(np.abs(w_xt))
    b_fluc_max = np.max(np.abs(b_fluc_xt))
    if reader.salinity:
        range_opt = [(-w_max, w_max), (0.0, np.max(S_xt)), (np.min(T_xt), np.max(T_xt)), (-b_fluc_max, b_fluc_max)]
        titles = [r"w", r"S", r"T", r"b'", r"b"]
        colors = ['RdBu', 'Blues', 'viridis', 'RdBu']
        labels = ['[m/s]', '[g/kg]', r'[$^\circ$C]', r'[m/s$^2$]']
        vars = [w_xt, S_xt, T_xt, b_fluc_xt]
        file = 'wSTb_xt.svg'
    else:
        range_opt = [(-w_max, w_max), (np.min(T_xt), np.max(T_xt)), (-b_fluc_max, b_fluc_max)]
        titles = [r"w", r"T", r"b'", r"b"]
        colors = ['RdBu', 'viridis', 'RdBu']
        labels = ['[m/s]', r'[$^\circ$C]', r'[m/s$^2$]']
        vars = [w_xt, T_xt, b_fluc_xt]
        file = 'wTb_xt.svg'
    case_opt = comparison_plot_opt(labels)
    if len(vars) <= 4:
        ncols = len(vars)
        nrows = 1
    else:
        ncols = 4
        nrows = int(np.ceil(len(vars)/ncols))
    for j, z_opt in enumerate(z_loc):
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4*nrows+1), sharex = True, sharey = True)

        ratio = ((time.max()/(3600*24))/lx[1])**-1
        axes = axes.ravel()
        fig.suptitle(f"z = {z_opt} m")
        plt.subplots_adjust(bottom = 0.1, top = 0.9)
        axes[0].set_ylabel("time [days]")
        for n, var in enumerate(vars):
            im = axes[n].imshow(var[:, :, j], extent=[x.min(), x.max(), time.min()/(3600*24), time.max()/(3600*24)], interpolation ='none', cmap=colors[n], vmin=range_opt[n][0], vmax=range_opt[n][1])
            axes[n].set_xlim(x.min(), x.max())
            axes[n].set_ylim(time.min()/(3600*24), time.max()/(3600*24))
            axes[n].set_xlabel("x [m]")
            axes[n].set_title(titles[n]+rf'(x, 0, {z_opt})')
            axes[n].set_aspect(ratio)
            cbar = fig.colorbar(im, ax = axes[n], anchor = (0.5, 0.9), orientation='horizontal', label=labels[n], shrink=0.8, aspect=30)
            cbar.formatter.set_useOffset(False)
            cbar.formatter.set_powerlimits((-3, 5))
            cbar.update_ticks() 
        frame_path = os.path.join(outdir, rf'z{z_loc[j]}_{file}')
        plt.savefig(frame_path)
        plt.close(fig)
    
if plot_xy_plane:
    hor_ranges = ranges.copy()
    hor_ranges['S'] = [S_tol, 0.2]
    hor_ranges['w'] = [-2*10**(-3), 2*10**(-3)]
    hor_ranges['Sw'] = [-2*10**(-4), 2*10**(-4)]
    vars =['S', 'w', 'Sw']

    xy_outdir = np.empty(len(vars), dtype=object)

    for it in range(nt):

        t = reader.t[it]
        Sw_hor = S_hor[it, :, :] * w_hor[it, :, :]

        xy_outdir[0] = plot_var_planeslice(t, it, hor_ranges, outdir, lx, x, y, S_hor[it, :, :], [''], vars[0], vars[0], colorbar_label='g/kg', cmap='Blues')
        xy_outdir[1] = plot_var_planeslice(t, it, hor_ranges, outdir, lx, x, y, w_hor[it, :, :], [''], vars[1], vars[1], colorbar_label='m/s', cmap='RdBu_r')
        xy_outdir[2] = plot_var_planeslice(t, it, hor_ranges, outdir, lx, x, y, Sw_hor, [''], vars[2], vars[2], colorbar_label='m/s', cmap='RdBu_r')
if plot_bin_plane:
    bin_dir = {}
    for it in range(nt):
        for n in range(len(range_names)):
            variables[var_names[n]] = {'title': var_names[n], 'label':var_units[n], 'cmap':'Blues', 'range':ranges[range_names[n]], 'range_name':range_names[n]}

        for dir, var in enumerate(var_names):
            l_bin = [lx[0]/2, lx[-1]]
            bin_dir[range_names[dir]] = plot_var_planeslice(time[it], it, outdir, lx, r, z, variables[var], ['',], plane='binning', ylim = (0, 32), zlim = (-60, -30))

if plot_1dz_stats:
    gridspec_kw={'height_ratios': [1, 1, 0.1]}
    width = 0.8
    labels = [r'$\text{w}_{(0, 0)}$', r"$\text{w}_{rms}$", r"$\text{w'}_{0, 0}$", r"$\text{b}_{rms}$", r"$\text{b'}_{0, 0}$"]
    case_opt = comparison_plot_opt(labels)
    case_handles = [Line2D([0], [0], color = case_opt['color'][i], linestyle=case_opt[case]['linestyle'], linewidth=width, label=labels[i]) for i in range(nvars)]

    for it in range(nt):
        td = time[it]/(3600*24)
        fig, axes = plt.subplots(3, 5, figsize=(12, 5), sharey = True, gridspec_kw=gridspec_kw)
        fig.suptitle(f"t = {td:.2f} days")
        for ax in axes[-1, :]:
            ax.remove()
        axes = axes.ravel()
        ax0 = axes[0]  # w, w' 
        ax1 = axes[1]  # dw/dz
        ax2 = axes[2]  # w rms
        ax3 = axes[3]  # dwrms/dz
        ax5 = axes[5]  # b'
        ax6 = axes[6]  # db'/dz
        ax7 = axes[7]  # b rms
        ax8 = axes[8]  # dbrms/dz
        fig.legend(handles=case_handles,
                loc='lower center',
                ncol=nvars,
                bbox_to_anchor=(0.52, 0.01))

        ax0.plot(mld_var, mld_array, color = case_opt['color'][0], label=r"$\text{h}_{ML}$", linewidth = width/2, linestyle = case_opt['linestyle'][1])
        ax0.plot(w_centerline[it, :], z, color = case_opt['color'][0], linewidth = width)
        ax0.set_xlim(ranges['w'])
        ax0.legend(loc='lower left')
        ax0.set_title("w")
        ax0.set_xlabel("[m/s]")
        ax0.set_ylabel("z [m]")

        ax1.plot(mld_var, mld_array, color = case_opt['color'][0], label=r"$\text{h}_{ML}$", linewidth = width/2, linestyle = case_opt['linestyle'][1])
        ax1.plot(dwdz_centerline[it, :], z, color = case_opt['color'][0], linewidth = width)
        ax1.set_xlim(ranges['gradw'])
        ax1.set_title("dw/dz")
        ax1.set_xlabel("dw/dz [1/s]")
        #ax1.set_ylabel("z [m]")

        ax2.plot(mld_var, mld_array, color = case_opt['color'][0], label=r"$\text{h}_{ML}$", linewidth = width/2, linestyle = case_opt['linestyle'][1])
        ax2.plot(w_rms[it, :], z, color = case_opt['color'][1], linewidth = width)
        ax2.set_xlim(ranges['vel_rms'])
        ax2.set_title(r"w$_{rms}$")
        ax2.set_xlabel(r"w$_{rms}$ [m/s]")

        ax3.plot(mld_var, mld_array, color = case_opt['color'][0], label=r"$\text{h}_{ML}$", linewidth = width/2, linestyle = case_opt['linestyle'][1])
        ax3.plot(dwrmsdz[it, :], z, color = case_opt['color'][1], linewidth = width)
        ax3.set_xlim(ranges['gradw'][0]*factor, ranges['gradw'][1]*factor)
        ax3.set_title(r"dw$_{rms}$/dz")
        ax3.set_xlabel("dw/dz [1/s]")
        #ax3.set_ylabel("z [m]")

        ax5.plot(mld_var, mld_array, color = case_opt['color'][0], label=r"$\text{h}_{ML}$", linewidth = width/2, linestyle = case_opt['linestyle'][1])
        ax5.plot(b_fluc_centerline[it, :], z, color = case_opt['color'][4], linewidth = width)
        ax5.set_xlim(ranges['b_fluc'])
        ax5.set_title("b'")
        ax5.set_xlabel(r"b' [m/s$^2$]")
        ax5.set_ylabel("z [m]")

        ax6.plot(mld_var, mld_array, color = case_opt['color'][0], label=r"$\text{h}_{ML}$", linewidth = width/2, linestyle = case_opt['linestyle'][1])
        ax6.plot(dbflucdz[it, :], z, color = case_opt['color'][4], linewidth = width)
        ax6.set_xlim(ranges['gradb'])
        ax6.set_title("db'/dz")
        ax6.set_xlabel(r"db'/dz [1/s$^2$]")

        ax7.plot(mld_var, mld_array, color = case_opt['color'][0], label=r"$\text{h}_{ML}$", linewidth = width/2, linestyle = case_opt['linestyle'][1])
        ax7.plot(b_rms[it, :], z, color = case_opt['color'][3], linewidth = width)
        ax7.set_xlim(ranges['b_rms'])
        ax7.set_title(r"b$_{rms}$")
        ax7.set_xlabel(r"b$_{rms}$ [m/s$^2$]")
        #ax7.set_ylabel("z [m]")

        ax8.plot(mld_var, mld_array, color = case_opt['color'][0], label=r"$\text{h}_{ML}$", linewidth = width/2, linestyle = case_opt['linestyle'][1])
        ax8.plot(dbrmsdz[it, :], z, color = case_opt['color'][3], linewidth = width)
        ax8.set_xlim(ranges['gradb'][0]*factor, ranges['gradb'][1]*factor)
        ax8.set_title(r"db$_{rms}$/dz")
        ax8.set_xlabel(r"db$_{rms}$/dz [1/s$^2$]")

        for ax in axes:
            ax.ticklabel_format(axis='x', style='sci', scilimits=(-1,2), useMathText=True)
        # --- Save Frame ---
        frame_path = os.path.join(fig_var_folder, f"variables_of_interest_{it:04d}.png")
        plt.savefig(frame_path)
        plt.close(fig)
if verify_outputs:
    outdir_verify = os.path.join(outdir, 'verify_outputs')
    os.makedirs(outdir_verify, exist_ok=True)
    width = 0.8
    for it in range(nt):
        td = time[it]/(3600*24)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey = True)
        axes = axes.ravel()
        plt.subplots_adjust(top = 0.9)
        fig.suptitle(f"t = {td:.2f} days")
        axes[0].plot(w_centerline[it, :], z, color = 'k', linewidth = width, label='high frequency')
        axes[0].plot(w_coarse_centerline[it, :], z, color = 'b', linewidth = width/2, linestyle = case_opt['linestyle'][1], label='coarse output')
        axes[0].set_title("w")
        axes[0].set_xlabel("[m/s]")
        axes[0].set_xlim(ranges['w'])
        axes[0].set_ylabel("z [m]")
        axes[0].legend(loc='lower left')

        axes[1].plot(b_centerline[it, :], z, color = 'k', linewidth = width, label='high frequency')
        axes[1].plot(b_coarse_centerline[it, :], z, color = 'b', linewidth = width/2, linestyle = case_opt['linestyle'][1], label='coarse output')
        axes[1].set_title("b")
        axes[1].set_xlabel(r"[m/s$^2$]")
        axes[1].set_xlim(ranges['b'])
        axes[1].legend(loc='upper left')

        if reader.salinity:
            axes[2].plot(S_centerline[it, :], z, color = 'k', linewidth = width, label='high frequency')
            axes[2].plot(S_coarse_centerline[it, :], z, color = 'b', linewidth = width/2, linestyle = case_opt['linestyle'][1], label='coarse output')
            axes[2].set_title("S")
            axes[2].set_xlabel(r"[g/kg]")
            axes[2].set_xlim(ranges['S'])
            axes[2].legend(loc='lower right')

        frame_path = os.path.join(outdir_verify, f'verify_outputs_{it}.png')
        plt.savefig(frame_path)
        plt.close(fig)

# creating videos
if video:
    if plot_1dz_stats:
        create_video(fig_var_folder, outdir, '', 'max penetration variables')
    if verify_outputs:
        create_video(outdir_verify, outdir, '', 'verify_outputs')
    if plot_xy_plane:
        for n, folder in enumerate(xy_outdir):
            create_video(folder, outdir, '', vars[n])
    if plot_bin_plane:
        for n, name in enumerate(range_names):
            create_video(bin_dir[range_names[n]], outdir, 'binning', name)
