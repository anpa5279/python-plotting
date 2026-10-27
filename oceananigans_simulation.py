import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

from reader import OceananigansData
from plotting_general import plot_format, create_video, comparison_plot_opt, plot_ranges
from plotting_planes import plot_var_planeslice, plot_debug_planeslice
from interpolation import point, interp1d_axis
# ==========================================================
# FLAGS
# ==========================================================
plot_xt = False
plot_yt = False
plot_zt = False
plot_rt = False
plot_xy_plane = False
plot_bin_plane = True
debugging = False

temporal_avg = False
video = True

if debugging:
    plot_bin_plane = True
if temporal_avg:
    nsteps =11 # number of time steps to average over for plotting, also trying 5


# ==========================================================
# MODEL INFORMATION
# ==========================================================
folder = '/Users/annapauls/Documents/Github repositories/3d_langmuir_gpu/localoutputs/scheme-tests/longer/WENO9/dx0.0625'

outdir = os.path.join(folder, 'figures')
reader = OceananigansData(folder, salinity = True, with_halos=True, read_fields=False)

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
if video or plot_zt or plot_yt or plot_xt:
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
if plot_zt:
    w_centerline = reader.field_centerline('w')
    b_avg, b_rms, b_centerline, b_fluc_centerline = reader.load_buoyancy()
    if reader.salinity:
        S_centerline = reader.field_centerline('S')
if plot_zt:
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
if plot_bin_plane and not debugging:
    range_names = ['T', 'ur', 'w', 'S', 'log w', 'log Tracer'] 
    var_names = ['T', 'ur', 'w', 'S', 'w', 'S'] 
    var_units = [r'[$^\circ$C]', '[m/s]', '[m/s]', '[g/kg]', '[m/s]', '[g/kg]']
    cmap_opt = ['Blues_r', 'RdBu_r', 'RdBu_r', 'Blues', 'RdBu_r', 'Blues']
elif plot_bin_plane and debugging:
    range_names = ['log Tracer'] 
    var_names = ['S'] 
    var_units = ['[g/kg]']
    cmap_opt = ['Blues']

if plot_bin_plane or debugging:
    variables = {}
    for n in range(len(range_names)):
        variables[range_names[n]] = {'var':reader.load_binning_var(var_names[n]), 'title': var_names[n], 'label':var_units[n], 'cmap': cmap_opt[n], 'range_name':range_names[n]}

# finding centerlines
if temporal_avg:
    n_start = (nsteps-1)//2
    n_end = nt - (nsteps-1)//2

    if plot_bin_plane or debugging:
        for n in range(len(range_names)):
            var_temp = np.empty((n_end-n_start, len(r), len(z)))
            for it, it_true in enumerate(np.arange(n_start, n_end)):
                if plot_bin_plane or debugging:
                    for n in range(len(range_names)):
                        var_temp[it, :, :] = np.mean(variables[range_names[n]]['var'][it_true-(nsteps-1)//2:it_true+(nsteps-1)//2, :, :], axis = 0)
            variables[range_names[n]]['var'] = var_temp
print("finished loading data")
# ==========================================================
# PLOTTING
# ==========================================================

mld_var = np.arange(-10**2, 10**2)
mld_array = -mld * np.ones(len(mld_var))
ranges = plot_ranges()
ranges['w'] = [-2*10**-3, 2*10**-3]
ranges['b_fluc'] = [-7*10**(-4), 7*10**(-4)]
ranges['gradw'] = [-0.05, 0.05]
ranges['gradb'] = [-0.0008, 0.0008]
ranges['b_rms'] = [0, 1.5*10**(-5)]
ranges['vel_rms'] = [0, 4*10**-3]
ranges['S'] = [0.0, 0.1]
ranges['log Tracer'] =[S_tol, 0.1]
ranges['log w'] = ranges['w']
ranges['T'] =[T0 - 0.7, T0 + 0.05]
ranges['ur'] = [-1*10**(-2), 1*10**(-2)]
factor = 10**(-2)
nvars = 7
plot_format(fontsize = 20)
os.makedirs(outdir, exist_ok=True)
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

        xy_outdir[0] = plot_var_planeslice(t, it, hor_ranges, outdir, x, y, S_hor[it, :, :], [''], vars[0], vars[0], colorbar_label='g/kg', cmap='Blues')
        xy_outdir[1] = plot_var_planeslice(t, it, hor_ranges, outdir, x, y, w_hor[it, :, :], [''], vars[1], vars[1], colorbar_label='m/s', cmap='RdBu_r')
        xy_outdir[2] = plot_var_planeslice(t, it, hor_ranges, outdir, x, y, Sw_hor, [''], vars[2], vars[2], colorbar_label='m/s', cmap='RdBu_r')
if plot_bin_plane:
    bin_dir = {}
    if not temporal_avg and not debugging:
        l_bin = [lx[0]/2, lx[-1]]
        for it in range(nt):
            for dir, var in enumerate(range_names):
                variables[var]['range'] = ranges[var]
                var_temp = variables[var].copy()
                var_temp['var'] = variables[var]['var'][it, :, :].T
                bin_dir[range_names[dir]] = plot_var_planeslice(time[it], it, outdir, r, z, var_temp, ['',], plane='binning', ylim = (0, 32), zlim = (np.min(z), 0.0))
    elif temporal_avg and not debugging:
        l_bin = [lx[0]/2, lx[-1]]
        for it in range(n_start, n_end):
            for dir, var in enumerate(range_names):
                variables[var]['range'] = ranges[var]
                var_temp = variables[var].copy()
                var_temp['var'] = variables[var]['var'][it-(nsteps-1)//2:it+(nsteps-1)//2, :, :]
                bin_dir[range_names[dir]] = plot_var_planeslice(time[it], it, outdir, r, z, var_temp, ['',], plane='binning', ylim = (0, 32), zlim = (-60, -30))
    elif temporal_avg and debugging:
        l_bin = [lx[0]/2, lx[-1]]
        for it in range(n_start, n_end):
            for dir, var in enumerate(range_names):
                variables[var]['range'] = ranges[var]
                var_temp = variables[var].copy()
                var_temp['var'] = variables[var]['var'][it-(nsteps-1)//2:it+(nsteps-1)//2, :, :]
                title = var_names[dir]+f" temporal average, t = {time[it-(nsteps-1)//2]/3600:.2f} to {time[it+(nsteps-1)//2]/3600:.2f} hours"
                bin_dir[var] = plot_debug_planeslice(it, title, outdir, r, z, var_temp, ['',], plane='binning', ylim = (0, 32), zlim = (-55, -35), loc_opt = np.array([-50, -40]))
# creating videos
if video:
    if plot_xy_plane:
        for n, folder in enumerate(xy_outdir):
            create_video(folder, outdir, '', vars[n])
    if plot_bin_plane and not debugging:
        for n, name in enumerate(range_names):
            create_video(bin_dir[range_names[n]], outdir, 'binning', name)
    elif plot_bin_plane and debugging:
        for n, name in enumerate(range_names):
            print(f"creating video for {name}")
            print(f"bin_dir[range_names[n]] = {bin_dir[range_names[n]]}")
            create_video(bin_dir[range_names[n]], outdir, 'binning', name+'debugging')
