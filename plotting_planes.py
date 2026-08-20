import os
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import colors

from plotting_general import create_video, save_frame
### -------------------------PLOTTING PLANE SLICES FUNCTIONS------------------------- ###
## variable vertical plane slice across all cases
def plot_variable_vert_slice(time, it, ranges, fig_folder, lx, hor, z, var, case_names, name, range_name, colorbar_label=None, loc = 0.0, cmap='RdBu_r', plane='YZ'):
    td = time / 3600 / 24
    if plane == 'YZ': #yz plane
        lhor = np.min(lx[1])
        lz = np.max(lx[2])
        ar = lhor/lz
        plane = 'YZ plane'
        xlabel = "y [m]"
        title = name + ', ' + plane + ', ' + f'{td:.2f} days'
        out_folder = f'x = {loc:.2f} m'
    elif plane == 'XZ': #xz plane
        lhor = np.max(lx[0])
        lz = np.max(lx[2])
        ar = lhor/lz
        plane = 'XZ plane'
        xlabel = "x [m]"
        title = name + ', ' + plane + ', ' + f'{td:.2f} days'
        out_folder = f'y = {loc:.2f} m'
    elif plane == 'binning':
        lhor = np.min(lx[0:1])/2
        lz = np.max(lx[2])
        ar = lhor/lz
        xlabel = "r [m]"
        title = name + ', ' + f'{td:.2f} days'
        out_folder = f''
    else:
        raise ValueError("Invalid plane specified. Choose from 'YZ', 'XZ', or 'binning'.")

    outdir = os.path.join(fig_folder, 'comparison plume analysis/', name, plane, out_folder)
    os.makedirs(outdir, exist_ok=True)
    num_cases = len(var)
    ncols = 4
    if num_cases < ncols*2:
        ncols = num_cases
    nrows = int(math.ceil(num_cases/ncols))
    #hor_len = 12.0
    #vert_len = hor_len * nrows / (ncols * ar) - 1.5 + 0.25 * nrows + 0.5
    size_in = (3 * num_cases, 6)
    fig, axes = plt.subplots(nrows, ncols, figsize=size_in, sharey = True, sharex = True, constrained_layout=True)
    fig.suptitle(title)
    axes = axes.ravel()
    # Force even pixel dimensions at 600 dpi
    w_px = int(fig.get_figwidth() * fig.dpi)
    h_px = int(fig.get_figheight() * fig.dpi)
    if w_px % 2 != 0:
        fig.set_figwidth((w_px + 1) / fig.dpi)
    if h_px % 2 != 0:
        fig.set_figheight((h_px + 1) / fig.dpi)
    if num_cases != (nrows*ncols):
        for i in range(num_cases, nrows*ncols):
            axes[i].remove() # remove extra subplots if number of cases is less than nrows*ncols
    for n in range(num_cases):
        if 'log' in range_name:
            var[n][var[n] <= 0] = 10**(-16)
            im = axes[n].imshow(var[n], extent =[hor[n].min(), hor[n].max(), z[n].min(), z[n].max()], interpolation ='none', origin ='lower', cmap = cmap, norm=colors.LogNorm(vmin=ranges[range_name][0], vmax=ranges[range_name][-1]))
        else:
            im = axes[n].imshow(var[n], vmin=ranges[range_name][0], vmax=ranges[range_name][-1], extent =[hor[n].min(), hor[n].max(), z[n].min(), z[n].max()], interpolation ='none', origin ='lower', cmap = cmap)
        axes[n].set_title(case_names[n])
        axes[n].set_xlim(-min(lx[:-1, :]/2), min(lx[:-1, :]/2))
        axes[n].set_ylim(min(lx[2, :]), 0)
        axes[n].set_aspect('equal')
        if n == 0 or n%ncols == 0:
            axes[n].set_ylabel("Depth [m]")
        if n >= (nrows - 1) * ncols:
            axes[n].set_xlabel(xlabel)

    active_axes = [axes[n] for n in range(num_cases)]
    cbar = fig.colorbar(im, ax = active_axes, shrink=0.9, aspect=50, label = colorbar_label)#, anchor = (0.5, 0.05), orientation='horizontal')
    if 'log' not in range_name: 
        cbar.formatter.set_useOffset(False)
        cbar.formatter.set_powerlimits((-2, 5))
        cbar.update_ticks() 

    # --- Save Frame ---
    save_frame(fig, outdir, it, size_in, file_name = "oc_plane_slices_")
    return outdir
## variable xy plane slice across all cases
def plot_variable_xy_slice(time, it, ranges, fig_folder, lx, x, y, var, case_names, name, range_name, colorbar_label=None, loc = 0.0, cmap='RdBu_r'):
    plane = 'XY plane'
    td = time / 3600 / 24
    num_cases = len(var)
    ncols = 3
    if num_cases < ncols*2:
        ncols = num_cases
    nrows = int(math.ceil(num_cases/ncols))
    hor_len = 12.0
    vert_len = hor_len * nrows / (ncols) + 0.5 * nrows + 1.1
    size_in = (hor_len, vert_len)
    fig, axes = plt.subplots(nrows, ncols, figsize = size_in, sharey = True, sharex = True, constrained_layout=True)
    fig.suptitle(rf'{name}, z = {loc:.2f} m, {td:.2f} days')
    if num_cases > 1:
        outdir = os.path.join(fig_folder, 'comparison plume analysis/', range_name, plane, f'z = {loc:.2f} m')
        axes = axes.ravel()
        if num_cases != (nrows*ncols):
            for i in range(num_cases, nrows*ncols):
                axes[i].remove() # remove extra subplots if number of cases is less than nrows*ncols
        for n in range(num_cases):
            if 'log' in range_name and 'w' not in range_name and 'neg S' not in range_name:
                im = axes[n].imshow(var[n].T, extent =[x[n].min(), x[n].max(), y[n].min(), y[n].max()], interpolation ='none', origin ='lower', cmap = cmap, norm=colors.LogNorm(vmin=ranges[range_name][0], vmax=ranges[range_name][-1]))
            elif 'log' in range_name and 'w' in range_name:
                im = axes[n].imshow(var[n].T, extent =[x[n].min(), x[n].max(), y[n].min(), y[n].max()], interpolation ='none', origin ='lower', cmap = cmap, norm=colors.SymLogNorm(linthresh=1e-5, vmin=ranges[range_name][0], vmax=ranges[range_name][-1]))
            elif 'log' in range_name and 'neg S' in range_name:
                im = axes[n].imshow(var[n].T, extent =[x[n].min(), x[n].max(), y[n].min(), y[n].max()], interpolation ='none', origin ='lower', cmap = cmap, norm=colors.SymLogNorm(linthresh=1e-8, vmin=ranges[range_name][0], vmax=ranges[range_name][-1]))
            else:
                im = axes[n].imshow(var[n].T, vmin=ranges[range_name][0], vmax=ranges[range_name][-1], extent =[x[n].min(), x[n].max(), y[n].min(), y[n].max()], interpolation ='none', origin ='lower', cmap = cmap)
            axes[n].set_title(case_names[n])
            axes[n].set_aspect('equal')
            axes[n].set_xlabel("x [m]")
            axes[n].set_ylabel("y [m]")
        active_axes = [axes[n] for n in range(num_cases)]
    else:
        outdir = os.path.join(fig_folder, name, plane, f'z = {loc:.2f} m')
        if name == 'Tracer' or name == 'log w' or name == 'S':
            im = axes.imshow(var.T, extent =[x.min(), x.max(), y.min(), y.max()], interpolation ='none', origin ='lower', cmap = cmap, norm=colors.LogNorm(vmin=ranges[range_name][0], vmax=ranges[range_name][-1]))
            
        else:
            im = axes.imshow(var.T, vmin=ranges[range_name][0], vmax=ranges[range_name][-1], extent =[x.min(), x.max(), y.min(), y.max()], interpolation ='none', origin ='lower', cmap = cmap)
        axes.set_aspect('equal')
        axes.set_xlabel("x [m]")
        axes.set_ylabel("y [m]")
        active_axes = [axes]
    os.makedirs(outdir, exist_ok=True)
    cbar = fig.colorbar(im, ax = active_axes, anchor = (0.5, -0.3), orientation='horizontal', label = colorbar_label, shrink=0.75, aspect=50)
    if 'log' not in range_name: 
        cbar.formatter.set_useOffset(False)
        cbar.formatter.set_powerlimits((-2, 5))
        cbar.update_ticks() 
    # --- Save Frame ---
    save_frame(fig, outdir, it, size_in, file_name = "oc_plane_slices_")
    return outdir

### -------------------------BINNING PLOTTING FUNCTIONS------------------------- ###
def plot_binning(S_rz, T_rz, hor_vel_rz, w_rz, r, z, time, output_folder, min_S = 10**(-6)):

    # ranges for plotting 
    frac = 0.7
    Smax = np.max(np.abs(S_rz))
    S_range = (min_S, Smax)
    S_rz[S_rz<min_S] = min_S # set values below threshold to threshold for log plotting
    T_range = (np.min(T_rz), 25.01)
    umax = np.max(np.abs(hor_vel_rz)) * frac**2
    u_range = (-umax, umax)
    wmax = np.max(np.abs(w_rz)) * frac
    w_range = (-wmax, wmax)
    size_in = (12, 11)

    outdir = os.path.join(output_folder, 'plotting')
    os.makedirs(outdir, exist_ok=True)

    # plotting results
    for it, t in enumerate(time):
        fig, axes = plt.subplots(2, 2, figsize=size_in, sharey = True, sharex = True, constrained_layout=True)
        axes = axes.ravel()
        td = t / 3600 / 24
        fig.suptitle(f'{td:.2f} days', y = 0.99)
        """
        ax0  # temperature
        ax1  # tracer
        ax2  # w velocity
        ax3  # horizontal velocity
        """

        im = axes[0].imshow(T_rz[:, :, it].T, vmin=T_range[0], vmax=T_range[1], extent =[r.min(), r.max(), z.min(), z.max()], interpolation ='none', origin ='lower')
        axes[0].set_ylabel("Depth [m]")
        #axes[0].set_xlabel("radial distance [m]")
        axes[0].set_title("Temperature")
        axes[0].set_aspect('equal')
        cbar = fig.colorbar(im, ax = axes[0], label=r"$^\circ$C", anchor = (0.5, -0.05), orientation='horizontal', shrink=0.75)

        im = axes[1].imshow(S_rz[:, :, it].T, extent =[r.min(), r.max(), z.min(), z.max()], interpolation ='none', origin ='lower', norm=colors.LogNorm(vmin=S_range[0], vmax=S_range[1]))
        #axes[1].set_xlabel("radial distance [m]")
        axes[1].set_title("Tracer")
        axes[1].set_aspect('equal')
        cbar = fig.colorbar(im, ax = axes[1], label=r"g/kg", anchor = (0.5, -0.05), orientation='horizontal', shrink=0.75)

        im = axes[2].imshow(w_rz[:, :, it].T, vmin=w_range[0], vmax=w_range[1], extent =[r.min(), r.max(), z.min(), z.max()], interpolation ='none', origin ='lower', cmap='RdBu_r')
        #axes[2].set_xlabel("radial distance [m]")
        axes[2].set_title("w")
        axes[2].set_aspect('equal')
        cbar = fig.colorbar(im, ax = axes[2], label=r"m/s", anchor = (0.5, -0.05), orientation='horizontal', shrink=0.75)
        cbar.formatter.set_powerlimits((-2, 2))
        cbar.update_ticks()

        im = axes[3].imshow(hor_vel_rz[:, :, it].T, vmin=u_range[0], vmax=u_range[1], extent =[r.min(), r.max(), z.min(), z.max()], interpolation ='none', origin ='lower', cmap='RdBu_r')
        axes[3].set_xlabel("radial distance [m]")
        axes[3].set_title("Horizontal Velocity")
        axes[3].set_aspect('equal')
        cbar = fig.colorbar(im, ax = axes[3], label=r"m/s", anchor = (0.5, -0.05), orientation='horizontal', shrink=0.75)
        cbar.formatter.set_powerlimits((-2, 2))
        cbar.update_ticks()
        
        
        # --- Save Frame ---
        save_frame(fig, outdir, it, size_in, file_name = "oc_plane_slices_")

    create_video(outdir, output_folder, '', 'binning_rtz')
