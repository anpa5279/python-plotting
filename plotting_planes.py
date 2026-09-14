import os
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D

from plotting_general import save_frame
### -------------------------PLOTTING PLANE SLICES FUNCTIONS------------------------- ###
## variable vertical plane slice across all cases
def plot_var_planeslice(time, it, fig_folder, lx, hor, z, var, case_names, loc = 0.0, plane='YZ'):
    planeslice = var['var']
    name = var['title']
    range_name = var['range_name']
    cmap = var['cmap']
    colorbar_label = var['label']
    range_var = var['range']
    td = time / 3600 / 24
    if plane == 'YZ': #yz plane
        lhor = np.min(lx[1])
        lz = np.max(np.abs(lx[2]))
        ar = lhor/lz
        plane = 'YZ plane'
        xlabel = "y [m]"
        ylabel = "Depth [m]"
        title = name + ', ' + plane + ', ' + f'{td:.2f} days'
        out_folder = f'x = {loc:.2f} m'
    elif plane == 'XZ': #xz plane
        lhor = np.max(lx[0])
        lz = np.max(np.abs(lx[2]))
        ar = lhor/lz
        plane = 'XZ plane'
        xlabel = "x [m]"
        ylabel = "Depth [m]"
        title = name + ', ' + plane + ', ' + f'{td:.2f} days'
        out_folder = f'y = {loc:.2f} m'
    elif plane == 'XY': #xz plane
        lhor = np.max(lx[0])
        lz = np.max(np.abs(lx[1]))
        ar = lhor/lz
        plane = 'XY plane'
        xlabel = "x [m]"
        ylabel = "y [m]"
        title = name + ', ' + plane + ', ' + f'{td:.2f} days'
        out_folder = f'z = {loc:.2f} m'
    elif plane == 'binning':
        lhor = np.min(lx[0:1])/2
        lz = np.max(np.abs(lx[2]))
        ar = lhor/lz
        xlabel = "r [m]"
        ylabel = "Depth [m]"
        title = name + ', ' + f'{td:.2f} days'
        out_folder = f''
    else:
        raise ValueError("Invalid plane specified. Choose from 'YZ', 'XZ', 'XY', or 'binning'.")

    outdir = os.path.join(fig_folder, 'comparison plume analysis/', range_name, plane, out_folder)
    os.makedirs(outdir, exist_ok=True)
    num_cases = len(planeslice)
    ncols = 4
    if num_cases < ncols*2:
        ncols = num_cases
    nrows = int(math.ceil(num_cases/ncols))
    hor_len = 12.0
    vert_len = hor_len * nrows / (ncols * ar) - 1.5 + 0.25 * nrows + 0.5
    size_in = (hor_len, vert_len)#(4 * num_cases, 8)
    fig, axes = plt.subplots(nrows, ncols, figsize=size_in, sharey = True, sharex = True, constrained_layout=True)
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
    fig.suptitle(title)
    for n in range(num_cases):
        if 'log' in range_name and 'w' not in range_name and 'neg' not in range_name:
            var[n][var[n] <= 0] = 10**(-16)
            im = axes[n].imshow(planeslice[n], extent =[hor[n].min(), hor[n].max(), z[n].min(), z[n].max()], interpolation ='none', origin ='lower', cmap = cmap, norm=colors.LogNorm(vmin=range_var[0], vmax=range_var[-1]))
        elif any(('log' in range_name, 'neg' in range_name)):
            im = axes[n].imshow(planeslice[n], extent =[hor[n].min(), hor[n].max(), z[n].min(), z[n].max()], interpolation ='none', origin ='lower', cmap = cmap, norm=colors.SymLogNorm(linthresh=1e-6, vmin=range_var[0], vmax=range_var[-1]))
        else:
            im = axes[n].imshow(planeslice[n], vmin=range_var[0], vmax=range_var[-1], extent =[hor[n].min(), hor[n].max(), z[n].min(), z[n].max()], interpolation ='none', origin ='lower', cmap = cmap)
        axes[n].set_title(case_names[n])
        if plane == 'binning':
            axes[n].set_xlim(0, hor[n].max())
            axes[n].set_ylim(z[n].min(), 0)
        else:
            axes[n].set_xlim(-np.min(lx[:-1, :]/2), np.min(lx[:-1, :]/2))
            axes[n].set_ylim(np.min(lx[2, :]), 0)
        
        axes[n].set_aspect('equal')
        if n == 0 or n%ncols == 0:
            axes[n].set_ylabel(ylabel)
        if n >= (nrows - 1) * ncols:
            axes[n].set_xlabel(xlabel)

    active_axes = [axes[n] for n in range(num_cases)]
    cbar = fig.colorbar(im, ax = active_axes, shrink=0.9, aspect=50, label = colorbar_label)#, anchor = (0.5, 0.05), orientation='horizontal')
    if all(('log' not in range_name, 'neg' not in range_name)):
        cbar.formatter.set_useOffset(False)
        cbar.formatter.set_powerlimits((-2, 5))
        cbar.update_ticks() 

    # --- Save Frame ---
    save_frame(fig, outdir, it, size_in, file_name = "oc_plane_slices_")
    return outdir

## tracer slice comparison across all cases
def plot_tracer_slice_comparison(time_sec, it, case_names, ranges, y, z, tracer_fields, Sval, fig_folder, ylim = (-5, 5), zlim = (-10, 0), binning = False, folder_name = "tracer_zoom_frames", negative = False):
    if negative:
        folder_name += "_log_neg"
    frame_dir = os.path.join(fig_folder, folder_name)
    os.makedirs(frame_dir, exist_ok = True)
    num_cases = len(z)
    hor_len = 4 * num_cases
    vert_len = 5 * (zlim[1] - zlim[0]) / (ylim[1] - ylim[0])
    fig, axes = plt.subplots(1, num_cases, figsize = (hor_len, vert_len), constrained_layout = True, sharey = True)

    if num_cases == 1:
        axes = [axes]
    if binning:
        xlabel = "r [m]"
    else:
        xlabel = "y [m]"

    levels = [0.005 * Sval, 0.01 * Sval, 0.05 * Sval]
    legend_lines = [Line2D([0], [0], color='orange', lw=2, label=r'0.5% S$_0$'),
                    Line2D([0], [0], color='red', lw=2, label=r'1% S$_0$'),
                    Line2D([0], [0], color='black', lw=2, label=r'5% S$_0$')]
    for n in range(num_cases):
        S = tracer_fields[n]
        if negative:
            im = axes[n].imshow(S.T, origin = "lower", interpolation = "none", norm=colors.SymLogNorm(linthresh=1e-8, vmin=ranges['log neg S'][0], vmax=ranges['log neg S'][-1]), extent = [y[n].min(), y[n].max(), z[n].min(), z[n].max()], aspect = "auto", cmap = "RdBu")
        else:
            im = axes[n].imshow(S.T, origin = "lower", interpolation = "none", norm = colors.LogNorm(vmin = ranges['Tracer'][0], vmax = ranges['Tracer'][1]), extent = [y[n].min(), y[n].max(), z[n].min(), z[n].max()], aspect = "auto", cmap = "Blues")
            axes[n].contour(y[n], z[n], S.T, levels = levels, colors = ["orange", "red", "black"])

        axes[n].set_xlim(ylim)
        axes[n].set_ylim(zlim)
        axes[n].set_title(case_names[n])
        axes[n].set_xlabel(xlabel)
        if n == 0:
            axes[n].set_ylabel("z [m]")
        axes[n].set_aspect('equal')
    if not negative:
        axes[0].legend(handles=legend_lines,loc='lower right')

    plt.colorbar(im, ax = axes, anchor = (0.5, 0.0), orientation='horizontal', shrink=0.75, aspect=80)
    fig.suptitle(f"t = {time_sec/3600:.2f} hr")
    fig.set_size_inches(hor_len, vert_len)
    save_frame(fig, frame_dir, it, (hor_len, vert_len))
    return frame_dir