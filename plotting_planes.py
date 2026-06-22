import os
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import colors

from plotting_general import create_video
### -------------------------PLOTTING PLANE SLICES FUNCTIONS------------------------- ###
## variable vertical plane slice across all cases
def plot_variable_vert_slice(time, it, ranges, fig_folder, lx, hor, z, var, case_names, name, range_name, colorbar_label=None, cmap='RdBu_r', plane='YZ'):
    td = time / 3600 / 24
    #print(hor)
    #print(z)
    if plane == 'YZ': #yz plane
        lhor = np.min(lx[1])
        lz = np.max(lx[2])
        ar = lhor/lz
        plane = 'YZ plane'
        xlabel = "y [m]"
        title = name + ', ' + plane + ', ' + f'{td:.2f} days'
    elif plane == 'XZ': #xz plane
        lhor = np.max(lx[0])
        lz = np.max(lx[2])
        ar = lhor/lz
        plane = 'XZ plane'
        xlabel = "x [m]"
        title = name + ', ' + plane + ', ' + f'{td:.2f} days'
    elif plane == 'binning':
        lhor = np.max(lx[0])/2
        lz = np.max(lx[2])
        ar = lhor/lz
        xlabel = "r [m]"
        title = name + ', ' + f'{td:.2f} days'
    else:
        raise ValueError("Invalid plane specified. Choose from 'YZ', 'XZ', or 'binning'.")
    #if plane != 'binning':
    #    for n, case_name in enumerate(case_names):
    #        var[n] = var[n].T
    outdir = os.path.join(fig_folder, 'comparison plume analysis/', name, plane)
    os.makedirs(outdir, exist_ok=True)
    num_cases = len(case_names)
    ncols = 3
    if num_cases < ncols*2:
        ncols = num_cases
    nrows = int(math.ceil(num_cases/ncols))
    hor_len = 12.0
    vert_len = hor_len * nrows / (ncols * ar) + 0.25 * nrows + 0.5

    fig, axes = plt.subplots(nrows, ncols, figsize=(hor_len, vert_len), sharey = True, sharex = True, constrained_layout=True, dpi = 600)
    fig.suptitle(title, fontsize=12)
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
    for n, case_name in enumerate(case_names):
        if name == 'Tracer':
            var[n][var[n] <= 0] = 10**(-16)
            im = axes[n].imshow(var[n], extent =[hor[n].min(), hor[n].max(), z[n].min(), z[n].max()], interpolation ='none', origin ='lower', cmap = cmap, norm=colors.LogNorm(vmin=ranges[range_name][0], vmax=ranges[range_name][-1]))
        else:
            im = axes[n].imshow(var[n], vmin=ranges[range_name][0], vmax=ranges[range_name][-1], extent =[hor[n].min(), hor[n].max(), z[n].min(), z[n].max()], interpolation ='none', origin ='lower', cmap = cmap)
        axes[n].set_title(case_name, fontsize=10)
        axes[n].set_aspect('equal')
        if n == 0 or n%ncols == 0:
            axes[n].set_ylabel("Depth [m]")
        if n >= (nrows - 1) * ncols:
            axes[n].set_xlabel(xlabel)

    active_axes = [axes[n] for n in range(num_cases)]
    cbar = fig.colorbar(im, ax = active_axes, anchor = (0.5, -0.3), orientation='horizontal', label = colorbar_label, shrink=0.75, aspect=50)
    if not name == 'Tracer': 
        cbar.formatter.set_useOffset(False)
        cbar.formatter.set_powerlimits((-2, 5))
        cbar.update_ticks() 

    # --- Save Frame ---
    frame_path = os.path.join(outdir, f"oc_plane_slices_{it:04d}.png")
    plt.savefig(frame_path)
    plt.close(fig)
    print(f"Time step {it} captured: {frame_path}")
    return outdir
## variable xy plane slice across all cases
def plot_variable_xy_slice(time, it, ranges, fig_folder, lx, x, y, var, case_names, name, range_name, colorbar_label=None, cmap='RdBu_r'):
    plane = 'XY plane'
    outdir = os.path.join(fig_folder, 'comparison plume analysis/', name, plane)
    os.makedirs(outdir, exist_ok=True)
    td = time / 3600 / 24
    num_cases = len(case_names)
    ncols = 3
    nrows = int(math.ceil(num_cases/ncols))
    hor_len = 12.0
    vert_len = hor_len * nrows / (ncols) + 0.5 * nrows + 1.1

    fig, axes = plt.subplots(nrows, 3, figsize=(hor_len, vert_len), sharey = True, sharex = True, constrained_layout=True, dpi = 300)
    fig.suptitle(name + ', ' + plane + ', ' + f'{td:.2f} days', fontsize=12)
    axes = axes.ravel()
    if num_cases != (nrows*ncols):
        for i in range(num_cases, nrows*ncols):
            axes[i].remove() # remove extra subplots if number of cases is less than nrows*ncols
    for n, case_name in enumerate(case_names):
        if name == 'Tracer':
            im = axes[n].imshow(var[n].T, extent =[x.min(), x.max(), y.min(), y.max()], interpolation ='none', origin ='lower', cmap = cmap, norm=colors.LogNorm(vmin=ranges[range_name][0], vmax=ranges[range_name][-1]))
        else:
            im = axes[n].imshow(var[n].T, vmin=ranges[range_name][0], vmax=ranges[range_name][-1], extent =[x.min(), x.max(), y.min(), y.max()], interpolation ='none', origin ='lower', cmap = cmap)
        axes[n].set_title(case_name, fontsize=10)
        axes[n].set_aspect('equal')
        axes[n].set_xlabel("x [m]")
        axes[n].set_ylabel("y [m]")
    cbar = fig.colorbar(im, ax = axes.tolist(), anchor = (0.5, -0.3), orientation='horizontal', label = colorbar_label, shrink=0.75, aspect=50)
    if not name == 'Tracer': 
        cbar.formatter.set_useOffset(False)
        cbar.formatter.set_powerlimits((-2, 5))
        cbar.update_ticks() 
    # --- Save Frame ---
    frame_path = os.path.join(outdir, f"oc_plane_slices_{it:04d}.png")
    plt.savefig(frame_path)
    plt.close(fig)
    print(f"Time step {it} captured: {frame_path}")
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

    outdir = os.path.join(output_folder, 'plotting')
    os.makedirs(outdir, exist_ok=True)

    # plotting results
    for it, t in enumerate(time):
        fig, axes = plt.subplots(2, 2, figsize=(10, 9.5), sharey = True, sharex = True, constrained_layout=True, dpi = 300)
        axes = axes.ravel()
        td = t / 3600 / 24
        fig.suptitle(f'{td:.2f} days', y = 0.99, fontsize=12)
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
        frame_path = os.path.join(outdir, f"oc_plane_slices_{it:04d}.png")
        plt.savefig(frame_path)
        plt.close(fig)
        print(f"Time step {it + 1} captured: {frame_path}")

        plt.close()

    create_video(outdir, output_folder, '', 'binning_rtz')
