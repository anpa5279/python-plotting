import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import colors

from plotting_functions import create_video
folder = '/Users/annapauls/Library/CloudStorage/OneDrive-UCB-O365/CU-Boulder/TESLa/Carbon Sequestration/Simulations/Oceananigans/NBP/salinity and temperature/no noise circle inlet/S0 = 0.1 dTdz = 0.01 MLD = 60'
output_folder = os.path.join(folder, 'binning')
with h5py.File(os.path.join(output_folder, 'binning_rtz.h5'), 'r') as f:
    S_rz = f['ccc/S_rz'][:]
    T_fluc_rz = f['ccc/T\'_rz'][:]
    T_rz = f['ccc/T_rz'][:]
    u_rz = f['ccc/u_rz'][:]
    v_rz = f['ccc/v_rz'][:]
    w_rz = f['ccc/w_rz'][:]
    r = f['ccc/dimensions/r_bin'][:]
    z = f['ccc/dimensions/z'][:]
    time = f['ccc/dimensions/time'][:]

# ranges for plotting 
frac = 0.7
min_S = 10**(-6)
Smax = np.max(np.abs(S_rz))
S_range = (min_S, Smax)
S_rz[S_rz<min_S] = min_S # set values below threshold to threshold for log plotting
T_flucmax = np.max(np.abs(T_fluc_rz)) * frac
T_fluc_range = (-T_flucmax, T_flucmax)
T_range = (np.min(T_rz), 25.05)
umax = np.max(np.abs(u_rz)) * frac
u_range = (-umax, umax)
vmax = np.max(np.abs(v_rz)) * frac
v_range = (-vmax, vmax)
wmax = np.max(np.abs(w_rz)) * frac
w_range = (-wmax, wmax)

outdir = os.path.join(output_folder, 'plotting')
os.makedirs(outdir, exist_ok=True)

# plotting results
for it, t in enumerate(time):
    fig, ax = plt.subplots(2, 3, figsize=(16, 9.5), sharey = True, sharex = True, constrained_layout=True, dpi = 300)
    ax = ax.ravel()
    td = t / 3600 / 24
    fig.suptitle(f'{td:.2f} days', y = 0.99, fontsize=12)
    """
    ax0  # perturbed temperature
    ax1  # temperature
    ax2  # tracer
    ax3  # u velocity
    ax4  # v velocity
    ax5  # w velocity
    """

    im = ax[0].imshow(T_fluc_rz[:, :, it].T, vmin=T_fluc_range[0], vmax=T_fluc_range[1], extent =[r.min(), r.max(), z.min(), z.max()], interpolation ='none', origin ='lower', cmap='RdBu_r')
    ax[0].set_ylabel("Depth [m]")
    ax[0].set_title("Perturbed Temperature")
    ax[0].set_aspect('equal')
    cbar = fig.colorbar(im, ax = ax[0], label=r"$^\circ$C", anchor = (0.5, -0.02), orientation='horizontal', shrink=0.75)

    im = ax[1].imshow(T_rz[:, :, it].T, vmin=T_range[0], vmax=T_range[1], extent =[r.min(), r.max(), z.min(), z.max()], interpolation ='none', origin ='lower')
    ax[1].set_title("Temperature")
    ax[1].set_aspect('equal')
    cbar = fig.colorbar(im, ax = ax[1], label=r"$^\circ$C", anchor = (0.5, -0.02), orientation='horizontal', shrink=0.75)

    im = ax[2].imshow(S_rz[:, :, it].T, extent =[r.min(), r.max(), z.min(), z.max()], interpolation ='none', origin ='lower', norm=colors.LogNorm(vmin=S_range[0], vmax=S_range[1]))
    ax[2].set_title("Tracer")
    ax[2].set_aspect('equal')
    cbar = fig.colorbar(im, ax = ax[2], label=r"g/kg", anchor = (0.5, -0.02), orientation='horizontal', shrink=0.75)

    im = ax[3].imshow(u_rz[:, :, it].T, vmin=u_range[0], vmax=u_range[1], extent =[r.min(), r.max(), z.min(), z.max()], interpolation ='none', origin ='lower', cmap='RdBu_r')
    ax[3].set_ylabel("Depth [m]")
    ax[3].set_xlabel("radial distance [m]")
    ax[3].set_title("u")
    ax[3].set_aspect('equal')
    cbar = fig.colorbar(im, ax = ax[3], label=r"m/s", anchor = (0.5, -0.05), orientation='horizontal', shrink=0.75)
    cbar.formatter.set_powerlimits((-2, 2))
    cbar.update_ticks()

    im = ax[4].imshow(v_rz[:, :, it].T, vmin=v_range[0], vmax=v_range[1], extent =[r.min(), r.max(), z.min(), z.max()], interpolation ='none', origin ='lower', cmap='RdBu_r')
    ax[4].set_xlabel("radial distance [m]")
    ax[4].set_title("v")
    ax[4].set_aspect('equal')
    cbar = fig.colorbar(im, ax = ax[4], label=r"m/s", anchor = (0.5, -0.05), orientation='horizontal', shrink=0.75)
    cbar.formatter.set_powerlimits((-2, 2))
    cbar.update_ticks()

    im = ax[5].imshow(w_rz[:, :, it].T, vmin=w_range[0], vmax=w_range[1], extent =[r.min(), r.max(), z.min(), z.max()], interpolation ='none', origin ='lower', cmap='RdBu_r')
    ax[5].set_xlabel("radial distance [m]")
    ax[5].set_title("w")
    ax[5].set_aspect('equal')
    cbar = fig.colorbar(im, ax = ax[5], label=r"m/s", anchor = (0.5, -0.05), orientation='horizontal', shrink=0.75)
    cbar.formatter.set_powerlimits((-2, 2))
    cbar.update_ticks()
    
    
    # --- Save Frame ---
    frame_path = os.path.join(outdir, f"oc_plane_slices_{it:04d}.png")
    plt.savefig(frame_path)
    plt.close(fig)
    print(f"Time step {it + 1} captured: {frame_path}")

    plt.close()

create_video(outdir, output_folder, '', 'binning_rtz')