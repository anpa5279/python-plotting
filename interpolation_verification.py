import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from reader import OceananigansData
from plotting_general import plot_format, create_video, comparison_plot_opt
from interpolation import velocities_to_center, point

video = True

# flags for how to read data
with_halos = False
closure = False
salinity = True
stokes = False

folder = '/Users/annapauls/Documents/TESLa /Simulations/Oceananigans/dense plume/salinity and temperature/no noise circle inlet/domain testing/Lz = 160m/S0 = 0.1 dTdz = 0.01 MLD = 60'
fig_folder = os.path.join(folder, 'interpolation_verification_figures')
reader = OceananigansData(folder, salinity = salinity)

# collecting model information for all cases
reader.load_time()
reader.load_grid()
x = reader.x
y = reader.y
z = reader.z
zf = reader.zf
nx = reader.nx
lx = reader.lx
nt = reader.nt
i_opt = [nx[0]//2 - 1, nx[0]//2]
j_opt = [nx[1]//2 - 1, nx[1]//2]
idx, idy = np.meshgrid(i_opt, j_opt, indexing = 'ij')
# video or not setup
if video:
    time = reader.time
else:
    time = reader.time[-1]

nvars = 4
# create necessary 3D fields
w = reader.lazy_field('w').compute()
# grab desired grid locations 
w = w[:, idx, idy, :]
w_center_function = velocities_to_center(w, -1)
w_averaging = (w[:, :, :, :-1] + w[:, :, :, 1:])/2
w_interpolation = np.empty_like(w_center_function)
for k in range(reader.nx[2]):
    for i in range(len(i_opt)):
        for j in range(len(j_opt)):
            w_interpolation[:, i, j, k] = point(w[:, i, j, :], zf, z0 = z[k])
############ PLOTTING ############
"""
    Plotting throughout time...
    columns: the 4 center grid cells around the center
    interpolated w averaged between with grid cells vs interpolated w using velocities_to_center vs interpolated w using interpolation.py
"""
color_opt, line_opt  = comparison_plot_opt(nvars)
plot_format()
os.makedirs(fig_folder, exist_ok=True)
ncols = idx.size
gridspec_kw={'height_ratios': [1, 1, 0.15]}
width = 0.8
labels = ['raw output', 'class function', 'averaging', 'point interpolation function']
case_handles = [Line2D([0], [0], color=color_opt[i], linestyle='solid', linewidth=width, label=labels[i]) for i in range(nvars)]
for it in range(nt):
    fig, axes = plt.subplots(3, ncols, figsize=(16, 8.5), sharey = True, gridspec_kw=gridspec_kw)
    for ax in axes[-1, :]:
        ax.remove()
    for ax in axes[0, :]:
        ax.set_ylim(-reader.lx[-1], 0)
        ax.set_xlim(-0.15, 0.15)
        ax.set_ylabel("z [m]")
        ax.set_xlabel("w [m/s]")
    for ax in axes[1, :]:
        ax.set_ylim(-reader.lx[-1], 0)
        ax.set_xlim(-0.15/10, 0.15/10)
        ax.set_ylabel("z [m]")
        ax.set_xlabel(r"(w$-$w$_{raw}$) [m/s]")
    axes = axes.ravel()
    fig.suptitle(f"Time = {time[it]/3600/24:.2f} days", fontsize=12)
    fig.legend(handles=case_handles,
            loc='lower center',
            ncol=nvars,
            bbox_to_anchor=(0.52, 0.005))
    count = 0
    for i, ix in enumerate(i_opt):
        for j, jy in enumerate(j_opt):
            axes[i + j + count].set_title(f"w at ({ix}, {jy})")
            axes[i + j + count].plot(w[it, i, j, :], zf, label = labels[0], color = color_opt[0], linestyle = line_opt[0])
            axes[i + j + count].plot(w_center_function[it, i, j, :], z, label = labels[1], color = color_opt[1], linestyle = line_opt[1])
            axes[i + j + count].plot(w_averaging[it, i, j, :], z, label = labels[2], color = color_opt[2], linestyle = line_opt[1])
            axes[i + j + count].plot(w_interpolation[it, i, j, :], z, label = labels[3], color = color_opt[3], linestyle = line_opt[1])
        count += 1
    labels_diff = ['manipulated vs w[:-1]', 'mainpulated vs w[1:]']
    count += ix.size + jy.size
    for i, ix in enumerate(i_opt):
        for j, jy in enumerate(j_opt):
            center_diff = (w_center_function[it, i, j, :] - w[it, i, j, :-1])
            avg_diff = (w_averaging[it, i, j, :] - w[it, i, j, :-1])
            interp_diff = (w_interpolation[it, i, j, :] - w[it, i, j, :-1])
            center_diff1 = (w_center_function[it, i, j, :] - w[it, i, j, 1:])
            avg_diff1 = (w_averaging[it, i, j, :] - w[it, i, j, 1:])
            interp_diff1 = (w_interpolation[it, i, j, :] - w[it, i, j, 1:])
            axes[i + j + count].set_title(f"Difference at ({ix}, {jy})")
            axes[i + j + count].plot(center_diff, z, label = labels_diff[0], color = color_opt[0], linestyle = line_opt[2])
            axes[i + j + count].plot(center_diff1, z, label = labels_diff[1], color = color_opt[0], linestyle = line_opt[3])

            axes[i + j + count].plot(center_diff, z, color = color_opt[1], linestyle = line_opt[2])
            axes[i + j + count].plot(avg_diff, z, color = color_opt[2], linestyle = line_opt[2])
            axes[i + j + count].plot(interp_diff, z, color = color_opt[3], linestyle = line_opt[2])
            axes[i + j + count].plot(center_diff1, z, color = color_opt[1], linestyle = line_opt[3])
            axes[i + j + count].plot(avg_diff1, z, color = color_opt[2], linestyle = line_opt[3])
            axes[i + j + count].plot(interp_diff1, z, color = color_opt[3], linestyle = line_opt[3])
            axes[i + j + count].legend(loc = 'lower right')
        count += 1
    # --- Save Frame ---
    frame_path = os.path.join(fig_folder, f"interpolation_testing_{it:04d}.png")
    plt.savefig(frame_path)
    plt.close(fig)
# creating videos
if video:
    create_video(fig_folder, folder, '', 'verifying interpolation')