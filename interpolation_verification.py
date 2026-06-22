import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from reader import OceananigansData
from plotting_general import plot_format, create_video, comparison_plot_opt
from interpolation import velocities_to_center, point

video = True
vertical_verification = False
horizontal_verification = True

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
nx = reader.nx
lx = reader.lx
nt = reader.nt
# video or not setup
if video:
    time = reader.t
else:
    time = reader.t[-1]

nvars = 3
if vertical_verification:
    fig_folder_vert = os.path.join(fig_folder, 'vertical_line')
    zf = reader.zf
    # create necessary 3D fields
    w = reader.lazy_field('w').compute()
    # grab desired grid locations 
    i_opt = [nx[0]//2 - 1, nx[0]//2]
    j_opt = [nx[1]//2 - 1, nx[1]//2]
    idx, idy = np.meshgrid(i_opt, j_opt, indexing = 'ij')
    w = w[:, idx, idy, :]
    w_center_function = velocities_to_center(w, -1)
    w_averaging = (w[:, :, :, :-1] + w[:, :, :, 1:])/2
    w_interpolation = np.empty_like(w_center_function)
    for k in range(reader.nx[2]):
        for i in range(len(i_opt)):
            for j in range(len(j_opt)):
                w_interpolation[:, i, j, k] = point(w[:, i, j, :], zf, z0 = z[k])
if horizontal_verification:
    fig_folder_horiz = os.path.join(fig_folder, 'horizontal_line')
    hml = 60
    hml_opt = np.argmin(np.abs(z + hml)) # approximate since we what to compare interpolation to raw output
    hml_opt = [hml_opt, hml_opt + 1]
    hor_opt = [nx[1]//2, nx[1]//2 + 1]
    hor_idx, hml_idx = np.meshgrid(hor_opt, hml_opt, indexing = 'ij')
    xf = reader.xf
    yf = reader.yf
    u = reader.lazy_field('u').compute()
    v = reader.lazy_field('v').compute()
    u_center_function = velocities_to_center(u, -3)
    v_center_function = velocities_to_center(v, -2)
    u = np.concatenate([u, np.take(u, [0], axis=-3)], axis=-3)
    v = np.concatenate([v, np.take(v, [0], axis=-2)], axis=-2)
    u = u[:, :, hor_idx, hml_idx]
    v = v[:, hor_idx, :, hml_idx]
    u_averaging = (u[:, :-1, :, :] + u[:, 1:, :, :])/2
    v_averaging = (v[:, :, :-1, :] + v[:, :, 1:, :])/2
    """
    u_interpolation = np.empty_like(u_center_function)
    v_interpolation = np.empty_like(v_center_function)
    for k in range(len(hml_opt)):
        for i in range(nx[0]):
            for j in range(len(hor_opt)):
                u_interpolation[:, i, j, k] = point(u[:, :, j, k], xf, x0 = x[i])
        for i in range(len(hor_opt)):
            for j in range(nx[1]):
                v_interpolation[:, i, j, k] = point(v[:, i, :, k], yf, y0 = y[j])
    """
############ PLOTTING ############
"""
    Plotting throughout time...
    columns: the 4 center grid cells around the center
    interpolated w averaged between with grid cells vs interpolated w using velocities_to_center vs interpolated w using interpolation.py
"""
color_opt, line_opt  = comparison_plot_opt(nvars)
plot_format()
os.makedirs(fig_folder, exist_ok=True)
gridspec_kw={'height_ratios': [1, 1, 0.15]}
width = 0.8
labels = ['raw output', 'class function', 'averaging']#, 'point interpolation function']
case_handles = [Line2D([0], [0], color=color_opt[i], linestyle='solid', linewidth=width, label=labels[i]) for i in range(nvars)]
if vertical_verification:
    ncols = idx.size
    os.makedirs(fig_folder_vert, exist_ok=True)
    for it in range(nt):
        fig, axes = plt.subplots(2, ncols, figsize=(16, 5), sharey = True, gridspec_kw=gridspec_kw)
        for ax in axes[-1, :]:
            ax.remove()
        for ax in axes[0, :]:
            ax.set_ylim(-reader.lx[-1], 0)
            ax.set_xlim(-0.15, 0.15)
            ax.set_ylabel("z [m]")
            ax.set_xlabel("w [m/s]")
        axes = axes.ravel()
        fig.suptitle(f"Time = {time[it]/3600/24:.2f} days", fontsize=12)
        fig.legend(handles=case_handles,
                loc='lower center',
                ncol=nvars,
                bbox_to_anchor=(0.52, 0.005))
        count = 0
        for i, ix in enumerate(i_opt):
            for j, jy in enumerate(j_opt):
                axes[i + j + count].set_title(f"w at (Nx = {ix}, Ny = {jy})")
                axes[i + j + count].plot(w[it, i, j, :], zf, label = labels[0], color = color_opt[0], linestyle = line_opt[0])
                axes[i + j + count].plot(w_center_function[it, i, j, :], z, label = labels[1], color = color_opt[1], linestyle = line_opt[1])
                axes[i + j + count].plot(w_averaging[it, i, j, :], z, label = labels[2], color = color_opt[2], linestyle = line_opt[1])
                axes[i + j + count].plot(w_interpolation[it, i, j, :], z, label = labels[3], color = color_opt[3], linestyle = line_opt[1])
            count += 1

        # --- Save Frame ---
        frame_path = os.path.join(fig_folder_vert, f"interpolation_testing_{it:04d}.png")
        plt.savefig(frame_path)
        plt.close(fig)
    # creating videos
    if video:
        create_video(fig_folder_vert, folder, '', 'verifying interpolation')
if horizontal_verification:
    ncols = hor_idx.size 
    for it in range(nt):
        fig, axes = plt.subplots(3, ncols, figsize=(16, 8.5), sharey = True, gridspec_kw=gridspec_kw)
        for ax in axes[-1, :]:
            ax.remove()
        for ax in axes[0, :]:
            ax.set_xlim(-reader.lx[-1], 0)
            ax.set_ylim(-0.15, 0.15)
            ax.set_xlabel("x [m]")
            ax.set_ylabel("u [m/s]")
        for ax in axes[1, :]:
            ax.set_xlim(-reader.lx[-1], 0)
            ax.set_ylim(-0.15/10, 0.15/10)
            ax.set_xlabel("y [m]")
            ax.set_ylabel("v [m/s]")
        axes = axes.ravel()
        fig.suptitle(f"Time = {time[it]/3600/24:.2f} days", fontsize=12)
        fig.legend(handles=case_handles,
                loc='lower center',
                ncol=nvars,
                bbox_to_anchor=(0.52, 0.005))
        count = 0
        for j, jy in enumerate(hor_opt):
            for k, kz in enumerate(hml_opt):
                axes[j + k + count].set_title(f"u at (Ny = {jy}, Nz = {kz})")
                axes[j + k + count].plot(xf, u[it, :, j, k], label = labels[0], color = color_opt[0], linestyle = line_opt[0])
                axes[j + k + count].plot(x, u_center_function[it, :, j, k], label = labels[1], color = color_opt[1], linestyle = line_opt[1])
                axes[j + k + count].plot(x, u_averaging[it, :, j, k], label = labels[2], color = color_opt[2], linestyle = line_opt[1])
                #axes[j + k + count].plot(x, u_interpolation[it, :, j, k], label = labels[3], color = color_opt[3], linestyle = line_opt[1])
            count += 1
        count += ix.size + jy.size
        for i, ix in enumerate(hor_opt):
            for k, kz in enumerate(hml_opt):
                axes[i + k + count].set_title(f"v at (Nx = {ix}, Nz = {kz})")
                axes[i + k + count].plot(yf, v[it, i, :, k], label = labels[0], color = color_opt[0], linestyle = line_opt[0])
                axes[i + k + count].plot(y, v_center_function[it, i, :, k], label = labels[1], color = color_opt[1], linestyle = line_opt[1])
                axes[i + k + count].plot(y, v_averaging[it, i, :, k], label = labels[2], color = color_opt[2], linestyle = line_opt[1])
                #axes[i + k + count].plot(y, v_interpolation[it, i, :, k], label = labels[3], color = color_opt[3], linestyle = line_opt[1])
            count += 1
        # --- Save Frame ---
        frame_path = os.path.join(fig_folder_horiz, f"interpolation_testing_{it:04d}.png")
        plt.savefig(frame_path)
        plt.close(fig)
    # creating videos
    if video:
        create_video(fig_folder_horiz, folder, '', 'verifying horizontal interpolation')