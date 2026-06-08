import os
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from reader import OceananigansData
from plotting_general import plot_format, create_video, comparison_plot_opt
from interpolation import velocities_to_center, vertical_line
from physics import buoyancy
"""
    what is the best way to find the maximum penetration depth of a plume via momentum?
    ways to consider w:
        1. w_avg = 1D array of shape (nt, nz) --> average w at each depth and time step
        2. w centerline = 1D array of shape (nt, nz) --> w at (0.0, 0.0) and time step
        3. w_fluc_avg = 1D array of shape (nt, nz) --> average w' at each depth and time step
        4. w_fluc_centerline = 1D array of shape (nt, nz) --> w' at (0.0, 0.0) and time step

    ways to consider buoyancy, b:
        1. b_fluc_avg = 1D array of shape (nt, nz) --> average b' at each depth and time step
        2. b_fluc_centerline = 1D array of shape (nt, nz) --> b' at (0.0, 0.0) and time step

    ways to percieve w and/or b:
        1. sign changes
        2. order of magnitude changes
        3. gradient changes
"""
# plotting flags
plot_variables = True
plot_depth = False
video = True

# flags for how to read data
with_halos = False
closure = False
salinity = True
stokes = False

folder = '/Users/annapauls/Documents/TESLa /Simulations/Oceananigans/dense plume/salinity and temperature/no noise circle inlet/domain testing/Lz = 160m/S0 = 0.1 dTdz = 0.01 MLD = 60'
outdir = os.path.join(folder, 'max penetration algorithm testing figures')
reader = OceananigansData(folder, salinity = salinity)

if plot_variables:
    fig_var_folder = os.path.join(folder, 'max penetration variables')
    os.makedirs(fig_var_folder, exist_ok=True)
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
# video or not setup
if video:
    time = reader.time
else:
    time = reader.time[-1]

# create necessary 3D fields
w = reader.lazy_field('w').compute()
w = velocities_to_center(w, -1)
b = buoyancy(reader, type = 'field').compute()

# calculating statistics
w_avg = np.mean(w, axis = (-3, -2))
w_fluc = w - w_avg[:, None, None, :]
w_fluc_avg = np.mean(w_fluc, axis = (-3, -2))

b_avg = np.mean(b, axis = (-3, -2))
b_fluc = b - b_avg[:, None, None, :]
b_fluc_avg = np.mean(b_fluc, axis = (-3, -2))

# finding centerlines
w_centerline = vertical_line(w, x, y)
w_fluc_centerline = vertical_line(w_fluc, x, y)

b_fluc_centerline = vertical_line(b_fluc, x, y)

# calculating gradients
dwdz = np.gradient(w_avg, z, axis = -1)
dwdz_centerline = np.gradient(w_centerline, z, axis = -1)
dwflucdz = np.gradient(w_fluc_avg, z, axis = -1)
dwflucdz_centerline = np.gradient(w_fluc_centerline, z, axis = -1)

dbflucdz = np.gradient(b_fluc_avg, z, axis = -1)
dbflucdz_centerline = np.gradient(b_fluc_centerline, z, axis = -1)

# calculating order of magnitude
w_centerline_mag = math.floor(math.log10(np.abs(w_centerline)))
w_avg_mag = math.floor(math.log10(np.abs(w_avg)))

w_fluc_centerline_mag = math.floor(math.log10(np.abs(w_fluc_centerline)))
w_fluc_avg_mag = math.floor(math.log10(np.abs(w_fluc_avg)))

b_fluc_centerline_mag = math.floor(math.log10(np.abs(b_fluc_centerline)))
b_fluc_avg_mag = math.floor(math.log10(np.abs(b_fluc_avg)))

# finding sign change of w
w_avg_sign_change = np.diff(np.sign(w_avg), axis = -1)
w_centerline_sign_change = np.diff(np.sign(w_centerline), axis = -1)
w_fluc_avg_sign_change = np.diff(np.sign(w_fluc_avg), axis = -1)
w_fluc_centerline_sign_change = np.diff(np.sign(w_fluc_centerline), axis = -1)

b_fluc_avg_sign_change = np.diff(np.sign(b_fluc_avg), axis = -1)
b_fluc_centerline_sign_change = np.diff(np.sign(b_fluc_centerline), axis = -1)
############ PLOTTING ############
"""
    Plotting throughout time...
    [0, 0]: w_avg, w centerline
    [0, 1]: w_fluc_avg, w flucuation centerline
    [0, 2]: dw/dz (w_avg, w centerline, w_fluc_avg, w_fluc_centerline)
    [1, 0]: b_fluc_avg, b_fluc_centerline
    [1, 1]: db/dz (b_fluc_avg, b_fluc_centerline)
    [1, 2]: sign of variables (w_avg, w centerline, w_fluc_avg, w_fluc_centerline, b_fluc_avg, b_fluc_centerline)
    [1, 3]: order of magnitude of variables (w_avg, w centerline, w_fluc_avg, w_fluc_centerline, b_fluc_avg, b_fluc_centerline)

"""
nvars = 6
color_opt, line_opt  = comparison_plot_opt(nvars)
plot_format()
os.makedirs(outdir, exist_ok=True)

gridspec_kw={'height_ratios': [1, 1, 0.15]}
width = 0.8
labels = [r'$\langle \text{w} \rangle_{xy}$', r'$\text{w}_{(0, 0)}$', r"$\langle \text{w'} \rangle_{xy}$", r"$\text{w'}_{0, 0}$", r"$\langle \text{b'} \rangle_{xy}$", r"$\text{b'}_{0, 0}$"]
case_handles = [Line2D([0], [0], color=color_opt[i], linestyle='solid', linewidth=width, label=labels[i]) for i in range(nvars)]
if plot_variables:
    for it in range(nt):
        fig = plt.figure(figsize=(16, 9))#, gridspec_kw=gridspec_kw)
        ax0 = plt.subplot2grid(shape = (2, 2), loc = (0, 0), rowspan = 2, colspan=2) # w
        ax1 = plt.subplot2grid(shape = (2, 2), loc = (0, 2), rowspan = 2, colspan=2) # w'
        ax2 = plt.subplot2grid(shape = (2, 2), loc = (0, 4), rowspan = 2, colspan=2) # dw/dz
        ax3 = plt.subplot2grid(shape = (2, 2), loc = (2, 0), rowspan = 2, colspan=2) # b'
        ax4 = plt.subplot2grid(shape = (2, 2), loc = (2, 2), rowspan = 2, colspan=2) # db'/dz
        ax5 = plt.subplot2grid(shape = (1, 2), loc = (2, 3), colspan=2) # sign
        ax6 = plt.subplot2grid(shape = (1, 2), loc = (2, 2), colspan=2) # magnitude

        ax7 = plt.subplot2grid(shape = (1, 2), loc = (4, 0)) 
        ax7.remove()

        ax0.plot(w_avg[it, :], z, color = color_opt[0], linewidth = width)
        ax0.plot(w_centerline[it, :], z, color = color_opt[1], linewidth = width)
        ax0.set_title("w")
        ax0.set_xlabel("w [m/s]")
        ax0.set_ylabel("z [m]")

        ax1.plot(w_fluc_avg[it, :], z, color = color_opt[2], linewidth = width)
        ax1.plot(w_fluc_centerline[it, :], z, color = color_opt[3], linewidth = width)
        ax1.set_title("w'")
        ax1.set_xlabel("w' [m/s]")
        ax1.set_ylabel("z [m]")

        ax2.plot(dwdz[it, :], z, color = color_opt[0], linewidth = width)
        ax2.plot(dwdz_centerline[it, :], z, color = color_opt[1], linewidth = width)
        ax2.plot(dwflucdz[it, :], z, color = color_opt[2], linewidth = width)
        ax2.plot(dwflucdz_centerline[it, :], z, color = color_opt[3], linewidth = width)
        ax2.set_title("dw/dz")
        ax2.set_xlabel("dw/dz [1/s]")
        ax2.set_ylabel("z [m]")

        ax3.plot(b_fluc_avg[it, :], z, color = color_opt[4], linewidth = width)
        ax3.plot(b_fluc_centerline[it, :], z, color = color_opt[5], linewidth = width)
        ax3.set_title("b'")
        ax3.set_xlabel("b' [m/s^2]")
        ax3.set_ylabel("z [m]")

        ax4.plot(dbflucdz[it, :], z, color = color_opt[4], linewidth = width)
        ax4.plot(dbflucdz_centerline[it, :], z, color = color_opt[5], linewidth = width)
        ax4.set_title("db'/dz")
        ax4.set_xlabel("db'/dz [1/s^2]")
        ax4.set_ylabel("z [m]")

        ax5.plot(np.sign(w_avg[it, :]), z, color = color_opt[0], linewidth = width)
        ax5.plot(np.sign(w_centerline[it, :]), z, color = color_opt[1], linewidth = width)
        ax5.plot(np.sign(w_fluc_avg[it, :]), z, color = color_opt[2], linewidth = width)
        ax5.plot(np.sign(w_fluc_centerline[it, :]), z, color = color_opt[3], linewidth = width)
        ax5.plot(np.sign(b_fluc_avg[it, :]), z, color = color_opt[4], linewidth = width)
        ax5.plot(np.sign(b_fluc_centerline[it, :]), z, color = color_opt[5], linewidth = width)
        ax5.set_title("Sign of variables")
        ax5.set_xlabel("Sign")
        ax5.set_ylabel("z [m]")

        ax6.plot(w_avg_mag[it, :], z, color = color_opt[0], linewidth = width)
        ax6.plot(w_centerline_mag[it, :], z, color = color_opt[1], linewidth = width)
        ax6.plot(w_fluc_avg_mag[it, :], z, color = color_opt[2], linewidth = width)
        ax6.plot(w_fluc_centerline_mag[it, :], z, color = color_opt[3], linewidth = width)
        ax6.plot(b_fluc_avg_mag[it, :], z, color = color_opt[4], linewidth = width)
        ax6.plot(b_fluc_centerline_mag[it, :], z, color = color_opt[5], linewidth = width)
        ax6.set_title("Order of magnitude of variables")
        ax6.set_xlabel("Order of magnitude")
        ax6.set_ylabel("z [m]")
        fig.legend(handles=case_handles,
                loc='lower center',
                ncol=nvars,
                bbox_to_anchor=(0.52, 0.005))
        # --- Save Frame ---
        frame_path = os.path.join(fig_var_folder, f"variables_of_interest_{it:04d}.png")
        plt.savefig(frame_path)
        plt.close(fig)
# creating videos
if video:
    create_video(fig_var_folder, outdir, '', 'max penetration variables')