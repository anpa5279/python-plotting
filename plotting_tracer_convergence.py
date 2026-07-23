import os

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

def interp_reference(z_case, z_ref, ref_profile):
    return np.interp(z_case, z_ref, ref_profile)

def percent_difference_interp(profile, z_profile, ref_profile, z_ref):
    ref_interp = np.interp(z_profile, z_ref, ref_profile)

    return (100 * (profile - ref_interp) / (np.abs(ref_interp) + 1e-12))

### ------------------------- TRACER CONVERGENCE PLOTTING FUNCTIONS ------------------------- ###
## tracer slice comparison across all cases
def plot_tracer_slice_comparison(time_sec, it, case_names, ranges, y, z, tracer_fields, Sval, fig_folder):
    frame_dir = os.path.join(fig_folder, "tracer_slice_frames")
    os.makedirs(frame_dir, exist_ok = True)
    ncases = len(case_names)

    fig, axes = plt.subplots(1, ncases, figsize = (5 * ncases, 6), constrained_layout = True, sharey = True)

    if ncases == 1:
        axes = [axes]

    levels = [0.005 * Sval, 0.01 * Sval, 0.05 * Sval]
    legend_lines = [Line2D([0], [0], color='white', lw=2, label=r'5% S$_0$'),
                    Line2D([0], [0], color='red', lw=2, label=r'1% S$_0$'),
                    Line2D([0], [0], color='black', lw=2, label=r'0.5% S$_0$')]

    for n in range(ncases):
        S = tracer_fields[n]
        im = axes[n].imshow(S.T, origin = "lower", interpolation = "none", norm = LogNorm(vmin = ranges['Tracer'][0], vmax = ranges['Tracer'][1]), extent = [y[n].min(), y[n].max(), z[n].min(), z[n].max()], aspect = "auto", cmap = "Blues")

        axes[n].contour(y[n], z[n], S.T, levels = levels, colors = ["white", "red", "black"])
        axes[n].set_title(case_names[n])
        axes[n].set_xlabel("y [m]")
        axes[n].set_ylabel("z [m]")
    axes[n].legend(handles=legend_lines,loc='upper right')

    plt.colorbar(im, ax = axes, label = "Tracer")
    fig.suptitle(f"t = {time_sec/3600:.2f} hr")
    fname = os.path.join(frame_dir, f"{it:06d}.png")
    fig.savefig(fname, dpi = 200)
    plt.close(fig)
    return frame_dir

## tracer zoom slice comparison across all cases
def plot_tracer_zoom_comparison(time_sec, it, case_names, ranges, y, z, tracer_fields, Sval, fig_folder, ylim = (-5, 5), zlim = (-10, 0)):

    frame_dir = os.path.join(fig_folder, "tracer_zoom_frames")
    os.makedirs(frame_dir, exist_ok = True)
    ncases = len(case_names)

    fig, axes = plt.subplots(1, ncases, figsize = (5 * ncases, 6), constrained_layout = True, sharey = True)

    if ncases == 1:
        axes = [axes]

    levels = [0.005 * Sval, 0.01 * Sval, 0.05 * Sval]
    legend_lines = [Line2D([0], [0], color='white', lw=2, label=r'5% S$_0$'),
                    Line2D([0], [0], color='red', lw=2, label=r'1% S$_0$'),
                    Line2D([0], [0], color='black', lw=2, label=r'0.5% S$_0$')]
    for n in range(ncases):
        S = tracer_fields[n]
        im = axes[n].imshow(S.T, origin = "lower", interpolation = "none", norm = LogNorm(vmin = ranges['Tracer'][0], vmax = ranges['Tracer'][1]), extent = [y[n].min(), y[n].max(), z[n].min(), z[n].max()], aspect = "auto", cmap = "Blues")

        axes[n].contour(y[n], z[n], S.T, levels = levels, colors = ["white", "red", "black"])

        axes[n].set_xlim(ylim)
        axes[n].set_ylim(zlim)
        axes[n].set_title(case_names[n])
    axes[n].legend(handles=legend_lines,loc='lower right')

    plt.colorbar(im, ax = axes)
    fig.suptitle(f"t = {time_sec/3600:.2f} hr")
    fname = os.path.join(frame_dir, f"{it:06d}.png")
    fig.savefig(fname, dpi = 200)
    plt.close(fig)
    return frame_dir

## turbulent statistics convergence plotting across all cases
def plot_turbulence_convergence(time_sec, it, case_names, ranges, z,u_rms, v_rms, w_rms, bw_fluc, fig_folder,):
    ref_u_rms = u_rms[-1]
    ref_v_rms = v_rms[-1]
    ref_w_rms = w_rms[-1]
    ref_bw = bw_fluc[-1]

    frame_dir = os.path.join(fig_folder, "turbulence_frames")
    os.makedirs(frame_dir, exist_ok = True)

    fig, axes = plt.subplots(2, 4, figsize = (24, 10), sharey = True)

    for i in range(len(case_names)):
        axes[0,0].plot(u_rms[i], z[i], label = case_names[i])
        axes[1,0].plot(percent_difference_interp(u_rms[i], z[i], ref_u_rms, z[-1]), z[i])

        axes[0,1].plot(v_rms[i], z[i])
        axes[1,1].plot(percent_difference_interp(v_rms[i], z[i], ref_v_rms, z[-1]), z[i])

        axes[0,2].plot(w_rms[i], z[i])
        axes[1,2].plot(percent_difference_interp(w_rms[i], z[i], ref_w_rms, z[-1]), z[i])

        axes[0,3].plot(bw_fluc[i], z[i])
        axes[1,3].plot(percent_difference_interp(bw_fluc[i], z[i], ref_bw, z[-1]), z[i])

    axes[0,0].set_title("u RMS")
    axes[0,0].set_ylabel("Depth [m]")
    axes[0,0].set_xlim(ranges['vel_rms'])

    axes[1,0].set_ylabel("Depth [m]")
    axes[1,0].set_title("% Difference u RMS")
    axes[1,0].set_xlim([-200, 200])

    axes[0,1].set_title("v RMS")
    axes[0,1].set_xlim(ranges['vel_rms'])

    axes[1,1].set_title("% Difference v RMS")
    axes[1,1].set_xlim([-200, 200])

    axes[0,2].set_title("w RMS")
    axes[0,2].set_xlim(ranges['vel_rms'])

    axes[1,2].set_title("% Difference w RMS")
    axes[1,2].set_xlim([-200, 200])

    axes[0,3].set_title(r"$\langle$b'w'$\rangle_{\text{xy}}$")
    axes[0,3].set_xlim(ranges['bw_fluc'])

    axes[1,3].set_title("% Difference b'w'")
    axes[1,3].set_xlim([-200, 200])

    axes[0,0].legend()
    for ax in axes.ravel():
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,3), useMathText=True)

    fig.suptitle(f"t = {time_sec/3600:.2f} hr")
    fname = os.path.join(frame_dir, f"{it:06d}.png")

    plt.savefig(fname, dpi = 200)
    plt.close()
    return frame_dir

## tracer convergence plotting across all cases
def plot_salinity_convergence(time_sec, it, case_names, ranges, z,S_avg, S_center, plume_radius, contour, fig_folder):

    frame_dir = os.path.join(fig_folder, "salinity_frames")
    os.makedirs(frame_dir, exist_ok = True)

    fig, axes = plt.subplots(2, 3, figsize = (15,10), sharey = True)

    ref_avg = S_avg[-1]
    ref_center = S_center[-1]
    ref_r = plume_radius[-1]

    for i in range(len(case_names)):
        axes[0,0].plot(S_avg[i], z[i], label = case_names[i])
        axes[0,1].plot(S_center[i], z[i])
        axes[0,2].plot(plume_radius[i], z[i])
        axes[1,0].plot(percent_difference_interp(S_avg[i], z[i], ref_avg, z[-1]), z[i])
        axes[1,1].plot(percent_difference_interp(S_center[i], z[i], ref_center, z[-1]), z[i])
        axes[1,2].plot(percent_difference_interp(plume_radius[i], z[i], ref_r, z[-1]), z[i])

    axes[0,0].legend()
    axes[0,0].set_title(r"$\langle$S$\rangle_{\text{xy}}$")
    axes[0,0].set_ylabel("Depth [m]")
    axes[0,0].set_xlim(ranges['S_avg'])

    axes[0,1].set_title("S(0, 0, z)")
    axes[0,1].set_xlim(ranges['Tracer'])

    axes[0,2].set_title(rf"Plume Radius (contour: S$_0\cdot${contour})")
    axes[0,2].set_xlim(ranges['plume_radius'])

    axes[1,0].set_title(r"% Difference $\langle$S$\rangle_{\text{xy}}$")
    axes[1,0].set_ylabel("Depth [m]")
    axes[1,0].set_xlim([-200, 200])

    axes[1,1].set_title("% Difference S(0, 0, z)")
    axes[1,1].set_xlim([-200, 200])

    axes[1,2].set_title("% Difference Plume Radius")
    axes[1,2].set_xlim([-200, 200])

    for ax in axes.ravel():
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,3), useMathText=True)

    fig.suptitle(f"t = {time_sec/3600:.2f} hr")
    fname = os.path.join(frame_dir, f"{it:06d}.png")

    plt.savefig(fname, dpi = 200)
    plt.close()

    return frame_dir

## temperature convergence plotting across all cases
def plot_temperature_convergence(time_sec, it, case_names, ranges, z,T_avg, T_fluc_center, fig_folder):

    frame_dir = os.path.join(fig_folder, "temperature_frames")
    os.makedirs(frame_dir, exist_ok = True)

    fig, axes = plt.subplots(2, 2, figsize = (12,10), sharey = True)

    ref_avg = T_avg[-1]
    ref_prime = T_fluc_center[-1]

    for i in range(len(case_names)):
        axes[0,0].plot(T_avg[i], z[i], label = case_names[i])
        axes[0,1].plot(T_fluc_center[i], z[i])
        axes[1,0].plot(percent_difference_interp(T_avg[i], z[i], ref_avg, z[-1]), z[i])
        axes[1,1].plot(percent_difference_interp(T_fluc_center[i], z[i], ref_prime, z[-1]), z[i])

    axes[0,0].set_title(r"$\langle$T$\rangle_{\text{xy}}$ [$^{\circ}$C]")
    axes[0,0].set_ylabel("Depth [m]")
    axes[0,0].set_xlim(ranges['T'])

    axes[0,1].set_title(r"T(0, 0, z)-$\langle$T$\rangle_{\text{xy}}$ [$^{\circ}$C]")
    axes[0,1].set_xlim(ranges['T_fluc'])

    axes[1,0].set_title(r"% Difference $\langle$T$\rangle_{\text{xy}}$")
    axes[1,0].set_ylabel("Depth [m]")
    axes[1,0].set_xlim([-200, 200])

    axes[1,1].set_title(r"% Difference T(0, 0, z)-$\langle$T$\rangle_{\text{xy}}$")
    axes[1,1].set_xlim([-200, 200])

    for ax in axes.ravel():
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,3), useMathText=True)

    fig.suptitle(f"t = {time_sec/3600:.2f} hr")
    fname = os.path.join(frame_dir, f"{it:06d}.png")
    plt.savefig(fname, dpi = 200)
    plt.close()
    return frame_dir