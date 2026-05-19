import os
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import imageio.v2 as imageio
import matplotlib.ticker as mticker

from matplotlib.lines import Line2D
from matplotlib import cm
from matplotlib import colors
from fractions import Fraction
### -------------------------PLOTTING 1D LINES FUNCTIONS------------------------- ###
## temporal analysis ###
def plume_temporal_analysis(time, ranges, color_opt, fig_folder, case_names, name, lx, start_neutral, mld, h_neutral, h_max, r_mld, r_neutral, r_hmax, w_mld, w_neutral, w_hmax, b_mld, b_neutral, b_hmax, T_mld, T_neutral, T_hmax, tracer_mld, tracer_neutral, tracer_hmax, tracerw_fluc_avg, Tw_fluc_avg, ND = False):
    num_cases = len(case_names)
    if num_cases==1:
        fig, ax = plt.subplots(2, 4, figsize=(12, 5))
        outdir = os.path.join(fig_folder, 'plume analysis/')
        os.makedirs(outdir, exist_ok=True)
    else:
        outdir = os.path.join(fig_folder, 'comparison plume analysis/')
        os.makedirs(outdir, exist_ok=True)
        gridspec_kw={'height_ratios': [1, 1, 0.1]} # add space for universal legend
        fig, ax = plt.subplots(3, 4, figsize=(12, 6.5), gridspec_kw=gridspec_kw)
        for a in ax[2, :]:
            a.remove()
        case_handles = [
            Line2D([0], [0], color=color_opt[i], linestyle='solid', label=case_names[i])
            for i in range(num_cases)
        ]
        fig.legend(handles=case_handles,
                loc='lower center',
                ncol=num_cases,
                bbox_to_anchor=(0.52, 0.015))
    #fig.tight_layout()
    ax1 = ax[0, 0] # depth of plume through time 
    ax2 = ax[0, 1] # max and average radius of plume through time 
    ax3 = ax[0, 2] # vertical velocity at depth through time
    ax4 = ax[0, 3] # perturbed buoyancy at depth through time
    ax5 = ax[1, 0] # perturbed Temperature at depth through time
    ax6 = ax[1, 1] # perturbed tracer at depth through time
    ax7 = ax[1, 2] # average tracer at MLD through time
    ax8 = ax[1, 3] # w_avg at MLD through time 
    if ND:
        ax1.set_ylabel(r"z/h$_{\text{MLD}}$")
        ax1.set_ylim(ymin = -lx[-1], ymax = 0)
        ax2.set_ylabel(r"$\langle$r$\rangle_{\text{xy}}$/l$_{j}$") #(r"$\langle$r$\rangle_{\text{xy}}$/h$_{\text{MLD}}$") #
        ax2.set_ylim(ymin = ranges['radius'][0], ymax = ranges['radius'][-1])
        ax3.set_ylabel(r"w/(h$_{\mathrm{MLD}_0} \sqrt{N^{2}})$")
        ax3.set_ylim(ymin = ranges['w'][0], ymax = ranges['w'][-1])
        ax4.set_ylabel(r"b'/(h$_{\mathrm{MLD}_0} N^{2}$)")
        ax4.set_ylim(ymin = ranges['b_fluc'][0], ymax = ranges['b_fluc'][-1])
        ax5.set_ylabel(r"$\langle$T$'\rangle_{\text{xy}}$/T$_{0}$")
        ax5.set_ylim(ymin = ranges['T_fluc'][0], ymax = ranges['T_fluc'][-1])
        ax6.set_ylabel(r"$\langle$C'$\rangle_{\text{xy}}$/S$_{\text{max}}$") #(\text{h}_{mld} \sqrt{N^{2}}$)/(F$_{\text{C}}$)") #(r"$\langle$C$'\sqrt{g\text{r}_{j}}$/(F$_{\text{C}}$)") #
        ax6.set_ylim(ymin = ranges['Tracer_fluc'][0], ymax = ranges['Tracer_fluc'][-1])
        ax7.set_ylabel(r"$\langle$C'w$\rangle_{\text{xy}}$/F$_{\text{C}}$") #(\text{h}_{mld}\sqrt{N^{2}}$)/(F$_{\text{C}}$)")# (r"$\langle$C$\rangle_{\text{xy}}$\sqrt{g\text{r}_{j}}$/(F$_{\text{C}}$)")#
        ax7.set_ylim(ymin = ranges['Sw_fluc'][0], ymax = ranges['Sw_fluc'][-1])
        ax8.set_ylabel(r"$\langle$T$'$w$\rangle_{\text{xy}}$/(h$_{\mathrm{MLD}_0} \sqrt{N^{2}}$)")
        ax8.set_ylim(ymin = ranges['Tw_fluc'][0], ymax = ranges['Tw_fluc'][-1])
    else:
        ax1.set_ylabel("[m]")
        ax1.set_ylim(ymin = -lx[-1], ymax = 0)
        ax2.set_ylabel("[m]")
        ax2.set_ylim(ymin = ranges['radius'][0], ymax = ranges['radius'][-1])
        ax3.set_ylabel("[m/s]")
        ax3.set_ylim(ymin = ranges['w'][0], ymax = ranges['w'][-1])
        ax4.set_ylabel(r"[m/s$^2$]")
        ax4.set_ylim(ymin = ranges['b_fluc'][0], ymax = ranges['b_fluc'][-1])
        ax5.set_ylabel(r"$\langle$T$'\rangle_{\text{xy}}$ [$^{\circ}$C]")
        ax5.set_ylim(ymin = ranges['T_fluc'][0], ymax = ranges['T_fluc'][-1])
        ax6.set_ylabel(r"$\langle$C$'\rangle_{\text{xy}}$ [g/kg]")
        ax6.set_ylim(ymin = ranges['Tracer_fluc'][0], ymax = ranges['Tracer_fluc'][-1])
        ax7.set_ylabel(r"$\langle$C$'\text{w}\rangle_{\text{xy}}$ [g/kg]")
        ax7.set_ylim(ymin = ranges['Sw_fluc'][0], ymax = ranges['Sw_fluc'][-1])
        ax8.set_ylabel(r"$\langle$T$'$w$\rangle_{\text{xy}}$ [$^{\circ}$C $\cdot$ m/s]")
        ax8.set_ylim(ymin = ranges['Tw_fluc'][0], ymax = ranges['Tw_fluc'][-1])
    # Depth of plume through time 
    for i in range(num_cases):
        if i == 0:
            ax1.plot(time/ 3600 / 24, -mld[:, i], label = r"h$_{\text{MLD}}$", linewidth = 0.75, linestyle = 'dashed', color = color_opt[i])
            ax1.plot(time[start_neutral[i]::]/ 3600 / 24, h_neutral[start_neutral[i]::, i], label = r"h$_{\text{neutral}}$", linewidth = 0.75, linestyle = 'dotted', color = color_opt[i])
            ax1.plot(time/ 3600 / 24, h_max[:, i], label = r"h$_{\text{intrusion}}$", linewidth = 0.75, linestyle = 'solid', color = color_opt[i])
        else: 
            ax1.plot(time/ 3600 / 24, -mld[:, i], linewidth = 0.75, linestyle = 'dashed', color = color_opt[i])
            ax1.plot(time/ 3600 / 24, h_max[:, i], linewidth = 0.75, linestyle = 'solid', color = color_opt[i])
            ax1.plot(time[start_neutral[i]::]/ 3600 / 24, h_neutral[start_neutral[i]::, i], linewidth = 0.75, linestyle = 'dotted', color = color_opt[i])
    ax1.set_title("Plume Depths", size = 10)
    #ax1.set_xlabel("Time [days]") 
    ax1.legend(loc='lower right', labelspacing = 0.25, handlelength=0.75)
    ax1.set_xlim([time.min() / 3600 / 24, time.max() / 3600 / 24])
    # radius of plume 
    for i in range(num_cases):
        if i == 0:
            ax2.plot(time/ 3600 / 24, r_mld[:, i], label = r"r$_{\text{MLD}}$", linewidth = 0.75, linestyle = 'dashed', color = color_opt[i])
            ax2.plot(time[start_neutral[i]::]/ 3600 / 24, r_neutral[start_neutral[i]::, i], label = r"r$_{\text{neutral}}$", linewidth = 0.75, linestyle = 'dotted', color = color_opt[i])
            ax2.plot(time/ 3600 / 24, r_hmax[:, i], label = r"r$_{\text{intrusion}}$", linewidth = 0.75, linestyle = 'solid', color = color_opt[i])
        else: 
            ax2.plot(time/ 3600 / 24, r_mld[:, i], linewidth = 0.75, linestyle = 'dashed', color = color_opt[i])
            ax2.plot(time[start_neutral[i]::]/ 3600 / 24, r_neutral[start_neutral[i]::, i], linewidth = 0.75, linestyle = 'dotted', color = color_opt[i])
            ax2.plot(time/ 3600 / 24, r_hmax[:, i], linewidth = 0.75, linestyle = 'solid', color = color_opt[i])
    ax2.set_xlim([time.min() / 3600 / 24, time.max() / 3600 / 24])
    ax2.set_title("Plume Radii", size = 10)
    #ax2.set_xlabel("Time [days]") 
    ax2.legend(loc='upper left', labelspacing = 0.25, handlelength=0.75)
    # vertical velocity
    for i in range(num_cases):
        if i == 0:
            ax3.plot(time/ 3600 / 24, w_mld[:, i], label = r"w$_{\text{MLD}}$", linewidth = 0.75, linestyle = 'dashed', color = color_opt[i])
            ax3.plot(time[start_neutral[i]::]/ 3600 / 24, w_neutral[start_neutral[i]::, i], label = r"w$_{\text{neutral}}$", linewidth = 0.75, linestyle = 'dotted', color = color_opt[i])
            ax3.plot(time/ 3600 / 24, w_hmax[:, i],label = r"w$_{\text{intrusion}}$", linewidth = 0.75, linestyle = 'solid', color = color_opt[i])
        else:
            ax3.plot(time/ 3600 / 24, w_mld[:, i], linewidth = 0.75, linestyle = 'dashed', color = color_opt[i])
            ax3.plot(time[start_neutral[i]::]/ 3600 / 24, w_neutral[start_neutral[i]::, i], linewidth = 0.75, linestyle = 'dotted', color = color_opt[i])
            ax3.plot(time/ 3600 / 24, w_hmax[:, i], linewidth = 0.75, linestyle = 'solid', color = color_opt[i])
    #ax3.set_xlabel("Time [days]") 
    ax3.set_title("Vertical Velocity", size = 10)
    ax3.legend(loc='upper right', labelspacing = 0.25, handlelength=0.75)
    ax3.set_xlim([time.min() / 3600 / 24, time.max() / 3600 / 24]) 
    # buoyancy perturbations 
    for i in range(num_cases):
        if i == 0:
            ax4.plot(time/ 3600 / 24, b_mld[:, i], label = r"b$'_{\text{MLD}}$", linewidth = 0.75, linestyle = 'dashed', color = color_opt[i])
            ax4.plot(time[start_neutral[i]::]/ 3600 / 24, b_neutral[start_neutral[i]::, i], label = r"b$'_{\text{neutral}}$", linewidth = 0.75, linestyle = 'dotted', color = color_opt[i])
            ax4.plot(time/ 3600 / 24, b_hmax[:, i], label = r"b$'_{\text{intrusion}}$", linewidth = 0.75, linestyle = 'solid', color = color_opt[i])
        else:
            ax4.plot(time/ 3600 / 24, b_mld[:, i], linewidth = 0.75, linestyle = 'dashed', color = color_opt[i])
            ax4.plot(time/ 3600 / 24, b_hmax[:, i], linewidth = 0.75, linestyle = 'solid', color = color_opt[i])
            ax4.plot(time[start_neutral[i]::]/ 3600 / 24, b_neutral[start_neutral[i]::, i], linewidth = 0.75, linestyle = 'dotted', color = color_opt[i])
    #ax4.set_xlabel("Time [days]") 
    ax4.set_title("Perturbed Buoyancy", size = 10)
    ax4.legend(loc='upper right', labelspacing = 0.25, handlelength=0.75)
    ax4.set_xlim([time.min() / 3600 / 24, time.max() / 3600 / 24])
    ax4.ticklabel_format(axis='y', style='sci', scilimits=(-3,2), useMathText=True)
    # temperature perturbations 
    for i in range(num_cases):
        if i == 0:
            ax5.plot(time/ 3600 / 24, T_mld[:, i], label = r"T$'_{\text{MLD}}$", linewidth = 0.75, linestyle = 'dashed', color = color_opt[i])
            ax5.plot(time[start_neutral[i]::]/ 3600 / 24, T_neutral[start_neutral[i]::, i], label = r"T$'_{\text{neutral}}$", linewidth = 0.75, linestyle = 'dotted', color = color_opt[i])
            ax5.plot(time/ 3600 / 24, T_hmax[:, i], label = r"T$'_{\text{intrusion}}$", linewidth = 0.75, linestyle = 'solid', color = color_opt[i])
        else:
            ax5.plot(time/ 3600 / 24, T_mld[:, i], linewidth = 0.75, linestyle = 'dashed', color = color_opt[i])
            ax5.plot(time/ 3600 / 24, T_hmax[:, i], linewidth = 0.75, linestyle = 'solid', color = color_opt[i])
            ax5.plot(time[start_neutral[i]::]/ 3600 / 24, T_neutral[start_neutral[i]::, i], linewidth = 0.75, linestyle = 'dotted', color = color_opt[i])
    ax5.set_xlabel("Time [days]") 
    ax5.set_title("Perturbed Temperature", size = 10)
    ax5.legend(loc='lower right', labelspacing = 0.25, handlelength=0.75)
    ax5.set_xlim([time.min() / 3600 / 24, time.max() / 3600 / 24])
    ax5.ticklabel_format(axis='y', style='sci', scilimits=(-3,2), useMathText=True)
    # tracer perturbations 
    for i in range(num_cases):
        if i == 0:
            ax6.plot(time/ 3600 / 24, tracer_mld[:, i], label = r"C$'_{\text{MLD}}$", linewidth = 0.75, linestyle = 'dashed', color = color_opt[i])
            ax6.plot(time[start_neutral[i]::]/ 3600 / 24, tracer_neutral[start_neutral[i]::, i], label = r"C$'_{\text{neutral}}$", linewidth = 0.75, linestyle = 'dotted', color = color_opt[i])
            ax6.plot(time/ 3600 / 24, tracer_hmax[:, i], label = r"C$'_{\text{intrusion}}$", linewidth = 0.75, linestyle = 'solid', color = color_opt[i])
        else:
            ax6.plot(time/ 3600 / 24, tracer_mld[:, i], linewidth = 0.75, linestyle = 'dashed', color = color_opt[i])
            ax6.plot(time/ 3600 / 24, tracer_hmax[:, i], linewidth = 0.75, linestyle = 'solid', color = color_opt[i])
            ax6.plot(time[start_neutral[i]::]/ 3600 / 24, tracer_neutral[start_neutral[i]::, i], linewidth = 0.75, linestyle = 'dotted', color = color_opt[i])
    ax6.set_xlabel("Time [days]") 
    ax6.set_title("Perturbed Tracer", size = 10)
    ax6.legend(loc='lower right', labelspacing = 0.25, handlelength=0.75)
    ax6.set_xlim([time.min() / 3600 / 24, time.max() / 3600 / 24])
    ax6.ticklabel_format(axis='y', style='sci', scilimits=(-3,2), useMathText=True)
    # average salinity at MLD
    for i in range(num_cases):
        ax7.plot(time/ 3600 / 24, tracerw_fluc_avg[:, i], linewidth = 0.75, color = color_opt[i])
    ax7.set_xlabel("Time [days]")
    ax7.set_title(r"$\langle$C$'$w$\rangle_{\text{xy}}$ at MLD", size = 10)
    ax7.ticklabel_format(axis='y', scilimits=(-1,1), useMathText=True)
    ax7.set_xlim([time.min() / 3600 / 24, time.max() / 3600 / 24])
    # root mean square w at MLD 
    for i in range(num_cases):
        ax8.plot(time/ 3600 / 24, Tw_fluc_avg[:, i], linewidth = 0.75, color = color_opt[i])
    ax8.set_xlabel("Time [days]")
    ax8.set_title(r"$\langle$T$'$w$\rangle_{\text{xy}}$ at MLD", size = 10)
    ax8.ticklabel_format(axis='y', style='sci', scilimits=(-3,2), useMathText=True)
    ax8.set_xlim([time.min() / 3600 / 24, time.max() / 3600 / 24])
    # --- Save Frame ---
    frame_path = os.path.join(outdir, f"{name}_temporal_comparison.png")
    plt.savefig(frame_path)
    plt.close(fig)
    print("Temporal Plot Saved: ", frame_path)
## spatial vertical analysis ###
def plot_plume_vertical_spatial(time, it, ranges, color_opt, fig_folder, case_names, name, lx, z, tracer_avg, u_rms, v_rms, w_rms, b_avg, b_center, r_profile, bu_fluc_avg, bv_fluc_avg, bw_fluc_avg, T_avg, T_fluc, tracer_fluc, ND = False, z_nd = r"(z - h$_{\mathrm{MLD}_0}$)/l$_{j}$"):
    num_cases = len(case_names)
    if num_cases==0:
        fig, ax = plt.subplots(2, 4, figsize=(12, 8))
        outdir = os.path.join(fig_folder, 'vertical centerline-' + name)
        os.makedirs(outdir, exist_ok=True)
    else:
        outdir = os.path.join(fig_folder, 'vertical centerline-' + name)
        os.makedirs(outdir, exist_ok=True)
        gridspec_kw={'height_ratios': [1, 1, 0.02]} # add space for universal legend
        fig, ax = plt.subplots(3, 4, figsize=(12, 10), gridspec_kw=gridspec_kw)
        for a in ax[2, :]:
            a.remove()
        case_handles = [
            Line2D([0], [0], color=color_opt[i], linestyle='solid', label=case_names[i])
            for i in range(num_cases)
        ]

        fig.legend(handles=case_handles,
                loc='lower center',
                ncol=num_cases,
                bbox_to_anchor=(0.52, 0.015))

    td = time / 3600 / 24
    fig.suptitle(f'{td:.2f} days', fontsize=12)

    ax1 = ax[0, 0]
    ax2 = ax[0, 1]
    ax3 = ax[0, 2]
    ax4 = ax[0, 3]
    ax5 = ax[1, 0]
    ax6 = ax[1, 1]
    ax7 = ax[1, 2]
    ax8 = ax[1, 3]

    if ND:
        ax1.set_ylabel(z_nd) 
        ax1.set_xlabel(r"(u$_{i}$/$\sqrt{\text{g l}_{j}}$)Ri$_{g}^{1/3}$")
        ax2.set_xlabel(r"($\langle$C$\rangle_{\text{xy}} \beta$)Ri$_{g}^{-1/3}$Fr$_{flux}^{-1}$")
        ax3.set_xlabel(r"(b/g)(Ri$_{g}^{1/3}$)Fr$_{flux}^{-1}$")
        ax4.set_xlabel(r"(C'$_{\text{centerline}} \beta$)Ri$_{g}^{-1/3}$Fr$_{flux}^{-1}$") 
        ax5.set_ylabel(z_nd)
        ax5.set_xlabel(r"(r/l$_{j}$)Fr$_{flux}^{-1}$")
        ax6.set_xlabel(r"($\langle$b'u'$_{i}\rangle_{xy}$/$\sqrt{\text{g}^3 \text{r}_{j}}$)Fr$_{flux}^{-1}$")
        ax7.set_xlabel(r"(b$_{rms}$/g)(Ri$_{g}^{1/3}$)Fr$_{flux}^{-1}$")
        ax8.set_xlabel(r"$(\text{T'}_{\text{centerline}}\alpha$)Ri$_{g}^{-1/3}$Fr$_{flux}^{-1}$")
    else:
        ax1.set_ylabel("Depth [m]")
        ax1.set_xlabel("[m/s]")
        ax2.set_xlabel(r"$\langle$C$\rangle_{\text{xy}}$ [g/kg]")
        ax3.set_xlabel(r"[m/s$^2$]")
        ax4.set_xlabel(r"C$'_{\text{centerline}}$ [g/kg]")
        ax5.set_ylabel("Depth [m]")
        ax5.set_xlabel("[m]")
        ax6.set_xlabel(r"[m$^2$/s$^3$]")
        ax7.set_xlabel(r"$\langle$T$\rangle_{\text{xy}}$ [$^{\circ}$ C]")
        ax8.set_xlabel(r"T$'_{\text{centerline}}$ [$^{\circ}$ C]")

    # velocity rms
    for i in range(num_cases):
        if i == 0:
            ax1.plot(u_rms[i], z[i], label=r"$\langle$u$_{\text{rms}}\rangle_{\text{xy}}$", color = color_opt[i], linestyle='dotted', linewidth = 0.75)
            ax1.plot(v_rms[i], z[i], label=r"$\langle$v$_{\text{rms}}\rangle_{\text{xy}}$", color = color_opt[i], linestyle='dashed', linewidth = 0.75)
            ax1.plot(w_rms[i], z[i], label=r"$\langle$w$_{\text{rms}}\rangle_{\text{xy}}$", color = color_opt[i], linestyle='solid', linewidth = 0.75)
        else:
            ax1.plot(u_rms[i], z[i], color = color_opt[i], linestyle='dotted', linewidth = 0.75)
            ax1.plot(v_rms[i], z[i], color = color_opt[i], linestyle='dashed', linewidth = 0.75)
            ax1.plot(w_rms[i], z[i], color = color_opt[i], linestyle='solid', linewidth = 0.75)
    ax1.set_title("Root Mean Square Velocities")
    #ax1.set_ylim(ymin = np.min(z), ymax = np.max(z))
    ax1.set_xlim(ranges['vel_rms'])
    ax1.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
    ax1.legend(loc='lower right')

    # tracer profile 
    for i in range(num_cases):
        ax2.plot(tracer_avg[i], z[i], color = color_opt[i], linestyle='solid', linewidth = 0.75)
    ax2.set_title('Tracer Profile')
    #ax2.set_ylim(ymin = np.min(z), ymax = np.max(z))
    ax2.set_xlim(ranges['Tracer_avg'])
    ax2.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)

    # buoyancy profiles
    for i in range(num_cases):
        if i == 0:
            ax3.plot(b_avg[i], z[i], color = color_opt[i], label = r"$\langle$b$\rangle_{\text{xy}}$", linestyle='solid', linewidth = 0.75)
            ax3.plot(b_center[i], z[i], color = color_opt[i], label = r"b$_{\text{centerline}}$", linestyle='dashed', linewidth = 0.75)
        else:
            ax3.plot(b_avg[i], z[i], color = color_opt[i], linestyle='solid', linewidth = 0.75)
            ax3.plot(b_center[i], z[i], color = color_opt[i], linestyle='dashed', linewidth = 0.75)
    ax3.set_title("Buoyancy Profile")
    #ax3.set_ylim(ymin = np.min(z), ymax = np.max(z))
    ax3.set_xlim(ranges['b_avg'])
    ax3.legend(loc='lower right')
    ax3.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)

    # temperature fluctuations 
    for i in range(num_cases):
        ax4.plot(tracer_fluc[i], z[i], color = color_opt[i], linestyle='solid', linewidth = 0.75)
    ax4.set_title("Perturbed Tracer")
    #ax4.set_ylim(ymin = np.min(z), ymax = np.max(z))
    ax4.set_xlim(ranges['Tracer_fluc'])
    ax4.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)

    # plume radius
    for i in range(num_cases):
        ax5.plot(r_profile[i], z[i], color = color_opt[i], linestyle='solid', linewidth = 0.75)
    ax5.set_title("Plume Radius with Depth")
    #ax5.set_ylim(ymin = np.min(z), ymax = np.max(z))
    ax5.set_xlim(0, lx[0]/2)

    # perturbed buoyancy flux 
    for i in range(num_cases):
        if i == 0:
            ax6.plot(bu_fluc_avg[i], z[i], color = color_opt[i], label = r"$\langle$b'u'$\rangle_{\text{xy}}$", linestyle='dotted', linewidth = 0.75)
            ax6.plot(bv_fluc_avg[i], z[i], color = color_opt[i], label = r"$\langle$b'v'$\rangle_{\text{xy}}$", linestyle='dashed', linewidth = 0.75)
            ax6.plot(bw_fluc_avg[i], z[i], color = color_opt[i], label = r"$\langle$b'w'$\rangle_{\text{xy}}$", linestyle='solid', linewidth = 0.75)
        else:
            ax6.plot(bu_fluc_avg[i], z[i], color = color_opt[i], linestyle='dotted', linewidth = 0.75)
            ax6.plot(bv_fluc_avg[i], z[i], color = color_opt[i], linestyle='dashed', linewidth = 0.75)
            ax6.plot(bw_fluc_avg[i], z[i], color = color_opt[i], linestyle='solid', linewidth = 0.75)
    ax6.legend(loc='lower right')
    ax6.set_title("Buoyancy Flux Fluctuations")
    #ax6.set_ylim(ymin = np.min(z), ymax = np.max(z))
    ax6.set_xlim(ranges['bw_fluc'])
    ax6.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True) 

    # buoyancy brms 
    for i in range(num_cases):
        ax7.plot(T_avg[i], z[i], color = color_opt[i], linestyle='solid', linewidth = 0.75)
    ax7.set_title("Temperature")
    #ax7.set_ylim(ymin = np.min(z), ymax = np.max(z))
    ax7.set_xlim(ranges['T'])
    ax7.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)

    # temperature fluctuations 
    for i in range(num_cases):
        ax8.plot(T_fluc[i], z[i], color = color_opt[i], linestyle='solid', linewidth = 0.75)
    ax8.set_title("Perturbed Temperature")
    #ax8.set_ylim(ymin = np.min(z), ymax = np.max(z))
    ax8.set_xlim(ranges['T_fluc'])
    ax8.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)

    # --- Save Frame ---
    frame_path = os.path.join(outdir, f"comparison_vert_buoyancy_{it:04d}.png")
    plt.savefig(frame_path)
    plt.close(fig)
    print(f"Time step {it + 1} captured: {frame_path}")

    return outdir # return the directory where frames are saved for video creation
## spatial horizontal analysis ###
def plot_plume_horizontal_spatial(time, it, ranges, color_opt, fig_folder, case_names, name, lx, y, u, v, w, b_center, bu_fluc, bv_fluc, bw_fluc, T, tracer, ND = False):
    num_cases = len(case_names)
    if num_cases==0:
        fig, ax = plt.subplots(2, 3, figsize=(12, 7))
        outdir = os.path.join(fig_folder, 'horizontal centerline-' + name)
        os.makedirs(outdir, exist_ok=True)
    else:
        outdir = os.path.join(fig_folder, 'horizontal centerline-' + name)
        os.makedirs(outdir, exist_ok=True)
        gridspec_kw={'height_ratios': [1, 1, 0.02]} # add space for universal legend
        fig, ax = plt.subplots(3, 3, figsize=(12, 9), gridspec_kw=gridspec_kw)
        for a in ax[2, :]:
            a.remove()
        case_handles = [
            Line2D([0], [0], color=color_opt[i], linestyle='solid', label=case_names[i])
            for i in range(num_cases)
        ]

        fig.legend(handles=case_handles,
                loc='lower center',
                ncol=num_cases,
                bbox_to_anchor=(0.52, 0.015))

    td = time[it] / 3600 / 24
    fig.suptitle(f'{td:.2f} days', fontsize=12)

    ax1 = ax[0, 0] # u, v, w through horizontal centerline
    ax2 = ax[0, 1] # horizontal buoyancy flux through horizontal centerline
    ax3 = ax[0, 2] # tracer through horizontal centerline
    ax4 = ax[1, 0] # perturbed buoyancy through horizontal centerline
    ax5 = ax[1, 1] # vertical buoyancy flux through horizontal centerline
    ax6 = ax[1, 2] # temperature through horizontal centerline

    if ND:
        ax1.set_xlabel(r"y/l$_{j}$") 
        ax1.set_ylabel(r"u$_{i}$/(F$_{\text{C}} \beta$ dT/dz l$_{j}$ T$_{0}$)") 
        ax2.set_xlabel(r"y/l$_{j}$")
        ax2.set_ylabel(r"(b' u'$_{i}$)/(F$_{\text{C}} \beta$ g dT/dz l$_{j}$/T$_{0}$)") 
        ax3.set_xlabel(r"y/l$_{j}$")
        ax3.set_ylabel(r"C$_{\text{centerline}} \sqrt{\text{g l}_{j}}$/(F$_{\text{C}}$)") 
        ax4.set_xlabel(r"y/l$_{j}$")
        ax4.set_ylabel(r"b'/(F$_{\text{C}} \beta \sqrt{\text{g dT/dz l}_{j}/\text{T}_{0}}$)")
        ax5.set_xlabel(r"y/l$_{j}$")
        ax5.set_ylabel(r"b'w'/(F$_{\text{C}} \beta$ g dT/dz l$_{j}$/T$_{0}$)")
        ax6.set_xlabel(r"y/l$_{j}$")
        ax6.set_ylabel(r"T$_{\text{centerline}\sqrt{\text{g l}_{j}}}$/(F$_{\text{C}}\beta$dT/dz l$_{j}$)")
    else:
        ax1.set_xlabel("x [m]")
        ax1.set_ylabel("[m/s]")
        ax2.set_xlabel("x [m]")
        ax2.set_ylabel(r"b'u'$_{i}$ [m$^2$/s$^3$]")
        ax3.set_xlabel("x [m]")
        ax3.set_ylabel(r"C$_{\text{centerline}}$ [g/kg]")
        ax4.set_xlabel("x [m]")
        ax4.set_ylabel(r"[m/s$^2$]")
        ax5.set_xlabel("x [m]")
        ax5.set_ylabel(r"b'w' [m$^2$/s$^3$]")
        ax6.set_xlabel("x [m]")
        ax6.set_ylabel(r"T$_{\text{centerline}}$ [$^{\circ}$ C]")

    # velocity
    for i in range(num_cases):
        if i == 0:
            ax1.plot(y, u[i], label=r"u$_{\text{centerline}}$", color = color_opt[i], linestyle='dotted', linewidth = 0.75)
            ax1.plot(y, v[i], label=r"v$_{\text{centerline}}$", color = color_opt[i], linestyle='dashed', linewidth = 0.75)
            ax1.plot(y, w[i], label=r"w$_{\text{centerline}}$", color = color_opt[i], linestyle='solid', linewidth = 0.75)
        else:
            ax1.plot(y, u[i], color = color_opt[i], linestyle='dotted', linewidth = 0.75)
            ax1.plot(y, v[i], color = color_opt[i], linestyle='dashed', linewidth = 0.75)
            ax1.plot(y, w[i], color = color_opt[i], linestyle='solid', linewidth = 0.75)
    ax1.set_title("Velocity")
    ax1.set_xlim(-lx[0][1]/2, lx[0][1]/2)
    ax1.set_ylim(ranges['w'])
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(-3,2), useMathText=True)
    ax1.legend(loc='lower right')

    # horizontal buoyancy flux 
    for i in range(num_cases):
        if i == 0:
            ax2.plot(y, bu_fluc[i], color = color_opt[i], label = r"b$'$u$'_{\text{centerline}}$", linestyle='dotted', linewidth = 0.75)
            ax2.plot(y, bv_fluc[i], color = color_opt[i], label = r"b$'$v$'_{\text{centerline}}$", linestyle='dashed', linewidth = 0.75)
        else:
            ax2.plot(y, bu_fluc[i], color = color_opt[i], linestyle='dotted', linewidth = 0.75)
            ax2.plot(y, bv_fluc[i], color = color_opt[i], linestyle='dashed', linewidth = 0.75)
    ax2.set_title('Horizontal Buoyancy Flux Fluctuations')
    ax2.set_xlim(-lx[0][1]/2, lx[0][1]/2)
    ax2.set_ylim(ymin = ranges['b_flux'][0], ymax = ranges['b_flux'][1])
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(-3,2), useMathText=True)

    # tracer  
    for i in range(num_cases):
        ax3.plot(y, tracer[i], color = color_opt[i], linestyle='solid', linewidth = 0.75)
    ax3.set_title("Tracer")
    ax3.set_xlim(-lx[0][1]/2, lx[0][1]/2)
    ax3.set_ylim(ranges['Tracer'])
    ax3.ticklabel_format(axis='y', style='sci', scilimits=(-3,2), useMathText=True)

    # Perturbed buoyancy 
    for i in range(num_cases):
        ax4.plot(y, b_center[i], color = color_opt[i], linestyle='solid', linewidth = 0.75)
    ax4.set_title("Perturbed Buoyancy")
    ax4.set_xlim(-lx[0][1]/2, lx[0][1]/2)
    ax4.set_ylim(ranges['b_fluc'])
    ax4.ticklabel_format(axis='y', style='sci', scilimits=(-3,2), useMathText=True)
    # perturbed buoyancy flux 
    for i in range(num_cases):
        ax5.plot(y, bw_fluc[i], color = color_opt[i], linestyle='solid', linewidth = 0.75)
    ax5.set_title("Vertical Buoyancy Flux Fluctuations")
    ax5.set_xlim(-lx[0][1]/2, lx[0][1]/2)
    ax5.set_ylim(ranges['bw_fluc'])
    ax5.ticklabel_format(axis='y', style='sci', scilimits=(-3,2), useMathText=True) 

    # temperature  
    for i in range(num_cases):
        ax6.plot(y, T[i], color = color_opt[i], linestyle='solid', linewidth = 0.75)
    ax6.set_title("Temperature")
    ax6.set_xlim(-lx[0][1]/2, lx[0][1]/2)
    ax6.set_ylim(ranges['T'])

    # --- Save Frame ---
    frame_path = os.path.join(outdir, f"hor_centerline_comparisons_{it:04d}.png")
    plt.savefig(frame_path)
    plt.close(fig)
    print(f"Time step {it + 1} captured: {frame_path}")

    return outdir # return the directory where frames are saved for video creation
## turbulent statistics plotting
def plot_turb_stats_bin(time, it, ranges, color_opt, fig_folder, case_names, z, u_rms, w_rms, uw, b_avg, bu_fluc_avg, bw_fluc_avg, Tu, Tw, Cu, Cw):
    num_cases = len(case_names)
    ncols = 3
    nrows = 2
    outdir = os.path.join(fig_folder, 'binning', 'turbulent statistics')
    if it == 0:
        os.makedirs(outdir, exist_ok=True)
    ar = np.ones(nrows + 1)
    ar[-1] = 0.02 # add space for universal legend
    gridspec_kw={'height_ratios': ar} # add space for universal legend
    fig, ax = plt.subplots(nrows + 1, ncols, figsize=(12, 10), gridspec_kw=gridspec_kw, sharey = True)
    for a in ax[2, :]:
        a.remove()
    case_handles = [
        Line2D([0], [0], color=color_opt[i], linestyle='solid', label=case_names[i])
        for i in range(num_cases)
    ]
    fig.legend(handles=case_handles,
            loc='lower center',
            ncol=num_cases,
            bbox_to_anchor=(0.52, 0.015))
    ax = ax.ravel()
    td = time / 3600 / 24
    fig.suptitle(f'{td:.2f} days', fontsize=12)
    """
    ax[0] = velocity rms
    ax[1] = Reynolds stresses
    ax[2] = buoyancy fluxes
    ax[3] = temperature fluxes
    ax[4] = tracer fluxes
    ax[5] = buoyancy profile
    """
    ax[0].set_ylabel("Depth [m]")
    ax[0].set_xlabel("[m/s]")
    ax[1].set_xlabel(r"[m$^2$/s$^2$]")
    ax[2].set_xlabel(r"[m$^2$/s$^3$]")

    ax[3].set_ylabel("Depth [m]")
    ax[3].set_xlabel(r"[$^{\circ}$ C$\cdot$m/s]")
    ax[4].set_xlabel(r"[g/kg$\cdot$m/s]")
    ax[5].set_xlabel(r"[m/s$^2$]")
    for a in ax:
        a.set_ylim(ymin = np.min(z), ymax = np.max(z))
    # velocity rms
    for i in range(num_cases):
        if i == 0:
            ax[0].plot(u_rms[i], z, label=r"$\langle$u$_{r,\text{rms}}\rangle_{\text{xy}}$", color = color_opt[i], linestyle='dotted', linewidth = 0.75)
            ax[0].plot(w_rms[i], z, label=r"$\langle$w$_{\text{rms}}\rangle_{\text{xy}}$", color = color_opt[i], linestyle='solid', linewidth = 0.75)
        else:
            ax[0].plot(u_rms[i], z, color = color_opt[i], linestyle='dotted', linewidth = 0.75)
            ax[0].plot(w_rms[i], z, color = color_opt[i], linestyle='solid', linewidth = 0.75)
    ax[0].set_title("Root Mean Square Velocities")
    ax[0].set_xlim(ranges['vel_rms'])
    ax[0].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
    ax[0].legend(loc='lower right')

    # reynolds stresses
    for i in range(num_cases):
        ax[1].plot(uw[i], z, color = color_opt[i], linestyle='solid', linewidth = 0.75)
    ax[1].set_title(r"Reynolds Stresses, $\langle$u$_r$'w'$\rangle_{\text{xy}}$")
    ax[1].set_xlim(ranges['restress'])
    ax[1].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)

    # perturbed buoyancy flux 
    for i in range(num_cases):
        if i == 0:
            ax[2].plot(bu_fluc_avg[i], z, color = color_opt[i], label = r"$\langle$b'u$_r\rangle_{\text{xy}}$", linestyle='dotted', linewidth = 0.75)
            ax[2].plot(bw_fluc_avg[i], z, color = color_opt[i], label = r"$\langle$b'w$\rangle_{\text{xy}}$", linestyle='solid', linewidth = 0.75)
        else:
            ax[2].plot(bu_fluc_avg[i], z, color = color_opt[i], linestyle='dotted', linewidth = 0.75)
            ax[2].plot(bw_fluc_avg[i], z, color = color_opt[i], linestyle='solid', linewidth = 0.75)
    ax[2].legend(loc='upper left')
    ax[2].set_title("Perturbed Buoyancy Flux")
    ax[2].set_xlim(ranges['bw_fluc'])
    ax[2].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True) 

    # perturbed temperature flux 
    for i in range(num_cases):
        if i == 0:
            ax[3].plot(Tu[i], z, color = color_opt[i], label = r"$\langle$T'u$_r\rangle_{\text{xy}}$", linestyle='dotted', linewidth = 0.75)
            ax[3].plot(Tw[i], z, color = color_opt[i], label = r"$\langle$T'w$\rangle_{\text{xy}}$", linestyle='solid', linewidth = 0.75)
        else:
            ax[3].plot(Tu[i], z, color = color_opt[i], linestyle='dotted', linewidth = 0.75)
            ax[3].plot(Tw[i], z, color = color_opt[i], linestyle='solid', linewidth = 0.75)
    ax[3].legend(loc='upper right')
    ax[3].set_title("Perturbed Temperature Flux")
    ax[3].set_xlim(ranges['Tw_fluc'])
    ax[3].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True) 

    # tracer flux 
    for i in range(num_cases):
        if i == 0:
            ax[4].plot(Cu[i], z, color = color_opt[i], label = r"$\langle$Cu$_r\rangle_{\text{xy}}$", linestyle='dotted', linewidth = 0.75)
            ax[4].plot(Cw[i], z, color = color_opt[i], label = r"$\langle$Cw$\rangle_{\text{xy}}$", linestyle='solid', linewidth = 0.75)
        else:
            ax[4].plot(Cu[i], z, color = color_opt[i], linestyle='dotted', linewidth = 0.75)
            ax[4].plot(Cw[i], z, color = color_opt[i], linestyle='solid', linewidth = 0.75)
    ax[4].legend(loc='lower right')
    ax[4].set_title("Tracer Flux")
    ax[4].set_xlim(ranges['Cw'])
    ax[4].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True) 

    # buoyancy profile
    for i in range(num_cases):
        ax[5].plot(b_avg[i], z, color = color_opt[i], linewidth = 0.75)
    ax[5].set_title("Buoyancy Profile")
    ax[5].set_xlim(ranges['b_avg'])
    ax[5].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True) 

    # --- Save Frame ---
    frame_path = os.path.join(outdir, f"comparison_vert_buoyancy_{it:04d}.png")
    plt.savefig(frame_path)
    plt.close(fig)
    print(f"Time step {it + 1} captured: {frame_path}")

    return outdir # return the directory where frames are saved for video creation
### -------------------------PLOTTING DENSE PLUME FUNCTIONS------------------------- ###
## buoyancy analysis 
def buoyancy_analysis_plot(time, it, ranges, fig_folder, lx, nx, z, zf, X, Z, mld, b_avg, w_avg, b_center, w_center, b_rms, bu_fluc_avg, bv_fluc_avg, bw_fluc_avg, b_fluc, rho_perturbed, Ri_avg, Ri_strat, Ri_plume, plume_depth_intrusion, plume_depth_neutral, w_neutral, w_intrusion, w_mld, rho_perturbed_neutral, rho_perturbed_intrusion, rho_perturbed_mld, bwfluc_neutral, bwfluc_intrusion, bwfluc_mld):

    outdir = os.path.join(fig_folder, 'NBP buoyancy analysis/')
    os.makedirs(outdir, exist_ok=True)
    td = time[it] / 3600 / 24

    fig = plt.figure()
    fig.set_figheight(7)
    fig.set_figwidth(22)
    fig.suptitle(f'{td:.2f} days', fontsize=12) 

    ax1 = plt.subplot2grid(shape=(4, 10), loc=(0, 0), rowspan=2, colspan=2)
    ax2 = plt.subplot2grid(shape=(4, 10), loc=(0, 2), rowspan=2, colspan=2)
    ax3 = plt.subplot2grid(shape=(4, 10), loc=(0, 4), rowspan=2, colspan=2)
    ax4 = plt.subplot2grid(shape=(4, 10), loc=(0, 6), rowspan=2, colspan=2)
    ax5 = plt.subplot2grid(shape=(4, 10), loc=(0, 8), rowspan=2, colspan=2)
    ax6 = plt.subplot2grid(shape=(4, 10), loc=(2, 0), rowspan=2, colspan=2)
    ax7 = plt.subplot2grid(shape=(4, 10), loc=(2, 2), rowspan=2, colspan=2)
    ax8 = plt.subplot2grid(shape=(4, 10), loc=(2, 4), rowspan=2, colspan=2)
    ax9 = plt.subplot2grid(shape=(4, 10), loc=(2, 6), rowspan=2, colspan=2)
    ax10 = plt.subplot2grid(shape=(4, 10), loc=(2, 8), rowspan=1, colspan=2)
    ax11 = plt.subplot2grid(shape=(4, 10), loc=(3, 8), rowspan=1, colspan=2, sharex=ax10)

    fig.subplots_adjust(hspace=0.05)

    levels = 500

    # Depth of plume through time
    ax1.plot(time/ 3600 / 24, -mld*np.ones(len(time)), linestyle='--', linewidth = 0.5, color = 'black', label = "MLD")
    ax1.plot(time[:it+1]/ 3600 / 24, plume_depth_intrusion, color = 'blue', label = r"h$_{\text{intrusion}}$")
    ax1.plot(time[:it+1]/ 3600 / 24, plume_depth_neutral, color = 'red', label = r"h$_{\text{neutral}}$")
    ax1.set_xlabel("Time [days]") 
    ax1.set_ylabel("[m]")
    ax1.set_title("Plume Depths")
    ax1.set_ylim(ymin = -lx[-1], ymax = 0)
    ax1.legend(loc='upper right', handlelength=0.9)
    ax1.set_xlim([0, time.max() / 3600 / 24])  

    # vertical velocity
    ax2.plot(time[:it+1]/ 3600 / 24, w_mld, color = 'black', label = r"w$_{\text{MLD}}$")
    ax2.plot(time[:it+1]/ 3600 / 24, w_intrusion, color = 'blue',label = r"w$_{\text{intrusion}}$")
    ax2.plot(time[:it+1]/ 3600 / 24, w_neutral, color = 'red', label = r"w$_{\text{neutral}}$")
    ax2.set_xlabel("Time [days]") 
    ax2.set_ylabel("[m/s]")
    ax2.set_title("w at Depth")
    ax2.set_ylim(ymin = ranges['w'][0]*10, ymax = ranges['w'][-1]*10)
    ax2.legend(loc='upper right', handlelength=0.9)
    ax2.set_xlim([0, time.max() / 3600 / 24])  

    # density perturbations 
    ax3.plot(time[:it+1]/ 3600 / 24, rho_perturbed_mld, color = 'black', label = r"$\rho$'$_{\text{MLD}}$")
    ax3.plot(time[:it+1]/ 3600 / 24, rho_perturbed_intrusion, color = 'blue', label = r"$\rho$'$_{\text{intrusion}}$")
    ax3.plot(time[:it+1]/ 3600 / 24, rho_perturbed_neutral, color = 'red', label = r"$\rho$'$_{\text{neutral}}$")
    ax3.set_xlabel("Time [days]") 
    ax3.set_ylabel(r"[kg/m$^3$]")
    ax3.set_title("Perturbed Density at Depth")
    ax3.legend(loc='upper right', handlelength=0.9)
    ax3.set_ylim(ymin = ranges['rho_fluc'][0]*2, ymax = ranges['rho_fluc'][-1]*2)
    ax3.set_xlim([0, time.max() / 3600 / 24])

    # buoyancy perturbations 
    ax4.plot(time[:it+1]/ 3600 / 24, bwfluc_mld, color = 'black', label = r"b'w$_{\text{MLD}}$")
    ax4.plot(time[:it+1]/ 3600 / 24, bwfluc_intrusion, color = 'blue', label = r"b'w$_{\text{intrusion}}$")
    ax4.plot(time[:it+1]/ 3600 / 24, bwfluc_neutral, color = 'red', label = r"b'w$_{\text{neutral}}$")
    ax4.set_xlabel("Time [days]") 
    ax4.set_ylabel(r"[m$^{2}$/s$^{3}$]")
    ax4.set_title("Perturbed Buoyancy Flux at Depth")
    ax4.legend(loc='lower right', handlelength=0.9)
    ax4.set_ylim(ymin = ranges['bw_fluc'][0]*10, ymax = ranges['bw_fluc'][-1]*10)
    ax4.set_xlim([0, time.max() / 3600 / 24])

    #Richardson profile
    ax5.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')
    ax5.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')
    ax5.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')
    ax5.set_xlim(ranges['Ri'])
    if len(np.shape(Ri_strat))>1: # if we have the Hassanzadeh Richardson number, plot that as well
        Ri_strat_h = Ri_strat[:, 1]
        Ri_avg_h = Ri_avg[:, 1]
        Ri_plume_h = Ri_plume[:, 1]
        Ri_strat = Ri_strat[:, 0]
        Ri_avg = Ri_avg[:, 0]
        Ri_plume = Ri_plume[:, 0]
        ax5.plot(Ri_strat_h, z, linestyle='--', label = r"Ri$_{\text{stratified, Hassanzadeh}}$")
        ax5.plot(Ri_avg_h, z, color = 'black', linestyle='--', label = r"Ri$_{\text{average, Hassanzadeh}}$")
        ax5.plot(Ri_plume_h, z, color = 'red', linestyle='--', label = r"Ri$_{\text{centerline, Hassanzadeh}}$")
    ax5.plot(Ri_strat, z, label = r"Ri$_{\text{stratified}}$")
    ax5.plot(Ri_avg, z, color = 'black', label = r"Ri$_{\text{average}}$")
    ax5.plot(Ri_plume, z, color = 'red', label = r"Ri$_{\text{centerline}}$")
    ax5.set_xlabel("Richardson Number") 
    ax5.set_ylabel("Depth [m]")
    ax5.set_title("Richardson Number")
    ax5.legend(loc='lower right', handlelength=0.9)
    ax5.set_ylim(-lx[2], 0)

    # buoyancy 
    ax6.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')
    ax6.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')
    ax6.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')
    ax6.set_xlim(ranges['b'])
    ax6.plot(b_avg, z, color='black', label = r"$\langle$b$\rangle_{\text{xy}}")
    ax6.plot(b_center, z, color='red', label = r"b$_{\text{centerline}}$")
    ax6.set_title("Buoyancy Profile")
    ax6.set_xlabel("[m/s$^{2}$]")
    ax6.set_ylabel("Depth [m]")
    ax6.set_ylim(-lx[2], 0)
    ax6.legend(loc='upper left', handlelength=0.9)
    ax6.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True)

    # w
    ax7.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')
    ax7.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')
    ax7.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')
    ax7.set_xlim(xmin = ranges['w'][0]*10, xmax = ranges['w'][-1]*1)
    ax7.plot(w_avg, zf, color='black', label = r"w$_{\text{average}}$")
    ax7.plot(w_center, zf, color='red', label = r"w$_{\text{centerline}}$")
    ax7.set_title("Vertical Velocity Profile")
    ax7.set_xlabel("[m/s]")
    ax7.set_ylim(-lx[2], 0)
    ax7.legend(loc='lower right', handlelength=0.9)
    ax7.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True)

    # RMS buoyancy 
    ax8.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')
    ax8.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')
    ax8.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')
    ax8.set_xlim(ranges['b_rms'])
    ax8.plot(b_rms, z, color='black')
    ax8.set_title("Buoyancy Root Mean Square Error")
    ax8.set_xlabel(r"[m/s$^{2}$]")
    ax8.set_ylabel("Depth [m]")
    ax8.set_ylim(-lx[2], 0)
    ax8.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True)

    # RMS buoyancy flux 
    ax9.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')
    ax9.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')
    ax9.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')
    ax9.set_xlim(ranges['bw_fluc'])
    ax9.plot(bu_fluc_avg, z, color='black', label = r"b'u'")
    ax9.plot(bv_fluc_avg, z, color='blue', label = r"b'v'")
    ax9.plot(bw_fluc_avg, z, color='red', label = r"b'w'")
    ax9.legend(loc='lower right', handlelength=0.9)
    ax9.set_xlabel(r"[m$^{2}$/s$^{3}$]")
    #ax9.set_ylabel("Depth [m]")
    ax9.set_title("Buoyancy Flux Fluctuations")
    ax9.set_ylim(-lx[2], 0)
    ax9.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True) 

    # Perturbed density
    norm = mcolors.Normalize(vmin=ranges['rho_fluc'][0], vmax=ranges['rho_fluc'][-1])
    ax10.contourf(X[int(nx[1]/2), :, :], Z[int(nx[1]/2), :, :], rho_perturbed[int(nx[1]/2), :, :], levels, norm=norm, cmap='RdBu_r')
    #ax10.set_xlabel("[m]")
    #ax10.set_ylabel("Depth [m]")
    ax10.set_ylim(-lx[2], 0)
    ax10.set_xlim(0, lx[1])
    ax10.set_aspect('equal')
    mappable = cm.ScalarMappable(norm=norm, cmap='RdBu_r')
    cbar = fig.colorbar(mappable, ax=ax10, label=r"[kg/m$^3$]", shrink=0.9, pad=0.1)#, fraction=0.046, pad=0.1) #anchor = (0.5, -0.4), orientation='horizontal', 
    cbar.formatter.set_scientific(True)

    # Buoyancy fluctuations
    norm = mcolors.Normalize(vmin=ranges['b_fluc'][0], vmax=ranges['b_fluc'][-1])
    ax11.contourf(X[int(nx[1]/2), :, :], Z[int(nx[1]/2), :, :], b_fluc[int(nx[1]/2), :, :], levels, norm=norm, cmap='RdBu_r')
    ax11.set_xlabel("[m]")
    #ax11.set_ylabel("Depth [m]")
    ax11.set_ylim(-lx[2], 0)
    ax11.set_xlim(0, lx[1])
    ax11.set_aspect('equal')
    mappable = cm.ScalarMappable(norm=norm, cmap='RdBu_r')
    cbar = fig.colorbar(mappable, ax=ax11, label=r"[m/s$^{2}$]", shrink=0.9, pad=0.1)
    cbar.formatter.set_powerlimits((-3, 2))
    cbar.update_ticks()

    # --- Save Frame ---
    frame_path = os.path.join(outdir, f"oc_frame_{it:04d}.png")
    plt.tight_layout()
    plt.savefig(frame_path)
    plt.close(fig)
    print(f"Time step {it + 1} captured: {frame_path}")
    plt.close(fig)
    return outdir # return the directory where frames are saved for video creation
## dense tracer buoyancy analysis 
def plot_tracer_plume(time, it, ranges, fig_folder, lx, nx, z, zf, Y, Z, mld, u_avg, v_avg, w_avg, uv_fluc, uw_fluc, vw_fluc, u_rms, v_rms, w_rms, dbdx, dbdy, dbdz, b_avg, b_center, b_rms, bu_fluc_avg, bv_fluc_avg, bw_fluc_avg, b_fluc, rho_perturbed, tracer_avg, rp, plume_depths, ws, rhos, bw_flucs, l_scale):
    plume_depth_intrusion = plume_depths[0]
    plume_depth_neutral = plume_depths[1]

    w_intrusion = ws[0]
    w_neutral = ws[1]
    w_mld = ws[2]

    rho_perturbed_intrusion = rhos[0]
    rho_perturbed_neutral = rhos[1]
    rho_perturbed_mld = rhos[2]

    bwfluc_intrusion = bw_flucs[0]
    bwfluc_neutral = bw_flucs[1]
    bwfluc_mld = bw_flucs[2]

    outdir = os.path.join(fig_folder, 'NBP analysis/')
    os.makedirs(outdir, exist_ok=True)
    td = time[it] / 3600 / 24

    fig = plt.figure()
    fig.set_figheight(12)
    fig.set_figwidth(22)
    fig.suptitle(f'{td:.2f} days', fontsize=12) 

    ax1a = plt.subplot2grid(shape=(6, 10), loc=(0, 0), rowspan=2, colspan=2) # velocity profiles
    ax2a = plt.subplot2grid(shape=(6, 10), loc=(0, 2), rowspan=2, colspan=2) # velocity rms w/ vertical velocity of plume
    ax3a = plt.subplot2grid(shape=(6, 10), loc=(0, 4), rowspan=2, colspan=2) # Reynolds stress
    ax4a = plt.subplot2grid(shape=(6, 10), loc=(0, 6), rowspan=2, colspan=2) # buoyancy gradients
    ax5a = plt.subplot2grid(shape=(6, 10), loc=(0, 8), rowspan=2, colspan=2) # buoyancy flux u, v, w
    ax1 = plt.subplot2grid(shape=(6, 10), loc=(2, 0), rowspan=2, colspan=2) # plume depths
    ax2 = plt.subplot2grid(shape=(6, 10), loc=(2, 2), rowspan=2, colspan=2) # w at depth
    ax3 = plt.subplot2grid(shape=(6, 10), loc=(2, 4), rowspan=2, colspan=2) # perturbed density at depth
    ax4 = plt.subplot2grid(shape=(6, 10), loc=(2, 6), rowspan=2, colspan=2) # perturbed buoyancy flux at depth
    ax5 = plt.subplot2grid(shape=(6, 10), loc=(2, 8), rowspan=2, colspan=2) # entrainment coefficient
    ax6 = plt.subplot2grid(shape=(6, 10), loc=(4, 0), rowspan=2, colspan=2) # max radius of plume 
    ax7 = plt.subplot2grid(shape=(6, 10), loc=(4, 2), rowspan=2, colspan=2) # buoyancy profiles
    ax8 = plt.subplot2grid(shape=(6, 10), loc=(4, 4), rowspan=2, colspan=2) # buoyancy rms
    ax9 = plt.subplot2grid(shape=(6, 10), loc=(4, 6), rowspan=2, colspan=2) # tracer plane slice with plume contour
    ax10 = plt.subplot2grid(shape=(6, 10), loc=(4, 8), rowspan=1, colspan=2) # perturbed density place slice
    ax11 = plt.subplot2grid(shape=(6, 10), loc=(5, 8), rowspan=1, colspan=2, sharex=ax10) # perturbed buoyancy plane slice

    fig.subplots_adjust(hspace=0.05)

    levels = 500

    # velocity profiles
    ax1a.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')#, label = "MLD")
    ax1a.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')#, label = "Intrusion Depth")
    ax1a.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')#, label = "Neutral Buoyancy")
    ax1a.plot(u_avg, z, label=r"$\langle$u$\rangle_{xy}$", color='green')
    ax1a.plot(v_avg, z, label=r"$\langle$v$\rangle_{xy}$", color='red')
    ax1a.plot(w_avg, zf, label=r"$\langle$w$\rangle_{xy}$", color='blue')
    ax1a.set_xlabel("[m/s]")
    ax1a.set_ylabel("Depth [m]")
    ax1a.set_title('Velocity Profiles')
    ax1a.set_ylim(-lx[2], 0)
    ax1a.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True)
    ax1a.set_xlim(ranges['vel'])
    ax1a.legend(loc='lower right', handlelength=0.9)

    # velocity rms
    ax2a.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')
    ax2a.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')
    ax2a.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')
    ax2a.plot(u_rms, z, label=r"$\langle$u$_{rms}$$\rangle_{xy}$", color='green')
    ax2a.plot(v_rms, z, label=r"$\langle$v$_{rms}$$\rangle_{xy}$", color='red')
    ax2a.plot(w_rms, zf, label=r"$\langle$w$_{rms}$$\rangle_{xy}$", color='blue')
    ax2a.set_xlabel("[m/s]")
    #ax2a.set_ylabel("Depth [m]")
    ax2a.set_title("Root Mean Square Velocities")
    ax2a.set_ylim(-lx[2], 0)
    ax2a.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True)
    ax2a.set_xlim(ranges['vel_rms'])
    ax2a.legend(loc='lower right', handlelength=0.9)

    # reynolds stresses 
    ax3a.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')
    ax3a.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')
    ax3a.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')
    ax3a.plot(uv_fluc, z, label=r"$\langle$uv$\rangle_{xy}$", color='green')
    ax3a.plot(uw_fluc, z, label=r"$\langle$uw$\rangle_{xy}$", color='red')
    ax3a.plot(vw_fluc, z, label=r"$\langle$vw$\rangle_{xy}$", color='blue')
    ax3a.set_xlabel(r"[m$^2$/s$^2$]")
    #ax3a.set_ylabel("Depth [m]")
    ax3a.set_title("Reynolds Stresses")
    ax3a.set_ylim(-lx[2], 0)
    ax3a.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True)
    ax3a.set_xlim(ranges['restress'])
    ax3a.legend(loc='lower right', handlelength=0.9)

    # buoyancy gradients 
    ax4a.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')
    ax4a.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')
    ax4a.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')
    ax4a.plot(dbdx, z, color='green', label = r"db/dx")
    ax4a.plot(dbdy, z, label=r"db/dy", color='red')
    ax4a.plot(dbdz, z, label=r"db/dz", color='blue')
    ax4a.set_xlabel(r"[1/s$^2$]")
    ax4a.set_title("Buoyancy Gradient")
    ax4a.set_ylim(-lx[2], 0)
    ax4a.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True)
    ax4a.set_xlim(ranges['gradb'])
    ax4a.legend(loc='lower left', handlelength=0.9)

    # perturbed buoyancy flux 
    ax5a.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')
    ax5a.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')
    ax5a.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')
    ax5a.set_xlim(ranges['bw_fluc'])
    ax5a.plot(bu_fluc_avg, z, color='green', label = r"b'u'")
    ax5a.plot(bv_fluc_avg, z, color='blue', label = r"b'v'")
    ax5a.plot(bw_fluc_avg, z, color='red', label = r"b'w'")
    ax5a.legend(loc='lower right', handlelength=0.9)
    ax5a.set_xlabel(r"[m$^{2}$/s$^{3}$]")
    ax5a.set_title("Buoyancy Flux Fluctuations")
    ax5a.set_ylim(-lx[2], 0)
    ax5a.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True) 

    # Depth of plume through time
    ax1.plot(time/ 3600 / 24, -mld*np.ones(len(time)), linestyle='--', linewidth = 0.5, color = 'black', label = "MLD")
    ax1.plot(time[:it+1]/ 3600 / 24, plume_depth_intrusion, color = 'cornflowerblue', label = r"h$_{\text{intrusion}}$")
    ax1.plot(time[:it+1]/ 3600 / 24, plume_depth_neutral, color = 'mediumblue', label = r"h$_{\text{neutral}}$")
    ax1.set_xlabel("Time [days]") 
    ax1.set_ylabel("[m]")
    ax1.set_title("Plume Depths")
    ax1.set_ylim(ymin = -lx[-1], ymax = 0)
    ax1.legend(loc='upper right', handlelength=0.9)
    ax1.set_xlim([0, time.max() / 3600 / 24])  

    # vertical velocity
    ax2.plot(time[:it+1]/ 3600 / 24, w_mld, color = 'black', label = r"w$_{\text{MLD}}$")
    ax2.plot(time[:it+1]/ 3600 / 24, w_intrusion, color = 'cornflowerblue',label = r"w$_{\text{intrusion}}$")
    ax2.plot(time[:it+1]/ 3600 / 24, w_neutral, color = 'mediumblue', label = r"w$_{\text{neutral}}$")
    ax2.set_xlabel("Time [days]") 
    ax2.set_ylabel("[m/s]")
    ax2.set_title("w at Depth")
    ax2.set_ylim(ymin = ranges['w'][0]*10.1, ymax = ranges['w'][-1]*10.1)
    ax2.legend(loc='upper right', handlelength=0.9)
    ax2.set_xlim([0, time.max() / 3600 / 24])  

    # density perturbations 
    ax3.plot(time[:it+1]/ 3600 / 24, rho_perturbed_mld, color = 'black', label = r"$\rho$'$_{\text{MLD}}$")
    ax3.plot(time[:it+1]/ 3600 / 24, rho_perturbed_intrusion, color = 'cornflowerblue', label = r"$\rho$'$_{\text{intrusion}}$")
    ax3.plot(time[:it+1]/ 3600 / 24, rho_perturbed_neutral, color = 'mediumblue', label = r"$\rho$'$_{\text{neutral}}$")
    ax3.set_xlabel("Time [days]") 
    ax3.set_ylabel(r"[kg/m$^3$]")
    ax3.set_title("Perturbed Density at Depth")
    ax3.legend(loc='upper right', handlelength=0.9)
    ax3.set_ylim(ymin = ranges['rho_fluc'][0]*4, ymax = ranges['rho_fluc'][-1]*4)
    ax3.set_xlim([0, time.max() / 3600 / 24])

    # buoyancy perturbations 
    ax4.plot(time[:it+1]/ 3600 / 24, bwfluc_mld, color = 'black', label = r"b'w$_{\text{MLD}}$")
    ax4.plot(time[:it+1]/ 3600 / 24, bwfluc_intrusion, color = 'cornflowerblue', label = r"b'w$_{\text{intrusion}}$")
    ax4.plot(time[:it+1]/ 3600 / 24, bwfluc_neutral, color = 'mediumblue', label = r"b'w$_{\text{neutral}}$")
    ax4.set_xlabel("Time [days]") 
    ax4.set_ylabel(r"[m$^{2}$/s$^{3}$]")
    ax4.set_title("Perturbed Buoyancy Flux at Depth")
    ax4.legend(loc='lower right', handlelength=0.9)
    ax4.set_ylim(ymin = ranges['bw_fluc'][0]*10**(3), ymax = ranges['bw_fluc'][-1]*10**(3))
    ax4.set_xlim([0, time.max() / 3600 / 24])

    # length scale ratio 
    ax5.plot(time[:it+1]/ 3600 / 24, l_scale)
    ax5.set_xlabel("Time [days]") 
    ax5.set_ylabel(r"r$_p$/h$_intrusion$")
    ax5.set_title("Lengthscale ratio")
    ax5.set_ylim(ymin = 0.0, ymax = 1.0)
    ax5.set_xlim([0, time.max() / 3600 / 24])

    # radius of plume 
    ax6.plot(time[:it+1]/ 3600 / 24, rp, color = 'black')
    ax6.set_xlim([0, time.max() / 3600 / 24])
    ax6.set_title("Horizontal Plume Spread")
    ax6.set_xlabel("Time [days]") 
    ax6.set_ylabel("Radius [m]")
    ax6.set_ylim(0, lx[0]/4)
    ax6.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True)

    # buoyancy
    ax7.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')
    ax7.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')
    ax7.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')
    ax7.set_xlim(ranges['b'])
    ax7.plot(b_avg, z, color='green', label = r"$\langle$b$\rangle_{\text{xy}}")
    ax7.plot(b_center, z, color='red', label = r"b$_{\text{centerline}}$")
    ax7.set_title("Buoyancy Profile")
    ax7.set_xlabel("[m/s$^{2}$]")
    ax7.set_ylabel("Depth [m]")
    ax7.set_ylim(-lx[2], 0)
    ax7.legend(loc='upper left', handlelength=0.9)
    ax7.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True)

    # RMS buoyancy 
    ax8.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')
    ax8.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')
    ax8.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')
    ax8.set_xlim(ranges['b_rms'])
    ax8.plot(b_rms, z, color='black')
    ax8.set_title("Buoyancy Root Mean Square Error")
    ax8.set_xlabel(r"[m/s$^{2}$]")
    ax8.set_ylim(-lx[2], 0)
    ax8.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True)

    # tracer average with depth  
    ax9.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')
    ax9.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')
    ax9.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')
    ax9.plot(tracer_avg, z, color='black')
    ax9.set_xlabel(r"S/S$_{source}$")
    ax9.set_ylabel("Depth [m]")
    ax9.set_title("Tracer")
    ax9.set_ylim(-lx[2], 0)
    ax9.set_xlim(xmin = ranges['Tracer'][0], xmax = ranges['Tracer'][-1]*10**(-2))
    ax9.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True)

    # Perturbed density plane slice
    norm = mcolors.Normalize(vmin=ranges['rho_fluc'][0], vmax=ranges['rho_fluc'][-1])
    ax10.contourf(Y[:, int(nx[1]/2), :], Z[:, int(nx[1]/2), :], rho_perturbed[:, int(nx[1]/2), :], levels, norm=norm, cmap='RdBu_r')
    ax10.set_ylim(-lx[2], 0)
    ax10.set_xlim(0, lx[1])
    ax10.set_ylabel("Depth [m]")
    ax10.set_aspect('equal')
    ax10.set_title("Perturbed Density")
    mappable = cm.ScalarMappable(norm=norm, cmap='RdBu_r')
    cbar = fig.colorbar(mappable, ax=ax10, label=r"[kg/m$^3$]", shrink=0.9, pad=0.1)#, fraction=0.046, pad=0.1) #anchor = (0.5, -0.4), orientation='horizontal', 
    cbar.formatter.set_scientific(True)

    # Buoyancy fluctuations plane slice
    norm = mcolors.Normalize(vmin=ranges['b_fluc'][0], vmax=ranges['b_fluc'][-1])
    ax11.contourf(Y[:, int(nx[1]/2), :], Z[:, int(nx[1]/2), :], b_fluc[:, int(nx[1]/2), :], levels, norm=norm, cmap='RdBu_r')
    ax11.set_xlabel("[m]")
    ax11.set_ylim(-lx[2], 0)
    ax11.set_xlim(0, lx[1])
    ax11.set_aspect('equal')
    ax11.set_title("Perturbed Buoyancy")
    mappable = cm.ScalarMappable(norm=norm, cmap='RdBu_r')
    cbar = fig.colorbar(mappable, ax=ax11, label=r"[m/s$^{2}$]", shrink=0.9, pad=0.1)
    cbar.formatter.set_powerlimits((-3, 2))
    cbar.update_ticks()

    # --- Save Frame ---
    frame_path = os.path.join(outdir, f"oc_frame_{it:04d}.png")
    plt.tight_layout()
    plt.savefig(frame_path)
    plt.close(fig)
    print(f"Time step {it + 1} captured: {frame_path}")
    plt.close(fig)
    return outdir # return the directory where frames are saved for video creation
## dense tracer buoyancy analysis via momentum
def plot_momentum_plume(time, it, ranges, fig_folder, lx, z, zf, mld, b_avg, tracer_avg, u_rms, v_rms, w_rms, b_rms, bu_fluc_avg, bv_fluc_avg, bw_fluc_avg, tracer_fluc, T_fluc, Q, M, F, B, wm, dm, bm, Ri, r_profile, b_center, plume_depths, ND = False):
    plume_depth_intrusion = plume_depths[0]
    plume_depth_neutral = plume_depths[1]

    outdir = os.path.join(fig_folder, 'plume momentum analysis/')
    os.makedirs(outdir, exist_ok=True)
    td = time[it] / 3600 / 24
    gridspec_kw={'height_ratios': [1, 1, 1, 1, 0.02]} # add space for universal legend
    fig, ax = plt.subplots(5, 4, figsize=(15, 18), gridspec_kw=gridspec_kw)
    for a in ax[4, :]:
        a.remove()
    fig.suptitle(f'{td:.2f} days', fontsize=12) 

    ax1 = ax[0, 0]
    ax2 = ax[0, 1]
    ax3 = ax[0, 2]
    ax4 = ax[0, 3]
    ax5 = ax[1, 0]
    ax6 = ax[1, 1]
    ax7 = ax[1, 2]
    ax8 = ax[1, 3]
    ax9 = ax[2, 0] # Q - volume flux
    ax10 = ax[2, 1] # M - momentum flux
    ax11 = ax[2, 2] # F - buoyancy flux
    ax12 = ax[2, 3] # B - buoyancy integral
    ax13 = ax[3, 0] # wm - characteristic velocity
    ax14 = ax[3, 1] # dm - characteristic length
    ax15 = ax[3, 2] # bm - characteristic buoyancy
    ax16 = ax[3, 3] # Ri - Richardson number

    if ND:
        ax1.set_ylabel(r"z/h$_{\text{MLD_{0}}}$")
        ax1.set_xlabel(r"u$_{i, rms}$/(h$_{\text{MLD_{0}}} \sqrt{N^{2}}$)")
        #ax2.set_ylabel(r"z/h$_{\text{MLD_{0}}}$")
        ax2.set_xlabel(r"S$(\text{h}_{mld} \sqrt{N^{2}}$)/(J$^{\text{S}}$)") #(r"S$\sqrt{g\text{r}_j}$/(J$^{\text{S}}$)")
        #ax3.set_ylabel(r"z/h$_{\text{MLD_{0}}}$")
        ax3.set_xlabel(r"b/(h$_{\text{MLD_{0}}} N^{2}$)")
        #ax4.set_ylabel(r"z/h$_{\text{MLD_{0}}}$")
        ax4.set_xlabel(r"S$'(\text{h}_{mld} \sqrt{N^{2}}$)/(J$^{\text{S}}$)") #(r"S$'\sqrt{g\text{r}_j}$/(J$^{\text{S}}$)")
        ax5.set_ylabel(r"z/h$_{\text{MLD_{0}}}$")
        ax5.set_xlabel(r"r/h$_{\text{MLD_{0}}}$") #(r"r/r$_{j}$")
        #ax6.set_ylabel(r"z/h$_{\text{MLD_{0}}}$")
        ax6.set_xlabel(r"$\langle$b'u'$_{i}\rangle_{\text{xy}}$/(h$_{\text{MLD_{0}}}^2 (N^{2})^{3/2}$)")
        #ax7.set_ylabel(r"z/h$_{\text{MLD_{0}}}$")
        ax7.set_xlabel(r"b$_{rms}$/(h$_{\text{MLD_{0}}} N^{2}$)")

        #ax8.set_ylabel(r"z/h$_{\text{MLD_{0}}}$")
        ax8.set_xlabel(r"$\langle$T$'\rangle_{\text{xy}}$/T$_{0}$")
    else:
        ax1.set_ylabel("Depth [m]")
        ax1.set_xlabel("[m/s]")
        #ax2.set_ylabel("[m]")
        ax2.set_xlabel(r"$\langle$C$\rangle_{\text{xy}}$[g/kg]")
        #ax3.set_ylabel("[m]")
        ax3.set_xlabel(r"[m/s$^2$]")
        #ax4.set_ylabel("[m]")
        ax4.set_xlabel(r"$\langle$C$'\rangle_{\text{xy}}$[g/kg]")
        ax5.set_ylabel("Depth [m]")
        ax5.set_xlabel("[m]")
        #ax6.set_ylabel("[m]")
        ax6.set_xlabel(r"[m$^2$/s$^3$]")
        #ax7.set_ylabel("[m]")
        ax7.set_xlabel(r"[m/s$^2$]")
        #ax8.set_ylabel("[m]")
        ax8.set_xlabel(r"$\langle$T$'\rangle_{\text{xy}}$[$^{\circ}$ C]")
        ax9.set_xlabel("[m]")
        ax9.set_xlabel(r"[m$^3$/s]")
        ax10.set_xlabel(r"[m$^4$/s$^2$]")
        ax11.set_xlabel(r"[m$^4$/s$^3$]")
        ax12.set_xlabel(r"[m$^3$/s$^2$]")
        ax13.set_xlabel(r"[m/s]")
        ax14.set_xlabel(r"[m]")
        ax15.set_xlabel(r"[-]")
        ax16.set_xlabel(r"[-]")

    # velocity rms
    mld_handle, = ax1.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')#, label = "MLD")
    intrusion_handle, = ax1.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')#, label = "Intrusion Depth")
    neutral_handle, = ax1.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')#, label = "Neutral Buoyancy")
    ax1.plot(u_rms, z, label=r"$\langle$u$_{\text{rms}}\rangle_{\text{xy}}$", color = 'blue')
    ax1.plot(v_rms, z, label=r"$\langle$v$_{\text{rms}}\rangle_{\text{xy}}$", color = 'green')
    ax1.plot(w_rms, zf, label=r"$\langle$w$_{\text{rms}}\rangle_{\text{xy}}$", color = 'red')
    ax1.set_title("Root Mean Square Velocities")
    ax1.set_ylim(-lx[2], 0)
    ax1.set_xlim(ranges['vel_rms'])
    ax1.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True)
    ax1.legend(loc='lower right')

    # tracer profile 
    ax2.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')#, label = "MLD")
    ax2.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')#, label = "Intrusion Depth")
    ax2.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')#, label = "Neutral Buoyancy")
    ax2.plot(tracer_avg, z, color = 'black')
    ax2.set_title('Tracer')
    ax2.set_ylim(-lx[2], 0)
    ax2.set_xlim(ranges['Tracer'])
    ax2.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True)

    # buoyancy profiles
    ax3.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')#, label = "MLD")
    ax3.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')#, label = "Intrusion Depth")
    ax3.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')#, label = "Neutral Buoyancy")
    ax3.plot(b_avg, z, color = 'black', label = r"$\langle$b$\rangle_{\text{xy}}$")
    ax3.plot(b_center, z, color = 'black', label = r"b$_{\text{centerline}}$", linestyle = 'dashed')
    ax3.set_title("Buoyancy Profile")
    ax3.set_ylim(-lx[2], 0)
    ax3.set_xlim(ranges['b_avg'])
    if ND:
        ax3.legend(loc='lower right')
    else:
        ax3.legend(loc='upper left')
    ax3.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True)

    # temperature fluctuations 
    ax4.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')#, label = "MLD")
    ax4.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')#, label = "Intrusion Depth")
    ax4.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')#, label = "Neutral Buoyancy")
    ax4.plot(tracer_fluc, z, color = 'black')
    ax4.set_title("Perturbed Tracer")
    ax4.set_ylim(-lx[2], 0)
    ax4.set_xlim(ranges['Tracer_fluc'])
    ax4.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)

    # tracer plume radius
    ax5.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')#, label = "MLD")
    ax5.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')#, label = "Intrusion Depth")
    ax5.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')#, label = "Neutral Buoyancy")
    ax5.plot(r_profile, z, color = 'black')
    ax5.set_title("Tracer Plume Radius with Depth")
    ax5.set_ylim(-lx[2], 0)
    ax5.set_xlim(0, lx[0]/4)

    # perturbed buoyancy flux 
    ax6.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')#, label = "MLD")
    ax6.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')#, label = "Intrusion Depth")
    ax6.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')#, label = "Neutral Buoyancy")
    ax6.plot(bu_fluc_avg, z, color = 'blue', label = r"$\langle$b'u'$\rangle_{\text{xy}}$")
    ax6.plot(bv_fluc_avg, z, color = 'green', label = r"$\langle$b'v'$\rangle_{\text{xy}}$")
    ax6.plot(bw_fluc_avg, z, color = 'red', label = r"$\langle$b'w'$\rangle_{\text{xy}}$")
    ax6.legend(loc='lower right')
    ax6.set_title("Buoyancy Flux Fluctuations")
    ax6.set_ylim(-lx[2], 0)
    ax6.set_xlim(ranges['bw_fluc'])
    ax6.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True) 

    # buoyancy brms 
    ax7.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')#, label = "MLD")
    ax7.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')#, label = "Intrusion Depth")
    ax7.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')#, label = "Neutral Buoyancy")
    ax7.plot(b_rms, z, color = 'black')
    ax7.set_title("Root Mean Square Buoyancy")
    ax7.set_ylim(-lx[2], 0)
    ax7.set_xlim(ranges['b_rms'])
    ax7.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True)

    # temperature fluctuations 
    ax8.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')#, label = "MLD")
    ax8.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')#, label = "Intrusion Depth")
    ax8.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')#, label = "Neutral Buoyancy")
    ax8.plot(T_fluc, z, color = 'black')
    ax8.set_title("Perturbed Temperature")
    ax8.set_ylim(-lx[2], 0)
    ax8.set_xlim(ranges['T_fluc'])

    # Q
    ax9.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')#, label = "MLD")
    ax9.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')#, label = "Intrusion Depth")
    ax9.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')#, label = "Neutral Buoyancy")
    ax9.plot(Q, z, color = 'black')
    ax9.set_title("Volume Flux")
    ax9.set_ylim(-lx[2], 0)
    ax9.set_xlim(ranges['Q'])

    # M
    ax10.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')#, label = "MLD")
    ax10.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')#, label = "Intrusion Depth")
    ax10.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')#, label = "Neutral Buoyancy")
    ax10.plot(M, z, color = 'black')
    ax10.set_title("Momentum Flux")
    ax10.set_ylim(-lx[2], 0)
    ax10.set_xlim(ranges['M'])

    # F perturbed 
    ax11.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')#, label = "MLD")
    ax11.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')#, label = "Intrusion Depth")
    ax11.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')#, label = "Neutral Buoyancy")
    ax11.plot(F, z, color = 'black')
    ax11.set_title("Buoyancy Perturbed Flux")
    ax11.set_ylim(-lx[2], 0)
    ax11.set_xlim(xmin = ranges['F'][0], xmax = ranges['F'][-1])

    # B
    ax12.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')#, label = "MLD")
    ax12.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')#, label = "Intrusion Depth")
    ax12.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')#, label = "Neutral Buoyancy")
    ax12.plot(B, z, color = 'black')
    ax12.set_title("Buoyancy Integral")
    ax12.set_ylim(-lx[2], 0)
    ax12.set_xlim(xmin = ranges['B'][0], xmax = ranges['B'][-1])

    # wm
    ax13.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')#, label = "MLD")
    ax13.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')#, label = "Intrusion Depth")
    ax13.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')#, label = "Neutral Buoyancy")
    ax13.plot(wm, z, color = 'red')
    ax13.set_title("Characteristic Velocity")
    ax13.set_ylim(-lx[2], 0)
    ax13.set_xlim(xmin = ranges['w'][0]*10.0, xmax = ranges['w'][-1]*10.0)

    # dm
    ax14.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')#, label = "MLD")
    ax14.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')#, label = "Intrusion Depth")
    ax14.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')#, label = "Neutral Buoyancy")
    ax14.plot(dm, z, color = 'black')
    ax14.set_title("Characteristic Length")
    ax14.set_ylim(-lx[2], 0)
    ax14.set_xlim(xmin = 0, xmax = lx[0]/4)

    # Ri
    ax15.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')#, label = "MLD")
    ax15.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')#, label = "Intrusion Depth")
    ax15.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')#, label = "Neutral Buoyancy")
    ax15.plot(Ri, z, color = 'black')
    ax15.set_title("Richardson Number")
    ax15.set_ylim(-lx[2], 0)
    ax15.set_xlim(ranges['Ri'])

    fig.legend(handles=[mld_handle, intrusion_handle, neutral_handle], labels=["MLD", "Intrusion Depth", "Neutral Buoyancy Depth"],
            loc='lower center',
            ncol=3,
            bbox_to_anchor=(0.52, 0.015))

    # --- Save Frame ---
    frame_path = os.path.join(outdir, f"plume_momentum_analysis_{it:04d}.png")
    plt.savefig(frame_path)
    plt.close(fig)
    print(f"Time step {it + 1} captured: {frame_path}")

    return outdir # return the directory where frames are saved for video creation
