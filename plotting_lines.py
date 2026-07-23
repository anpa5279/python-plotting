import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from matplotlib.lines import Line2D
from matplotlib import cm
### -------------------------PLOTTING 1D LINES FUNCTIONS------------------------- ###
## temporal average ###
def temporal_avg(t_range,ranges, color_opt, fig_folder, case_names, lx, z, w_center, S_center, b_fluc_center, w_rms, b_rms, h_ml=None):
    num_cases = len(case_names)
    if num_cases==1:
        fig, axes = plt.subplots(1, 5, figsize=(12, 4), sharey = True)
        outdir = os.path.join(fig_folder)
        os.makedirs(outdir, exist_ok=True)
    else:
        outdir = os.path.join(fig_folder, 'temporal analysis/')
        os.makedirs(outdir, exist_ok=True)
        gridspec_kw={'height_ratios': [1, 0.1]} # add space for universal legend
        fig, axes = plt.subplots(2, 5, figsize=(12, 6), gridspec_kw=gridspec_kw, sharey = True)
        for a in axes[-1, :]:
            a.remove()
        case_handles = [
            Line2D([0], [0], color=color_opt[i], linestyle='solid', label=case_names[i])
            for i in range(num_cases)
        ]
        if num_cases > 4:
            ncols = 4
        else:
            ncols = num_cases
        fig.legend(handles=case_handles,
                loc='lower center',
                ncol=ncols,
                bbox_to_anchor=(0.52, 0.0))
    axes = axes.ravel()
    axes[0].set_title(rf"$\langle$w(0, 0, z)$\rangle_{{\text{{t={t_range[0]:.2f}-{t_range[1]:.2f} days}}}}$")
    axes[0].set_ylabel("Depth [m]")
    axes[0].set_xlabel("[m/s]")
    axes[0].set_ylim(ymin = -max(lx[2]), ymax = 0)
    axes[0].set_xlim(xmin = ranges['w'][0], xmax = ranges['w'][1])
    axes[1].set_title(rf"$\langle$S(0, 0, z)$\rangle_{{\text{{t={t_range[0]:.2f}-{t_range[1]:.2f} days}}}}$")
    axes[1].set_xlabel("[g/kg]")
    axes[1].set_xlim(xmin = ranges['S'][0], xmax = ranges['S'][1])
    axes[2].set_title(rf"$\langle$b'(0, 0, z)$\rangle_{{\text{{t={t_range[0]:.2f}-{t_range[1]:.2f} days}}}}$")
    axes[2].set_xlabel(r"[m/s$^2$]")
    axes[2].set_xlim(xmin = ranges['b_fluc'][0], xmax = ranges['b_fluc'][1])
    axes[3].set_title(rf"$\langle$w$_{{rms}}\rangle_{{\text{{t={t_range[0]:.2f}-{t_range[1]:.2f} days}}}}$")
    axes[3].set_xlabel("[m/s]")
    axes[3].set_xlim(xmin = ranges['vel_rms'][0], xmax = ranges['vel_rms'][1])
    axes[4].set_title(rf"$\langle$b$_{{rms}}\rangle_{{\text{{t={t_range[0]:.2f}-{t_range[1]:.2f} days}}}}$")
    axes[4].set_xlabel(r"[m/s$^2$]")
    axes[4].set_xlim(xmin = ranges['b_rms'][0], xmax = ranges['b_rms'][1])
    if h_ml is not None:
        h_ml = np.asarray(h_ml)
        h_ml = np.unique(h_ml)
        for opt, h in enumerate(h_ml):
            for ax in axes:
                ax.plot([-1*10**6, 1*10**6], [-h, -h], color_opt[opt], linewidth = 0.5, linestyle = 'dashed')
    for n in range(num_cases):
        axes[0].plot(w_center[n], z[n], color = color_opt[n], linewidth = 0.75)
        axes[1].plot(S_center[n], z[n], color = color_opt[n], linewidth = 0.75)
        axes[2].plot(b_fluc_center[n], z[n], color = color_opt[n], linewidth = 0.75)
        axes[3].plot(w_rms[n], z[n], color = color_opt[n], linewidth = 0.75)
        axes[4].plot(b_rms[n], z[n], color = color_opt[n], linewidth = 0.75)

    frame_path = os.path.join(outdir, f"{t_range[0]:.2f}-{t_range[1]:.2f}-days-avg.svg")
    plt.savefig(frame_path)
    plt.close(fig)
## temporal analysis ###
def plume_temporal_analysis(time, ranges, color_opt, fig_folder, case_names, name, lx, start_neutral, mld, h_neutral, h_max, r_mld, r_neutral, r_hmax, w_mld, w_neutral, w_hmax, b_mld, b_neutral, b_hmax, T_mld, T_neutral, T_hmax, tracer_mld, tracer_neutral, tracer_hmax, tracerw_fluc_avg, Tw_fluc_avg, ND = False):
    num_cases = len(case_names)
    if num_cases==1:
        fig, axes = plt.subplots(2, 4, figsize=(12, 5))
        outdir = os.path.join(fig_folder, 'plume analysis/')
        os.makedirs(outdir, exist_ok=True)
    else:
        outdir = os.path.join(fig_folder, 'comparison plume analysis/')
        os.makedirs(outdir, exist_ok=True)
        gridspec_kw={'height_ratios': [1, 1, 0.1]} # add space for universal legend
        fig, axes = plt.subplots(3, 4, figsize=(12, 6.5), gridspec_kw=gridspec_kw)
        for a in axes[2, :]:
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
    ax1 = axes[0, 0] # depth of plume through time 
    ax2 = axes[0, 1] # max and average radius of plume through time 
    ax3 = axes[0, 2] # vertical velocity at depth through time
    ax4 = axes[0, 3] # perturbed buoyancy at depth through time
    ax5 = axes[1, 0] # perturbed Temperature at depth through time
    ax6 = axes[1, 1] # perturbed tracer at depth through time
    ax7 = axes[1, 2] # average tracer at MLD through time
    ax8 = axes[1, 3] # w_avg at MLD through time 
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
## turbulent statistics plotting
def plot_turb_stats_bin(time, it, ranges, color_opt, fig_folder, case_names, z, u_rms, w_rms, uw, b_avg, bur_fluc_avg, bw_fluc_avg, Tu, Tw, Cu, Cw):
    num_cases = len(case_names)
    ncols = 3
    nrows = 2
    outdir = os.path.join(fig_folder, 'binning', 'turbulent statistics')
    if it == 0:
        os.makedirs(outdir, exist_ok=True)
    ar = np.ones(nrows + 1)
    ar[-1] = 0.02 # add space for universal legend
    gridspec_kw={'height_ratios': ar} # add space for universal legend
    fig, axes = plt.subplots(nrows + 1, ncols, figsize=(15, 14), gridspec_kw=gridspec_kw, sharey = True)
    plt.subplots_adjust(bottom=0.05)
    for a in axes[2, :]:
        a.remove()
    case_handles = [
        Line2D([0], [0], color=color_opt[i], linestyle='solid', label=case_names[i])
        for i in range(num_cases)
    ]
    fig.legend(handles=case_handles,
            loc='lower center',
            ncol=num_cases,
            bbox_to_anchor=(0.52, 0.001))
    axes = axes.ravel()
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
    axes[0].set_ylabel("Depth [m]")
    axes[0].set_xlabel("[m/s]")
    axes[1].set_xlabel(r"[m$^2$/s$^2$]")
    axes[2].set_xlabel(r"[m$^2$/s$^3$]")

    axes[3].set_ylabel("Depth [m]")
    axes[3].set_xlabel(r"[$^{\circ}$ C$\cdot$m/s]")
    axes[4].set_xlabel(r"[g/kg$\cdot$m/s]")
    axes[5].set_xlabel(r"[m/s$^2$]")
    for i in range(num_cases):
        zmin_i = np.min(z[i])
        z_min = zmin_i if i == 0 else min(z_min, zmin_i)
        zmax_i = np.max(z[i])
        z_max = zmax_i if i == 0 else max(z_max, zmax_i)
    for a in axes:
        a.set_ylim(ymin = z_min, ymax = z_max)
    # velocity rms
    for i in range(num_cases):
        if i == 0:
            axes[0].plot(u_rms[i], z[i], label=r"$\langle$u$_{r,\text{rms}}\rangle_{\text{xy}}$", color = color_opt[i], linestyle='dotted', linewidth = 0.75)
            axes[0].plot(w_rms[i], z[i], label=r"$\langle$w$_{\text{rms}}\rangle_{\text{xy}}$", color = color_opt[i], linestyle='solid', linewidth = 0.75)
        else:
            axes[0].plot(u_rms[i], z[i], color = color_opt[i], linestyle='dotted', linewidth = 0.75)
            axes[0].plot(w_rms[i], z[i], color = color_opt[i], linestyle='solid', linewidth = 0.75)
    axes[0].set_title("Root Mean Square Velocities")
    axes[0].set_xlim(ranges['vel_rms'])
    axes[0].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
    axes[0].legend(loc='lower right')

    # reynolds stresses
    for i in range(num_cases):
        axes[1].plot(uw[i], z[i], color = color_opt[i], linestyle='solid', linewidth = 0.75)
    axes[1].set_title(r"Reynolds Stresses, $\langle$u$_r$'w'$\rangle_{\text{xy}}$")
    axes[1].set_xlim(ranges['restress'])
    axes[1].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)

    # perturbed buoyancy flux 
    for i in range(num_cases):
        if i == 0:
            axes[2].plot(bur_fluc_avg[i], z[i], color = color_opt[i], label = r"$\langle$b'u$_r\rangle_{\text{xy}}$", linestyle='dotted', linewidth = 0.75)
            axes[2].plot(bw_fluc_avg[i], z[i], color = color_opt[i], label = r"$\langle$b'w$\rangle_{\text{xy}}$", linestyle='solid', linewidth = 0.75)
        else:
            axes[2].plot(bur_fluc_avg[i], z[i], color = color_opt[i], linestyle='dotted', linewidth = 0.75)
            axes[2].plot(bw_fluc_avg[i], z[i], color = color_opt[i], linestyle='solid', linewidth = 0.75)
    axes[2].legend(loc='upper left')
    axes[2].set_title("Perturbed Buoyancy Flux")
    axes[2].set_xlim(ranges['bw_fluc'])
    axes[2].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True) 

    # perturbed temperature flux 
    for i in range(num_cases):
        if i == 0:
            axes[3].plot(Tu[i], z[i], color = color_opt[i], label = r"$\langle$T'u$_r\rangle_{\text{xy}}$", linestyle='dotted', linewidth = 0.75)
            axes[3].plot(Tw[i], z[i], color = color_opt[i], label = r"$\langle$T'w$\rangle_{\text{xy}}$", linestyle='solid', linewidth = 0.75)
        else:
            axes[3].plot(Tu[i], z[i], color = color_opt[i], linestyle='dotted', linewidth = 0.75)
            axes[3].plot(Tw[i], z[i], color = color_opt[i], linestyle='solid', linewidth = 0.75)
    axes[3].legend(loc='upper right')
    axes[3].set_title("Perturbed Temperature Flux")
    axes[3].set_xlim(ranges['Tw_fluc'])
    axes[3].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True) 

    # tracer flux 
    for i in range(num_cases):
        if i == 0:
            axes[4].plot(Cu[i], z[i], color = color_opt[i], label = r"$\langle$Cu$_r\rangle_{\text{xy}}$", linestyle='dotted', linewidth = 0.75)
            axes[4].plot(Cw[i], z[i], color = color_opt[i], label = r"$\langle$Cw$\rangle_{\text{xy}}$", linestyle='solid', linewidth = 0.75)
        else:
            axes[4].plot(Cu[i], z[i], color = color_opt[i], linestyle='dotted', linewidth = 0.75)
            axes[4].plot(Cw[i], z[i], color = color_opt[i], linestyle='solid', linewidth = 0.75)
    axes[4].legend(loc='lower right')
    axes[4].set_title("Tracer Flux")
    axes[4].set_xlim(ranges['Cw'])
    axes[4].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True) 

    # buoyancy profile
    for i in range(num_cases):
        axes[5].plot(b_avg[i], z[i], color = color_opt[i], linewidth = 0.75)
    axes[5].set_title("Buoyancy Profile")
    axes[5].set_xlim(ranges['b_avg'])
    axes[5].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True) 

    # --- Save Frame ---
    frame_path = os.path.join(outdir, f"comparison_vert_buoyancy_{it:04d}.png")
    plt.savefig(frame_path)
    plt.close(fig)
    print(f"Time step {it + 1} captured: {frame_path}")

    return outdir # return the directory where frames are saved for video creation
### -------------------------PLOTTING DENSE PLUME FUNCTIONS------------------------- ###
## buoyancy analysis 
def buoyancy_analysis_plot(time, it, ranges, fig_folder, lx, nx, z, zf, X, Z, mld, b_avg, w_avg, b_center, w_center, b_rms, bur_fluc_avg, bw_fluc_avg, b_fluc, rho_perturbed, Ri_avg, Ri_strat, Ri_plume, plume_depth_intrusion, plume_depth_neutral, w_neutral, w_intrusion, w_mld, rho_perturbed_neutral, rho_perturbed_intrusion, rho_perturbed_mld, bwfluc_neutral, bwfluc_intrusion, bwfluc_mld):

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
    ax9.plot(bur_fluc_avg, z, color='black', label = r"b'u$_r$")
    ax9.plot(bw_fluc_avg, z, color='red', label = r"b'w")
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
## spatial vertical analysis ###
def plot_plume_vertical_spatial(time, ranges, color_opt, fig_folder, case_names, name, lx, z, tracer_avg, u_rms, v_rms, w_rms, b_avg, b_center, r_profile, bur_fluc_avg, bw_fluc_avg, T_avg, T_fluc, tracer, ND = False, z_nd = r"(z - h$_{\mathrm{MLD}_0}$)/l$_{j}$"):
    num_cases = len(case_names)
    outdir = os.path.join(fig_folder, 'vertical centerline-' + name)
    os.makedirs(outdir, exist_ok=True)
    if num_cases>1:
        case_handles = [
            Line2D([0], [0], color=color_opt[i], linestyle='solid', label=case_names[i])
            for i in range(num_cases)
        ]
    for it, t in enumerate(time):
        if num_cases==1:
            fig, axes = plt.subplots(2, 4, figsize=(1, 8))
        else:
            gridspec_kw={'height_ratios': [1, 1, 0.05]} # add space for universal legend
            fig, axes = plt.subplots(3, 4, figsize=(13, 10), gridspec_kw=gridspec_kw)
            for a in axes[2, :]:
                a.remove()
            fig.legend(handles=case_handles,
                    loc='lower center',
                    ncol=num_cases,
                    bbox_to_anchor=(0.52, 0.0))

        td = t / 3600 / 24
        fig.suptitle(f'{td:.2f} days', fontsize=12)

        ax1 = axes[0, 0]
        ax2 = axes[0, 1]
        ax3 = axes[0, 2]
        ax4 = axes[0, 3]
        ax5 = axes[1, 0]
        ax6 = axes[1, 1]
        ax7 = axes[1, 2]
        ax8 = axes[1, 3]
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
            ax4.set_xlabel(r"C$_{\text{centerline}}$ [g/kg]")
            ax5.set_ylabel("Depth [m]")
            ax5.set_xlabel("[m]")
            ax6.set_xlabel(r"[m$^2$/s$^3$]")
            ax7.set_xlabel(r"$\langle$T$\rangle_{\text{xy}}$ [$^{\circ}$ C]")
            ax8.set_xlabel(r"T$'_{\text{centerline}}$ [$^{\circ}$ C]")

        # velocity rms
        for i in range(num_cases):
            if i == 0:
                ax1.plot(u_rms[i][it], z[i], label=r"$\langle$u$_{\text{rms}}\rangle_{\text{xy}}$", color = color_opt[i], linestyle='dotted', linewidth = 0.75)
                ax1.plot(v_rms[i][it], z[i], label=r"$\langle$v$_{\text{rms}}\rangle_{\text{xy}}$", color = color_opt[i], linestyle='dashed', linewidth = 0.75)
                ax1.plot(w_rms[i][it], z[i], label=r"$\langle$w$_{\text{rms}}\rangle_{\text{xy}}$", color = color_opt[i], linestyle='solid', linewidth = 0.75)
            else:
                ax1.plot(u_rms[i][it], z[i], color = color_opt[i], linestyle='dotted', linewidth = 0.75)
                ax1.plot(v_rms[i][it], z[i], color = color_opt[i], linestyle='dashed', linewidth = 0.75)
                ax1.plot(w_rms[i][it], z[i], color = color_opt[i], linestyle='solid', linewidth = 0.75)
        ax1.set_title("Root Mean Square Velocities")
        ax1.set_ylim(ymin = -min(lx[-1, :]), ymax = 0.0)
        ax1.set_xlim(ranges['vel_rms'])
        ax1.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax1.legend(loc='lower right')

        # tracer profile 
        for i in range(num_cases):
            ax2.plot(tracer_avg[i][it], z[i], color = color_opt[i], linestyle='solid', linewidth = 0.75)
        ax2.set_title('Tracer Profile')
        ax2.set_ylim(ymin = -min(lx[-1, :]), ymax = 0.0)
        ax2.set_xlim(ranges['Tracer_avg'])
        ax2.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)

        # buoyancy profiles
        for i in range(num_cases):
            if i == 0:
                ax3.plot(b_avg[i][it], z[i], color = color_opt[i], label = r"$\langle$b$\rangle_{\text{xy}}$", linestyle='solid', linewidth = 0.75)
                ax3.plot(b_center[i][it], z[i], color = color_opt[i], label = r"b$_{\text{centerline}}$", linestyle='dashed', linewidth = 0.75)
            else:
                ax3.plot(b_avg[i][it], z[i], color = color_opt[i], linestyle='solid', linewidth = 0.75)
                ax3.plot(b_center[i][it], z[i], color = color_opt[i], linestyle='dashed', linewidth = 0.75)
        ax3.set_title("Buoyancy Profile")
        ax3.set_ylim(ymin = -min(lx[-1, :]), ymax = 0.0)
        ax3.set_xlim(ranges['b_avg'])
        ax3.legend(loc='upper left')
        ax3.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)

        # temperature fluctuations 
        for i in range(num_cases):
            ax4.plot(tracer[i][it], z[i], color = color_opt[i], linestyle='solid', linewidth = 0.75)
        ax4.set_title("Tracer")
        ax4.set_ylim(ymin = -min(lx[-1, :]), ymax = 0.0)
        ax4.set_xlim(ranges['Tracer'])
        ax4.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)

        # plume radius
        for i in range(num_cases):
            ax5.plot(r_profile[i][:, it], z[i], color = color_opt[i], linestyle='solid', linewidth = 0.75)
        ax5.set_title("Plume Radius with Depth")
        ax5.set_ylim(ymin = -min(lx[-1, :]), ymax = 0.0)
        ax5.set_xlim(0, min(lx[0, :])/1.9)

        # perturbed buoyancy flux 
        for i in range(num_cases):
            if i == 0:
                ax6.plot(bur_fluc_avg[i][it], z[i], color = color_opt[i], label = r"$\langle$b'u$_r\rangle_{\text{xy}}$", linestyle='dotted', linewidth = 0.75)
                ax6.plot(bw_fluc_avg[i][it], z[i], color = color_opt[i], label = r"$\langle$b'w$\rangle_{\text{xy}}$", linestyle='solid', linewidth = 0.75)
            else:
                ax6.plot(bur_fluc_avg[i][it], z[i], color = color_opt[i], linestyle='dotted', linewidth = 0.75)
                ax6.plot(bw_fluc_avg[i][it], z[i], color = color_opt[i], linestyle='solid', linewidth = 0.75)
        ax6.legend(loc='lower right')
        ax6.set_title("Buoyancy Flux Fluctuations")
        ax6.set_ylim(ymin = -min(lx[-1, :]), ymax = 0.0)
        ax6.set_xlim(ranges['bw_fluc'])
        ax6.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True) 

        # average temperature
        for i in range(num_cases):
            ax7.plot(T_avg[i][it], z[i], color = color_opt[i], linestyle='solid', linewidth = 0.75)
        ax7.set_title("Temperature")
        ax7.set_ylim(ymin = -min(lx[-1, :]), ymax = 0.0)
        ax7.set_xlim(ranges['T'])
        ax7.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)

        # temperature fluctuations 
        for i in range(num_cases):
            ax8.plot(T_fluc[i][it], z[i], color = color_opt[i], linestyle='solid', linewidth = 0.75)
        ax8.set_title("Perturbed Temperature")
        ax8.set_ylim(ymin = -min(lx[-1, :]), ymax = 0.0)
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
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        outdir = os.path.join(fig_folder, 'horizontal centerline-' + name)
        os.makedirs(outdir, exist_ok=True)
    else:
        outdir = os.path.join(fig_folder, 'horizontal centerline-' + name)
        os.makedirs(outdir, exist_ok=True)
        gridspec_kw={'height_ratios': [1, 1, 0.02]} # add space for universal legend
        fig, axes = plt.subplots(3, 3, figsize=(12, 9), gridspec_kw=gridspec_kw)
        for a in axes[2, :]:
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

    ax1 = axes[0, 0] # u, v, w through horizontal centerline
    ax2 = axes[0, 1] # horizontal buoyancy flux through horizontal centerline
    ax3 = axes[0, 2] # tracer through horizontal centerline
    ax4 = axes[1, 0] # perturbed buoyancy through horizontal centerline
    ax5 = axes[1, 1] # vertical buoyancy flux through horizontal centerline
    ax6 = axes[1, 2] # temperature through horizontal centerline

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
## plume depths
def plot_plume_depths(time, color_opt, fig_folder, case_names, lx, zp, zneutral, zc, contour, trend = True):
    num_cases = len(case_names)
    ncols = 3
    nrows = 2
    outdir = os.path.join(fig_folder, 'binning', 'plume_depths')
    os.makedirs(outdir, exist_ok=True)
    ar = np.ones(nrows)
    ar[-1] = 0.05 # add space for universal legend
    gridspec_kw={'height_ratios': ar} # add space for universal legend
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 6), gridspec_kw=gridspec_kw, sharey = True)
    for a in axes[-1, :]:
        a.remove()
    case_handles = [
        Line2D([0], [0], color=color_opt[i], linestyle='solid', label=case_names[i])
        for i in range(num_cases)
    ]
    fig.legend(handles=case_handles,
            loc='lower center',
            ncol=num_cases,
            bbox_to_anchor=(0.52, 0.01))
    ax = axes.ravel()
    """
    ax[0] = zp, where w = 0
    ax[1] = zneutral, where buoyancy = 0
    ax[2] = zc, where tracer is at the contour value
    """
    ax[0].set_ylabel("Depth [m]")
    ax[0].set_xlabel("Time [hrs]")
    ax[1].set_xlabel("Time [hrs]")
    ax[2].set_xlabel("Time [hrs]")

    ax[0].set_title("Depth of w = 0")
    ax[1].set_title("Depth of Neutral Buoyancy")
    ax[2].set_title(rf"Depth of Tracer Contour = {contour:.4f}")

    for i in range(num_cases):
        if i == 0:
            tmax = time[i].max() / 3600
            ax[0].plot(time[i]/3600, zp[i], label=r"z$_{w=0}$", color = color_opt[i], linewidth = 0.75)
            ax[1].plot(time[i]/3600, zneutral[i], label=r"z$_{b=0}$", color = color_opt[i], linewidth = 0.75)
            ax[2].plot(time[i]/3600, zc[i], label=rf"z$_{{contour = {contour:.3f}}}$", color = color_opt[i], linewidth = 0.75)
        else:
            tmax = max(tmax, time[i].max() / 3600)
            ax[0].plot(time[i]/3600, zp[i], color = color_opt[i], linewidth = 0.75)
            ax[1].plot(time[i]/3600, zneutral[i], color = color_opt[i], linewidth = 0.75)
            ax[2].plot(time[i]/3600, zc[i], color = color_opt[i], linewidth = 0.75)
        if trend:
            start = 10
            print(time[i][start:]/3600)
            print(zp[i][start:])
            vars = np.polyfit(time[i][start:]/3600, zp[i][start:], 1)
            z_trend = time[i]/3600 * vars[0] + vars[1]
            ax[0].plot(time[i]/3600, z_trend, label=rf"z = {vars[0]:.2f}t + {vars[1]:.2f}", color = color_opt[i], linestyle = '--', linewidth = 0.5)
            ax[0].plot(time[i]/3600, z_trend, label=rf"z = {vars[0]:.2f}t + {vars[1]:.2f}", color = color_opt[i], linestyle = '--', linewidth = 0.5)
    for a in ax:
        a.set_ylim(ymin = -lx[-1].max(), ymax = 0.0)
        a.set_xlim(0, tmax)
        a.legend(loc='upper right')

    # --- Save Frame ---
    frame_path = os.path.join(outdir, f"comparison_depths_{contour}.svg")
    plt.savefig(frame_path)
    plt.close(fig)
    return outdir # return the directory where frames are saved for video creation 