import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import imageio.v2 as imageio
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
### ---------------------- PLOTTING PARAMETERS--------------------------- ###
def plot_format(ncases):
    colors = ['black', 'red', 'blue', 'green', 'orange', 'purple', 'pink', 'gray', 'olive', 'cyan', 'magenta']
    line_styles = ['solid', 'dashed', 'dotted', 'dashdot', 'dashdotted']

    return colors[:ncases], line_styles[:ncases]

### ---------------------- TURB STATS ----------------------------- ###
def turb_stats_multi(time, it, ranges, color_opt, line_opt, fig_folder, case_names, name, lx, z, zf, u_avg, v_avg, w_avg, u_rms, v_rms, w_rms, uv_fluc, uw_fluc, vw_fluc, bu_fluc_avg, bv_fluc_avg, bw_fluc_avg, b_rms, rho, ND = False):
    outdir = os.path.join(fig_folder, 'comparison turb stats/')
    os.makedirs(outdir, exist_ok=True)
    num_cases = len(case_names)
    if num_cases==0:
        fig, ax = plt.subplots(2, 3, figsize=(11, 8))
    else:
        gridspec_kw={'height_ratios': [1, 1, 0.02]} # add space for universal legend
        fig, ax = plt.subplots(3, 3, figsize=(11, 10), gridspec_kw=gridspec_kw)
        for a in ax[2, :]:
            a.remove()

    if it > 0:
        td = time[it] / 3600 / 24
        fig.suptitle(f'{td:.2f} days', fontsize=12)

    case_handles = [
        Line2D([0], [0], color=color_opt[i], linestyle='solid', label=case_names[i])
        for i in range(num_cases)
    ]

    fig.legend(handles=case_handles,
            loc='lower center',
            ncol=num_cases,
            bbox_to_anchor=(0.52, 0.015))

    ax1 = ax[0, 0]
    ax2 = ax[0, 1]
    ax3 = ax[0, 2]
    ax4 = ax[1, 0]
    ax5 = ax[1, 1]
    ax6 = ax[1, 2]
    # velocity profiles
    for i in range(num_cases):
        if i == 0:
            ax1.plot(u_avg[i, :], z, label=r"$\langle$u$\rangle_{\text{xy}}$", color = color_opt[i], linestyle=line_opt[i])
            ax1.plot(v_avg[i, :], z, label=r"$\langle$v$\rangle_{\text{xy}}$", color = color_opt[i], linestyle=line_opt[i])
            ax1.plot(w_avg[i, :], zf, label=r"$\langle$w$\rangle_{\text{xy}}$", color = color_opt[i], linestyle=line_opt[i])
        else:
            ax1.plot(u_avg[i, :], z, color = color_opt[i], linestyle=line_opt[i])
            ax1.plot(v_avg[i, :], z, color = color_opt[i], linestyle=line_opt[i])
            ax1.plot(w_avg[i, :], zf, color = color_opt[i], linestyle=line_opt[i])
    ax1.set_xlabel("[m/s]")
    ax1.set_ylabel("Depth [m]")
    ax1.set_title('Velocity Profiles')
    ax1.set_ylim(ymin = np.min(zf), ymax = np.max(zf))
    ax1.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
    ax1.set_xlim(ranges['vel'])
    ax1.legend(loc='lower right', handlelength=0.75)

    # velocity rms
    for i in range(num_cases):
        if i == 0:
            ax2.plot(u_rms[i, :], z, label=r"$\langle$u$_{rms}\rangle_{\text{xy}}$", color = color_opt[i], linestyle=line_opt[i])
            ax2.plot(v_rms[i, :], z, label=r"$\langle$v$_{rms}\rangle_{\text{xy}}$", color = color_opt[i], linestyle=line_opt[i])
            ax2.plot(w_rms[i, :], zf, label=r"$\langle$w$_{rms}\rangle_{\text{xy}}$", color = color_opt[i], linestyle=line_opt[i])
        else:
            ax2.plot(u_rms[i, :], z, color = color_opt[i], linestyle=line_opt[i])
            ax2.plot(v_rms[i, :], z, color = color_opt[i], linestyle=line_opt[i])
            ax2.plot(w_rms[i, :], zf, color = color_opt[i], linestyle=line_opt[i])
    ax2.set_xlabel("[m/s]")
    ax2.set_title("Root Mean Square Velocities")
    ax2.set_ylim(ymin = np.min(zf), ymax = np.max(zf))
    ax2.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
    ax2.set_xlim(ranges['vel_rms'])
    ax2.legend(loc='lower right', handlelength=0.75)

    # reynolds stresses 
    for i in range(num_cases):
        if i == 0:
            ax3.plot(uv_fluc[i, :], z, label=r"$\langle$u'v'$\rangle_{\text{xy}}$", color = color_opt[i], linestyle=line_opt[i])
            ax3.plot(uw_fluc[i, :], z, label=r"$\langle$u'w'$\rangle_{\text{xy}}$", color = color_opt[i], linestyle=line_opt[i])
            ax3.plot(vw_fluc[i, :], z, label=r"$\langle$v'w'$\rangle_{\text{xy}}$", color = color_opt[i], linestyle=line_opt[i])
        else:
            ax3.plot(uv_fluc[i, :], z, color = color_opt[i], linestyle=line_opt[i])
            ax3.plot(uw_fluc[i, :], z, color = color_opt[i], linestyle=line_opt[i])
            ax3.plot(vw_fluc[i, :], z, color = color_opt[i], linestyle=line_opt[i])
    ax3.set_xlabel(r"[m$^2$/s$^2$]")
    ax3.set_title("Reynolds Stresses")
    ax3.set_ylim(ymin = np.min(zf), ymax = np.max(zf))
    ax3.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
    ax3.set_xlim(ranges['restress'])
    ax3.legend(loc='lower right', handlelength=0.75)

    # density profiles
    for i in range(num_cases):
        ax4.plot(rho[i, :], z, color = color_opt[i], linestyle=line_opt[i])
    ax4.set_ylabel("Depth [m]")
    ax4.set_title("Density Profiles")
    ax4.set_xlabel(r"[kg/m$^3$]")
    ax4.set_ylim(ymin = np.min(zf), ymax = np.max(zf))
    ax4.set_xlim(ranges['rho'])

    # perturbed buoyancy flux 
    for i in range(num_cases):
        if i == 0:
            ax5.plot(bu_fluc_avg[i, :], z, color = color_opt[i], label = r"$\langle$b'u'$\rangle_{\text{xy}}$", linestyle=line_opt[i])
            ax5.plot(bv_fluc_avg[i, :], z, color = color_opt[i], label = r"$\langle$b'v'$\rangle_{\text{xy}}$", linestyle=line_opt[i])
            ax5.plot(bw_fluc_avg[i, :], z, color = color_opt[i], label = r"$\langle$b'w'$\rangle_{\text{xy}}$", linestyle=line_opt[i])
        else:
            ax5.plot(bu_fluc_avg[i, :], z, color = color_opt[i], linestyle=line_opt[i])
            ax5.plot(bv_fluc_avg[i, :], z, color = color_opt[i], linestyle=line_opt[i])
            ax5.plot(bw_fluc_avg[i, :], z, color = color_opt[i], linestyle=line_opt[i])
    ax5.legend(loc='lower right', handlelength=0.75)
    ax5.set_xlabel(r"[m$^{2}$/s$^{3}$]")
    ax5.set_title("Buoyancy Flux Fluctuations")
    ax5.set_ylim(ymin = np.min(zf), ymax = np.max(zf))
    ax5.set_xlim(ranges['bw_fluc'])
    ax5.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True) 

    # buoyancy brms 
    for i in range(num_cases):
        ax6.plot(b_rms[i, :], z, color = color_opt[i], linestyle=line_opt[i])
    ax6.set_xlabel(r"[m/s$^2$]")
    ax6.set_title("Root Mean Square Buoyancy")
    ax6.set_ylim(ymin = np.min(zf), ymax = np.max(zf))
    ax6.set_xlim(ranges['b_rms'])
    ax6.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)

    # --- Save Frame ---
    frame_path = os.path.join(outdir, f"{name}_comparison_turb_stats_{it:04d}.png")
    plt.savefig(frame_path)
    plt.close(fig)
    print(f"Time step {it + 1} captured: {frame_path}")

    return outdir # return the directory where frames are saved for video creation
### ---------------------- DENSE PLUME ----------------------------- ###
### temporal analysis ###
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
        ax2.set_ylabel(r"$\langle$r$\rangle_{\text{xy}}$/r$_{j}$") #(r"$\langle$r$\rangle_{\text{xy}}$/h$_{\text{MLD}}$") #
        ax2.set_ylim(ymin = ranges['radius'][0], ymax = ranges['radius'][-1])
        ax3.set_ylabel(r"w/(h$_{\mathrm{MLD}_0} \sqrt{N^{2}})$")
        ax3.set_ylim(ymin = ranges['w'][0], ymax = ranges['w'][-1])
        ax4.set_ylabel(r"b'/(h$_{\mathrm{MLD}_0} N^{2}$)")
        ax4.set_ylim(ymin = ranges['b_fluc'][0], ymax = ranges['b_fluc'][-1])
        ax5.set_ylabel(r"$\langle$T$'\rangle_{\text{xy}}$/T$_{0}$")
        ax5.set_ylim(ymin = ranges['T_fluc'][0], ymax = ranges['T_fluc'][-1])
        ax6.set_ylabel(r"$\langle$C'$\rangle_{\text{xy}}$/S$_{\text{max}}$") #(\text{h}_{mld} \sqrt{N^{2}}$)/(F$^{\text{C}}$)") #(r"$\langle$C$'\sqrt{g\text{r}_j}$/(F$^{\text{C}}$)") #
        ax6.set_ylim(ymin = ranges['S_fluc'][0], ymax = ranges['S_fluc'][-1])
        ax7.set_ylabel(r"$\langle$C'w$\rangle_{\text{xy}}$/F$^{\text{C}}$") #(\text{h}_{mld}\sqrt{N^{2}}$)/(F$^{\text{C}}$)")# (r"$\langle$C$\rangle_{\text{xy}}$\sqrt{g\text{r}_j}$/(F$^{\text{C}}$)")#
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
        ax6.set_ylim(ymin = ranges['S_fluc'][0], ymax = ranges['S_fluc'][-1])
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
### spatial vertical analysis ###
def plume_vertical_spatial_plot(time, it, ranges, color_opt, fig_folder, case_names, name, lx, z, zf, tracer_avg, u_rms, v_rms, w_rms, b_avg, b_center, r_profile, bu_fluc_avg, bv_fluc_avg, bw_fluc_avg, b_rms, T_fluc, tracer_fluc, ND = False, z_nd = r"(z - h$_{\mathrm{MLD}_0}$)/r$_{j}$"):
    num_cases = len(case_names)
    if num_cases==0:
        fig, ax = plt.subplots(2, 4, figsize=(12, 8))
        outdir = os.path.join(fig_folder, name)
        os.makedirs(outdir, exist_ok=True)
    else:
        outdir = os.path.join(fig_folder, name)
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

    td = time[it] / 3600 / 24
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
        ax1.set_xlabel(r"u$_{i}$/(F$^{\text{C}} \beta$)")
        ax2.set_xlabel(r"$\langle$C$\rangle_{\text{xy}} \sqrt{\text{g r}_{j}}$/(F$^{\text{C}}$)")
        ax3.set_xlabel(r"br$_j$/(F$^{\text{C}}\beta\sqrt{\text{g r}_{j}}$)")
        ax4.set_xlabel(r"C'$_{\text{centerline}} \sqrt{\text{g r}_{j}}$/(F$^{\text{C}}$)") 
        ax5.set_ylabel(z_nd)
        ax5.set_xlabel(r"r $\sqrt{\text{g r}_{j}}$/(F$^{\text{C}} \beta$r$_{j}$)")
        ax6.set_xlabel(r"$\langle$b'u'$_{i}\rangle_{xy}$/(F$^{\text{C}} \beta$ g)")
        ax7.set_xlabel(r"b$_{rms}$r$_j$/(F$^{\text{C}}\beta\sqrt{\text{g r}_{j}}$)")
        ax8.set_xlabel(r"T'$_{\text{centerline}}\sqrt{\text{g r}_{j}}$/(F$^{\text{C}}\beta$T$_{0}$)")
    else:
        ax1.set_ylabel("Depth [m]")
        ax1.set_xlabel("[m/s]")
        ax2.set_xlabel(r"$\langle$C$\rangle_{\text{xy}}$ [g/kg]")
        ax3.set_xlabel(r"[m/s$^2$]")
        ax4.set_xlabel(r"C$'_{\text{centerline}}$ [g/kg]")
        ax5.set_ylabel("Depth [m]")
        ax5.set_xlabel("[m]")
        ax6.set_xlabel(r"[m$^2$/s$^3$]")
        ax7.set_xlabel(r"[m/s$^2$]")
        ax8.set_xlabel(r"T$'_{\text{centerline}}$ [$^{\circ}$ C]")

    # velocity rms
    for i in range(num_cases):
        if i == 0:
            ax1.plot(u_rms[:, i], z[:, i], label=r"$\langle$u$_{\text{rms}}\rangle_{\text{xy}}$", color = color_opt[i], linestyle='dotted')
            ax1.plot(v_rms[:, i], z[:, i], label=r"$\langle$v$_{\text{rms}}\rangle_{\text{xy}}$", color = color_opt[i], linestyle='dashed')
            ax1.plot(w_rms[:, i], zf[:, i], label=r"$\langle$w$_{\text{rms}}\rangle_{\text{xy}}$", color = color_opt[i], linestyle='solid')
        else:
            ax1.plot(u_rms[:, i], z[:, i], color = color_opt[i], linestyle='dotted')
            ax1.plot(v_rms[:, i], z[:, i], color = color_opt[i], linestyle='dashed')
            ax1.plot(w_rms[:, i], zf[:, i], color = color_opt[i], linestyle='solid')
    ax1.set_title("Root Mean Square Velocities")
    ax1.set_ylim(ymin = np.min(zf), ymax = np.max(zf))
    ax1.set_xlim(ranges['vel_rms'])
    ax1.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
    ax1.legend(loc='lower right')

    # tracer profile 
    for i in range(num_cases):
        ax2.plot(tracer_avg[:, i], z[:, i], color = color_opt[i], linestyle='solid')
    ax2.set_title('Tracer')
    ax2.set_ylim(ymin = np.min(zf), ymax = np.max(zf))
    ax2.set_xlim(ranges['S'])
    ax2.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)

    # buoyancy profiles
    for i in range(num_cases):
        if i == 0:
            ax3.plot(b_avg[:, i], z[:, i], color = color_opt[i], label = r"$\langle$b$\rangle_{\text{xy}}$", linestyle='solid')
            ax3.plot(b_center[:, i], z[:, i], color = color_opt[i], label = r"b$_{\text{centerline}}$", linestyle='dashed')
        else:
            ax3.plot(b_avg[:, i], z[:, i], color = color_opt[i], linestyle='solid')
            ax3.plot(b_center[:, i], z[:, i], color = color_opt[i], linestyle='dashed')
    ax3.set_title("Buoyancy Profile")
    ax3.set_ylim(ymin = np.min(zf), ymax = np.max(zf))
    ax3.set_xlim(ranges['b_avg'])
    ax3.legend(loc='upper left')
    ax3.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)

    # temperature fluctuations 
    for i in range(num_cases):
        ax4.plot(tracer_fluc[:, i], z[:, i], color = color_opt[i], linestyle='solid')
    ax4.set_title("Perturbed Tracer")
    ax4.set_ylim(ymin = np.min(zf), ymax = np.max(zf))
    ax4.set_xlim(ranges['S_fluc'])
    ax4.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)

    # plume radius
    for i in range(num_cases):
        ax5.plot(r_profile[:, i], z[:, i], color = color_opt[i], linestyle='solid')
    ax5.set_title("Plume Radius with Depth")
    ax5.set_ylim(ymin = np.min(zf), ymax = np.max(zf))
    ax5.set_xlim(0, lx[0]/4)

    # perturbed buoyancy flux 
    for i in range(num_cases):
        if i == 0:
            ax6.plot(bu_fluc_avg[:, i], z[:, i], color = color_opt[i], label = r"$\langle$b'u'$\rangle_{\text{xy}}$", linestyle='dotted')
            ax6.plot(bv_fluc_avg[:, i], z[:, i], color = color_opt[i], label = r"$\langle$b'v'$\rangle_{\text{xy}}$", linestyle='dashed')
            ax6.plot(bw_fluc_avg[:, i], z[:, i], color = color_opt[i], label = r"$\langle$b'w'$\rangle_{\text{xy}}$", linestyle='solid')
        else:
            ax6.plot(bu_fluc_avg[:, i], z[:, i], color = color_opt[i], linestyle='dotted')
            ax6.plot(bv_fluc_avg[:, i], z[:, i], color = color_opt[i], linestyle='dashed')
            ax6.plot(bw_fluc_avg[:, i], z[:, i], color = color_opt[i], linestyle='solid')
    ax6.legend(loc='lower right')
    ax6.set_title("Buoyancy Flux Fluctuations")
    ax6.set_ylim(ymin = np.min(zf), ymax = np.max(zf))
    ax6.set_xlim(ranges['bw_fluc'])
    ax6.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True) 

    # buoyancy brms 
    for i in range(num_cases):
        ax7.plot(b_rms[:, i], z[:, i], color = color_opt[i], linestyle='solid')
    ax7.set_title("Root Mean Square Buoyancy")
    ax7.set_ylim(ymin = np.min(zf), ymax = np.max(zf))
    ax7.set_xlim(ranges['b_rms'])
    ax7.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)

    # temperature fluctuations 
    for i in range(num_cases):
        ax8.plot(T_fluc[:, i], z[:, i], color = color_opt[i], linestyle='solid')
    ax8.set_title("Perturbed Temperature")
    ax8.set_ylim(ymin = np.min(zf), ymax = np.max(zf))
    ax8.set_xlim(ranges['T_fluc'])
    ax8.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)

    # --- Save Frame ---
    frame_path = os.path.join(outdir, f"{name}_comparison_vert_buoyancy_{it:04d}.png")
    plt.savefig(frame_path)
    plt.close(fig)
    print(f"Time step {it + 1} captured: {frame_path}")

    return outdir # return the directory where frames are saved for video creation
### spatial horizontal analysis ###
def plume_horizontal_spatial_plot(time, it, ranges, color_opt, fig_folder, case_names, name, lx, y, u, v, w, b_center, bu_fluc, bv_fluc, bw_fluc, T, tracer, ND = False):
    num_cases = len(case_names)
    if num_cases==0:
        fig, ax = plt.subplots(2, 3, figsize=(12, 7))
        outdir = os.path.join(fig_folder, name)
        os.makedirs(outdir, exist_ok=True)
    else:
        outdir = os.path.join(fig_folder, name)
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
        ax1.set_xlabel(r"y/r$_{j}$") 
        ax1.set_ylabel(r"u$_{i}$/(F$^{\text{C}} \beta$ dT/dz r$_j$ T$_{0}$)") 
        ax2.set_xlabel(r"y/r$_{j}$")
        ax2.set_ylabel(r"(b' u'$_{i}$)/(F$^{\text{C}} \beta$ g dT/dz r$_j$/T$_{0}$)") 
        ax3.set_xlabel(r"y/r$_{j}$")
        ax3.set_ylabel(r"C$_{\text{centerline}} \sqrt{\text{g r}_{j}}$/(F$^{\text{C}}$)") 
        ax4.set_xlabel(r"y/r$_{j}$")
        ax4.set_ylabel(r"b'/(F$^{\text{C}} \beta \sqrt{\text{g dT/dz r}_j/\text{T}_{0}}$)")
        ax5.set_xlabel(r"y/r$_{j}$")
        ax5.set_ylabel(r"b'w'/(F$^{\text{C}} \beta$ g dT/dz r$_j$/T$_{0}$)")
        ax6.set_xlabel(r"y/r$_{j}$")
        ax6.set_ylabel(r"T$_{\text{centerline}\sqrt{\text{g r}_{j}}}$/(F$^{\text{C}}\beta$dT/dz r$_j$)")
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
            ax1.plot((y - lx[1]/2), u[:, i], label=r"u$_{\text{centerline}}$", color = color_opt[i], linestyle='dotted')
            ax1.plot((y - lx[1]/2), v[:, i], label=r"v$_{\text{centerline}}$", color = color_opt[i], linestyle='dashed')
            ax1.plot((y - lx[1]/2), w[:, i], label=r"w$_{\text{centerline}}$", color = color_opt[i], linestyle='solid')
        else:
            ax1.plot((y - lx[1]/2), u[:, i], color = color_opt[i], linestyle='dotted')
            ax1.plot((y - lx[1]/2), v[:, i], color = color_opt[i], linestyle='dashed')
            ax1.plot((y - lx[1]/2), w[:, i], color = color_opt[i], linestyle='solid')
    ax1.set_title("Velocity")
    ax1.set_xlim(-lx[1]/2, lx[1]/2)
    ax1.set_ylim(ranges['w'])
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(-3,2), useMathText=True)
    ax1.legend(loc='lower right')

    # horizontal buoyancy flux 
    for i in range(num_cases):
        if i == 0:
            ax2.plot((y - lx[1]/2), bu_fluc[:, i], color = color_opt[i], label = r"b$'$u$'_{\text{centerline}}$", linestyle='dotted')
            ax2.plot((y - lx[1]/2), bv_fluc[:, i], color = color_opt[i], label = r"b$'$v$'_{\text{centerline}}$", linestyle='dashed')
        else:
            ax2.plot((y - lx[1]/2), bu_fluc[:, i], color = color_opt[i], linestyle='dotted')
            ax2.plot((y - lx[1]/2), bv_fluc[:, i], color = color_opt[i], linestyle='dashed')
    ax2.set_title('Horizontal Buoyancy Flux Fluctuations')
    ax2.set_xlim(-lx[1]/2, lx[1]/2)
    ax2.set_ylim(ymin = ranges['b_flux'][0], ymax = ranges['b_flux'][1])
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(-3,2), useMathText=True)

    # tracer  
    for i in range(num_cases):
        ax3.plot((y - lx[1]/2), tracer[:, i], color = color_opt[i], linestyle='solid')
    ax3.set_title("Tracer")
    ax3.set_xlim(-lx[1]/2, lx[1]/2)
    ax3.set_ylim(ranges['S'])
    ax3.ticklabel_format(axis='y', style='sci', scilimits=(-3,2), useMathText=True)

    # Perturbed buoyancy 
    for i in range(num_cases):
        ax4.plot((y - lx[1]/2), b_center[:, i], color = color_opt[i], linestyle='solid')
    ax4.set_title("Perturbed Buoyancy")
    ax4.set_xlim(-lx[1]/2, lx[1]/2)
    ax4.set_ylim(ranges['b_fluc'])
    ax4.ticklabel_format(axis='y', style='sci', scilimits=(-3,2), useMathText=True)
    # perturbed buoyancy flux 
    for i in range(num_cases):
        ax5.plot((y - lx[1]/2), bw_fluc[:, i], color = color_opt[i], linestyle='solid')
    ax5.set_title("Vertical Buoyancy Flux Fluctuations")
    ax5.set_xlim(-lx[1]/2, lx[1]/2)
    ax5.set_ylim(ranges['bw_fluc'])
    ax5.ticklabel_format(axis='y', style='sci', scilimits=(-3,2), useMathText=True) 

    # temperature  
    for i in range(num_cases):
        ax6.plot((y - lx[1]/2), T[:, i], color = color_opt[i], linestyle='solid')
    ax6.set_title("Temperature")
    ax6.set_xlim(-lx[1]/2, lx[1]/2)
    ax6.set_ylim(ranges['T'])

    # --- Save Frame ---
    frame_path = os.path.join(outdir, f"{name}_hor_centerline_comparisons_{it:04d}.png")
    plt.savefig(frame_path)
    plt.close(fig)
    print(f"Time step {it + 1} captured: {frame_path}")

    return outdir # return the directory where frames are saved for video creation
### transient MLD analysis ###
def mld_temporal_analysis(time, ranges, color_opt, fig_folder, case_names, name, lx, mld, mld0, bw_mld, Tw_mld, tracerw_mld, tracer_ratio, w_rms, ND = False):
    num_cases = len(case_names)
    if num_cases==1:
        fig, ax = plt.subplots(2, 3, figsize=(10, 5))
        outdir = os.path.join(fig_folder, 'MLD analysis/')
        os.makedirs(outdir, exist_ok=True)
    else:
        outdir = os.path.join(fig_folder, 'comparison MLD analysis/')
        os.makedirs(outdir, exist_ok=True)
        gridspec_kw={'height_ratios': [1, 1, 0.1]} # add space for universal legend
        fig, ax = plt.subplots(3, 3, figsize=(12, 6.5), gridspec_kw=gridspec_kw)
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

    ax1 = ax[0, 0] # MLD depth through time 
    ax2 = ax[0, 1] # tracer flux average at the MLD through time
    ax3 = ax[0, 2] # Temperature flux average at the MLD through time
    ax4 = ax[1, 0] # vertical buoyancy flux average at the MLD through time 
    ax5 = ax[1, 1] # ratio of tracer in the mixed layer 
    ax6 = ax[1, 2] # w_rms at MLD through time

    if ND:
        ax1.set_ylabel(r"z/h$_{\text{MLD}}$")
        ax1.set_ylim(ymin = -lx[-1], ymax = 0)
        ax2.set_ylabel(r"C$_{plume}$w/F$^{\text{C}}$") 
        ax2.set_ylim(ymin = ranges['Sw_fluc'][0], ymax = ranges['Sw_fluc'][-1])
        ax3.set_ylabel(r"T'w$_{\text{plume}}$/(T$_{0}$h$_{\mathrm{MLD}_0} \sqrt{N^{2}}$)")
        ax3.set_ylim(ymin = ranges['Tw_fluc'][0], ymax = ranges['Tw_fluc'][-1])
        ax4.set_ylabel(r"b'w/(h$_{\mathrm{MLD}_0}^2 (N^{2})^{3/2}$)")
        ax4.set_ylim(ymin = ranges['bw_fluc'][0], ymax = ranges['bw_fluc'][-1])
        ax5.set_ylabel(r"M$_{ML}$/M$_{total}$")
        ax5.set_ylim(ymin = ranges['mass_ratio'][0], ymax = ranges['mass_ratio'][-1])
        ax6.set_ylabel(r"w$_{\text{rms}}$/(h$_{\mathrm{MLD}_0} \sqrt{N^{2}})$") 
        ax6.set_ylim(ymin = ranges['vel_rms'][0], ymax = ranges['vel_rms'][-1])
    else:
        ax1.set_ylabel("[m]")
        ax1.set_ylim(ymin = -lx[-1], ymax = 0)
        ax2.set_ylabel("[g/kg m/s]")
        ax2.set_ylim(ymin = ranges['Sw_fluc'][0], ymax = ranges['Sw_fluc'][-1])
        ax3.set_ylabel(r"$^{\circ}$C$\cdot$/s]")
        ax3.set_ylim(ymin = ranges['Tw_fluc'][0], ymax = ranges['Tw_fluc'][-1])
        ax4.set_ylabel(r"[m$^2$/s$^3$]")
        ax4.set_ylim(ymin = ranges['bw_fluc'][0], ymax = ranges['bw_fluc'][-1])
        ax5.set_ylabel(r"M$_{ML}$/M$_{total}$")
        ax5.set_ylim(ymin = ranges['mass_ratio'][0], ymax = ranges['mass_ratio'][-1])
        ax6.set_ylabel("[m/s]")
        ax6.set_ylim(ymin = ranges['vel_rms'][0], ymax = ranges['vel_rms'][-1])
    # Depth of plume through time 
    for i in range(num_cases):
        if i == 0:
            ax1.plot(time/ 3600 / 24, -mld0[i]*np.ones(len(time)), label = r"MLD$_{\text{t=0}}$", linewidth = 0.75, linestyle = 'dashed', color = color_opt[i])
            ax1.plot(time/ 3600 / 24, -mld[:, i], label = r"h$_{\text{MLD}}$", linewidth = 0.75, linestyle = 'solid', color = color_opt[i])
        else: 
            ax1.plot(time/ 3600 / 24, -mld0[i]*np.ones(len(time)), linewidth = 0.75, linestyle = 'dashed', color = color_opt[i])
            ax1.plot(time/ 3600 / 24, -mld[:, i], linewidth = 0.75, linestyle = 'solid', color = color_opt[i])
    ax1.set_title("MLD temporal change", size = 10)
    ax1.legend(loc='lower right', labelspacing = 0.25, handlelength=0.75)
    ax1.set_xlim([time.min() / 3600 / 24, time.max() / 3600 / 24])
    # tracer flux average at the MLD through time
    for i in range(num_cases):
        ax2.plot(time/ 3600 / 24, tracerw_mld[:, i], linewidth = 0.75, linestyle = 'dashed', color = color_opt[i])
    ax2.set_xlim([time.min() / 3600 / 24, time.max() / 3600 / 24])
    ax2.set_title("Vertical Trace Flux", size = 10)
    # Temperature flux average at the MLD through time
    for i in range(num_cases):
        ax3.plot(time/ 3600 / 24, Tw_mld[:, i], linewidth = 0.75, linestyle = 'solid', color = color_opt[i])
    ax3.set_title("Vertical Temperature Flux", size = 10)
    ax3.set_xlim([time.min() / 3600 / 24, time.max() / 3600 / 24]) 
    # vertical buoyancy flux average at the MLD through time
    for i in range(num_cases):
        ax4.plot(time/ 3600 / 24, bw_mld[:, i], linewidth = 0.75, color = color_opt[i])
    ax4.set_xlabel("Time [days]") 
    ax4.set_title("Perturbed Buoyancy Flux", size = 10)
    ax4.set_xlim([time.min() / 3600 / 24, time.max() / 3600 / 24])
    ax4.ticklabel_format(axis='y', style='sci', scilimits=(-3,2), useMathText=True)
    # ratio of tracer in the mixed layer  
    for i in range(num_cases):
        ax5.plot(time/ 3600 / 24, tracer_ratio[:, i], linewidth = 0.75, linestyle = 'solid', color = color_opt[i])
    ax5.set_xlabel("Time [days]") 
    ax5.set_title("MLD Tracer Ratio", size = 10)
    ax5.set_xlim([time.min() / 3600 / 24, time.max() / 3600 / 24])
    ax5.ticklabel_format(axis='y', style='sci', scilimits=(-3,2), useMathText=True)
    # w_rms at MLD through time
    for i in range(num_cases):
        ax6.plot(time/ 3600 / 24, w_rms[:, i], linewidth = 0.75, linestyle = 'solid', color = color_opt[i])
    ax6.set_xlabel("Time [days]") 
    ax6.set_title(r"w$_{\text{rms}}$", size = 10)
    ax6.set_xlim([time.min() / 3600 / 24, time.max() / 3600 / 24])
    ax6.ticklabel_format(axis='y', style='sci', scilimits=(-3,2), useMathText=True)
    # --- Save Frame ---
    frame_path = os.path.join(outdir, f"{name}_ml_temporal_comparison.png")
    plt.savefig(frame_path)
    plt.close(fig)
    print("Temporal Plot Saved: ", frame_path)

### ---------------------- CONVERGENCE TESTS ----------------------------- ###
def convergence_tests(time, it, ranges, fig_folder, lx, nx, x, y, z, zf, cases_sorted, matrix_N, ver, hor, 
                      b, b_avg, b_rms_sign, w_rms, b_rms, bw_fluc, b_flux_avg, b_max_sign_change_to_negative_loc, L_ozmidov, L_ozmidov_background, idx_neg, plot_points = False):

    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))

    levels = 500

    colors = ['black', 'blue', 'green', 'red', 'purple', 'pink', 'gray', 'orange', 'cyan', 'olive']
    td = time[it] / 3600 / 24

    ## buoyancy analysis profiles
    outdir0 = [fig_folder + 'convergence tests buoyancy profiles/']
    os.makedirs(outdir0, exist_ok=True)

    fig, ax = plt.subplots(3, 5, figsize=(20, 8), height_ratios = [1, 0.2, 1])

    fig.text(0.5, 1.08, f'{td:.2f} days', ha="center", fontsize=12) 
    # Titles for each row
    fig.text(0.5, 1.05, "Vertical resolution convergence", ha="center", fontsize=14)
    fig.text(0.5, 0.52, "Horizontal resolution convergence", ha="center", fontsize=14)
    
    ax1 = ax[0, 0]
    ax6 = ax[2, 0]
    for a in ax[1, :]:
        a.remove()
    for caseindex, case in enumerate(cases_sorted):
        if ver[caseindex] and hor[caseindex]:
            name_case = case.replace('flux b tracer ', "")
            # buoyancy profile
            ax[0, 0].plot(b_avg[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex], label = name_case)
            # RMS buoyancy 
            ax[0, 1].plot(b_rms[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex])
            # RMS w
            ax[0, 2].plot(w_rms[0:nx[2, caseindex]+1, caseindex], zf[0:nx[2, caseindex]+1, caseindex], color = colors[caseindex])#, linestyle = '--', label = r"$\langle$w$_{rms}rangle_{\text{xy}}$")
            # RMS buoyancy flux 
            ax[0, 3].plot(bw_fluc[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex])
            # buoyancy profile
            ax[2, 0].plot(b_avg[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex], label = name_case)
            # RMS buoyancy 
            ax[2, 1].plot(b_rms[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex])
            # RMS w 
            ax[2, 2].plot(w_rms[0:nx[2, caseindex]+1, caseindex], zf[0:nx[2, caseindex]+1, caseindex], color = colors[caseindex])#, linestyle = '--', label = r"$\langle$w$_{rms}\rangle_{\text{xy}}$")
            # RMS buoyancy flux 
            ax[2, 3].plot(bw_fluc[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex])
        elif ver[caseindex] and not hor[caseindex]:
            name_case = case.replace('flux b tracer ', "")
            # buoyancy profile
            ax[0, 0].plot(b_avg[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex], label = name_case)
            # RMS buoyancy 
            ax[0, 1].plot(b_rms[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex])
            # RMS w 
            ax[0, 2].plot(w_rms[0:nx[2, caseindex]+1, caseindex], zf[0:nx[2, caseindex]+1, caseindex], color = colors[caseindex])#, linestyle = '--')
            # RMS buoyancy flux 
            ax[0, 3].plot(bw_fluc[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex])
        elif hor[caseindex] and not ver[caseindex]:
            name_case = case.replace('flux b tracer ', "")
            # buoyancy profile
            ax[2, 0].plot(b_avg[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex], label = name_case)
            # RMS buoyancy 
            ax[2, 1].plot(b_rms[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex])
            # RMS w 
            ax[2, 2].plot(w_rms[0:nx[2, caseindex]+1, caseindex], zf[0:nx[2, caseindex]+1, caseindex], color = colors[caseindex])#, linestyle = ':')
            # RMS buoyancy flux 
            ax[2, 3].plot(bw_fluc[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex])

    # Ozmidov length scale
    ax[2, 4].plot(nx[0, hor], L_ozmidov_background[it, hor], marker = '+', label = r"b$_{\text{stratified}, 3}$ L$_{O}$", linestyle = 'none')
    ax[2, 4].plot(nx[0, hor], L_ozmidov[it, hor], marker = 'o', label = r"b$_{\text{average}, 3}$ L$_{O}$", linestyle = 'none')
    ax[0, 4].plot(nx[2, ver], L_ozmidov_background[it, ver], marker = '+', label = r"b$_{\text{stratified}, 3}$ L$_{O}$", linestyle = 'none')
    ax[0, 4].plot(nx[2, ver], L_ozmidov[it, ver], marker = 'o', label = r"b$_{\text{average}, 3}$ L$_{O}$", linestyle = 'none')

    ax[0, 0].set_xlabel("[m/s$^{2}$]")
    ax[0, 0].set_ylim([-lx[2], 0])
    ax[0, 0].set_title("Buoyancy")
    ax[0, 0].set_ylim([-lx[2], 0])
    ax[0, 0].set_ylabel("y [m]")
    ax[0, 0].set_xlim(ranges['b_avg'])
    ax[0, 0].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
    
    ax[0, 1].set_xlabel("[m/s$^{2}$]")
    ax[0, 1].set_title("Buoyancy RMS")
    ax[0, 1].set_xlim(ranges['b_rms'])
    ax[0, 1].set_ylim([-lx[2], 0])

    ax[0, 2].set_xlabel("[m/s]")
    ax[0, 2].set_title("w RMS")
    ax[0, 2].set_xlim(ranges['vel_rms'])
    ax[0, 2].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
    ax[0, 2].set_ylim([-lx[2], 0])

    ax[0, 3].set_xlabel("[m$^{2}$/s$^{3}$]")
    ax[0, 3].set_title("Buoyancy Flux Flucts")
    ax[0, 3].set_xlim(ranges['bflux_rms'])
    ax[0, 3].set_ylim([-lx[2], 0])

    ax[0, 4].legend(loc='upper right', handlelength=0.75)
    ax[0, 4].set_ylabel("Length Scale [m]")
    ax[0, 4].set_title("Ozmidov Length Scale")
    ax[0, 4].set_ylim(ranges['b_avg'])
    ax[0, 4].set_xlabel("Time [days]")
    ax[0, 4].set_xlim([0, matrix_N +10])
    
    ax[2, 0].set_xlabel("[m/s$^{2}$]")
    ax[2, 0].set_xlim(ranges['b_avg'])
    ax[2, 0].set_title("Buoyancy")
    ax[2, 0].set_ylabel("Depth [m]")
    ax[2, 0].set_ylim([-lx[2], 0])
    ax[2, 0].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)

    ax[2, 1].set_xlabel("[m/s$^{2}$]")
    ax[2, 1].set_title("Buoyancy RMS")
    ax[2, 1].set_xlim(ranges['b_rms'])
    ax[2, 1].set_ylim([-lx[2], 0])

    ax[2, 2].set_xlabel("[m/s]")
    ax[2, 2].set_title("w RMS")
    ax[2, 2].set_xlim(ranges['vel_rms'])
    ax[2, 2].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
    ax[2, 2].set_ylim([-lx[2], 0]) 

    ax[2, 3].set_xlabel("[m$^{2}$/s$^{3}$]")
    ax[2, 3].set_title("Buoyancy Flux Flucts")
    ax[2, 3].set_xlim(ranges['bflux_rms'])
    ax[2, 3].set_ylim([-lx[2], 0])

    ax[2, 4].set_title("Ozmidov Length Scale")
    ax[2, 4].set_ylabel("Length Scale [m]")
    ax[2, 4].set_ylim(ranges['lengthscale'])
    ax[2, 4].set_xlabel("Time [days]")
    ax[2, 4].set_xlim([0, matrix_N +10])
    ax[2, 4].legend(loc='upper right', handlelength=0.75)

    # universal legend
    handles0, labels0 = ax1.get_legend_handles_labels()
    handles1, labels1 = ax6.get_legend_handles_labels()

    fig.legend(handles0, labels0, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.99))
    fig.legend(handles1, labels1, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.46))

    # --- Save Frame ---
    frame_path = os.path.join(outdir0, f"convergence_test_{it:04d}.png")
    plt.savefig(frame_path, bbox_inches="tight")
    print(f"Time step {it + 1} captured")
    plt.close(fig)

    ## buoyancy plane slices ##
    outdir1 = fig_folder + 'buoyancy planeslices/'
    os.makedirs(outdir1, exist_ok=True)
    fig, ax = plt.subplots(3, 3, figsize=(18, 10), constrained_layout=True)
    fig.suptitle(f'{td:.2f} days') 
    norm = mcolors.Normalize(vmin=ranges['b_avg'][0], vmax=ranges['b_avg'][-1])
    mappable = cm.ScalarMappable(norm=norm) 
    hor_plot = 0
    ver_plot = 0
    for caseindex, case in enumerate(cases_sorted):
        X, Y, Z = np.meshgrid(x[0:nx[0, caseindex], caseindex] , y[0:nx[1, caseindex], caseindex] , z[0:nx[2, caseindex], caseindex])
        name_case = case.replace('flux b tracer ', "")
        ax[hor_plot, ver_plot].contourf(X[int(nx[0, caseindex]/2), :, :], Z[int(nx[0, caseindex]/2), :, :], b[caseindex, int(nx[0, caseindex]/2), 0:nx[1, caseindex], 0:nx[2, caseindex]], levels, norm=norm)
        ax[hor_plot, ver_plot].set_title(name_case)
        ax[hor_plot, ver_plot].set_xlabel("y [m]")
        ax[hor_plot, ver_plot].set_ylabel("z [m]")
        ax[hor_plot, ver_plot].set_aspect('equal')
        if ver_plot > 1:
            hor_plot += 1
            ver_plot = 0
        else:
            ver_plot += 1

    cbar = fig.colorbar(mappable, ax=ax, label=r"m/s$^2$", location='bottom', shrink=0.5, orientation='horizontal')
    
    # --- Save Frame ---
    frame_path = os.path.join(outdir1, f"planeslices_{it:04d}.png")
    plt.savefig(frame_path, bbox_inches="tight")
    print(f"Time step {it + 1} captured")
    plt.close(fig)

    if plot_points:
        ranges['brms_sign'] = [0, b_rms_sign.max()]
        ranges['bflux_rms'] = [b_flux_avg.min(), b_flux_avg.max()]
        ranges['z_sign'] = [b_max_sign_change_to_negative_loc.min(), b_max_sign_change_to_negative_loc.max()]

        outdir2 = fig_folder + 'convergence testing/'
        os.makedirs(outdir2, exist_ok=True)
        td = time[it] / 3600 / 24
        fig = plt.figure(figsize=(20, 8))
        fig.tight_layout()
        fig.suptitle(f'{td:.2f} days', fontsize=12) 
        # Titles for each row
        fig.text(0.5, 0.94, "Vertical resolution convergence", 
                ha="center", va="center", fontsize=14)

        fig.text(0.5, 0.48, "Horizontal resolution convergence", 
                ha="center", va="center", fontsize=14)

        # z location of buoyancy sign change as a function of resolution
        ax1 = fig.add_subplot(2, 5,  1)
        ax1.plot(nx[2, ver], b_max_sign_change_to_negative_loc[ver], marker='o', linestyle='none')
        ax1.set_ylabel("[m]")
        ax1.set_title("Neutrally buoyant depth")
        ax1.set_ylim(ranges['z_sign'])

        ax4 = fig.add_subplot(2, 5,  6)
        ax4.plot(nx[1, hor], b_max_sign_change_to_negative_loc[hor], marker='o', linestyle='none')
        ax4.set_ylabel("[m]")
        ax4.set_title("Neutrally buoyant depth")
        ax4.set_ylim(ranges['z_sign'])

        # RMS buoyancy as a function of resolution 
        ax2 = fig.add_subplot(2, 5,  2)
        ax2.plot(nx[2, ver], b_rms_sign[ver], marker='o', linestyle='none', label = "at neutrally buoyant depth")
        ax2.plot(nx[2, ver], b_rms_sign[ver-1], marker='o', linestyle='none', color = color_opt[i], label = "above neutrally buoyant depth")
        ax2.legend(loc='upper right', handlelength=0.75)
        ax2.set_ylabel("[m/s$^{2}$]")
        ax2.set_title("Buoyancy RMS")
        ax2.set_ylim(ranges['brms_sign'])

        ax5 = fig.add_subplot(2, 5,  7)
        ax5.plot(nx[1, hor], b_rms_sign[hor], marker='o', linestyle='none', label = "at neutrally buoyant depth")
        ax5.plot(nx[1, hor], b_rms_sign[hor-1], marker='o', linestyle='none', color = color_opt[i], label = "above neutrally buoyant depth")
        ax5.legend(loc='upper right', handlelength=0.75)
        ax5.set_ylabel("[m/s$^{2}$]")
        ax5.set_title("Buoyancy RMS")
        ax5.set_ylim(ranges['brms_sign'])

        # RMS w as a function of resolution
        ax2 = fig.add_subplot(2, 5,  3)
        ax2.plot(nx[2, ver], w_rms[idx_neg[ver], ver], marker='o', linestyle='none', label = "at neutrally buoyant depth")
        ax2.plot(nx[2, ver], w_rms[idx_neg[ver]-1, ver], marker='o', linestyle='none', color = color_opt[i], label = "above neutrally buoyant depth")
        ax2.legend(loc='upper right', handlelength=0.75)
        ax2.set_ylabel("[m/s]")
        ax2.set_title("w RMS")
        ax2.set_ylim(ranges['vel_rms'])
        ax2.ticklabel_format(axis='y', style='sci', scilimits=(-3,2), useMathText=True)

        ax5 = fig.add_subplot(2, 5,  8)
        ax5.plot(nx[1, hor], w_rms[idx_neg[hor], hor], marker='o', linestyle='none', label = "at neutrally buoyant depth")
        ax5.plot(nx[1, hor], w_rms[idx_neg[hor]-1, hor], marker='o', linestyle='none', color = color_opt[i], label = "above neutrally buoyant depth")
        ax5.legend(loc='upper right', handlelength=0.75)
        ax5.set_ylabel("[m/s]")
        ax5.set_title("w RMS")
        ax5.set_ylim(ranges['vel_rms'])
        ax5.ticklabel_format(axis='y', style='sci', scilimits=(-3,2), useMathText=True)

        # RMS buoyancy flux as a function of resolution
        ax4 = fig.add_subplot(2, 5,  4)
        ax4.plot(nx[2, ver], bw_fluc[idx_neg[ver], ver], marker='o', linestyle='none', label = "at neutrally buoyant depth")
        ax4.plot(nx[2, ver], bw_fluc[idx_neg[ver]-1, ver], marker='o', linestyle='none', color = color_opt[i], label = "above neutrally buoyant depth")
        ax4.legend(loc='upper right', handlelength=0.75)
        ax4.set_ylabel("[m$^{2}$/s$^{3}$]")
        ax4.set_title("Buoyancy Flux Flucts")
        ax4.set_ylim(ranges['bflux_rms'])

        ax8 = fig.add_subplot(2, 5,  9)
        ax8.plot(nx[1, hor], bw_fluc[idx_neg[hor], hor], marker='o', linestyle='none', label = "at neutrally buoyant depth")
        ax8.plot(nx[1, hor], bw_fluc[idx_neg[hor]-1, hor], marker='o', linestyle='none', color = color_opt[i], label = "above neutrally buoyant depth")
        ax8.legend(loc='upper right', handlelength=0.75)
        ax8.set_ylabel("[m$^{2}$/s$^{3}$]")
        ax8.set_title("Buoyancy Flux Flucts")
        ax8.set_ylim(ranges['bflux_rms'])

        # RMS buoyancy flux as a function of resolution
        ax5 = fig.add_subplot(2, 5, 5)
        ax5.plot(nx[2, ver], L_ozmidov[it, ver], marker='o', linestyle='none', color = color_opt[i], label = r"b$_{\text{average}, 3}$ L$_{O}$")
        ax5.plot(nx[2, ver], L_ozmidov_background[it, ver], marker='o', linestyle='none', color = 'blue', label = r"b$_{\text{stratified}, 3}$ L$_{O}$")
        ax5.legend(loc='upper right', handlelength=0.75)
        ax5.set_ylabel("[m]")
        ax5.set_title("Ozmidov Length Scale")
        ax5.set_ylim(ranges['lengthscale'])
        ax10 = fig.add_subplot(2, 5,  10)
        ax10.plot(nx[1, hor], L_ozmidov[it, hor], marker='o', linestyle='none', color = color_opt[i], label = r"b$_{\text{average}, 3}$ L$_{O}$")
        ax10.plot(nx[1, hor], L_ozmidov_background[it, hor], marker='o', linestyle='none', color = 'blue', label = r"b$_{\text{stratified}, 3}$ L$_{O}$")
        ax10.legend(loc='upper right', handlelength=0.75)
        ax10.set_title("Ozmidov Length Scale")
        ax10.set_ylabel("[m]")
        ax10.set_ylim(ranges['lengthscale'])

        fig.supxlabel("Number of Grid Cells", fontsize=12)
        # --- Save Frame ---
        frame_path = os.path.join(outdir, f"convergence_test_{it:04d}.png")
        plt.tight_layout()
        plt.savefig(frame_path)
        print(f"Time step {it + 1} captured")
        plt.close(fig)
        return outdir0, outdir1, outdir2
    return outdir0, outdir1 # return the directory where frames are saved for video