import os
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import scipy


from plotting_general import save_frame
### -------------------------PLOTTING 1D LINES FUNCTIONS------------------------- ###
## xy averages ##
def plot_lines(title, it, case_opt, fig_folder, z, lines_var_it, save_folder = 'profiles', error_norm = None):
    """
    Plot rms, xy averages, or vertical centerlines for each variable
    comparing all cases on the same axes, one subplot per variable.

    Called once per timestep from an external time loop; saves a single
    frame image per call and returns the full sorted list of frame paths
    so far, for use with create_video.

    Parameters
    ----------
    it : int
        Current timestep index.
    ranges : dict
        Axis range dictionary (expects keys matching the variable names,
        e.g. ranges['T'], ranges['u'], etc.).
    case_opt : list
        Per-case color/linestyle options from comparison_plot_opt.
    fig_folder : str
        Base output directory.
    case_names : list of str
        Legend labels, one per case.
    z : list of ndarray
        Vertical coordinate array per case.
    lines_var_it : dict
        Maps variable name -> list of 1D arrays (one per case), i.e. the
        z-profile of reader.load_averages(var)[it, :] for that case.

    Returns
    -------
    frame_paths : list of str
        Sorted list of all saved frame file paths (for create_video).
    """
    if error_norm is not None:
        with_error = True
        save_folder = os.path.join(save_folder, 'with error')
    else:
        with_error = False
        save_folder = os.path.join(save_folder)
    outdir = os.path.join(fig_folder, save_folder)
    os.makedirs(outdir, exist_ok=True)

    var_names = list(lines_var_it.keys())
    n_vars = len(var_names)
    case_names = list(case_opt.keys())
    num_cases = len(case_names)

    ncols = n_vars
    nrows = int(3) if with_error else int(2)  # extra row for universal legend
    gridspec_kw =np.ones(nrows)
    gridspec_kw[-1] = 0.4 # add space for universal legend
    gridspec_kw = {'height_ratios': gridspec_kw}
    hor_len = 12.0
    vert_len = hor_len * (nrows-1) / (ncols) + 2.5
    size_in = (hor_len, vert_len)
    fig, axes = plt.subplots(nrows, ncols, figsize=size_in, sharey=True, constrained_layout=True, gridspec_kw=gridspec_kw)
    for a in axes[-1, :]:
        a.remove()
    axes = axes.ravel()
    fig.suptitle(title)
    #if n_vars == 1:
    #    axes = [axes]
    # Force even pixel dimensions at 200 dpi
    w_px = int(fig.get_figwidth() * fig.dpi)
    h_px = int(fig.get_figheight() * fig.dpi)
    if w_px % 2 != 0:
        fig.set_figwidth((w_px + 1) / fig.dpi)
    if h_px % 2 != 0:
        fig.set_figheight((h_px + 1) / fig.dpi)
            
    for ax, var_name in zip(axes, var_names):
        profiles = lines_var_it[var_name]['var']
        title = lines_var_it[var_name]['title']
        label = lines_var_it[var_name]['label']
        range_var = lines_var_it[var_name]['range']
        for n, case in enumerate(case_names):
            if num_cases==1 or n == num_cases-1:
                ax.plot(profiles[n], z[n], color = case_opt[case]['color'], linestyle=case_opt[case]['linestyle'], linewidth = 2.0, label=case)
            else:
                ax.plot(profiles[n], z[n], color = case_opt[case]['color'], linestyle=case_opt[case]['linestyle'], linewidth = 2.0, label=case)
        ax.set_title(title)
        ax.set_xlabel(label)
        ax.set_xlim(range_var)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2), useMathText=True) 
    axes[0].set_ylim(ymin=-40, ymax=0.0)#(ymin = min(z[-1]), ymax = 0.0)
    axes[0].set_ylabel('z [m]')
    axes[n_vars].set_ylabel('z [m]')

    if with_error:
        for a, var_name in enumerate(var_names):
            comparison_profile = lines_var_it[var_name]['var'][-1]
            profiles = lines_var_it[var_name]['var']
            normalize_error = error_norm[var_name]

            for n, case in enumerate(case_names[:-1]):
                stepping = int(len(comparison_profile)/len(profiles[n]))
                error = (profiles[n] - comparison_profile[::stepping])/ np.max(np.abs(comparison_profile[::stepping]))#normalize_error
                axes[a+n_vars].plot(error, z[n], color = case_opt[case]['color'], linestyle=case_opt[case]['linestyle'], linewidth = 2.0)
                #print(f"it: {it}, var: {lines_var_it[var_name]['title']}, case: {case}, abs max error: {np.max(np.abs(error))}, abs avg error: {np.mean(np.abs(error))}")
            axes[a+n_vars].set_xlabel(r"$\Delta \phi/\phi_{0}$")
            axes[a+n_vars].set_xlim([-0.3, 0.3])
    if num_cases > 1:
        case_handles = [Line2D([0], [0], color = case_opt[case]['color'], linestyle=case_opt[case]['linestyle'], linewidth = 2.0, label=case)for n, case in enumerate(case_names)]
        fig.legend(handles=case_handles,
                loc='lower center',
                handlelength = 0.9,
                ncols=np.ceil(num_cases/2).astype(int),
                bbox_to_anchor=(0.5, 0.0))

    # --- Save Frame ---
    save_frame(fig, outdir, it, size_in, file_name = 'profiles_frame_')
    return outdir
## horizontal lines ##
def plot_lines_h(title, it, case_opt, fig_folder, coord, lines_var_it, loc, xlabel = 'y [m]', error_norm = None, save_folder = 'field outputs'):
    """
        Plot horizontal line profiles (field value vs coord) for each panel,
        comparing all cases on the same axes, one subplot per panel.

        Called once per timestep from an external time loop; saves a single
        frame image per call and returns the output directory, for use with
        create_video.

        Parameters
        ----------
        time : ndarray
            Full time array; time[it] gives the current time for the title.
        it : int
            Current timestep index.
        fig_folder : str
            Base output directory.
        coord : list of ndarray
            Horizontal coordinate array (e.g. y or x) per case
        lines_var_it : dict
            Maps panel name -> dict with:
            'var' : dict mapping series label -> list of 1D arrays
                        (one array per case). A panel with a single series
                        draws one line per case (no series legend); multiple
                        series overlay with dotted/dashed/solid/dashdot styles
                        and get their own legend, e.g.
                        {'u_centerline': [...], 'v_centerline': [...], 'w_centerline': [...]}
            'title'  : subplot title
            'label'  : y-axis label (field units)
            'range'  : y-axis limits, e.g. ranges['w']

        Returns
        -------
        outdir : str
            Directory the frame was saved to.
    """
    if error_norm is not None:
        with_error = True
        save_folder = os.path.join(save_folder, 'with error')
    else:
        with_error = False
        save_folder = os.path.join(save_folder)
    outdir = os.path.join(fig_folder, save_folder, loc)
    os.makedirs(outdir, exist_ok=True)

    var_names = list(lines_var_it.keys())
    n_vars = len(var_names)
    num_cases = len(coord)

    ncols = n_vars
    nrows = int(3) if with_error else int(2)  # extra row for universal legend
    gridspec_kw = np.ones(nrows)
    gridspec_kw[-1] = 0.3
    gridspec_kw = {'height_ratios': gridspec_kw}

    hor_len = 12.0
    vert_len = hor_len * (nrows-1) / ncols + 1
    size_in = (hor_len, vert_len)

    fig, axes = plt.subplots(nrows, ncols, figsize=size_in, constrained_layout=True, gridspec_kw=gridspec_kw)
    for ax in axes[-1, :]:
            ax.remove()
    axes = axes.ravel()
    fig.suptitle(title)

    for a, var_name in enumerate(var_names):
        profiles = lines_var_it[var_name]['var']
        title = lines_var_it[var_name]['title']
        label = lines_var_it[var_name]['label']
        range_var = lines_var_it[var_name]['range']
        for n, case in enumerate(case_opt.keys()):
            axes[a].plot(coord[n], profiles[n], color = case_opt[case]['color'], label=case)
        axes[a].set_title(title)
        axes[a].set_ylabel(label)
        axes[a].set_ylim(range_var)
        axes[a].set_xlabel(xlabel)
        axes[a].set_xlim([min(coord[-1]), max(coord[-1])/2])
        axes[a].ticklabel_format(axis='y', style='sci', scilimits=(-3,2), useMathText=True) 

    if with_error:
        for a, var_name in enumerate(var_names):
            comparison_profile = lines_var_it[var_name]['var'][-1]
            profiles = lines_var_it[var_name]['var']
            normalize_error = error_norm[var_name]

            for n in range(num_cases-1):
                stepping = int(len(comparison_profile)/len(profiles[n]))
                error = (profiles[n] - comparison_profile[::stepping])/normalize_error
                axes[a+n_vars].plot(coord[n], error, color = case_opt[case]['color'], linestyle=case_opt[case]['linestyle'], linewidth = 2.0)
                #print(f"it: {it}, var: {lines_var_it[var_name]['title']}, case: {case}, abs max error: {np.max(np.abs(error))}, abs avg error: {np.mean(np.abs(error))}")
            axes[a+n_vars].set_ylabel(r"$\frac{\Delta \phi}{\phi_{0}}$")
            axes[a+n_vars].set_ylim([-0.2, 0.2])
            axes[a+n_vars].set_xlabel(xlabel)
            axes[a+n_vars].set_xlim([min(coord[-1]), max(coord[-1])])
    #print('\n')
    plt.subplots_adjust(left=0.05, bottom=0.30)
    if num_cases > 1:
        case_handles = [Line2D([0], [0], color = case_opt[case]['color'], linewidth = 2.0, label=case) for n, case in enumerate(case_opt.keys())]
        fig.legend(handles=case_handles, loc='lower center',
                   ncols=np.ceil(num_cases/2).astype(int), bbox_to_anchor=(0.5, 0.0))
    # --- Save Frame ---#
    if with_error:
        file_name = 'hor_centerline_frame_with_error_'
    else:
        file_name = 'hor_centerline_frame_'
    save_frame(fig, outdir, it, size_in, file_name=file_name)
    return outdir
### -------------------------PLOTTING DENSE PLUME FUNCTIONS------------------------- ###
## scaling analysis ##
def plot_scaling_analysis(time, case_opt, fig_folder, F, eta, loc_z):
    def _F(eta, alpha):
        return np.exp(-alpha * eta**2)

    def _F_transverse(eta, c_delta, alpha):
        return (c_delta/(2*alpha))*(1/eta)*((2*alpha*eta**2 + 1)*np.exp(-alpha*eta**2) - 1)

    nz = len(loc_z)
    nt = len(time)
    size_in = (12, 6)

    outdir = os.path.join(fig_folder, 'scaling analysis/')
    os.makedirs(outdir, exist_ok=True)

    for it in range(nt):
        fig, ax = plt.subplots(1, 1, figsize=size_in)

        for n, case in zip(np.arange(nz), case_opt.keys()):
            F_it_k = F[it, :, n].ravel()
            eta_it_k = eta[it, :, n].ravel()

            n_keep = np.where(~np.isnan(F_it_k))
            F_it_k = F_it_k[n_keep]
            eta_it_k = eta_it_k[n_keep]

            alpha = scipy.optimize.curve_fit(_F, eta_it_k, F_it_k)[0][0]
            ax.plot(eta_it_k, _F(eta_it_k, alpha), color = case_opt[case]['color'],
                    label=rf"z={loc_z[n]:.2f} m, $\alpha={alpha:.2f}$")
            ax.scatter(eta_it_k, F_it_k, marker='o', s=10, color = case_opt[case]['color'])

        ax.set_xlabel(r"$\eta$")
        ax.set_ylabel(r"$F(\eta)$")
        ax.set_title(f"t = {time[it]/3600:.2f} hours")
        ax.legend(loc='upper right', fontsize = 20)
        save_frame(fig, outdir, it, size_in)

    return outdir
## spatial vertical analysis ##
def plot_plume_vertical_spatial(time, ranges, case_opt, fig_folder, name, lx, z, tracer_avg, u_rms, v_rms, w_rms, b_avg, b_center, r_profile, bur_fluc_avg, bw_fluc_avg, T_avg, T_fluc, tracer, ND = False, z_nd = r"(z - h$_{\mathrm{MLD}_0}$)/l$_{j}$"):
    num_cases = case_opt.keys()
    outdir = os.path.join(fig_folder, 'vertical centerline-' + name)
    os.makedirs(outdir, exist_ok=True)
    if num_cases>1:
        case_handles = [
            Line2D([0], [0], color = case_opt[case]['color'], linestyle=case_opt[case]['linestyle'], label=case) for case in case_opt.keys()]
    for it, t in enumerate(time):
        if num_cases==1:
            size_in = (12, 8)
            fig, axes = plt.subplots(2, 4, figsize=size_in)
        else:
            gridspec_kw={'height_ratios': [1, 1, 0.05]} # add space for universal legend
            size_in = (12, 10)
            fig, axes = plt.subplots(3, 4, figsize=size_in, gridspec_kw=gridspec_kw)
            for a in axes[2, :]:
                a.remove()
            fig.legend(handles=case_handles,
                    loc='lower center',
                    ncols=num_cases,
                    bbox_to_anchor=(0.52, 0.0))

        td = t / 3600 / 24
        fig.suptitle(f'{td:.2f} days', )

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
        for n, case in enumerate(case_opt.keys()):
            if n == 0:
                ax1.plot(u_rms[n][it], z[n], label=r"$\langle$u$_{\text{rms}}\rangle_{\text{xy}}$", color = case_opt[case]['color'], linestyle='dotted', linewidth = 2.0)
                ax1.plot(v_rms[n][it], z[n], label=r"$\langle$v$_{\text{rms}}\rangle_{\text{xy}}$", color = case_opt[case]['color'], linestyle=case_opt['line_styles'], linewidth = 2.0)
                ax1.plot(w_rms[n][it], z[n], label=r"$\langle$w$_{\text{rms}}\rangle_{\text{xy}}$", color = case_opt[case]['color'], linestyle=case_opt[case]['linestyle'], linewidth = 2.0)
            else:
                ax1.plot(u_rms[n][it], z[n], color = case_opt[case]['color'], linestyle='dotted', linewidth = 2.0)
                ax1.plot(v_rms[n][it], z[n], color = case_opt[case]['color'], linestyle=case_opt['line_styles'], linewidth = 2.0)
                ax1.plot(w_rms[n][it], z[n], color = case_opt[case]['color'], linestyle=case_opt[case]['linestyle'], linewidth = 2.0)
        ax1.set_title("Root Mean Square Velocities")
        ax1.set_ylim(ymin = -min(lx[-1, :]), ymax = 0.0)
        ax1.set_xlim(ranges['vel_rms'])
        ax1.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax1.legend(loc='lower right')

        # tracer profile 
        for n, case in enumerate(case_opt.keys()):
            ax2.plot(tracer_avg[n][it], z[n], color = case_opt[case]['color'], linestyle=case_opt[case]['linestyle'], linewidth = 2.0)
        ax2.set_title('Tracer Profile')
        ax2.set_ylim(ymin = -min(lx[-1, :]), ymax = 0.0)
        ax2.set_xlim(ranges['Tracer_avg'])
        ax2.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)

        # buoyancy profiles
        for n, case in enumerate(case_opt.keys()):
            if n == 0:
                ax3.plot(b_avg[n][it], z[n], color = case_opt[case]['color'], label = r"$\langle$b$\rangle_{\text{xy}}$", linestyle=case_opt[case]['linestyle'], linewidth = 2.0)
                ax3.plot(b_center[n][it], z[n], color = case_opt[case]['color'], label = r"b$_{\text{centerline}}$", linestyle=case_opt['line_styles'], linewidth = 2.0)
            else:
                ax3.plot(b_avg[n][it], z[n], color = case_opt[case]['color'], linestyle=case_opt[case]['linestyle'], linewidth = 2.0)
                ax3.plot(b_center[n][it], z[n], color = case_opt[case]['color'], linestyle=case_opt['line_styles'], linewidth = 2.0)
        ax3.set_title("Buoyancy Profile")
        ax3.set_ylim(ymin = -min(lx[-1, :]), ymax = 0.0)
        ax3.set_xlim(ranges['b_avg'])
        ax3.legend(loc='upper left')
        ax3.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)

        # temperature fluctuations 
        for n, case in enumerate(case_opt.keys()):
            ax4.plot(tracer[n][it], z[n], color = case_opt[case]['color'], linestyle=case_opt[case]['linestyle'], linewidth = 2.0)
        ax4.set_title("Tracer")
        ax4.set_ylim(ymin = -min(lx[-1, :]), ymax = 0.0)
        ax4.set_xlim(ranges['Tracer'])
        ax4.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)

        # plume radius
        for n, case in enumerate(case_opt.keys()):
            ax5.plot(r_profile[n][:, it], z[n], color = case_opt[case]['color'], linestyle=case_opt[case]['linestyle'], linewidth = 2.0)
        ax5.set_title("Plume Radius with Depth")
        ax5.set_ylim(ymin = -min(lx[-1, :]), ymax = 0.0)
        ax5.set_xlim(0, min(lx[0, :])/1.9)

        # perturbed buoyancy flux 
        for n, case in enumerate(case_opt.keys()):
            if n == 0:
                ax6.plot(bur_fluc_avg[n][it], z[n], color = case_opt[case]['color'], label = r"$\langle$b'u$_r\rangle_{\text{xy}}$", linestyle='dotted', linewidth = 2.0)
                ax6.plot(bw_fluc_avg[n][it], z[n], color = case_opt[case]['color'], label = r"$\langle$b'w$\rangle_{\text{xy}}$", linestyle=case_opt[case]['linestyle'], linewidth = 2.0)
            else:
                ax6.plot(bur_fluc_avg[n][it], z[n], color = case_opt[case]['color'], linestyle='dotted', linewidth = 2.0)
                ax6.plot(bw_fluc_avg[n][it], z[n], color = case_opt[case]['color'], linestyle=case_opt[case]['linestyle'], linewidth = 2.0)
        ax6.legend(loc='lower right')
        ax6.set_title("Buoyancy Flux Fluctuations")
        ax6.set_ylim(ymin = -min(lx[-1, :]), ymax = 0.0)
        ax6.set_xlim(ranges['bw_fluc'])
        ax6.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True) 

        # average temperature
        for n, case in enumerate(case_opt.keys()):
            ax7.plot(T_avg[n][it], z[n], color = case_opt[case]['color'], linestyle=case_opt[case]['linestyle'], linewidth = 2.0)
        ax7.set_title("Temperature")
        ax7.set_ylim(ymin = -min(lx[-1, :]), ymax = 0.0)
        ax7.set_xlim(ranges['T'])
        ax7.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)

        # temperature fluctuations 
        for n, case in enumerate(case_opt.keys()):
            ax8.plot(T_fluc[n][it], z[n], color = case_opt[case]['color'], linestyle=case_opt[case]['linestyle'], linewidth = 2.0)
        ax8.set_title("Perturbed Temperature")
        ax8.set_ylim(ymin = -min(lx[-1, :]), ymax = 0.0)
        ax8.set_xlim(ranges['T_fluc'])
        ax8.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)

        # --- Save Frame ---
        save_frame(fig, outdir, it, size_in, file_name = 'comparison_vert_buoyancy_')

    return outdir # return the directory where frames are saved for video creation
## spatial horizontal analysis ##
def plot_plume_horizontal_spatial(time, it, ranges, case_opt, fig_folder, name, lx, y, u, v, w, b_center, bu_fluc, bv_fluc, bw_fluc, T, tracer, ND = False):
    case_names = list(case_opt.keys())
    num_cases = len(case_names)
    if num_cases==0:
        size_in = (12, 7)
        fig, axes = plt.subplots(2, 3, figsize=size_in)
        outdir = os.path.join(fig_folder, 'horizontal centerline-' + name)
        os.makedirs(outdir, exist_ok=True)
    else:
        outdir = os.path.join(fig_folder, 'horizontal centerline-' + name)
        os.makedirs(outdir, exist_ok=True)
        gridspec_kw={'height_ratios': [1, 1, 0.02]} # add space for universal legend
        size_in = (12, 9)
        fig, axes = plt.subplots(3, 3, figsize=size_in, gridspec_kw=gridspec_kw)
        for a in axes[2, :]:
            a.remove()
        case_handles = [
            Line2D([0], [0], color = case_opt[case]['color'], linestyle=case_opt[case]['linestyle'], label=case) for case in case_names]

        fig.legend(handles=case_handles,
                loc='lower center',
                ncols=num_cases,
                bbox_to_anchor=(0.52, 0.015))

    td = time[it] / 3600 / 24
    fig.suptitle(f'{td:.2f} days', )

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
    for n, case in enumerate(case_opt.keys()):
        if n == 0:
            ax1.plot(y, u[n], label=r"u$_{\text{centerline}}$", color = case_opt[case]['color'], linestyle='dotted', linewidth = 2.0)
            ax1.plot(y, v[n], label=r"v$_{\text{centerline}}$", color = case_opt[case]['color'], linestyle=case_opt['line_styles'], linewidth = 2.0)
            ax1.plot(y, w[n], label=r"w$_{\text{centerline}}$", color = case_opt[case]['color'], linestyle=case_opt[case]['linestyle'], linewidth = 2.0)
        else:
            ax1.plot(y, u[n], color = case_opt[case]['color'], linestyle='dotted', linewidth = 2.0)
            ax1.plot(y, v[n], color = case_opt[case]['color'], linestyle=case_opt['line_styles'], linewidth = 2.0)
            ax1.plot(y, w[n], color = case_opt[case]['color'], linestyle=case_opt[case]['linestyle'], linewidth = 2.0)
    ax1.set_title("Velocity")
    ax1.set_xlim(-lx[0][1]/2, lx[0][1]/2)
    ax1.set_ylim(ranges['w'])
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(-3,2), useMathText=True)
    ax1.legend(loc='lower right')

    # horizontal buoyancy flux 
    for n, case in enumerate(case_opt.keys()):
        if n == 0:
            ax2.plot(y, bu_fluc[n], color = case_opt[case]['color'], label = r"b$'$u$'_{\text{centerline}}$", linestyle='dotted', linewidth = 2.0)
            ax2.plot(y, bv_fluc[n], color = case_opt[case]['color'], label = r"b$'$v$'_{\text{centerline}}$", linestyle=case_opt['line_styles'], linewidth = 2.0)
        else:
            ax2.plot(y, bu_fluc[n], color = case_opt[case]['color'], linestyle='dotted', linewidth = 2.0)
            ax2.plot(y, bv_fluc[n], color = case_opt[case]['color'], linestyle=case_opt['line_styles'], linewidth = 2.0)
    ax2.set_title('Horizontal Buoyancy Flux Fluctuations')
    ax2.set_xlim(-lx[0][1]/2, lx[0][1]/2)
    ax2.set_ylim(ymin = ranges['b_flux'][0], ymax = ranges['b_flux'][1])
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(-3,2), useMathText=True)

    # tracer  
    for n, case in enumerate(case_opt.keys()):
        ax3.plot(y, tracer[n], color = case_opt[case]['color'], linestyle=case_opt[case]['linestyle'], linewidth = 2.0)
    ax3.set_title("Tracer")
    ax3.set_xlim(-lx[0][1]/2, lx[0][1]/2)
    ax3.set_ylim(ranges['Tracer'])
    ax3.ticklabel_format(axis='y', style='sci', scilimits=(-3,2), useMathText=True)

    # Perturbed buoyancy 
    for n, case in enumerate(case_opt.keys()):
        ax4.plot(y, b_center[n], color = case_opt[case]['color'], linestyle=case_opt[case]['linestyle'], linewidth = 2.0)
    ax4.set_title("Perturbed Buoyancy")
    ax4.set_xlim(-lx[0][1]/2, lx[0][1]/2)
    ax4.set_ylim(ranges['b_fluc'])
    ax4.ticklabel_format(axis='y', style='sci', scilimits=(-3,2), useMathText=True)
    # perturbed buoyancy flux 
    for n, case in enumerate(case_opt.keys()):
        ax5.plot(y, bw_fluc[n], color = case_opt[case]['color'], linestyle=case_opt[case]['linestyle'], linewidth = 2.0)
    ax5.set_title("Vertical Buoyancy Flux Fluctuations")
    ax5.set_xlim(-lx[0][1]/2, lx[0][1]/2)
    ax5.set_ylim(ranges['bw_fluc'])
    ax5.ticklabel_format(axis='y', style='sci', scilimits=(-3,2), useMathText=True) 

    # temperature  
    for n, case in enumerate(case_opt.keys()):
        ax6.plot(y, T[n], color = case_opt[case]['color'], linestyle=case_opt[case]['linestyle'], linewidth = 2.0)
    ax6.set_title("Temperature")
    ax6.set_xlim(-lx[0][1]/2, lx[0][1]/2)
    ax6.set_ylim(ranges['T'])

    # --- Save Frame ---
    save_frame(fig, outdir, it, size_in, file_name = 'hor_centerline_comparisons_')

    return outdir # return the directory where frames are saved for video creation
## plume depths ##
def plot_plume_depths(time, case_opt, fig_folder, lx, zp, zneutral, zc, contour, trend = True):
    num_cases = case_opt.keys()
    ncols = 3
    nrows = 2
    outdir = os.path.join(fig_folder, 'binning', 'plume_depths')
    os.makedirs(outdir, exist_ok=True)
    ar = np.ones(nrows)
    ar[-1] = 0.05 # add space for universal legend
    gridspec_kw={'height_ratios': ar} # add space for universal legend
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 8), gridspec_kw=gridspec_kw, sharey = True)
    for a in axes[-1, :]:
        a.remove()
    case_handles = [
        Line2D([0], [0], color = case_opt[case]['color'], linestyle=case_opt[case]['linestyle'], label=case)
        for i in range(num_cases)
    ]
    fig.legend(handles=case_handles,
            loc='lower center',
            ncols=num_cases,
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

    for n, case in enumerate(case_opt.keys()):
        if n == 0:
            tmax = time[n].max() / 3600
            ax[0].plot(time[n]/3600, zp[n], label=r"z$_{w=0}$", color = case_opt[case]['color'], linewidth = 2.0)
            ax[1].plot(time[n]/3600, zneutral[n], label=r"z$_{b=0}$", color = case_opt[case]['color'], linewidth = 2.0)
            ax[2].plot(time[n]/3600, zc[n], label=rf"z$_{{contour = {contour:.2f}}}$", color = case_opt[case]['color'], linewidth = 2.0)
        else:
            tmax = max(tmax, time[n].max() / 3600)
            ax[0].plot(time[n]/3600, zp[n], color = case_opt[case]['color'], linewidth = 2.0)
            ax[1].plot(time[n]/3600, zneutral[n], color = case_opt[case]['color'], linewidth = 2.0)
            ax[2].plot(time[n]/3600, zc[n], color = case_opt[case]['color'], linewidth = 2.0)
        if trend:
            start = 10
            print(time[n][start:]/3600)
            print(zp[n][start:])
            vars = np.polyfit(time[n][start:]/3600, zp[n][start:], 1)
            z_trend = time[n]/3600 * vars[0] + vars[1]
            ax[0].plot(time[n]/3600, z_trend, label=rf"z = {vars[0]:.2f}t + {vars[1]:.2f}", color = case_opt[case]['color'], linestyle = '--', linewidth = 2.0)
            ax[0].plot(time[n]/3600, z_trend, label=rf"z = {vars[0]:.2f}t + {vars[1]:.2f}", color = case_opt[case]['color'], linestyle = '--', linewidth = 2.0)
    for a in ax:
        a.set_ylim(ymin = -lx[-1].max(), ymax = 0.0)
        a.set_xlim(0, tmax)
        a.legend(loc='upper right')

    # --- Save Frame ---
    frame_path = os.path.join(outdir, f"comparison_depths_{contour}.svg")
    plt.savefig(frame_path)
    plt.close(fig)
    return outdir # return the directory where frames are saved for video creation 