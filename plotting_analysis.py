import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker

from matplotlib.lines import Line2D
from matplotlib import cm
from fractions import Fraction

from plotting_general import create_video
### -------------------------PLOTTING ND FUNCTIONS------------------------- ###
## ND Richardson 
def plot_rig_exponents(color_opt, title, file_name, fig_folder, w_rms, b_center, bw, rp, T, S, z_nd, Ri_g, case_names, exponents = [-0.5, -1/3, -0.25, 0.0, 0.25, 1/3, 0.5], z_str = rf"(z-h$_{{ML}}$)h$_{{ML}}^{{1/3}}$/L$_N^{{4/3}}$"):
    num_cases = len(case_names)
    scale = np.ones(7) 
    scale[-1] = 0.02
    gridspec_kw={'height_ratios': scale}
    fig, axes = plt.subplots(7, len(exponents), figsize=(len(exponents)*3, 25), sharey=True, gridspec_kw = gridspec_kw)
    plt.subplots_adjust(top=0.9)
    for a in axes[-1, :]:
        a.remove()
    case_handles = [Line2D([0], [0], color=color_opt[i], linestyle='solid', label=case_names[i])for i in range(num_cases)]
    fig.legend(handles=case_handles,
            loc='lower center',
            ncol=num_cases,
            bbox_to_anchor=(0.52, 0.005), )
    fig.suptitle(title,  y = 0.99)
    """
    axes[0, :] = ND rms velocity vs z_nd varied exponent of Ri_g
    axes[1, :] = ND centerline buoyancy vs z_nd varied exponent of Ri_g
    axes[2, :] = ND average buoyancy flux vs z_nd varied exponent of Ri_g
    axes[3, :] = ND radius of plume vs z_nd varied exponent of Ri_g
    axes[4, :] = ND perturbed temperature vs z_nd varied exponent of Ri_g
    axes[5, :] = ND average salinity vs z_nd varied exponent of Ri_g
    """
    for ax in axes[:, 0]:
        ax.set_ylabel(z_str, )
    for ax, exp in zip(axes[0, :], exponents):
        for i in range(num_cases):
            correction = Ri_g[i]**exp
            ax.plot(w_rms[:, i] * (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'Ri$_g^{{{exp:.2f}}}$', )
        ax.set_xlabel(rf"$w_{{rms}}/\sqrt{{\text{{g r}}_{{j}}}} \cdot Ri_g^{{{exp:.2f}}}$", )

    for ax, exp in zip(axes[1, :], exponents):
        for i in range(num_cases):
            correction = Ri_g[i]**exp
            ax.plot(b_center[:, i] *(correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'Ri$_g^{{{exp:.2f}}}$', )
        ax.set_xlabel(rf"$b_{{\text{{centerline}}}}/g \cdot Ri_g^{{{exp:.2f}}}$", )

    for ax, exp in zip(axes[2, :], exponents):
        for i in range(num_cases):
            correction = Ri_g[i]**exp
            ax.plot(bw[:, i] * (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'Ri$_g^{{{exp:.2f}}}$', )
        ax.set_xlabel(rf"$(\langle b'w'\rangle_{{xy}}/\sqrt{{\text{{g}}^3 \text{{r}}_{{j}}}})\cdot Ri_g^{{{exp:.2f}}}$", )

    for ax, exp in zip(axes[3, :], exponents):
        for i in range(num_cases):
            correction = Ri_g[i]**exp
            ax.plot(rp[:, i] * (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'Ri$_g^{{{exp:.2f}}}$', )
        ax.set_xlabel(rf"(r/r$_{{j}})\cdot Ri_g^{{{exp:.2f}}}$", )
    
    for ax, exp in zip(axes[4, :], exponents):
        for i in range(num_cases):
            correction = Ri_g[i]**exp
            ax.plot(T[:, i] * (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'Ri$_g^{{{exp:.2f}}}$', )
        ax.set_xlabel(rf"$(\text{{T'}}_{{\text{{centerline}}}}\alpha)\cdot Ri_g^{{{exp:.2f}}}$", )
    
    for ax, exp in zip(axes[5, :], exponents):
        for i in range(num_cases):
            correction = Ri_g[i]**exp
            ax.plot(S[:, i] * (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'Ri$_g^{{{exp:.2f}}}$', )
        ax.set_xlabel(rf"($\langle$C$\rangle_{{\text{{xy}}}} \beta)\cdot Ri_g^{{{exp:.2f}}}$", )
    #axes[5, 0].legend()

    plt.tight_layout()

    # --- Save Frame ---
    str_exp = list(map(str, exponents))
    str_exp = '_'.join(f"{x:.3g}" for x in exponents)
    frame_path = os.path.join(fig_folder, f"Ri_{file_name} _pow{str_exp}.png")
    plt.savefig(frame_path)
    plt.close(fig)
## ND Froude
def plot_Fr_exponents(color_opt, title, file_name, fig_folder, w_rms, b_center, bw, rp, T, S, z_nd, Fr, case_names, exponents = [-0.5, -1/3, -0.25, 0.0, 0.25, 1/3, 0.5], z_str = rf"(z-h$_{{ML}}$)h$_{{ML}}^{{1/3}}$/L$_N^{{4/3}}$"):
    num_cases = len(case_names)
    scale = np.ones(7) 
    scale[-1] = 0.02
    gridspec_kw={'height_ratios': scale}
    fig, axes = plt.subplots(7, len(exponents), figsize=(len(exponents)*3, 25), sharey=True, gridspec_kw = gridspec_kw)
    plt.subplots_adjust(top=0.9)
    for a in axes[-1, :]:
        a.remove()
    case_handles = [Line2D([0], [0], color=color_opt[i], linestyle='solid', label=case_names[i])for i in range(num_cases)]
    fig.legend(handles=case_handles,
            loc='lower center',
            ncol=num_cases,
            bbox_to_anchor=(0.52, 0.005), )
    fig.suptitle(title,  y = 0.99)
    """
    axes[0, :] = ND rms velocity vs z_nd varied exponent of Fr
    axes[1, :] = ND centerline buoyancy vs z_nd varied exponent of Fr
    axes[2, :] = ND average buoyancy flux vs z_nd varied exponent of Fr
    axes[3, :] = ND radius of plume vs z_nd varied exponent of Fr
    axes[4, :] = ND perturbed temperature vs z_nd varied exponent of Fr
    axes[5, :] = ND average salinity vs z_nd varied exponent of Fr
    """
    for ax in axes[:, 0]:
        ax.set_ylabel(z_str, )
    for ax, exp in zip(axes[0, :], exponents):
        for i in range(num_cases):
            correction = Fr[i]**exp
            ax.plot(w_rms[:, i] * (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'Fr$^{{{exp:.2f}}}$', )
        ax.set_xlabel(rf"$w_{{rms}}/\sqrt{{\text{{g r}}_{{j}}}} \cdot Fr^{{{exp:.2f}}}$", )
    
    for ax, exp in zip(axes[1, :], exponents):
        for i in range(num_cases):
            correction = Fr[i]**exp
            ax.plot(b_center[:, i] * (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'Fr$^{{{exp:.2f}}}$', )
        ax.set_xlabel(rf"$b_{{\text{{centerline}}}}/g \cdot Fr^{{{exp:.2f}}}$", )

    for ax, exp in zip(axes[2, :], exponents):
        for i in range(num_cases):
            correction = Fr[i]**exp
            ax.plot(bw[:, i] * (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'Fr$^{{{exp:.2f}}}$', )
        ax.set_xlabel(rf"$(\langle b'w'\rangle_{{xy}}/\sqrt{{\text{{g}}^3 \text{{r}}_{{j}}}})\cdot Fr^{{{exp:.2f}}}$", )
    #axes[2, 0].legend()

    for ax, exp in zip(axes[3, :], exponents):
        for i in range(num_cases):
            correction = Fr[i]**exp
            ax.plot(rp[:, i] * (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'Fr$^{{{exp:.2f}}}$', )
        ax.set_xlabel(rf"(r/r$_{{j}})\cdot Fr^{{{exp:.2f}}}$", )
    #axes[3, 0].legend()

    for ax, exp in zip(axes[4, :], exponents):
        for i in range(num_cases):
            correction = Fr[i]**exp
            ax.plot(T[:, i] * (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'Fr$^{{{exp:.2f}}}$', )
        ax.set_xlabel(rf"$(\text{{T'}}_{{\text{{centerline}}}}\alpha)\cdot Fr^{{{exp:.2f}}}$", )
    #axes[4, 0].legend()

    for ax, exp in zip(axes[5, :], exponents):
        for i in range(num_cases):
            correction = Fr[i]**exp
            ax.plot(S[:, i] * (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'Fr$^{{{exp:.2f}}}$', )
        ax.set_xlabel(rf"($\langle$C$\rangle_{{\text{{xy}}}} \beta)\cdot Fr^{{{exp:.2f}}}$", )

    plt.tight_layout()

    # --- Save Frame ---
    str_exp = '_'.join(f"{x:.3g}" for x in exponents)
    frame_path = os.path.join(fig_folder, f"Fr_{file_name} _pow{str_exp}.png")
    plt.savefig(frame_path)
    plt.close(fig)
## ND MLD
def plot_mld_exponents(color_opt, title, file_name, fig_folder, w_rms, b_center, bw, rp, T, S, z_nd, mld, case_names, exponents = [-0.5, -1/3, -0.25, 0.0, 0.25, 1/3, 0.5], z_str = rf"(z-h$_{{ML}}$)h$_{{ML}}^{{1/3}}$/L$_N^{{4/3}}$"):
    num_cases = len(case_names)
    scale = np.ones(7) 
    scale[-1] = 0.02
    gridspec_kw={'height_ratios': scale}
    fig, axes = plt.subplots(7, len(exponents), figsize=(len(exponents)*3, 25), sharey=True, gridspec_kw = gridspec_kw)
    plt.subplots_adjust(top=0.9)
    for a in axes[-1, :]:
        a.remove()
    case_handles = [Line2D([0], [0], color=color_opt[i], linestyle='solid', label=case_names[i])for i in range(num_cases)]
    fig.legend(handles=case_handles,
            loc='lower center',
            ncol=num_cases,
            bbox_to_anchor=(0.52, 0.005), )
    fig.suptitle(title,  y = 0.99)
    """
    axes[0, :] = ND rms velocity vs z_nd varied exponent of MLD
    axes[1, :] = ND centerline buoyancy vs z_nd varied exponent of MLD
    axes[2, :] = ND average buoyancy flux vs z_nd varied exponent of MLD
    axes[3, :] = ND radius of plume vs z_nd varied exponent of MLD
    axes[4, :] = ND perturbed temperature vs z_nd varied exponent of MLD
    axes[5, :] = ND average salinity vs z_nd varied exponent of MLD
    """

    for ax in axes[:, 0]:
        ax.set_ylabel(z_str, )

    for ax, exp in zip(axes[0, :], exponents):
        for i in range(num_cases):
            correction = mld[i]**exp
            ax.plot(w_rms[:, i] * (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'$\hat{{h}}_{{ML}}^{{{exp:.2f}}}$', )
        ax.set_xlabel(rf"$w_{{rms}}/\sqrt{{\text{{g}} \text{{r}}_{{j}}}} \cdot \hat{{h}}_{{ML}}^{{{exp:.2f}}}$", )

    for ax, exp in zip(axes[1, :], exponents):
        for i in range(num_cases):
            correction = mld[i]**exp
            ax.plot(b_center[:, i] *(correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'$\hat{{h}}_{{ML}}^{{{exp:.2f}}}$', )
        ax.set_xlabel(rf"$b_{{\text{{centerline}}}}/g \cdot \hat{{h}}_{{ML}}^{{{exp:.2f}}}$", )

    for ax, exp in zip(axes[2, :], exponents):
        for i in range(num_cases):
            correction = mld[i]**exp
            ax.plot(bw[:, i] * (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'$\hat{{h}}_{{ML}}^{{{exp:.2f}}}$', )
        ax.set_xlabel(rf"$(\langle b'w'\rangle_{{\text{{xy}}}}/\sqrt{{\text{{g}}^3 \text{{r}}_{{j}}}})\cdot \hat{{h}}_{{ML}}^{{{exp:.2f}}}$", )

    for ax, exp in zip(axes[3, :], exponents):
        for i in range(num_cases):
            correction = mld[i]**exp
            ax.plot(rp[:, i] * (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'$\hat{{h}}_{{ML}}^{{{exp:.2f}}}$', )
        ax.set_xlabel(rf"(r/r$_{{j}})\cdot \hat{{h}}_{{ML}}^{{{exp:.2f}}}$", )

    for ax, exp in zip(axes[4, :], exponents):
        for i in range(num_cases):
            correction = mld[i]**exp
            ax.plot(T[:, i] * (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'$\hat{{h}}_{{ML}}^{{{exp:.2f}}}$', )
        ax.set_xlabel(rf"$(\text{{T'}}_{{\text{{centerline}}}}\alpha)\cdot \hat{{h}}_{{ML}}^{{{exp:.2f}}}$", )

    for ax, exp in zip(axes[5, :], exponents):
        for i in range(num_cases):
            correction = mld[i]**exp
            ax.plot(S[:, i] * (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'$\hat{{h}}_{{ML}}^{{{exp:.2f}}}$', )
        ax.set_xlabel(rf"($\langle$C$\rangle_{{\text{{xy}}}} \beta)\cdot \hat{{h}}_{{ML}}^{{{exp:.2f}}}$", )

    plt.tight_layout()

    # --- Save Frame ---
    str_exp = '_'.join(f"{x:.3g}" for x in exponents)
    frame_path = os.path.join(fig_folder, f"MLD_{file_name} _pow{str_exp}.png")
    plt.savefig(frame_path)
    plt.close(fig)
## All ND
def plot_combo_exponents(color_opt, title, file_name, fig_folder, w_rms, b_center, bw, rp, T, S, z_nd, vars_exps, Ri_g, Fr, mld, case_names, z_str = rf"(z-h$_{{ML}}$)h$_{{ML}}^{{1/3}}$/L$_N^{{4/3}}$"):
    NDs = [rf"Ri$_g^", rf"Fr$^", rf"$\hat{{h}}_{{ML}}^"] 
    NDs_filtered = [[("" if str(Fraction(x).limit_denominator()) == '0' 
                else NDs[j] + "{"+str(Fraction(x).limit_denominator())+"}$")
                for j, x in enumerate(row)] for row in vars_exps]
    vars_str = [''.join(row) for row in NDs_filtered]
    num_cases = len(case_names)
    if num_cases > 4:
        gridspec_kw={'height_ratios': [1, 0.3]}
        n_col = num_cases//2
    else:
        gridspec_kw={'height_ratios': [1, 0.1]}
        n_col = num_cases

    fig, axes = plt.subplots(2, len(NDs_filtered), figsize=(12, 4), sharey=True, gridspec_kw = gridspec_kw)
    plt.subplots_adjust(top=0.9)
    for a in axes[-1, :]:
        a.remove()
    case_handles = [Line2D([0], [0], color=color_opt[i], linestyle='solid', label=case_names[i])for i in range(num_cases)]
    fig.legend(handles=case_handles,
            loc='lower center',
            ncol=n_col,
            bbox_to_anchor=(0.52, 0.005), )
    fig.suptitle(title,  y = 0.99)
    """
    axes[0] = ND rms velocity vs z_nd varied exponent of all
    axes[1] = ND centerline buoyancy vs z_nd varied exponent of all
    axes[2] = ND average buoyancy flux vs z_nd varied exponent of all
    axes[3] = ND radius of plume vs z_nd varied exponent of all
    axes[4] = ND perturbed temperature vs z_nd varied exponent of all
    axes[5] = ND average salinity vs z_nd varied exponent of all
    """

    for ax in axes[:, 0]:
        ax.set_ylabel(z_str, )

    for i in range(num_cases):
        axes[0, 0].plot(w_rms[:, i] * mld[i]**vars_exps[0, 2] * Ri_g[i]**vars_exps[0, 0] * Fr[i]**vars_exps[0, 1], 
                z_nd[:, i], color=color_opt[i])
    axes[0, 0].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
    axes[0, 0].set_xlabel(rf"$w_{{rms}}/\sqrt{{\text{{g r}}_{{j}}}}\cdot$ {vars_str[0]}", )

    for i in range(num_cases):
        axes[0, 1].plot(b_center[:, i] * mld[i]**vars_exps[1, 2] * Ri_g[i]**vars_exps[1, 0] * Fr[i]**vars_exps[1, 1], 
                z_nd[:, i], color=color_opt[i])
    axes[0, 1].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
    axes[0, 1].set_xlabel(rf"$b_{{\text{{centerline}}}}/g \cdot$ {vars_str[1]}", )

    for i in range(num_cases):
        axes[0, 2].plot(bw[:, i] * mld[i]**vars_exps[2, 2] * Ri_g[i]**vars_exps[2, 0] * Fr[i]**vars_exps[2, 1], 
                z_nd[:, i], color=color_opt[i])
    axes[0, 2].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
    axes[0, 2].set_xlabel(rf"$(\langle b'w'\rangle_{{xy}}/\sqrt{{\text{{g}}^3 \text{{r}}_{{j}}}})\cdot$ {vars_str[2]}", )

    for i in range(num_cases):
        axes[0, 3].plot(rp[:, i] * mld[i]**vars_exps[3, 2] * Ri_g[i]**vars_exps[3, 0] * Fr[i]**vars_exps[3, 1], 
                z_nd[:, i], color=color_opt[i])
    axes[0, 3].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
    axes[0, 3].set_xlabel(rf"(r/r$_{{j}})\cdot$ {vars_str[3]}", )

    for i in range(num_cases):
        axes[0, 4].plot(T[:, i] * mld[i]**vars_exps[4, 2] * Ri_g[i]**vars_exps[4, 0] * Fr[i]**vars_exps[4, 1],
                z_nd[:, i], color=color_opt[i])
    axes[0, 4].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
    axes[0, 4].set_xlabel(rf"$(\text{{T'}}_{{\text{{centerline}}}}\alpha)\cdot$ {vars_str[4]}", )

    for i in range(num_cases):
        axes[0, 5].plot(S[:, i] * mld[i]**vars_exps[5, 2] * Ri_g[i]**vars_exps[5, 0] * Fr[i]**vars_exps[5, 1], 
                z_nd[:, i], color=color_opt[i])
    axes[0, 5].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
    axes[0, 5].set_xlabel(rf"($\langle$C$\rangle_{{\text{{xy}}}} \beta)\cdot$ {vars_str[5]}", )

    plt.tight_layout()

    # --- Save Frame ---
    frame_path = os.path.join(fig_folder, f"{file_name} _combined.png")
    i = 0
    while True:
        i += 1
        frame_path = os.path.join(fig_folder, f"{file_name} _combined_{i}.png")
        if os.path.exists(frame_path):
            continue
        plt.savefig(frame_path)
        break
    plt.close(fig)

### -------------------------PLOTTING R FUNCTIONS------------------------- ###
def plot_r_at_depth_in_time(color_opt, fig_folder, case_names, time, r, tol, neutral, r_max, z_max, lz, best_fit, fit_exp, ND = False, log_auto = True):
    num_cases = len(case_names)
    outdir = os.path.join(fig_folder)
    os.makedirs(outdir, exist_ok=True)
    ncols = 1 + len(tol)
    nrows = num_cases
    hor_len = 12.0
    vert_len = nrows * hor_len/ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(hor_len, vert_len), sharex = True)
    rmin = np.min(np.concatenate(r))
    rmax = np.max(np.concatenate(r_max))
    width = 0.8
    if ND and log_auto:
        x_label = r"Time, tN"
        y_label = r"Radius, r/r$_j$"
        z_label = r"Depth, z/r$_j$"
    elif ND and not log_auto:
        x_label = r"Time, log(tN)"
        y_label = r"Radius, log(r/r$_j$)"
        z_label = r"Depth, (z/r$_j$)"
    elif not ND and log_auto:
        x_label = r"Time, log(seconds)"
        y_label = r"Radius, log(m)"
        z_label = r"Depth, (m)"
    else:
        x_label = r"Time (seconds)"
        y_label = r"Radius (m)"
        z_label = r"Depth (m)"
    for row in range(nrows):
        td = time[row] #/ 3600 / 24
        plt.tight_layout()
        fig.subplots_adjust(hspace=0.15, top=0.96)
        ax_row = axes[row, :] if nrows > 1 else axes
        for i, ax in enumerate(ax_row):
            if i == 0:
                for n in range(0, len(color_opt)):
                    ax.plot(td, z_max[row][:, n], color=color_opt[n], linestyle='--', linewidth=width)
                    ax.plot(td, neutral[row][:, n], color=color_opt[n])
                ax.plot(td, z_max[row][:, 0], color=color_opt[0], linestyle='--', label=r'max radius', linewidth=width)
                ax.plot(td, neutral[row][:, 0], color=color_opt[0], label=r'neutral')
                ax.set_ylabel(z_label)
                ax.legend(loc='upper right', handlelength=0.9)
                ax.set_ylim(lz)
            else:
                if log_auto:
                    ax.set_xscale('log')
                    ax.set_yscale('log')
                ax.plot(td, r_max[row][:, i - 1], color=color_opt[i-1], linestyle='--')
                ax.plot(td, r[row][:, i-1], color=color_opt[i-1])
                ax.plot(td, best_fit[0][row][:, i-1], color=color_opt[i-1], linestyle=':', linewidth=width, label = rf"t$^{{{fit_exp[row][0][i-1, 0]:.2f}}}-{{{fit_exp[row][0][i-1, 1]:.2f}}}$") 
                ax.plot(td, best_fit[1][row][:, i-1], color=color_opt[i-1], linestyle='-.', linewidth=width, label = rf"t$^{{{fit_exp[row][1][i-1, 0]:.2f}}}-{{{fit_exp[row][1][i-1, 1]:.2f}}}$") 
                ax.set_ylim(rmin, rmax)
                ax.set_ylabel(y_label)
                ax.legend(loc='lower right', handlelength=0.9)
            if row == nrows - 1:
                ax.set_xlabel(x_label)
            elif row == 0:
                ax.set_title(f"Contour = {tol[i-1]:.2e} ")
        x_center = 0.5
        y_pos = (nrows - row - 0.01*(nrows - row)) / nrows  # approximate vertical center of row
        fig.text(x_center, y_pos, case_names[row], ha='center', va='center', fontweight='bold', transform=fig.transFigure)
    # --- Save Frame ---
    file_name = 'ND_log_tracer_radius.svg' if ND else 'log_tracer_radius.svg'
    frame_path = os.path.join(outdir, file_name)
    plt.savefig(frame_path, bbox_inches='tight')
    plt.close(fig)

### ---------------------- CONVERGENCE TESTS ----------------------------- ###
def convergence_tests(time, it, ranges, fig_folder, lx, nx, x, y, z, cases_sorted, matrix_N, ver, hor, 
                      b, b_avg, b_rms_sign, w_rms, b_rms, bw_fluc, b_flux_avg, b_max_sign_change_to_negative_loc, L_ozmidov, L_ozmidov_background, idx_neg, plot_points = False):

    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))

    levels = 500

    colors = ['black', 'blue', 'green', 'red', 'purple', 'pink', 'gray', 'orange', 'cyan', 'olive']
    td = time[it] / 3600 / 24

    ## buoyancy analysis profiles
    outdir = [fig_folder + 'convergence tests buoyancy profiles/']
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(3, 5, figsize=(12, 5), height_ratios = [1, 0.2, 1])

    fig.text(0.5, 1.08, f'{td:.2f} days', ha="center", ) 
    # Titles for each row
    fig.text(0.5, 1.05, "Vertical resolution convergence", ha="center", fontsize=14)
    fig.text(0.5, 0.52, "Horizontal resolution convergence", ha="center", fontsize=14)
    
    ax1 = axes[0, 0]
    ax6 = axes[2, 0]
    for a in axes[1, :]:
        a.remove()
    for caseindex, case in enumerate(cases_sorted):
        if ver[caseindex] and hor[caseindex]:
            name_case = case.replace('flux b tracer ', "")
            # buoyancy profile
            axes[0, 0].plot(b_avg[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex], label = name_case)
            # RMS buoyancy 
            axes[0, 1].plot(b_rms[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex])
            # RMS w
            axes[0, 2].plot(w_rms[0:nx[2, caseindex]+1, caseindex], z[0:nx[2, caseindex]+1, caseindex], color = colors[caseindex])#, linestyle = '--', label = r"$\langle$w$_{rms}rangle_{\text{xy}}$")
            # RMS buoyancy flux 
            axes[0, 3].plot(bw_fluc[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex])
            # buoyancy profile
            axes[2, 0].plot(b_avg[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex], label = name_case)
            # RMS buoyancy 
            axes[2, 1].plot(b_rms[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex])
            # RMS w 
            axes[2, 2].plot(w_rms[0:nx[2, caseindex]+1, caseindex], z[0:nx[2, caseindex]+1, caseindex], color = colors[caseindex])#, linestyle = '--', label = r"$\langle$w$_{rms}\rangle_{\text{xy}}$")
            # RMS buoyancy flux 
            axes[2, 3].plot(bw_fluc[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex])
        elif ver[caseindex] and not hor[caseindex]:
            name_case = case.replace('flux b tracer ', "")
            # buoyancy profile
            axes[0, 0].plot(b_avg[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex], label = name_case)
            # RMS buoyancy 
            axes[0, 1].plot(b_rms[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex])
            # RMS w 
            axes[0, 2].plot(w_rms[0:nx[2, caseindex]+1, caseindex], z[0:nx[2, caseindex]+1, caseindex], color = colors[caseindex])#, linestyle = '--')
            # RMS buoyancy flux 
            axes[0, 3].plot(bw_fluc[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex])
        elif hor[caseindex] and not ver[caseindex]:
            name_case = case.replace('flux b tracer ', "")
            # buoyancy profile
            axes[2, 0].plot(b_avg[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex], label = name_case)
            # RMS buoyancy 
            axes[2, 1].plot(b_rms[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex])
            # RMS w 
            axes[2, 2].plot(w_rms[0:nx[2, caseindex]+1, caseindex], z[0:nx[2, caseindex]+1, caseindex], color = colors[caseindex])#, linestyle = ':')
            # RMS buoyancy flux 
            axes[2, 3].plot(bw_fluc[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex])

    # Ozmidov length scale
    axes[2, 4].plot(nx[0, hor], L_ozmidov_background[it, hor], marker = '+', label = r"b$_{\text{stratified}, 3}$ L$_{O}$", linestyle = 'none')
    axes[2, 4].plot(nx[0, hor], L_ozmidov[it, hor], marker = 'o', label = r"b$_{\text{average}, 3}$ L$_{O}$", linestyle = 'none')
    axes[0, 4].plot(nx[2, ver], L_ozmidov_background[it, ver], marker = '+', label = r"b$_{\text{stratified}, 3}$ L$_{O}$", linestyle = 'none')
    axes[0, 4].plot(nx[2, ver], L_ozmidov[it, ver], marker = 'o', label = r"b$_{\text{average}, 3}$ L$_{O}$", linestyle = 'none')

    axes[0, 0].set_xlabel("[m/s$^{2}$]")
    axes[0, 0].set_ylim([-lx[2], 0])
    axes[0, 0].set_title("Buoyancy")
    axes[0, 0].set_ylim([-lx[2], 0])
    axes[0, 0].set_ylabel("y [m]")
    axes[0, 0].set_xlim(ranges['b_avg'])
    axes[0, 0].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
    
    axes[0, 1].set_xlabel("[m/s$^{2}$]")
    axes[0, 1].set_title("Buoyancy RMS")
    axes[0, 1].set_xlim(ranges['b_rms'])
    axes[0, 1].set_ylim([-lx[2], 0])

    axes[0, 2].set_xlabel("[m/s]")
    axes[0, 2].set_title("w RMS")
    axes[0, 2].set_xlim(ranges['vel_rms'])
    axes[0, 2].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
    axes[0, 2].set_ylim([-lx[2], 0])

    axes[0, 3].set_xlabel("[m$^{2}$/s$^{3}$]")
    axes[0, 3].set_title("Buoyancy Flux Flucts")
    axes[0, 3].set_xlim(ranges['bflux_rms'])
    axes[0, 3].set_ylim([-lx[2], 0])

    axes[0, 4].legend(loc='upper right', handlelength=0.75)
    axes[0, 4].set_ylabel("Length Scale [m]")
    axes[0, 4].set_title("Ozmidov Length Scale")
    axes[0, 4].set_ylim(ranges['b_avg'])
    axes[0, 4].set_xlabel("Time [days]")
    axes[0, 4].set_xlim([0, matrix_N +10])
    
    axes[2, 0].set_xlabel("[m/s$^{2}$]")
    axes[2, 0].set_xlim(ranges['b_avg'])
    axes[2, 0].set_title("Buoyancy")
    axes[2, 0].set_ylabel("Depth [m]")
    axes[2, 0].set_ylim([-lx[2], 0])
    axes[2, 0].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)

    axes[2, 1].set_xlabel("[m/s$^{2}$]")
    axes[2, 1].set_title("Buoyancy RMS")
    axes[2, 1].set_xlim(ranges['b_rms'])
    axes[2, 1].set_ylim([-lx[2], 0])

    axes[2, 2].set_xlabel("[m/s]")
    axes[2, 2].set_title("w RMS")
    axes[2, 2].set_xlim(ranges['vel_rms'])
    axes[2, 2].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
    axes[2, 2].set_ylim([-lx[2], 0]) 

    axes[2, 3].set_xlabel("[m$^{2}$/s$^{3}$]")
    axes[2, 3].set_title("Buoyancy Flux Flucts")
    axes[2, 3].set_xlim(ranges['bflux_rms'])
    axes[2, 3].set_ylim([-lx[2], 0])

    axes[2, 4].set_title("Ozmidov Length Scale")
    axes[2, 4].set_ylabel("Length Scale [m]")
    axes[2, 4].set_ylim(ranges['lengthscale'])
    axes[2, 4].set_xlabel("Time [days]")
    axes[2, 4].set_xlim([0, matrix_N +10])
    axes[2, 4].legend(loc='upper right', handlelength=0.75)

    # universal legend
    handles0, labels0 = ax1.get_legend_handles_labels()
    handles1, labels1 = ax6.get_legend_handles_labels()

    fig.legend(handles0, labels0, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.99))
    fig.legend(handles1, labels1, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.46))

    # --- Save Frame ---
    frame_path = os.path.join(outdir, f"convergence_test_{it:04d}.png")
    plt.savefig(frame_path, bbox_inches="tight")
    print(f"Time step {it + 1} captured")
    plt.close(fig)

    ## buoyancy plane slices ##
    outdir1 = fig_folder + 'buoyancy planeslices/'
    os.makedirs(outdir1, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(12, 5), constrained_layout=True)
    fig.suptitle(f'{td:.2f} days') 
    norm = mcolors.Normalize(vmin=ranges['b_avg'][0], vmax=ranges['b_avg'][-1])
    mappable = cm.ScalarMappable(norm=norm) 
    hor_plot = 0
    ver_plot = 0
    for caseindex, case in enumerate(cases_sorted):
        X, Y, Z = np.meshgrid(x[0:nx[0, caseindex], caseindex] , y[0:nx[1, caseindex], caseindex] , z[0:nx[2, caseindex], caseindex])
        name_case = case.replace('flux b tracer ', "")
        axes[hor_plot, ver_plot].contourf(X[int(nx[0, caseindex]/2), :, :], Z[int(nx[0, caseindex]/2), :, :], b[caseindex, int(nx[0, caseindex]/2), 0:nx[1, caseindex], 0:nx[2, caseindex]], levels, norm=norm)
        axes[hor_plot, ver_plot].set_title(name_case)
        axes[hor_plot, ver_plot].set_xlabel("y [m]")
        axes[hor_plot, ver_plot].set_ylabel("z [m]")
        axes[hor_plot, ver_plot].set_aspect('equal')
        if ver_plot > 1:
            hor_plot += 1
            ver_plot = 0
        else:
            ver_plot += 1

    cbar = fig.colorbar(mappable, ax=axes, label=r"m/s$^2$", location='bottom', shrink=0.5, orientation='horizontal')
    
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
        fig = plt.figure(figsize=(12, 4))
        fig.tight_layout()
        fig.suptitle(f'{td:.2f} days', ) 
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

        fig.supxlabel("Number of Grid Cells", )
        # --- Save Frame ---
        frame_path = os.path.join(outdir, f"convergence_test_{it:04d}.png")
        plt.tight_layout()
        plt.savefig(frame_path)
        print(f"Time step {it + 1} captured")
        plt.close(fig)
        return outdir, outdir1, outdir2
    return outdir, outdir1 # return the directory where frames are saved for video
