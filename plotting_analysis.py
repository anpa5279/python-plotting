import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from fractions import Fraction

### -------------------------PLOTTING ND FUNCTIONS------------------------- ###
## ND Richardson 
def plot_rig_exponents(case_opt, title, file_name, fig_folder, w_rms, b_center, bw, rp, T, S, z_nd, Ri_g, exponents = [-0.5, -1/3, -0.25, 0.0, 0.25, 1/3, 0.5], z_str = rf"(z-h$_{{ML}}$)h$_{{ML}}^{{1/3}}$/L$_N^{{4/3}}$"):
    case_names = list(case_opt.keys())
    num_cases = len(case_names)
    scale = np.ones(7) 
    scale[-1] = 0.02
    gridspec_kw={'height_ratios': scale}
    fig, axes = plt.subplots(7, len(exponents), figsize=(len(exponents)*3, 25), sharey=True, gridspec_kw = gridspec_kw)
    plt.subplots_adjust(top=0.9)
    for a in axes[-1, :]:
        a.remove()
    case_handles = [Line2D([0], [0], color = case_opt[case]['color'], linestyle=case_opt[case]['linestyle'], label=case_opt.keys())for n, case in enumerate(case_names)]
    fig.legend(handles=case_handles,
            loc='lower center',
            ncols=num_cases,
            bbox_to_anchor=(0.52, 0.005), )
    fig.suptitle(title, y = 0.99)
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
        for n, case in enumerate(case_opt):
            correction = Ri_g[n]**exp
            ax.plot(w_rms[:, n] * (correction), 
                    z_nd[:, n], color = case_opt[case]['color'])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'Ri$_g^{{{exp:.2f}}}$', )
        ax.set_xlabel(rf"$w_{{rms}}/\sqrt{{\text{{g r}}_{{j}}}} \cdot Ri_g^{{{exp:.2f}}}$", )

    for ax, exp in zip(axes[1, :], exponents):
        for n, case in enumerate(case_opt):
            correction = Ri_g[n]**exp
            ax.plot(b_center[:, n] *(correction), 
                    z_nd[:, n], color = case_opt[case]['color'])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'Ri$_g^{{{exp:.2f}}}$', )
        ax.set_xlabel(rf"$b_{{\text{{centerline}}}}/g \cdot Ri_g^{{{exp:.2f}}}$", )

    for ax, exp in zip(axes[2, :], exponents):
        for n, case in enumerate(case_opt):
            correction = Ri_g[n]**exp
            ax.plot(bw[:, n] * (correction), 
                    z_nd[:, n], color = case_opt[case]['color'])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'Ri$_g^{{{exp:.2f}}}$', )
        ax.set_xlabel(rf"$(\langle b'w'\rangle_{{xy}}/\sqrt{{\text{{g}}^3 \text{{r}}_{{j}}}})\cdot Ri_g^{{{exp:.2f}}}$", )

    for ax, exp in zip(axes[3, :], exponents):
        for n, case in enumerate(case_opt):
            correction = Ri_g[n]**exp
            ax.plot(rp[:, n] * (correction), 
                    z_nd[:, n], color = case_opt[case]['color'])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'Ri$_g^{{{exp:.2f}}}$', )
        ax.set_xlabel(rf"(r/r$_{{j}})\cdot Ri_g^{{{exp:.2f}}}$", )
    
    for ax, exp in zip(axes[4, :], exponents):
        for n, case in enumerate(case_opt):
            correction = Ri_g[n]**exp
            ax.plot(T[:, n] * (correction), 
                    z_nd[:, n], color = case_opt[case]['color'])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'Ri$_g^{{{exp:.2f}}}$', )
        ax.set_xlabel(rf"$(\text{{T'}}_{{\text{{centerline}}}}\alpha)\cdot Ri_g^{{{exp:.2f}}}$", )
    
    for ax, exp in zip(axes[5, :], exponents):
        for n, case in enumerate(case_opt):
            correction = Ri_g[n]**exp
            ax.plot(S[:, n] * (correction), 
                    z_nd[:, n], color = case_opt[case]['color'])
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
def plot_Fr_exponents(case_opt, title, file_name, fig_folder, w_rms, b_center, bw, rp, T, S, z_nd, Fr, exponents = [-0.5, -1/3, -0.25, 0.0, 0.25, 1/3, 0.5], z_str = rf"(z-h$_{{ML}}$)h$_{{ML}}^{{1/3}}$/L$_N^{{4/3}}$"):
    case_names = list(case_opt.keys())
    num_cases = len(case_names)
    scale = np.ones(7) 
    scale[-1] = 0.02
    gridspec_kw={'height_ratios': scale}
    fig, axes = plt.subplots(7, len(exponents), figsize=(len(exponents)*3, 25), sharey=True, gridspec_kw = gridspec_kw)
    plt.subplots_adjust(top=0.9)
    for a in axes[-1, :]:
        a.remove()
    case_handles = [Line2D([0], [0], color = case_opt[case]['color'], linestyle=case_opt[case]['linestyle'], label=case_opt.keys())for n, case in enumerate(case_names)]
    fig.legend(handles=case_handles,
            loc='lower center',
            ncols=num_cases,
            bbox_to_anchor=(0.52, 0.005), )
    fig.suptitle(title, y = 0.99)
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
        for n, case in enumerate(case_opt):
            correction = Fr[n]**exp
            ax.plot(w_rms[:, n] * (correction), 
                    z_nd[:, n], color = case_opt[case]['color'])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'Fr$^{{{exp:.2f}}}$', )
        ax.set_xlabel(rf"$w_{{rms}}/\sqrt{{\text{{g r}}_{{j}}}} \cdot Fr^{{{exp:.2f}}}$", )
    
    for ax, exp in zip(axes[1, :], exponents):
        for n, case in enumerate(case_opt):
            correction = Fr[n]**exp
            ax.plot(b_center[:, n] * (correction), 
                    z_nd[:, n], color = case_opt[case]['color'])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'Fr$^{{{exp:.2f}}}$', )
        ax.set_xlabel(rf"$b_{{\text{{centerline}}}}/g \cdot Fr^{{{exp:.2f}}}$", )

    for ax, exp in zip(axes[2, :], exponents):
        for n, case in enumerate(case_opt):
            correction = Fr[n]**exp
            ax.plot(bw[:, n] * (correction), 
                    z_nd[:, n], color = case_opt[case]['color'])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'Fr$^{{{exp:.2f}}}$', )
        ax.set_xlabel(rf"$(\langle b'w'\rangle_{{xy}}/\sqrt{{\text{{g}}^3 \text{{r}}_{{j}}}})\cdot Fr^{{{exp:.2f}}}$", )
    #axes[2, 0].legend()

    for ax, exp in zip(axes[3, :], exponents):
        for n, case in enumerate(case_opt):
            correction = Fr[n]**exp
            ax.plot(rp[:, n] * (correction), 
                    z_nd[:, n], color = case_opt[case]['color'])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'Fr$^{{{exp:.2f}}}$', )
        ax.set_xlabel(rf"(r/r$_{{j}})\cdot Fr^{{{exp:.2f}}}$", )
    #axes[3, 0].legend()

    for ax, exp in zip(axes[4, :], exponents):
        for n, case in enumerate(case_opt):
            correction = Fr[n]**exp
            ax.plot(T[:, n] * (correction), 
                    z_nd[:, n], color = case_opt[case]['color'])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'Fr$^{{{exp:.2f}}}$', )
        ax.set_xlabel(rf"$(\text{{T'}}_{{\text{{centerline}}}}\alpha)\cdot Fr^{{{exp:.2f}}}$", )
    #axes[4, 0].legend()

    for ax, exp in zip(axes[5, :], exponents):
        for n, case in enumerate(case_opt):
            correction = Fr[n]**exp
            ax.plot(S[:, n] * (correction), 
                    z_nd[:, n], color = case_opt[case]['color'])
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
def plot_mld_exponents(case_opt, title, file_name, fig_folder, w_rms, b_center, bw, rp, T, S, z_nd, mld, exponents = [-0.5, -1/3, -0.25, 0.0, 0.25, 1/3, 0.5], z_str = rf"(z-h$_{{ML}}$)h$_{{ML}}^{{1/3}}$/L$_N^{{4/3}}$"):
    case_names = list(case_opt.keys())
    num_cases = len(case_names)
    scale = np.ones(7) 
    scale[-1] = 0.02
    gridspec_kw={'height_ratios': scale}
    fig, axes = plt.subplots(7, len(exponents), figsize=(len(exponents)*3, 25), sharey=True, gridspec_kw = gridspec_kw)
    plt.subplots_adjust(top=0.9)
    for a in axes[-1, :]:
        a.remove()
    case_handles = [Line2D([0], [0], color = case_opt[case]['color'], linestyle=case_opt[case]['linestyle'], label=case_opt.keys())for n, case in enumerate(case_names)]
    fig.legend(handles=case_handles,
            loc='lower center',
            ncols=num_cases,
            bbox_to_anchor=(0.52, 0.005), )
    fig.suptitle(title, y = 0.99)
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
        for n, case in enumerate(case_opt):
            correction = mld[n]**exp
            ax.plot(w_rms[:, n] * (correction), 
                    z_nd[:, n], color = case_opt[case]['color'])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'$\hat{{h}}_{{ML}}^{{{exp:.2f}}}$', )
        ax.set_xlabel(rf"$w_{{rms}}/\sqrt{{\text{{g}} \text{{r}}_{{j}}}} \cdot \hat{{h}}_{{ML}}^{{{exp:.2f}}}$", )

    for ax, exp in zip(axes[1, :], exponents):
        for n, case in enumerate(case_opt):
            correction = mld[n]**exp
            ax.plot(b_center[:, n] *(correction), 
                    z_nd[:, n], color = case_opt[case]['color'])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'$\hat{{h}}_{{ML}}^{{{exp:.2f}}}$', )
        ax.set_xlabel(rf"$b_{{\text{{centerline}}}}/g \cdot \hat{{h}}_{{ML}}^{{{exp:.2f}}}$", )

    for ax, exp in zip(axes[2, :], exponents):
        for n, case in enumerate(case_opt):
            correction = mld[n]**exp
            ax.plot(bw[:, n] * (correction), 
                    z_nd[:, n], color = case_opt[case]['color'])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'$\hat{{h}}_{{ML}}^{{{exp:.2f}}}$', )
        ax.set_xlabel(rf"$(\langle b'w'\rangle_{{\text{{xy}}}}/\sqrt{{\text{{g}}^3 \text{{r}}_{{j}}}})\cdot \hat{{h}}_{{ML}}^{{{exp:.2f}}}$", )

    for ax, exp in zip(axes[3, :], exponents):
        for n, case in enumerate(case_opt):
            correction = mld[n]**exp
            ax.plot(rp[:, n] * (correction), 
                    z_nd[:, n], color = case_opt[case]['color'])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'$\hat{{h}}_{{ML}}^{{{exp:.2f}}}$', )
        ax.set_xlabel(rf"(r/r$_{{j}})\cdot \hat{{h}}_{{ML}}^{{{exp:.2f}}}$", )

    for ax, exp in zip(axes[4, :], exponents):
        for n, case in enumerate(case_opt):
            correction = mld[n]**exp
            ax.plot(T[:, n] * (correction), 
                    z_nd[:, n], color = case_opt[case]['color'])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'$\hat{{h}}_{{ML}}^{{{exp:.2f}}}$', )
        ax.set_xlabel(rf"$(\text{{T'}}_{{\text{{centerline}}}}\alpha)\cdot \hat{{h}}_{{ML}}^{{{exp:.2f}}}$", )

    for ax, exp in zip(axes[5, :], exponents):
        for n, case in enumerate(case_opt):
            correction = mld[n]**exp
            ax.plot(S[:, n] * (correction), 
                    z_nd[:, n], color = case_opt[case]['color'])
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
def plot_combo_exponents(case_opt, title, file_name, fig_folder, w_rms, b_center, bw, rp, T, S, z_nd, vars_exps, Ri_g, Fr, mld, z_str = rf"(z-h$_{{ML}}$)h$_{{ML}}^{{1/3}}$/L$_N^{{4/3}}$"):
    NDs = [rf"Ri$_g^", rf"Fr$^", rf"$\hat{{h}}_{{ML}}^"] 
    NDs_filtered = [[("" if str(Fraction(x).limit_denominator()) == '0' 
                else NDs[j] + "{"+str(Fraction(x).limit_denominator())+"}$")
                for j, x in enumerate(row)] for row in vars_exps]
    vars_str = [''.join(row) for row in NDs_filtered]
    case_names = list(case_opt.keys())
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
    case_handles = [Line2D([0], [0], color = case_opt[case]['color'], linestyle=case_opt[case]['linestyle'], label=case_opt.keys())for n, case in enumerate(case_names)]
    fig.legend(handles=case_handles,
            loc='lower center',
            ncols=n_col,
            bbox_to_anchor=(0.52, 0.005), )
    fig.suptitle(title, y = 0.99)
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

    for n, case in enumerate(case_opt):
        axes[0, 0].plot(w_rms[:, n] * mld[n]**vars_exps[0, 2] * Ri_g[n]**vars_exps[0, 0] * Fr[n]**vars_exps[0, 1], 
                z_nd[:, n], color = case_opt[case]['color'])
    axes[0, 0].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
    axes[0, 0].set_xlabel(rf"$w_{{rms}}/\sqrt{{\text{{g r}}_{{j}}}}\cdot$ {vars_str[0]}", )

    for n, case in enumerate(case_opt):
        axes[0, 1].plot(b_center[:, n] * mld[n]**vars_exps[1, 2] * Ri_g[n]**vars_exps[1, 0] * Fr[n]**vars_exps[1, 1], 
                z_nd[:, n], color = case_opt[case]['color'])
    axes[0, 1].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
    axes[0, 1].set_xlabel(rf"$b_{{\text{{centerline}}}}/g \cdot$ {vars_str[1]}", )

    for n, case in enumerate(case_opt):
        axes[0, 2].plot(bw[:, n] * mld[n]**vars_exps[2, 2] * Ri_g[n]**vars_exps[2, 0] * Fr[n]**vars_exps[2, 1], 
                z_nd[:, n], color = case_opt[case]['color'])
    axes[0, 2].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
    axes[0, 2].set_xlabel(rf"$(\langle b'w'\rangle_{{xy}}/\sqrt{{\text{{g}}^3 \text{{r}}_{{j}}}})\cdot$ {vars_str[2]}", )

    for n, case in enumerate(case_opt):
        axes[0, 3].plot(rp[:, n] * mld[n]**vars_exps[3, 2] * Ri_g[n]**vars_exps[3, 0] * Fr[n]**vars_exps[3, 1], 
                z_nd[:, n], color = case_opt[case]['color'])
    axes[0, 3].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
    axes[0, 3].set_xlabel(rf"(r/r$_{{j}})\cdot$ {vars_str[3]}", )

    for n, case in enumerate(case_opt):
        axes[0, 4].plot(T[:, n] * mld[n]**vars_exps[4, 2] * Ri_g[n]**vars_exps[4, 0] * Fr[n]**vars_exps[4, 1],
                z_nd[:, n], color = case_opt[case]['color'])
    axes[0, 4].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
    axes[0, 4].set_xlabel(rf"$(\text{{T'}}_{{\text{{centerline}}}}\alpha)\cdot$ {vars_str[4]}", )

    for n, case in enumerate(case_opt):
        axes[0, 5].plot(S[:, n] * mld[n]**vars_exps[5, 2] * Ri_g[n]**vars_exps[5, 0] * Fr[n]**vars_exps[5, 1], 
                z_nd[:, n], color = case_opt[case]['color'])
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
def plot_r_at_depth_in_time(case_opt, fig_folder, time, r, tol, neutral, r_max, z_max, lz, best_fit, fit_exp, ND = False, log_auto = True):
    case_names = list(case_opt.keys())
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
                for n, case in enumerate(case_names):
                    ax.plot(td, z_max[row][:, n], color = case_opt[case]['color'], linestyle='--', linewidth=width)
                    ax.plot(td, neutral[row][:, n], color = case_opt[case]['color'])
                ax.plot(td, z_max[row][:, 0], color = case_opt['color'][0], linestyle='--', label=r'max radius', linewidth=width)
                ax.plot(td, neutral[row][:, 0], color = case_opt['color'][0], label=r'neutral')
                ax.set_ylabel(z_label)
                ax.legend(loc='upper right', handlelength=0.9)
                ax.set_ylim(lz)
            else:
                if log_auto:
                    ax.set_xscale('log')
                    ax.set_yscale('log')
                ax.plot(td, r_max[row][:, i - 1], color = case_opt['color'][i-1], linestyle='--')
                ax.plot(td, r[row][:, i-1], color = case_opt['color'][i-1])
                ax.plot(td, best_fit[0][row][:, i-1], color = case_opt['color'][i-1], linestyle=':', linewidth=width, label = rf"t$^{{{fit_exp[row][0][i-1, 0]:.2f}}}-{{{fit_exp[row][0][i-1, 1]:.2f}}}$") 
                ax.plot(td, best_fit[1][row][:, i-1], color = case_opt['color'][i-1], linestyle='-.', linewidth=width, label = rf"t$^{{{fit_exp[row][1][i-1, 0]:.2f}}}-{{{fit_exp[row][1][i-1, 1]:.2f}}}$") 
                ax.set_ylim(rmin, rmax)
                ax.set_ylabel(y_label)
                ax.legend(loc='lower right', handlelength=0.9)
            if row == nrows - 1:
                ax.set_xlabel(x_label)
            elif row == 0:
                ax.set_title(f"Contour = {tol[i-1]:.2e} ")
        x_center = 0.5
        y_pos = (nrows - row - 0.01*(nrows - row)) / nrows  # approximate vertical center of row
        fig.text(x_center, y_pos[row], ha='center', va='center', fontweight='bold', transform=fig.transFigure)
    # --- Save Frame ---
    file_name = 'ND_log_tracer_radius.svg' if ND else 'log_tracer_radius.svg'
    frame_path = os.path.join(outdir, file_name)
    plt.savefig(frame_path, bbox_inches='tight')
    plt.close(fig)
