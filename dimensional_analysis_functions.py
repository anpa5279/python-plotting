import os
import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
from matplotlib.lines import Line2D

def plot_rig_exponents(color_opt, title, fig_folder, w_rms, b_center, bw, rp, T, S, z_nd, zf_nd, Ri_g, case_names, exponents = [-0.5, -1/3, -0.25, 0.0, 0.25, 1/3, 0.5]):
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
            bbox_to_anchor=(0.52, 0.005), fontsize = 16)
    fig.suptitle(title, fontsize = 20, y = 0.99)
    """
    axes[0, :] = ND rms velocity vs z_nd varied exponent of Ri_g
    axes[1, :] = ND centerline buoyancy vs z_nd varied exponent of Ri_g
    axes[2, :] = ND average buoyancy flux vs z_nd varied exponent of Ri_g
    axes[3, :] = ND radius of plume vs z_nd varied exponent of Ri_g
    axes[4, :] = ND perturbed temperature vs z_nd varied exponent of Ri_g
    axes[5, :] = ND average salinity vs z_nd varied exponent of Ri_g
    """
    for ax, exp in zip(axes[0, :], exponents):
        for i in range(num_cases):
            correction = Ri_g[i]**exp
            ax.plot(w_rms[:, i] * (correction), 
                    zf_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(f'Ri_g^{exp:.2f}', fontsize = 16)
        ax.set_xlabel(r"$w_{{rms}}/\sqrt{\text{g r}_{j}} \cdot Ri_g^{exp}$", fontsize = 16)
    #axes[0, 0].legend()

    for ax, exp in zip(axes[1, :], exponents):
        for i in range(num_cases):
            correction = Ri_g[i]**exp
            ax.plot(b_center[:, i] *(correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(f'Ri_g^{exp:.2f}', fontsize = 16)
        ax.set_xlabel(r"$b_{centerline}/g \cdot Ri_g^{exp}$", fontsize = 16)
    #axes[1, 0].legend()

    for ax, exp in zip(axes[2, :], exponents):
        for i in range(num_cases):
            correction = Ri_g[i]**exp
            ax.plot(bw[:, i] * (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(f'Ri_g^{exp:.2f}', fontsize = 16)
        ax.set_xlabel(r"$(\langle b'w'\rangle_{xy}/\sqrt{\text{g}^3 \text{r}_{j}})\cdot Ri_g^{exp}$", fontsize = 16)
    #axes[2, 0].legend()

    for ax, exp in zip(axes[3, :], exponents):
        for i in range(num_cases):
            correction = Ri_g[i]**exp
            ax.plot(rp[:, i] * (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(f'Ri_g^{exp:.2f}', fontsize = 16)
        ax.set_xlabel(r"(r/r$_{j})\cdot Ri_g^{exp}$", fontsize = 16)
    #axes[3, 0].legend()

    for ax, exp in zip(axes[4, :], exponents):
        for i in range(num_cases):
            correction = Ri_g[i]**exp
            ax.plot(T[:, i] * (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(f'Ri_g^{exp:.2f}', fontsize = 16)
        ax.set_xlabel(r"$(\text{T'}_{\text{centerline}}\alpha)\cdot Ri_g^{exp}$", fontsize = 16)
    #axes[4, 0].legend()

    for ax, exp in zip(axes[5, :], exponents):
        for i in range(num_cases):
            correction = Ri_g[i]**exp
            ax.plot(S[:, i] * (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(f'Ri_g^{exp:.2f}', fontsize = 16)
        ax.set_xlabel(r"($\langle$C$\rangle_{\text{xy}} \beta)\cdot Ri_g^{exp}$", fontsize = 16)
    #axes[5, 0].legend()

    plt.tight_layout()

    # --- Save Frame ---
    str_exp = list(map(str, exponents))
    str_exp = '_'.join(f"{x:.3g}" for x in exponents)
    frame_path = os.path.join(fig_folder, f"Ri_{title}_pow{str_exp}.png")
    plt.savefig(frame_path)
    plt.close(fig)
    
def plot_Fr_exponents(color_opt, title, fig_folder, w_rms, b_center, bw, rp, T, S, z_nd, zf_nd, Fr, case_names, exponents = [-0.5, -1/3, -0.25, 0.0, 0.25, 1/3, 0.5]):
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
            bbox_to_anchor=(0.52, 0.005), fontsize = 16)
    fig.suptitle(title, fontsize = 20, y = 0.99)
    """
    axes[0, :] = ND rms velocity vs z_nd varied exponent of Fr
    axes[1, :] = ND centerline buoyancy vs z_nd varied exponent of Fr
    axes[2, :] = ND average buoyancy flux vs z_nd varied exponent of Fr
    axes[3, :] = ND radius of plume vs z_nd varied exponent of Fr
    axes[4, :] = ND perturbed temperature vs z_nd varied exponent of Fr
    axes[5, :] = ND average salinity vs z_nd varied exponent of Fr
    """
    for ax, exp in zip(axes[0, :], exponents):
        for i in range(num_cases):
            correction = Fr[i]**exp
            ax.plot(w_rms[:, i] * (correction), 
                    zf_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(f'Fr^{exp:.2f}', fontsize = 16)
        ax.set_xlabel(r"$w_{{rms}}/\sqrt{\text{g r}_{j}} \cdot Fr^{exp}$", fontsize = 16)
    #axes[0, 0].legend()

    for ax, exp in zip(axes[1, :], exponents):
        for i in range(num_cases):
            correction = Fr[i]**exp
            ax.plot(b_center[:, i] * (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(f'Fr^{exp:.2f}', fontsize = 16)
        ax.set_xlabel(r"$b_{centerline}/g \cdot Fr^{exp}$", fontsize = 16)
    #axes[1, 0].legend()

    for ax, exp in zip(axes[2, :], exponents):
        for i in range(num_cases):
            correction = Fr[i]**exp
            ax.plot(bw[:, i] * (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(f'Fr^{exp:.2f}', fontsize = 16)
        ax.set_xlabel(r"$(\langle b'w'\rangle_{xy}/\sqrt{\text{g}^3 \text{r}_{j}})\cdot Fr^{exp}$", fontsize = 16)
    #axes[2, 0].legend()

    for ax, exp in zip(axes[3, :], exponents):
        for i in range(num_cases):
            correction = Fr[i]**exp
            ax.plot(rp[:, i] * (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(f'Fr^{exp:.2f}', fontsize = 16)
        ax.set_xlabel(r"(r/r$_{j})\cdot Fr^{exp}$", fontsize = 16)
    #axes[3, 0].legend()

    for ax, exp in zip(axes[4, :], exponents):
        for i in range(num_cases):
            correction = Fr[i]**exp
            ax.plot(T[:, i] * (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(f'Fr^{exp:.2f}', fontsize = 16)
        ax.set_xlabel(r"$(\text{T'}_{\text{centerline}}\alpha)\cdot Fr^{exp}$", fontsize = 16)
    #axes[4, 0].legend()

    for ax, exp in zip(axes[5, :], exponents):
        for i in range(num_cases):
            correction = Fr[i]**exp
            ax.plot(S[:, i] * (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(f'Fr^{exp:.2f}', fontsize = 16)
        ax.set_xlabel(r"($\langle$C$\rangle_{\text{xy}} \beta)\cdot Fr^{exp}$", fontsize = 16)
    #axes[5, 0].legend()

    plt.tight_layout()

    # --- Save Frame ---
    str_exp = '_'.join(f"{x:.3g}" for x in exponents)
    frame_path = os.path.join(fig_folder, f"Fr_{title}_pow{str_exp}.png")
    plt.savefig(frame_path)
    plt.close(fig)

def plot_mld_exponents(color_opt, title, fig_folder, w_rms, b_center, bw, rp, T, S, z_nd, zf_nd, mld, case_names, exponents = [-0.5, -1/3, -0.25, 0.0, 0.25, 1/3, 0.5]):
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
            bbox_to_anchor=(0.52, 0.005), fontsize = 16)
    fig.suptitle(title, fontsize = 20, y = 0.99)
    """
    axes[0, :] = ND rms velocity vs z_nd varied exponent of MLD
    axes[1, :] = ND centerline buoyancy vs z_nd varied exponent of MLD
    axes[2, :] = ND average buoyancy flux vs z_nd varied exponent of MLD
    axes[3, :] = ND radius of plume vs z_nd varied exponent of MLD
    axes[4, :] = ND perturbed temperature vs z_nd varied exponent of MLD
    axes[5, :] = ND average salinity vs z_nd varied exponent of MLD
    """
    for ax, exp in zip(axes[0, :], exponents):
        for i in range(num_cases):
            correction = mld[i]**exp
            ax.plot(w_rms[:, i] * (correction), 
                    zf_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(f'mld^{exp:.2f}', fontsize = 16)
        ax.set_xlabel(r"$w_{{rms}}/\sqrt{\text{g r}_{j}} \cdot mld^{exp}$", fontsize = 16)
    #axes[0, 0].legend()

    for ax, exp in zip(axes[1, :], exponents):
        for i in range(num_cases):
            correction = mld[i]**exp
            ax.plot(b_center[:, i] *(correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(f'mld^{exp:.2f}', fontsize = 16)
        ax.set_xlabel(r"$b_{centerline}/g \cdot mld^{exp}$", fontsize = 16)
    #axes[1, 0].legend()

    for ax, exp in zip(axes[2, :], exponents):
        for i in range(num_cases):
            correction = mld[i]**exp
            ax.plot(bw[:, i] * (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(f'mld^{exp:.2f}', fontsize = 16)
        ax.set_xlabel(r"$(\langle b'w'\rangle_{xy}/\sqrt{\text{g}^3 \text{r}_{j}})\cdot mld^{exp}$", fontsize = 16)
    #axes[2, 0].legend()

    for ax, exp in zip(axes[3, :], exponents):
        for i in range(num_cases):
            correction = mld[i]**exp
            ax.plot(rp[:, i] * (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(f'mld^{exp:.2f}', fontsize = 16)
        ax.set_xlabel(r"(r/r$_{j})\cdot mld^{exp}$", fontsize = 16)
    #axes[3, 0].legend()

    for ax, exp in zip(axes[4, :], exponents):
        for i in range(num_cases):
            correction = mld[i]**exp
            ax.plot(T[:, i] * (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(f'mld^{exp:.2f}', fontsize = 16)
        ax.set_xlabel(r"$(\text{T'}_{\text{centerline}}\alpha)\cdot mld^{exp}$", fontsize = 16)
    #axes[4, 0].legend()

    for ax, exp in zip(axes[5, :], exponents):
        for i in range(num_cases):
            correction = mld[i]**exp
            ax.plot(S[:, i] * (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(f'mld^{exp:.2f}', fontsize = 16)
        ax.set_xlabel(r"($\langle$C$\rangle_{\text{xy}} \beta)\cdot mld^{exp}$", fontsize = 16)
    #axes[5, 0].legend()

    plt.tight_layout()

    # --- Save Frame ---
    str_exp = '_'.join(f"{x:.3g}" for x in exponents)
    frame_path = os.path.join(fig_folder, f"MLD_{title}_pow{str_exp}.png")
    plt.savefig(frame_path)
    plt.close(fig)
    

def plot_combo_exponents(color_opt, title, fig_folder, w_rms, b_center, bw, rp, T, S, z_nd, zf_nd, vars_exps, Ri_g, Fr, mld, case_names):
    NDs = [rf"Ri$_g^", rf"Fr$^", rf"MLD$^"] 
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

    fig, axes = plt.subplots(2, len(NDs_filtered), figsize=(25, 7), sharey=True, gridspec_kw = gridspec_kw)
    plt.subplots_adjust(top=0.9)
    for a in axes[-1, :]:
        a.remove()
    case_handles = [Line2D([0], [0], color=color_opt[i], linestyle='solid', label=case_names[i])for i in range(num_cases)]
    fig.legend(handles=case_handles,
            loc='lower center',
            ncol=n_col,
            bbox_to_anchor=(0.52, 0.005), fontsize = 16)
    fig.suptitle(title, fontsize = 20, y = 0.99)
    """
    axes[0] = ND rms velocity vs z_nd varied exponent of all
    axes[1] = ND centerline buoyancy vs z_nd varied exponent of all
    axes[2] = ND average buoyancy flux vs z_nd varied exponent of all
    axes[3] = ND radius of plume vs z_nd varied exponent of all
    axes[4] = ND perturbed temperature vs z_nd varied exponent of all
    axes[5] = ND average salinity vs z_nd varied exponent of all
    """
    for i in range(num_cases):
        axes[0, 0].plot(w_rms[:, i] * mld[i]**vars_exps[0, 2] * Ri_g[i]**vars_exps[0, 0] * Fr[i]**vars_exps[0, 1], 
                zf_nd[:, i], color=color_opt[i])
    axes[0, 0].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
    axes[0, 0].set_xlabel(rf"$w_{{rms}}/\sqrt{{\text{{g r}}_{{j}}}}\cdot$ {vars_str[0]}", fontsize = 16)

    for i in range(num_cases):
        axes[0, 1].plot(b_center[:, i] * mld[i]**vars_exps[1, 2] * Ri_g[i]**vars_exps[1, 0] * Fr[i]**vars_exps[1, 1], 
                z_nd[:, i], color=color_opt[i])
    axes[0, 1].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
    axes[0, 1].set_xlabel(rf"$b_{{centerline}}/g \cdot$ {vars_str[1]}", fontsize = 16)

    for i in range(num_cases):
        axes[0, 2].plot(bw[:, i] * mld[i]**vars_exps[2, 2] * Ri_g[i]**vars_exps[2, 0] * Fr[i]**vars_exps[2, 1], 
                z_nd[:, i], color=color_opt[i])
    axes[0, 2].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
    axes[0, 2].set_xlabel(rf"$(\langle b'w'\rangle_{{xy}}/\sqrt{{\text{{g}}^3 \text{{r}}_{{j}}}})\cdot$ {vars_str[2]}", fontsize = 16)

    for i in range(num_cases):
        axes[0, 3].plot(rp[:, i] * mld[i]**vars_exps[3, 2] * Ri_g[i]**vars_exps[3, 0] * Fr[i]**vars_exps[3, 1], 
                z_nd[:, i], color=color_opt[i])
    axes[0, 3].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
    axes[0, 3].set_xlabel(rf"(r/r$_{{j}})\cdot$ {vars_str[3]}", fontsize = 16)

    for i in range(num_cases):
        axes[0, 4].plot(T[:, i] * mld[i]**vars_exps[4, 2] * Ri_g[i]**vars_exps[4, 0] * Fr[i]**vars_exps[4, 1],
                z_nd[:, i], color=color_opt[i])
    axes[0, 4].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
    axes[0, 4].set_xlabel(rf"$(\text{{T'}}_{{\text{{centerline}}}}\alpha)\cdot$ {vars_str[4]}", fontsize = 16)

    for i in range(num_cases):
        axes[0, 5].plot(S[:, i] * mld[i]**vars_exps[5, 2] * Ri_g[i]**vars_exps[5, 0] * Fr[i]**vars_exps[5, 1], 
                z_nd[:, i], color=color_opt[i])
    axes[0, 5].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
    axes[0, 5].set_xlabel(rf"($\langle$C$\rangle_{{\text{{xy}}}} \beta)\cdot$ {vars_str[5]}", fontsize = 16)


    plt.tight_layout()

    # --- Save Frame ---
    frame_path = os.path.join(fig_folder, f"{title}_combined.png")
    i = 0
    while True:
        i += 1
        frame_path = os.path.join(fig_folder, f"{title}_combined_{i}.png")
        if os.path.exists(frame_path):
            continue
        plt.savefig(frame_path)
        break
    plt.close(fig)
    

