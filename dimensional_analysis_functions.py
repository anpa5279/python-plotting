import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_rig_exponents(color_opt, time, it, fig_folder, w_rms, b_center, bw, rp, T, S, z_nd, zf_nd, Ri_g, case_names, exponents = [-0.5, -1/3, -0.25, 0.0, 0.25, 1/3, 0.5]):
    num_cases = len(case_names)
    outdir = os.path.join(fig_folder, "ND strat influence")
    os.makedirs(outdir, exist_ok=True)
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
    td = time[it] / 3600 / 24
    fig.suptitle(f'{td:.2f} days', fontsize = 20, y = 0.99)
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
            ax.plot(w_rms[:, i] / (correction), 
                    zf_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(f'Ri_g^{exp:.2f}', fontsize = 16)
        ax.set_xlabel(r'$w_{{rms}}/\sqrt{\text{g l}_{j}} \cdot Ri_g^{{{exp}}}$', fontsize = 16)
    #axes[0, 0].legend()

    for ax, exp in zip(axes[1, :], exponents):
        for i in range(num_cases):
            correction = Ri_g[i]**exp
            ax.plot(b_center[:, i] / (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(f'Ri_g^{exp:.2f}', fontsize = 16)
        ax.set_xlabel(r'$b_{centerline}/g \cdot Ri_g^{{{exp}}}$', fontsize = 16)
    #axes[1, 0].legend()

    for ax, exp in zip(axes[2, :], exponents):
        for i in range(num_cases):
            correction = Ri_g[i]**exp
            ax.plot(bw[:, i] / (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(f'Ri_g^{exp:.2f}', fontsize = 16)
        ax.set_xlabel(r"$(\langle b'w'\rangle_{xy}/\sqrt{\text{g}^3 \text{l}_{j}})\cdot Ri_g^{{{exp}}}$", fontsize = 16)
    #axes[2, 0].legend()

    for ax, exp in zip(axes[3, :], exponents):
        for i in range(num_cases):
            correction = Ri_g[i]**exp
            ax.plot(rp[:, i] / (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(f'Ri_g^{exp:.2f}', fontsize = 16)
        ax.set_xlabel(r"(r/l$_{j})\cdot Ri_g^{{{exp}}}$", fontsize = 16)
    #axes[3, 0].legend()

    for ax, exp in zip(axes[4, :], exponents):
        for i in range(num_cases):
            correction = Ri_g[i]**exp
            ax.plot(T[:, i] / (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(f'Ri_g^{exp:.2f}', fontsize = 16)
        ax.set_xlabel(r"$(\text{T'}_{\text{centerline}}\alpha)\cdot Ri_g^{{{exp}}}$", fontsize = 16)
    #axes[4, 0].legend()

    for ax, exp in zip(axes[5, :], exponents):
        for i in range(num_cases):
            correction = Ri_g[i]**exp
            ax.plot(S[:, i] / (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(f'Ri_g^{exp:.2f}', fontsize = 16)
        ax.set_xlabel(r"($\langle$C$\rangle_{\text{xy}} \beta)\cdot Ri_g^{{{exp}}}$", fontsize = 16)
    #axes[5, 0].legend()

    plt.tight_layout()

    # --- Save Frame ---
    frame_path = os.path.join(outdir, f"{exponents}_{it:04d}.png")
    plt.savefig(frame_path)
    plt.close(fig)
    print(f"Time step {it + 1} captured: {frame_path}")
    return outdir # return the directory where frames are saved for video creation


def plot_Fr_exponents(color_opt, time, it, fig_folder, w_rms, b_center, bw, rp, T, S, z_nd, zf_nd, Fr, case_names, exponents = [-0.5, -1/3, -0.25, 0.0, 0.25, 1/3, 0.5]):
    num_cases = len(case_names)
    outdir = os.path.join(fig_folder, "ND flux influence")
    os.makedirs(outdir, exist_ok=True)
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
    td = time[it] / 3600 / 24
    fig.suptitle(f'{td:.2f} days', fontsize = 20, y = 0.99)
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
            ax.plot(w_rms[:, i] / (correction), 
                    zf_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(f'Fr^{exp:.2f}', fontsize = 16)
        ax.set_xlabel(r'$w_{{rms}}/\sqrt{\text{g l}_{j}} \cdot Fr^{{{exp}}}$', fontsize = 16)
    #axes[0, 0].legend()

    for ax, exp in zip(axes[1, :], exponents):
        for i in range(num_cases):
            correction = Fr[i]**exp
            ax.plot(b_center[:, i] / (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(f'Fr^{exp:.2f}', fontsize = 16)
        ax.set_xlabel(r'$b_{centerline}/g \cdot Fr^{{{exp}}}$', fontsize = 16)
    #axes[1, 0].legend()

    for ax, exp in zip(axes[2, :], exponents):
        for i in range(num_cases):
            correction = Fr[i]**exp
            ax.plot(bw[:, i] / (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(f'Fr^{exp:.2f}', fontsize = 16)
        ax.set_xlabel(r"$(\langle b'w'\rangle_{xy}/\sqrt{\text{g}^3 \text{l}_{j}})\cdot Fr^{{{exp}}}$", fontsize = 16)
    #axes[2, 0].legend()

    for ax, exp in zip(axes[3, :], exponents):
        for i in range(num_cases):
            correction = Fr[i]**exp
            ax.plot(rp[:, i] / (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(f'Fr^{exp:.2f}', fontsize = 16)
        ax.set_xlabel(r"(r/l$_{j})\cdot Fr^{{{exp}}}$", fontsize = 16)
    #axes[3, 0].legend()

    for ax, exp in zip(axes[4, :], exponents):
        for i in range(num_cases):
            correction = Fr[i]**exp
            ax.plot(T[:, i] / (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(f'Fr^{exp:.2f}', fontsize = 16)
        ax.set_xlabel(r"$(\text{T'}_{\text{centerline}}\alpha)\cdot Fr^{{{exp}}}$", fontsize = 16)
    #axes[4, 0].legend()

    for ax, exp in zip(axes[5, :], exponents):
        for i in range(num_cases):
            correction = Fr[i]**exp
            ax.plot(S[:, i] / (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(f'Fr^{exp:.2f}', fontsize = 16)
        ax.set_xlabel(r"($\langle$C$\rangle_{\text{xy}} \beta)\cdot Fr^{{{exp}}}$", fontsize = 16)
    #axes[5, 0].legend()

    plt.tight_layout()

    # --- Save Frame ---
    frame_path = os.path.join(outdir, f"{exponents}_{it:04d}.png")
    plt.savefig(frame_path)
    plt.close(fig)
    print(f"Time step {it + 1} captured: {frame_path}")
    return outdir # return the directory where frames are saved for video creation


def plot_mld_exponents(color_opt, time, it, fig_folder, w_rms, b_center, bw, rp, T, S, z_nd, zf_nd, mld, case_names, exponents = [-0.5, -1/3, -0.25, 0.0, 0.25, 1/3, 0.5]):
    num_cases = len(case_names)
    outdir = os.path.join(fig_folder, "ND MLD influence")
    os.makedirs(outdir, exist_ok=True)
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
    td = time[it] / 3600 / 24
    fig.suptitle(f'{td:.2f} days', fontsize = 20, y = 0.99)
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
            ax.plot(w_rms[:, i] / (correction), 
                    zf_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(f'mld^{exp:.2f}', fontsize = 16)
        ax.set_xlabel(r'$w_{{rms}}/\sqrt{\text{g l}_{j}} \cdot mld^{{{exp}}}$', fontsize = 16)
    #axes[0, 0].legend()

    for ax, exp in zip(axes[1, :], exponents):
        for i in range(num_cases):
            correction = mld[i]**exp
            ax.plot(b_center[:, i] / (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(f'mld^{exp:.2f}', fontsize = 16)
        ax.set_xlabel(r'$b_{centerline}/g \cdot mld^{{{exp}}}$', fontsize = 16)
    #axes[1, 0].legend()

    for ax, exp in zip(axes[2, :], exponents):
        for i in range(num_cases):
            correction = mld[i]**exp
            ax.plot(bw[:, i] / (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(f'mld^{exp:.2f}', fontsize = 16)
        ax.set_xlabel(r"$(\langle b'w'\rangle_{xy}/\sqrt{\text{g}^3 \text{l}_{j}})\cdot mld^{{{exp}}}$", fontsize = 16)
    #axes[2, 0].legend()

    for ax, exp in zip(axes[3, :], exponents):
        for i in range(num_cases):
            correction = mld[i]**exp
            ax.plot(rp[:, i] / (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(f'mld^{exp:.2f}', fontsize = 16)
        ax.set_xlabel(r"(r/l$_{j})\cdot mld^{{{exp}}}$", fontsize = 16)
    #axes[3, 0].legend()

    for ax, exp in zip(axes[4, :], exponents):
        for i in range(num_cases):
            correction = mld[i]**exp
            ax.plot(T[:, i] / (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(f'mld^{exp:.2f}', fontsize = 16)
        ax.set_xlabel(r"$(\text{T'}_{\text{centerline}}\alpha)\cdot mld^{{{exp}}}$", fontsize = 16)
    #axes[4, 0].legend()

    for ax, exp in zip(axes[5, :], exponents):
        for i in range(num_cases):
            correction = mld[i]**exp
            ax.plot(S[:, i] / (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(f'mld^{exp:.2f}', fontsize = 16)
        ax.set_xlabel(r"($\langle$C$\rangle_{\text{xy}} \beta)\cdot mld^{{{exp}}}$", fontsize = 16)
    #axes[5, 0].legend()

    plt.tight_layout()

    # --- Save Frame ---
    frame_path = os.path.join(outdir, f"{exponents}_{it:04d}.png")
    plt.savefig(frame_path)
    plt.close(fig)
    print(f"Time step {it + 1} captured: {frame_path}")
    return outdir # return the directory where frames are saved for video creation

