import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# flags for plotting
with_mld = True

# ── Paths ─────────────────────────────────────────────────────────────────────
output_folder = '/Users/annapauls/Documents/TESLa /Simulations/Oceananigans/dense plume/salinity and temperature/literature review/'
os.makedirs(output_folder, exist_ok=True)
name_mod = ''
# ── Paper data ─────────────────────────────────────────────────────────────────
rp    = 5.0 
g     = 9.80665
alpha = 2e-4
# proposed cases
h_ml = np.array([1.0, 3.2, 10.0, 32.0, 10.0, 10.0, 10.0, 10.0, 3.2, 10.0, 32.0])*rp
Ln = np.array([1.0, 1.0, 1.0, 1.0, 0.1, 10.0, 1e2, 1e3, 10, 10, 10])*rp
N = (g * alpha * 0.01)**0.5 * np.ones(len(h_ml))
F0 = (N)**3 * Ln**4
# 2016 experiment
r_exp = 0.4572/100.0
rho_b = 1.0572*1000
rho_t = 1.0492*1000
rho_j = 0.9972*1000
drho1 = 0.012
N_exp = np.array([39.79, 66.25]) #59.44, 87.62, 
Q = np.array([4.4, 15])*1e-6 #4.4, 4.4, 
F_exp = Q*g*(rho_b-rho_j)/rho_b
hml_exp = np.array([7.7, 1.2])/100
# 2017 LES
F_17 = np.array([94.65, 91.77, 129.63])
N_17 = 0.007 * np.ones(len(F_17))
hml_17 = np.ones(len(N_17)) * 160.0 
# 2026 preprint
h_nd = np.array([0.02, 0.06, 0.11, 0.21, 0.53, 1.06, 2.11, 3.17, 4.23, 5.29, 7.93, 8.89, 15.8, 29.7, 50.0, 88.9, 167, 281])
hml = 500.0*np.ones(len(h_nd))
"""
    # personal cases
    cases_info = comparison_info('all')
    beta  = 7.8e-4
    F0_current    = -((rp**2) * np.pi) * g * beta * cases_info['F_s']
    N_current     = np.sqrt(g * alpha * cases_info['dTdz'])
        {
            'title': 'Current Cases',
            'type': 'Implicit LES',
            'r0':  rp * np.ones(cases_info['num_cases']),
            'F0':  F0_current,
            'N':   N_current,
            'hml': cases_info['mld'],
            'Ln':  (F0_current / N_current**3)**(1/4),
        },
"""
papers = [
    {
        'title': 'Camassa, et al., 2016',
        'type': 'Experiment',
        'r0':  r_exp*np.ones(len(N_exp)),
        'F0':  F_exp,
        'N':   N_exp,
        'hml': hml_exp,
        'Ln':  (F_exp/N_exp**3)**(1/4),
    },
    {
        'title': 'Ezhova, et al., 2017',
        'type': 'LES w/ Smagorinsky',
        'r0':  np.array([5, 10, 10]),
        'F0':  F_17,
        'N':   N_17,
        'hml': 160.0 * np.ones(3),
        'Ln':  (F_17 / N_17**3)**(1/4),
    },
    {
        'title': 'Powell, et al., 2024',
        'type': 'LES w/ AMD',
        'r0':  0.005,
        'F0':  3.96e-07,
        'N':   1.0,
        'hml': 0.2,
        'Ln':  0.025,
    },
    {
        'title': 'Powell, et al., 2025',
        'type': 'LES w/ AMD',
        'r0':  (0.005) * np.ones(3),
        'F0':  (5e-6)  * np.ones(3),
        'N':   np.array([1.0, 10.0, 100.0]),
        'hml': 0.2 * np.ones(3),
        'Ln':  np.array([0.0473, 0.0084, 0.0015]),
    },

    {
        'title': 'Wang, et al., 2026',
        'type': 'Implicit LES',
        'r0':  27.42*np.ones(len(h_nd)),
        'F0':  1e-5,
        'N':   5e-5,
        'hml': hml,
        'Ln':  hml/h_nd,
    },
    {
        'title': 'Proposed Cases',
        'type': 'Implicit LES',
        'r0':  rp * np.ones(len(h_ml)),
        'F0':  F0,
        'N':   N,
        'hml': h_ml,
        'Ln':  Ln,
    },
]

# ── Color ──────────────────────────────────────────────────
sim_colors  = ['#4363d8', '#42d4f4', '#3cb44b', '#469990', '#911eb4', '#000075', '#c0c000', '#dcbeff', '#008080', '#aaffc3']
exp_colors = ['#f58231', "#ff0000", '#ffe119', "#7F2424", "#ea8989", '#f032e6', "#f0326b", "#ffad72", "#ffe772", "#AE6A2B"]
MARKERS = ['o', 's', '^', 'D', 'P', 'v', 'X', 'h', '<']

if with_mld:
    # ── marker assignment ──────────────────────────────────────────────────
    sim_count = 0
    exp_count = 0
    paper_style = {}
    for i, paper in enumerate(papers):
        if i > len(MARKERS) - 1:
            option = i - len(MARKERS)
        else:
            option = i
        if paper['type'] != 'Experiment':
            paper_style[paper['title']] = {
                'color':   sim_colors[sim_count],
                'marker':  MARKERS[option],
                'size':    55,
                'zorder':  3,
                'lw_edge': 0.4,
            }
            sim_count += 1
        else:
            paper_style[paper['title']] = {
                'color':   exp_colors[exp_count],
                'marker':  MARKERS[option],
                'size':    55,
                'zorder':  3,
                'lw_edge': 0.4,
            }
            exp_count += 1
    # personal cases get special styling
    paper_style['Proposed Cases'].update({
        'color': 'black', 'size': 150, 'zorder': 6, 'lw_edge': 0.6, 'marker': '*',
    })

    # ── Figure ─────────────────────────────────────────────────────────────────────
    scale = [1, 0.25]
    gridspec_kw = {'height_ratios': scale}
    fig, axes_grid = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw=gridspec_kw)

    # Remove the dummy bottom row axes used for legend space
    axes_grid[-1].remove()
    ax = axes_grid[0]
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(left=0.13, right=0.97, top=0.92, bottom=0.30)

    legend_handles = []

    for paper in papers:
        title = paper['title']
        style = paper_style[title]

        r0  = np.atleast_1d(np.asarray(paper['r0'],  dtype=float))
        hml = np.atleast_1d(np.asarray(paper['hml'], dtype=float))
        Ln  = np.atleast_1d(np.asarray(paper['Ln'],  dtype=float))

        x = Ln  / r0    # Ln / r0
        y = hml / r0    # hml / r0

        ax.scatter(
            x, y,
            color=style['color'], marker=style['marker'],
            s=style['size'], edgecolors='white',
            linewidths=style['lw_edge'], zorder=style['zorder'],
        )

        legend_handles.append(mlines.Line2D(
            [], [],
            marker=style['marker'], color='none',
            markerfacecolor=style['color'],
            markeredgecolor='white', markeredgewidth=0.5,
            markersize=9 if title == 'Proposed Cases' else 7,
            label=title,
        ))

    # ── Axis formatting ────────────────────────────────────────────────────────────
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$L_n / r_0$', )
    ax.set_ylabel(r'$h_{ml} / r_0$', )
    ax.tick_params(labelsize=10)
    ax.grid(True, which='both', linestyle='--', alpha=0.35, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.suptitle('Literature Review', )

    # ── Legend ─────────────────────────────────────────────────────────────────────
    fig.legend(
        handles=legend_handles,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.0),
        ncol=2,
        frameon=True, framealpha=0.92, edgecolor='#cccccc',
        handletextpad=0.5, borderpad=0.7,
        title='Studies', title_fontsize=12,
    )

    # ── Save ───────────────────────────────────────────────────────────────────────
    out_path = os.path.join(output_folder, name_mod +'mld_vs_r_lit_review_plot.svg')
    fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
    print(f'Saved -> {out_path}')

    # ── Figure ─────────────────────────────────────────────────────────────────────
    scale = [1, 0.25]
    gridspec_kw = {'height_ratios': scale}
    fig, axes_grid = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw=gridspec_kw)

    # Remove the dummy bottom row axes used for legend space
    axes_grid[-1].remove()
    ax = axes_grid[0]
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(left=0.13, right=0.97, top=0.92, bottom=0.30)

    for paper in papers:
        title = paper['title']
        style = paper_style[title]

        r0  = np.atleast_1d(np.asarray(paper['r0'],  dtype=float))
        hml = np.atleast_1d(np.asarray(paper['hml'], dtype=float))
        Ln  = np.atleast_1d(np.asarray(paper['Ln'],  dtype=float))

        x = Ln  / r0    # Ln / r0
        y = hml / Ln    # hml / Ln

        ax.scatter(
            x, y,
            color=style['color'], marker=style['marker'],
            s=style['size'], edgecolors='white',
            linewidths=style['lw_edge'], zorder=style['zorder'],
        )
    # ── Axis formatting ────────────────────────────────────────────────────────────
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$L_n / r_0$', )
    ax.set_ylabel(r'$h_{ml} / L_n$', )
    ax.tick_params(labelsize=10)
    ax.grid(True, which='both', linestyle='--', alpha=0.35, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.suptitle('Literature Review', )

    # ── Legend ─────────────────────────────────────────────────────────────────────
    fig.legend(
        handles=legend_handles,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.0),
        ncol=2,
        frameon=True, framealpha=0.92, edgecolor='#cccccc',
        handletextpad=0.5, borderpad=0.7,
        title='Studies', title_fontsize=12,
    )

    # ── Save ───────────────────────────────────────────────────────────────────────
    out_path = os.path.join(output_folder, name_mod +'mld_vs_Ln_lit_review_plot.svg')
    fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
    print(f'Saved -> {out_path}')
else:
    # ── additional paper data ───────────────────────────────
    # 2025 paper
    r_25 = np.array([0.014, 0.014, 0.014, 0.014, 0.014, 0.02, 0.02, 0.028, 0.028, 0.028, 0.028, 0.028, 0.028, 0.028, 0.028, 0.028, 0.028, 0.028])
    N_25 = 1.63e-3 * np.ones(len(r_25))
    F_25 = np.array([2.1e-3, 7.4e-4, 7.4e-5, 3e-4, 3.7e-5, 2.1e-3, 1.5e-3, 4.1e-3, 7.1e-4, 2.1e-3, 2.1e-3, 8.3e-3, 3e-4, 3e-3, 1.1e-5, 3.2e-4, 3.3e-3, 1.6e-4])

    no_mld_papers = [
        {
            'title': 'Mirajkar and Balasubramanian, 2017',
            'type': 'Experiment',
            'r0':  6.35e-3*np.ones(4),
            'F0':  np.array([1.67e-6, 5.24e-6, 1.16e-5, 2.02e-5]),
            'N':   np.array([0.37, 0.65, 0.89, 1.17]),
            'Ln':  (np.array([1.67e-6, 5.24e-6, 1.16e-5, 2.02e-5]) / np.array([0.37, 0.65, 0.89, 1.17])**3)**(1/4),
        },
        {
            'title': 'Mirajkar, et al., 2019',
            'type': 'Experiment',
            'r0':  6.35e-3,
            'F0':  1.5e-6,
            'N':   0.4,
            'Ln':  (1.5e-6 / 0.4**3)**(1/4),
        },
        {
            'title': 'Kumar, et al., 2022',
            'type': 'RANS',
            'r0':  6.35e-3*np.ones(4),
            'F0':  np.array([1.5e-6, 1.5e-6, 1.5e-6, 3.0e-6]),
            'N':   np.array([0.2, 0.4, 0.7, 0.4]),
            'Ln':  (np.array([1.5e-6, 1.5e-6, 1.5e-6, 3.0e-6]) / np.array([0.2, 0.4, 0.7, 0.4])**3)**(1/4),
        },
        {
            'title': 'Mukherjee, et al., 2022',
            'type': 'Experiment',
            'r0':  6.25e-3*np.ones(2),
            'F0':  np.array([1.84, 3.46])*1e-6,
            'N':   np.array([0.4, 0.6]),
            'Ln':  (np.array([1.84, 3.46])*1e-6 / np.array([0.4, 0.6])**3)**(1/4),
        },
        {
            'title': 'Mirajkar, et al., 2023',
            'type': 'Experiment',
            'r0':  6.25e-3,
            'F0':  1.5e-6,
            'N':   0.4,
            'Ln':  (1.5e-6 / 0.4**3)**(1/4),
        },
        {
            'title': 'Lemaréchal, et al., 2025',
            'type': 'LES',
            'r0':  r_25,
            'F0':  F_25,
            'N':   N_25,
            'Ln':  (F_25 / N_25**3)**(1/4),
        },
        {
            'title': 'Proposed Cases',
            'type': 'Implicit LES',
            'r0':  rp,
            'F0':  papers[-1]['F0'][2],
            'N':   papers[-1]['N'][2],
            'Ln':  (papers[-1]['F0'][2] / papers[-1]['N'][2]**3)**(1/4),
        },
    ]

    # ── marker assignment ──────────────────────────────────────────────────
    sim_count = 0
    exp_count = 0
    marker_count = 0
    paper_style = {}
    for i, paper in enumerate(papers):
        if paper['type'] != 'Experiment':
            paper_style[paper['title']] = {
                'color':   sim_colors[sim_count],
                'marker':  MARKERS[marker_count],
                'size':    55,
                'zorder':  3,
                'lw_edge': 0.4,
            }
            sim_count += 1
        else:
            paper_style[paper['title']] = {
                'color':   exp_colors[exp_count],
                'marker':  MARKERS[marker_count],
                'size':    55,
                'zorder':  3,
                'lw_edge': 0.4,
            }
            exp_count += 1
        if marker_count > len(MARKERS) - 2:
            marker_count = 0
        else:
            marker_count += 1
    # personal cases get special styling
    paper_style['Proposed Cases'].update({
        'color': 'black', 'size': 150, 'zorder': 6, 'lw_edge': 0.6, 'marker': '*',
    })
    for i, paper in enumerate(no_mld_papers):
        if paper['type'] != 'Experiment':
            paper_style[paper['title']] = {
                'color':   sim_colors[sim_count],
                'marker':  MARKERS[marker_count],
                'size':    55,
                'zorder':  3,
                'lw_edge': 0.4,
            }
            sim_count += 1
        else:
            paper_style[paper['title']] = {
                'color':   exp_colors[exp_count],
                'marker':  MARKERS[marker_count],
                'size':    55,
                'zorder':  3,
                'lw_edge': 0.4,
            }
            exp_count += 1
        if marker_count > len(MARKERS) - 2:
            marker_count = 0
        else:
            marker_count += 1
    # personal cases get special styling
    paper_style['Proposed Cases'].update({
        'color': 'black', 'size': 150, 'zorder': 6, 'lw_edge': 0.6, 'marker': '*',
    })
    # ── Figure ─────────────────────────────────────────────────────────────────────
    scale = [0.15, 0.9, 0.05]
    gridspec_kw = {'height_ratios': scale}
    fig, axes_grid = plt.subplots(3, 1, figsize=(12, 7), sharex = True,  gridspec_kw=gridspec_kw)

    # Remove the dummy bottom row axes used for legend space
    axes_grid[0].remove()
    ax0 = axes_grid[1]
    ax1 = axes_grid[2]
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(left=0.13, right=0.97, top=0.92, bottom=0.30)

    legend_handles = []

    for paper in papers:
        title = paper['title']
        style = paper_style[title]

        r0  = np.atleast_1d(np.asarray(paper['r0'],  dtype=float))
        hml = np.atleast_1d(np.asarray(paper['hml'], dtype=float))
        Ln  = np.atleast_1d(np.asarray(paper['Ln'],  dtype=float))

        x = Ln  / r0    # Ln / r0
        y = hml / r0    # hml / r0

        ax0.scatter(
            x, y,
            color=style['color'], marker=style['marker'],
            s=style['size'], edgecolors='white',
            linewidths=style['lw_edge'], zorder=style['zorder'],
        )
        if title != 'Proposed Cases':
            legend_handles.append(mlines.Line2D(
                [], [],
                marker=style['marker'], color='none',
                markerfacecolor=style['color'],
                markeredgecolor='white', markeredgewidth=0.5,
                markersize=7,
                label=title,
            ))


    # ── Axis formatting ────────────────────────────────────────────────────────────
    ax0.set_yscale('log')
    ax0.set_ylabel(r'$h_{ml} / r_0$', )
    ax0.tick_params(labelsize=10)
    ax0.grid(True, which='both', linestyle='--', alpha=0.35, zorder=0)
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)

    for paper in no_mld_papers:
        title = paper['title']
        style = paper_style[title]

        r0  = np.atleast_1d(np.asarray(paper['r0'],  dtype=float))
        Ln  = np.atleast_1d(np.asarray(paper['Ln'],  dtype=float))

        x = Ln  / r0    # Ln / r0

        ax1.scatter(
            x, np.zeros_like(x),
            color=style['color'], marker=style['marker'],
            s=style['size'], edgecolors='white',
            linewidths=style['lw_edge'], zorder=style['zorder'],
        )

        legend_handles.append(mlines.Line2D(
            [], [],
            marker=style['marker'], color='none',
            markerfacecolor=style['color'],
            markeredgecolor='white', markeredgewidth=0.5,
            markersize=10 if title == 'Proposed Cases' else 7,
            label=title,
        ))

    # ── Axis formatting ────────────────────────────────────────────────────────────
    ax1.yaxis.set_major_locator(ticker.NullLocator())
    ax1.xaxis.set_major_locator(ticker.LogLocator())
    ax1.set_ylim(-0.01, 0.01)
    ax1.set_xscale('log')
    ax1.set_xlabel(r'$L_n / r_0$', )
    ax1.set_yticks([-0.01], labels=[r'$h_{ml} / r_0$=0'], )
    ax1.tick_params(labelsize=10)
    ax1.grid(True, which='both', linestyle='--', alpha=0.35, zorder=0)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)

    # ── Legend ─────────────────────────────────────────────────────────────────────
    fig.legend(
        handles=legend_handles,
        loc='upper center',
        bbox_to_anchor=(0.525, 0.95),
        ncol=4,
        frameon=True, framealpha=0.92, edgecolor='#cccccc',
        handletextpad=0.5, borderpad=0.7,
        title='Studies', title_fontsize=12,
    )

    # ── Save ───────────────────────────────────────────────────────────────────────
    out_path = os.path.join(output_folder, 'lit_review_plot.svg')
    fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
    print(f'Saved -> {out_path}')

