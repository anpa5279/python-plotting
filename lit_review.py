import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import openpyxl

from diagnostics import comparison_info

# ── Paths ─────────────────────────────────────────────────────────────────────
output_folder = '/Users/annapauls/Documents/TESLa /Simulations/Oceananigans/dense plume/salinity and temperature/literature review/'
os.makedirs(output_folder, exist_ok=True)

# ── Paper data ─────────────────────────────────────────────────────────────────
# personal cases
cases_info = comparison_info('all')
rp    = 5.0 
g     = 9.80665
beta  = 7.8e-4
alpha = 2e-4
F0_current    = -((rp**2) * np.pi) * g * beta * cases_info['F_s']
N_current     = np.sqrt(g * alpha * cases_info['dTdz'])
# proposed cases
h_ml = np.array([0.0, 1.0, 2.0, 5.0, 10.0, 5.0, 5.0, 5.0, 5.0])*rp
Ln = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 10.0, 1e2, 1e3])*rp
N = N_current[0] * np.ones(len(h_ml))
F0 = N**3 * Ln**4
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
# 2025 no hml paper
r_25 = np.array([0.014, 0.014, 0.014, 0.014, 0.014, 0.02, 0.02, 0.028, 0.028, 0.028, 0.028, 0.028, 0.028, 0.028, 0.028, 0.028, 0.028, 0.028])
N_25 = 1.63e-3 * np.ones(len(r_25))
F_25 = np.array([2.1e-3, 7.4e-4, 7.4e-5, 3e-4, 3.7e-5, 2.1e-3, 1.5e-3, 4.1e-3, 7.1e-4, 2.1e-3, 2.1e-3, 8.3e-3, 3e-4, 3e-3, 1.1e-5, 3.2e-4, 3.3e-3, 1.6e-4])
# 2026 preprint
h_nd = np.array([0.02, 0.06, 0.11, 0.21, 0.53, 1.06, 2.11, 3.17, 4.23, 5.29, 7.93, 8.89, 15.8, 29.7, 50.0, 88.9, 167, 281])
hml = 500.0*np.ones(len(h_nd))
"""
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
        'title': 'Lemaréchal, et al., 2025',
        'type': 'LES',
        'r0':  r_25,
        'F0':  F_25,
        'N':   N_25,
        'hml': np.zeros(len(r_25)), 
        'Ln':  (F_25 / N_25**3)**(1/4),
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
        'Ln':  (F0 / N**3)**(1/4),
    },
]

# ── Color / marker assignment ──────────────────────────────────────────────────
COLORS  = [ '#f58231', '#4363d8', '#42d4f4', '#3cb44b', '#911eb4', '#469990',
           '#000075', '#c0c000', '#dcbeff', '#008080', '#aaffc3']
MARKERS = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', '<']

paper_style = {}
for i, paper in enumerate(papers):
    paper_style[paper['title']] = {
        'color':   COLORS[i % len(COLORS)],
        'marker':  MARKERS[i % len(MARKERS)],
        'size':    55,
        'zorder':  3,
        'lw_edge': 0.4,
    }
# personal cases get special styling
"""
paper_style['Current Cases'].update({
    'color': '#000075', 'size': 90, 'zorder': 6, 'lw_edge': 0.6,
})
"""
paper_style['Proposed Cases'].update({
    'color': 'black', 'size': 150, 'zorder': 6, 'lw_edge': 0.6,
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
ax.set_xlabel(r'$L_n / r_0$', fontsize=12)
ax.set_ylabel(r'$h_{ml} / r_0$', fontsize=12)
ax.tick_params(labelsize=10)
ax.grid(True, which='both', linestyle='--', alpha=0.35, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.suptitle('Literature Review', fontsize=12)

# ── Legend ─────────────────────────────────────────────────────────────────────
fig.legend(
    handles=legend_handles,
    loc='lower center',
    bbox_to_anchor=(0.5, 0.0),
    ncol=2,
    fontsize=9,
    frameon=True, framealpha=0.92, edgecolor='#cccccc',
    handletextpad=0.5, borderpad=0.7,
    title='Studies', title_fontsize=12,
)

# ── Save ───────────────────────────────────────────────────────────────────────
out_path = os.path.join(output_folder, 'mld_review_plot.svg')
fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
print(f'Saved -> {out_path}')
