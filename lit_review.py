import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import openpyxl

from diagnostics import comparison_info

# ── Paths ─────────────────────────────────────────────────────────────────────
output_folder = '/Users/annapauls/Documents/TESLa /Simulations/Oceananigans/dense plume/salinity and temperature/literature review/'
os.makedirs(output_folder, exist_ok=True)

# ── paper data ────────────────────────────────────────────────────────────────
cases_info = comparison_info('all')
rp = 5.0*np.ones(cases_info['num_cases'])
g = 9.80665  # gravity in m/s^2
beta = 7.8e-4
alpha = 2e-4
F0 = -((rp**2)*np.pi)*g*beta*cases_info['F_s']
N = np.sqrt(g*alpha*cases_info['dTdz'])

papers = [
    {
        'title': 'Ezhova, et al., 2017',
        'type': 'LES w/ Smagorinsky',
        'r0': np.array([5, 10, 10]),
        'F0': np.array([94.65, 91.77, 129.63]),
        'N': 0.34*np.ones(3),
        'hml': 160.0*np.ones(3),
        'Ln': np.array([7.00, 6.95, 7.57]), 
    },
    {
        'title': 'Powell, et al., 2024',
        'type': 'LES w/ AMD',
        'r0': 0.005,
        'F0': 3.96e-07,
        'N': 1.0,
        'hml': 0.2,
        'Ln': 0.025, 
    },
    {
        'title': 'Powell, et al., 2025',
        'type': 'LES w/ AMD',
        'r0': (0.005)*np.ones(3),
        'F0': (5e-6)*np.ones(3),
        'N': np.array([1.0, 10.0, 100.0]),
        'hml': 0.2*np.ones(3),
        'Ln': np.array([0.0473, 0.0084, 0.0015]), 
    },
    {
        'title': 'Wang, et al., 2026',
        'type': 'Implicit LES',
        'r0': 27.42,
        'F0': 1e-5,
        'N': 5e-5,
        'hml': 500,
        'Ln': 94.57, 
    },
    {
        'title': 'Current Cases',
        'type': 'Implicit LES',
        'r0': rp,
        'F0': F0,
        'N': N,
        'hml': cases_info['mld'],
        'Ln': (F0/N**3)**(1/4), 
    }
]

# ── Color / marker assignment ─────────────────────────────────────────────────
COLORS = ['#4363d8', '#42d4f4', '#3cb44b', '#911eb4', '#469990', '#000075',
              '#c0c000', '#dcbeff', '#008080', '#aaffc3']
MARKERS    = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', '<', '>', 'p', 'H', '8', 'd']

count = 0
paper_style = {}
for paper in papers:
    color  = COLORS[count % len(COLORS)]
    marker = MARKERS[count % len(MARKERS)]
    count += 1
    paper_style[paper['title']] = {'color': color, 'marker': marker}
paper_style[paper['title']] = {'color': 'black', 'marker': 'o'}

# ── Figure ────────────────────────────────────────────────────────────────────
scale = [1, 0.3]
gridspec_kw = {'height_ratios': scale}
fig, axes_grid = plt.subplots(1, 2, figsize=(12, 7), gridspec_kw=gridspec_kw)

# Remove the dummy bottom row axes used for legend space
for a in axes_grid[-1, :]:
    a.remove()
ax = axes_grid[0, :]
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

    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if not np.any(mask):
        continue

    ax.scatter(
        x[mask], y[mask],
        color=style['color'], marker=style['marker'],
        s=style['size'], edgecolors='white',
        linewidths=style['lw_edge'], zorder=style['zorder'],
    )

    legend_handles.append(mlines.Line2D(
        [], [],
        marker=style['marker'], color='none',
        markerfacecolor=style['color'],
        markeredgecolor='white', markeredgewidth=0.5,
        markersize=9 if title == 'Current Cases' else 7,
        label=title,
    ))



# ── Axis formatting ────────────────────────────────────────────────────────────
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$L_n / r_0$', fontsize=13)
ax.set_ylabel(r'$h_{ml} / r_0$', fontsize=13)
ax.tick_params(labelsize=10)
ax.grid(True, which='both', linestyle='--', alpha=0.35, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.suptitle('Literature Review', fontsize=13)

# ── Legend ─────────────────────────────────────────────────────────────────────
fig.legend(
    handles=legend_handles,
    loc='lower center',
    bbox_to_anchor=(0.5, 0.0),
    ncol=2,
    fontsize=9,
    frameon=True, framealpha=0.92, edgecolor='#cccccc',
    handletextpad=0.5, borderpad=0.7,
    title='Studies', title_fontsize=10,
)

# ── Save ───────────────────────────────────────────────────────────────────────
out_path = os.path.join(output_folder, 'mld_review_plot.svg')
fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
print(f'Saved -> {out_path}')