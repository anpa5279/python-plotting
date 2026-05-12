import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import openpyxl

# ── Paths ─────────────────────────────────────────────────────────────────────

file_archive = '/Users/annapauls/Library/CloudStorage/OneDrive-UCB-O365/CU-Boulder/TESLa/Carbon Sequestration/Simulations/Oceananigans/NBP/salinity and temperature/literature review.xlsx'
output_folder = '/Users/annapauls/Documents/TESLa /Simulations/Oceananigans/dense plume/salinity and temperature/literature review/'

os.makedirs(output_folder, exist_ok=True)

# ── Load sheet ────────────────────────────────────────────────────────────────
wb    = openpyxl.load_workbook(file_archive, data_only=True)
sheet = wb['paper with calculations']

all_rows = list(sheet.iter_rows(min_row=1, values_only=True))
headers  = all_rows[0]
data     = all_rows[1:]

n_rows = len(all_rows)
n_cols = sum(1 for h in headers if h is not None)  

# Pairs: [y_col, x_col] — last 6 non-None columns grouped as 3 pairs
PLOT_COLS = np.array([[-6, -5], [-4, -3], [-2, -1]]) + n_cols

COL_LABELS = [
    [r"$M^{3/4}F_0^{-1/2}\ [\mathrm{m}]$",
     r"$F_0^{1/4}N^{-3/4}\ [\mathrm{m}]$"],
    [r"$F_0/(\pi r_j^2)\sqrt{g/r_{j}}$",
     r"$N^2 r_{j}/g$"],
    [r"$h_{ml}/L_n\ [F_0^{-1/4}N^{3/4}h_{ml}]$",
     r"$\Gamma_0\ [Q_0^2 F_0 M^{-5/2}]$"],
]

USE_LOG = [[True, True], [True, True], [False, True]]

nplots = len(PLOT_COLS)

# Current-case rows: sheet rows
CURRENT_IDX = set(np.arange(n_rows - 10, n_rows - 1))

# ── Helper ────────────────────────────────────────────────────────────────────
def to_float(v):
    if v is None or isinstance(v, str):
        return np.nan
    try:
        f = float(v)
        return np.nan if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return np.nan

# ── Collect unique papers ─────────────────────────────────────────────────────
seen_papers = {}
for i, row in enumerate(data):
    if i in CURRENT_IDX:
        continue
    paper    = str(row[0]).strip() if row[0] else ''
    sim_type = str(row[2]).strip().lower() if row[3] else ''
    if paper and paper not in seen_papers:
        seen_papers[paper] = {'type': sim_type}

unique_papers = list(seen_papers.keys())

# ── Color / marker assignment ─────────────────────────────────────────────────
EXP_COLORS = ['#e6194b', '#f58231', '#c8850a', '#fabebe', '#a03020', '#e05c00', '#ffaa00']
SIM_COLORS = ['#4363d8', '#42d4f4', '#3cb44b', '#911eb4', '#469990', '#000075',
              '#c0c000', '#dcbeff', '#008080', '#aaffc3']
MARKERS    = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', '<', '>', 'p', 'H', '8', 'd']

exp_count = sim_count = 0
paper_style = {}
for paper in unique_papers:
    is_exp = seen_papers[paper]['type'] == 'experiment'
    if is_exp:
        color  = EXP_COLORS[exp_count % len(EXP_COLORS)]
        marker = MARKERS[exp_count % len(MARKERS)]
        exp_count += 1
    else:
        color  = SIM_COLORS[sim_count % len(SIM_COLORS)]
        marker = MARKERS[sim_count % len(MARKERS)]
        sim_count += 1
    paper_style[paper] = {'color': color, 'marker': marker, 'is_exp': is_exp}

def shorten(title, n=56):
    return title if len(title) <= n else title[:n].rstrip() + '…'

# ── Figure ────────────────────────────────────────────────────────────────────
scale = [1, 0.3]
gridspec_kw = {'height_ratios': scale}
fig, axes_grid = plt.subplots(2, nplots, figsize=(22, 7), gridspec_kw=gridspec_kw)

# Remove the dummy bottom row axes used for legend space
for a in axes_grid[-1, :]:
    a.remove()

# Only keep the top row
axes = axes_grid[0, :]

fig.patch.set_facecolor('white')

exp_papers_seen = []
sim_papers_seen = []
papers_seen_set = set()

for i, row in enumerate(data):
    paper  = str(row[0]).strip() if row[0] else ''
    is_cur = i in CURRENT_IDX

    if is_cur:
        color   = 'black'
        marker  = 'o'
        size    = 110
        zorder  = 6
        lw_edge = 0.6
    else:
        if paper not in paper_style:
            continue
        style   = paper_style[paper]
        color   = style['color']
        marker  = style['marker']
        size    = 55
        zorder  = 3
        lw_edge = 0.4

    any_plotted = False
    for ax_i, col_pair in enumerate(PLOT_COLS):
        y = to_float(row[col_pair[0]])
        x = to_float(row[col_pair[1]])
        if not np.isnan(y) and y > 0 and not np.isnan(x) and x > 0:
            axes[ax_i].scatter(
                x, y,
                color=color, marker=marker, s=size,
                edgecolors='white', linewidths=lw_edge,
                zorder=zorder,
            )
            any_plotted = True

    if any_plotted and not is_cur and paper not in papers_seen_set:
        papers_seen_set.add(paper)
        if paper_style[paper]['is_exp']:
            exp_papers_seen.append(paper)
        else:
            sim_papers_seen.append(paper)

# ── Axis formatting ───────────────────────────────────────────────────────────
for ax_i in range(nplots):
    ax = axes[ax_i]
    if USE_LOG[ax_i][0]:
        ax.set_yscale('log')
    if USE_LOG[ax_i][1]:
        ax.set_xscale('log')
    ax.set_ylabel(COL_LABELS[ax_i][0], fontsize=12, labelpad=4)
    ax.set_xlabel(COL_LABELS[ax_i][1], fontsize=12, labelpad=4)
    ax.tick_params(labelsize=7)
    ax.grid(True, which='both', linestyle='--', alpha=0.35, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ── Legend ────────────────────────────────────────────────────────────────────
def make_handle(style_dict, label):
    return mlines.Line2D(
        [], [],
        marker=style_dict['marker'], color='none',
        markerfacecolor=style_dict['color'],
        markeredgecolor='white', markeredgewidth=0.5,
        markersize=7, label=label,
    )

exp_handles = [make_handle(paper_style[p], shorten(p)) for p in exp_papers_seen]
sim_handles = [make_handle(paper_style[p], shorten(p)) for p in sim_papers_seen]

cur_handle = mlines.Line2D(
    [], [],
    marker='o', color='none',
    markerfacecolor='black',
    markeredgecolor='white', markeredgewidth=0.6,
    markersize=10, label='Current cases',
)
sim_handles.append(cur_handle)

leg_exp = fig.legend(
    handles=exp_handles,
    loc='lower left',
    bbox_to_anchor=(0.3, 0.0),
    ncol=1,
    fontsize=10,
    frameon=True, framealpha=0.92, edgecolor='#cccccc',
    handletextpad=0.5, borderpad=0.7,
    title='Experiments', title_fontsize = 12,
)

leg_sim = fig.legend(
    handles=sim_handles,
    loc='lower center',
    bbox_to_anchor=(0.60, 0.0),
    ncol=2,
    fontsize=10,
    frameon=True, framealpha=0.92, edgecolor='#cccccc',
    handletextpad=0.5, borderpad=0.7,
    title='Simulations', title_fontsize = 12,
)

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(output_folder, 'literature_review_plot.png')
fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
print(f'Saved -> {out_path}')