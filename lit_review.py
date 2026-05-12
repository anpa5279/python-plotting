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
all_cols = list(sheet.iter_cols(min_col=1, values_only=True))

PLOT_COLS  = np.array([[-6, -5], [-4, -3], [-2, -1]]) + len(all_cols)
COL_LABELS = [
    [r"$M^{3/4}F_0^{-1/2} [m]$",
    r"$F_0^{1/4}N^{-3/4} [m]$"],
    [r"$F_0/(\pi r_j^2) \sqrt{g/h_{ml}}$",
    r"$N^2 h_{ml}/g$"],
    [r"$h_{ml}/L_n\ [F_0^{-1/4}N^{3/4}h_{ml}]$",
    r"$\Gamma_0\ [Q_0^2 F_0 M^{-5/2}]$"]
]
nplots =PLOT_COLS.size//2

# All columns span >4 orders of magnitude → log scale on all
USE_LOG = [[True, True], [True, True], [True, True]]

# my cases: 
CURRENT_IDX = set(range(len(all_rows)-9, len(all_rows)))

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
    sim_type = str(row[3]).strip().lower() if row[3] else ''
    if paper and paper not in seen_papers:
        seen_papers[paper] = {'type': sim_type}

unique_papers = list(seen_papers.keys())

# ── Color / marker assignment ─────────────────────────────────────────────────
EXP_COLORS = ['#e6194b', '#f58231', '#c8850a', '#fabebe', '#a03020', '#e05c00', '#ffaa00']
SIM_COLORS = ['#4363d8', '#42d4f4', '#3cb44b', '#911eb4', '#469990', '#000075',
              '#bfef45', '#dcbeff', '#008080', '#aaffc3']
MARKERS    = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', '<', '>', 'p', 'H', '8', 'd']

exp_count = sim_count = 0
paper_style = {}
for paper in unique_papers:
    is_exp = seen_papers[paper]['type'] == 'experiment'
    print(paper, ': \n', seen_papers[paper]['type'], '\tis_exp =', is_exp)
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

# ── Figure: 5 subplots in one row ─────────────────────────────────────────────
scale = [1, 0.3]
gridspec_kw={'height_ratios': scale}
fig, axes = plt.subplots(2, nplots, figsize=(22, 7), gridspec_kw = gridspec_kw)
for a in axes[-1, :]:
        a.remove()
axes = axes.ravel()
fig.patch.set_facecolor('white')
plt.subplots_adjust(top=0.91)

exp_papers_seen  = []
sim_papers_seen  = []
papers_seen_set  = set()

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
    for ax_i, col_i in enumerate(PLOT_COLS):
        y = to_float(row[col_i[0]])
        x = to_float(row[col_i[1]])
        if not np.isnan(y) and y > 0 and not np.isnan(x) and x > 0:          # log scale requires y > 0
            axes[ax_i].scatter(
                x, y,
                color=color, marker=marker, s=size,
                edgecolors='white', linewidths=lw_edge,
                zorder=zorder,
            )
            any_plotted = True

    if any_plotted and not is_cur and paper not in papers_seen_set:
        papers_seen_set.add(paper)
        print(paper)
        print(paper_style[paper]['is_exp'])
        if paper_style[paper]['is_exp']:
            exp_papers_seen.append(paper)
        else:
            sim_papers_seen.append(paper)

# ── Axis formatting ───────────────────────────────────────────────────────────
for ax_i in range(nplots):
    if USE_LOG[ax_i][0]:
        axes[ax_i].set_yscale('log')
    if USE_LOG[ax_i][1]:
        axes[ax_i].set_xscale('log')
    axes[ax_i].set_ylabel(COL_LABELS[ax_i][0], fontsize=9, labelpad=4)
    axes[ax_i].set_xlabel(COL_LABELS[ax_i][1], fontsize=9, labelpad=4)
    axes[ax_i].tick_params(labelsize=7)
    axes[ax_i].grid(True, which='both', linestyle='--', alpha=0.35, zorder=0)
    axes[ax_i].spines['top'].set_visible(False)
    axes[ax_i].spines['right'].set_visible(False)
    axes[ax_i].set_title(headers[PLOT_COLS[ax_i][0]
                                 ], fontsize=8.5, pad=5)

fig.suptitle('Literature Review',
             fontsize=12, y=0.97)

# ── Legend: two groups side by side ──────────────────────────────────────────
def make_handle(style_dict, label):
    return mlines.Line2D(
        [], [],
        marker=style_dict['marker'], color='none',
        markerfacecolor=style_dict['color'],
        markeredgecolor='white', markeredgewidth=0.5,
        markersize=7, label=label,
    )

exp_handles = []
for p in exp_papers_seen:
    exp_handles.append(make_handle(paper_style[p], shorten(p)))

sim_handles = []
for p in sim_papers_seen:
    sim_handles.append(make_handle(paper_style[p], shorten(p)))
cur_handle = mlines.Line2D(
    [], [],
    marker='o', color='none',
    markerfacecolor='black',
    markeredgecolor='white', markeredgewidth=0.6,
    markersize=10, label=r'$\bf{Current\ cases}$',
)
sim_handles.append(cur_handle)

# Place two separate legends side by side below the subplots
n_exp = len(exp_handles)
n_sim = len(sim_handles)

leg_exp = fig.legend(
    handles=exp_handles,
    loc='lower left',
    bbox_to_anchor=(0.1, 0.0),
    ncol=1,
    fontsize=7.2,
    frameon=True, framealpha=0.92, edgecolor='#cccccc',
    handletextpad=0.5, borderpad=0.7,
    title='Experiments', title_fontsize=8,
)

leg_sim = fig.legend(
    handles=sim_handles,
    loc='lower center',
    bbox_to_anchor=(0.6, 0.0),
    ncol=2,
    fontsize=7.2,
    frameon=True, framealpha=0.92, edgecolor='#cccccc',
    handletextpad=0.5, borderpad=0.7,
    title='Simulations', title_fontsize=8,
)

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(output_folder, 'literature_review_plot.png')
fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
print(f'Saved -> {out_path}')