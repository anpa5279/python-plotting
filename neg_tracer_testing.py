import os
import numpy as np
import math
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

from plotting_general import plot_format, comparison_plot_opt
from diagnostics import comparison_info
from reader import OceananigansData
salinity = True
# Set up folder and simulation parameters
universal_folder = '/glade/derecho/scratch/apauls/outputs/version109/horizontal-domain/'

variations = 'horizontal resolution'
cases_info = comparison_info(variations, universal_folder = universal_folder)
num_cases = cases_info['num_cases']
case_names = cases_info['case_names']
readers = []
for name in cases_info["folder_names"]:
    folder = os.path.join(universal_folder, name)
    readers.append(OceananigansData(folder, salinity = salinity))
    readers[-1].load_grid()
    readers[-1].load_time()

x = readers[0].x
y = readers[0].y
z = readers[0].z
nx = readers[0].nx
nt = readers[0].nt

domain = math.prod(nx)

percents = []
S_min = []
neg_avg = []
S_avg = []
S_sum = []
S_div = []
t = []

# Load data from files
for n, reader in enumerate(readers):
    t.append(reader.time / 3600 / 24)
    # load tracer
    S = reader.lazy_field('S')
    S_neg = S
    S_neg[S>=0] = 0
    # negative number of negative values appearing in domain 
    S_neg_sum = np.sum(S_neg, axis = (0, 1))
    percents.append(S_neg_sum/domain*100)
    neg_avg.append(np.mean(S_neg, axis = (0, 1)))
    # average S value in domain
    S_avg.append(np.mean(S, axis = (0, 1)))
    # sum of S values in domain
    S_sum.append(np.sum(S, axis = (0, 1)))
    # minimum negative S value in domain
    S_min.append(np.min(S, axis = (0, 1)))

color_opt, line_opt = comparison_plot_opt(num_cases)
plot_format()
scale = [1, 1, 0.02]
gridspec_kw={'height_ratios': scale}
fig, ax = plt.subplots(3, 3, figsize=(12, 8), dpi = 300, gridspec_kw = gridspec_kw)
for a in ax[-1, :]:
        a.remove()
ax = ax.ravel()
plt.subplots_adjust(top=0.9)
case_handles = [Line2D([0], [0], color=color_opt[i], linestyle='solid', label=case_names[i]) for i in range(num_cases)]
fig.legend(handles=case_handles,
        loc='lower center',
        ncol=num_cases,
        bbox_to_anchor=(0.52, 0.005), fontsize = 12)

ax[0].set_title('Percent of negative S values', fontsize = 12)
ax[0].set_xlabel('Time (days)', fontsize = 12)
ax[0].set_ylabel('% of domain', fontsize = 12)

ax[1].set_title('Maximum negative S value', fontsize = 12)
ax[1].set_xlabel('Time (days)', fontsize = 12)
ax[1].set_ylabel('[g/kg]', fontsize = 12)

ax[2].set_title('Average negative S value', fontsize = 12)
ax[2].set_xlabel('Time (days)', fontsize = 12)
ax[2].set_ylabel('[g/kg]', fontsize = 12)

ax[3].set_title('Average S value', fontsize = 12)
ax[3].set_xlabel('Time (days)', fontsize = 12)
ax[3].set_ylabel('[g/kg]', fontsize = 12)

ax[4].set_title('Sum of S in domain', fontsize = 12)
ax[4].set_xlabel('Time (days)', fontsize = 12)
ax[4].set_ylabel('[g/kg]', fontsize = 12)

for n in range(num_cases):
    ax[0].plot(t[n], percents[n], color = color_opt[n], label=case_names[n])
    ax[1].plot(t[n], S_min[n], color = color_opt[n], label=case_names[n])
    ax[2].plot(t[n], neg_avg[n], color = color_opt[n], label=case_names[n])
    ax[3].plot(t[n], S_avg[n], color = color_opt[n], label=case_names[n])
    ax[4].plot(t[n], S_sum[n], color = color_opt[n], label=case_names[n])

outdir = os.path.join(universal_folder, 'callback comparisons')
os.makedirs(outdir, exist_ok=True)
plt.savefig(os.path.join(outdir, variations + ' comparisons.svg'))