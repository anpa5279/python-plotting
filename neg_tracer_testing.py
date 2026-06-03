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
universal_folder = '/glade/derecho/scratch/apauls/outputs/version109/default/horizontal-domain'
#'/Users/annapauls/Documents/TESLa /Simulations/Oceananigans/dense plume/salinity and temperature/no noise circle inlet/version109/default/horizontal domain/'
#'/glade/derecho/scratch/apauls/outputs/version109/default/horizontal-domain'

variations = 'horizontal resolution'
cases_info = comparison_info(variations, universal_folder = universal_folder)
num_cases = cases_info['num_cases']
case_names = cases_info['case_names']
readers = []
for name in cases_info["folder_names"]:
    folder = os.path.join(universal_folder, name)
    readers.append(OceananigansData(folder, salinity = salinity))

# collecting model information for all cases
nx = np.empty((3, num_cases), dtype=object)
lx = np.empty((3, num_cases), dtype=object)
nt = np.empty(num_cases, dtype=int)
time  = []
grid_specs = False*np.ones(num_cases)
grid_specs[2] = True # flag for whether to plot grid specs in title
for i, reader in enumerate(readers):
    reader.load_time()
    reader.load_grid(grid_specs = grid_specs[i])
    time.append(reader.time)
    nx[:, i] = reader.nx
    lx[:, i] = reader.lx
    nt[i] = reader.nt

percents = []
S_min = []
neg_avg = []
S_neg_sum = []
S_avg = []
S_sum = []
S_div = []
dSdt = []
dSavgdt = []
t = []

# Load data from files
for n, reader in enumerate(readers):
    t.append(reader.time / 3600 / 24)
    # load tracer
    if "/glade" in universal_folder:
        domain = math.prod(reader.nx)
        S = np.array(reader.lazy_field('S'))
        # average S value in domain
        S_avg.append(np.mean(S, axis = (1, 2, 3)))
        # sum of S values in domain
        S_sum.append(np.sum(S, axis = (1, 2, 3)))
        # minimum -S value in domain
        S_min.append(np.min(S, axis = (1, 2, 3)))
    else:
        domain = nx[0, n]//2*nx[2, n]
        S = reader.load_binning_var('S')
        # average S value in domain
        S_avg.append(np.mean(S, axis = (0, 1)))
        # sum of S values in domain
        S_sum.append(np.sum(S, axis = (0, 1)))
        # minimum -S value in domain
        S_min.append(np.min(S, axis = (0, 1)))
    dSdt.append(np.gradient(S_sum[n], t[n]))
    dSavgdt.append(np.gradient(S_avg[n], t[n]))
    S_neg = S
    S_neg[S>=0] = None
    # negative number of negative values appearing in domain 
    if S_neg.all() == None:
        neg_avg.append(np.zeros(nt[n]))
        S_neg_sum.append(np.zeros(nt[n]))
        S_neg_count = np.zeros(nt[n])
    else:
        if "/glade" not in universal_folder:
            neg_avg.append(np.nanmean(S_neg, axis = (0, 1)))
            S_neg_count = np.sum(S<0, axis = (0, 1))
            S_neg_sum.append(np.nansum(S_neg, axis = (0, 1)))
        else:
            neg_avg.append(np.mean(S_neg, axis = (1, 2, 3)))
            S_neg_count = np.sum(S_neg, axis = (1, 2, 3))
            S_neg_sum.append(np.sum(S_neg, axis = (1, 2, 3)))
    percents.append(S_neg_count/domain*100)
    print(case_names[n], ' domain: ', domain)
    print(case_names[n], ' S_neg: ', S_neg)
print('negative average: ', neg_avg)
print('negative sum: ', S_neg_sum)
print('% negative: ', percents)

color_opt, line_opt = comparison_plot_opt(num_cases)
plot_format()
scale = [1, 1, 0.02]
gridspec_kw={'height_ratios': scale}
fig, ax = plt.subplots(3, 4, figsize=(16, 8), dpi = 300, gridspec_kw = gridspec_kw, sharex = True)
for a in ax[-1, :]:
        a.remove()
ax = ax.ravel()
plt.subplots_adjust(top=0.9)
case_handles = [Line2D([0], [0], color=color_opt[i], linestyle='solid', label=case_names[i]) for i in range(num_cases)]
fig.legend(handles=case_handles,
        loc='lower center',
        ncol=num_cases,
        bbox_to_anchor=(0.52, 0.005), fontsize = 12)


ax[0].set_title(r'-S$_{avg}$', fontsize = 12)
ax[0].set_yscale('log')
ax[0].set_xlabel('Time (days)', fontsize = 12)
ax[0].set_ylabel('[g/kg]', fontsize = 12)

ax[1].set_title('Maximum magnitude of -S values', fontsize = 12)
ax[1].set_yscale('log')
ax[1].set_xlabel('Time (days)', fontsize = 12)
ax[1].set_ylabel('[g/kg]', fontsize = 12)

ax[3].set_title('Percent of -S values', fontsize = 12)
ax[3].set_xlabel('Time (days)', fontsize = 12)
ax[3].set_ylabel('% of domain', fontsize = 12)

ax[2].set_title('Sum of -S in domain', fontsize = 12)
ax[2].set_xlabel('Time (days)', fontsize = 12)
ax[2].set_ylabel('[g/kg]', fontsize = 12)

ax[4].set_title(r'S$_{avg}$', fontsize = 12)
ax[4].set_xlabel('Time (days)', fontsize = 12)
ax[4].set_ylabel('[g/kg]', fontsize = 12)

ax[5].set_title(r'dS$_{avg}$/dt', fontsize = 12)
ax[5].set_xlabel('Time (days)', fontsize = 12)
ax[5].set_ylabel('[g/kg/days]', fontsize = 12)

ax[6].set_title('Sum of S in domain', fontsize = 12)
ax[6].set_xlabel('Time (days)', fontsize = 12)
ax[6].set_ylabel('[g/kg]', fontsize = 12)

ax[7].set_title(r'dS$_{sum}$/dt', fontsize = 12)
ax[7].set_xlabel('Time (days)', fontsize = 12)
ax[7].set_ylabel('[g/kg/days]', fontsize = 12)

for n in range(num_cases):
    ax[0].plot(t[n], np.abs(neg_avg[n]), color = color_opt[n], label=case_names[n])
    ax[1].plot(t[n], np.abs(S_min[n]), color = color_opt[n], label=case_names[n])
    ax[3].plot(t[n], percents[n], color = color_opt[n], label=case_names[n])
    ax[2].plot(t[n], S_neg_sum[n], color = color_opt[n], label=case_names[n])
    ax[4].plot(t[n], S_avg[n], color = color_opt[n], label=case_names[n])
    ax[5].plot(t[n], dSavgdt[n], color = color_opt[n], label=case_names[n])
    ax[6].plot(t[n], S_sum[n], color = color_opt[n], label=case_names[n])
    ax[7].plot(t[n], dSdt[n], color = color_opt[n], label=case_names[n])

for a in ax[:8]:
    if a.get_yscale() != 'log':
        a.ticklabel_format(axis='y', style='sci', scilimits=(0, 3))

outdir = os.path.join(universal_folder, 'callback comparisons')
os.makedirs(outdir, exist_ok=True)
if "/glade" in universal_folder:
    variations += ' with fields'
else:
    variations += ' with binned data'
plt.savefig(os.path.join(outdir, variations + ' comparisons.svg'))