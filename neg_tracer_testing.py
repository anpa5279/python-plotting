import os
import numpy as np
import math
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

from plotting_functions import plot_format, comparison_plot_opt
from reader import OceananigansData
# Set up folder and simulation parameters
universal_folder = '/Users/annapauls/Documents/TESLa /Simulations/Oceananigans/dense plume/salinity and temperature/no noise circle inlet'
folder1 = os.path.join(universal_folder, 'S0 = 0.1 dTdz = 0.01 MLD = 60')
folder2 = os.path.join(universal_folder, 'S0 = 0.1 dTdz = 0.01 MLD = 60 WENO mod')
folder3 = os.path.join(universal_folder, 'S0 = 0.1 dTdz = 0.01 MLD = 60 WENO mod callback')
case_names = ['Original', 'WENO mod', 'WENO mod with 0 forced callback']
num_cases = len(case_names)

reader1 = OceananigansData(folder1)
reader2 = OceananigansData(folder2)
reader3 = OceananigansData(folder3)
reader1.load_grid()
reader2.load_grid()
reader3.load_grid()
reader1.load_time()
reader2.load_time()
reader3.load_time()

domain = math.prod(reader1.nx)
percents = np.empty((3, len(reader1.t_save)))
max_neg = np.empty((3, len(reader1.t_save)))
neg_avg = np.empty((3, len(reader1.t_save)))
S_avg = np.empty((3, len(reader1.t_save)))

nt = len(reader1.t_save)
for it in range(nt):
    # Load data from files
    S = reader1.lazy_field('S', reader1.t_save[it])
    S_sum = np.sum(S<0)
    S_avg[0, it] = np.mean(S)
    max_neg[0, it] = np.min(S)
    if S_sum>0:
        percents[0, it] = S_sum/domain*100
        neg_avg[0, it] = np.mean(S[S<0])
    else:
        percents[0, it] = 0
        neg_avg[0, it] = 0

    Smod = reader2.lazy_field('S', reader2.t_save[it])
    Smod_sum = np.sum(Smod<0)
    S_avg[1, it] = np.mean(Smod)
    max_neg[1, it] = np.min(Smod)
    if Smod_sum>0:
        percents[1, it] = Smod_sum/domain*100
        neg_avg[1, it] = np.mean(Smod[Smod<0])
    else:
        percents[1, it] = 0
        neg_avg[1, it] = 0
    
    Scall = reader3.lazy_field('S', reader3.t_save[it])
    Scall_sum = np.sum(Scall<0)
    S_avg[2, it] = np.mean(Scall)
    max_neg[2, it] = np.min(Scall)
    if Scall_sum>0:
        percents[2, it] = Scall_sum/domain*100
        neg_avg[2, it] = np.mean(Scall[Scall<0])
    else:
        percents[2, it] = 0
        neg_avg[2, it] = 0

t = reader1.time / 3600 / 24
color_opt, line_opt = comparison_plot_opt(num_cases)
plot_format()
scale = [1, 0.02]
gridspec_kw={'height_ratios': scale}
fig, ax = plt.subplots(2, 4, figsize=(12, 4), dpi = 300, gridspec_kw = gridspec_kw)
for a in ax[-1, :]:
        a.remove()
ax = ax.ravel()
plt.subplots_adjust(top=0.9)
case_handles = [Line2D([0], [0], color=color_opt[i], linestyle='solid', label=case_names[i])for i in range(num_cases)]
fig.legend(handles=case_handles,
        loc='lower center',
        ncol=num_cases,
        bbox_to_anchor=(0.52, 0.005), fontsize = 12)
ax[0].plot(t, percents[0, :], color = color_opt[0], label=case_names[0])
ax[0].plot(t, percents[1, :], color = color_opt[1], label=case_names[1])
ax[0].plot(t, percents[2, :], color = color_opt[2], label=case_names[2])
ax[0].set_title('Percent of negative S values', fontsize = 12)
ax[0].set_xlabel('Time (days)', fontsize = 12)
ax[0].set_ylabel('% of domain', fontsize = 12)

ax[1].plot(t, max_neg[0, :], color = color_opt[0], label=case_names[0])
ax[1].plot(t, max_neg[1, :], color = color_opt[1], label=case_names[1])
ax[1].plot(t, max_neg[2, :], color = color_opt[2], label=case_names[2])
ax[1].set_title('Maximum negative S value', fontsize = 12)
ax[1].set_xlabel('Time (days)', fontsize = 12)
ax[1].set_ylabel('Most negative S value', fontsize = 12)

ax[2].plot(t, neg_avg[0, :], color = color_opt[0], label=case_names[0])
ax[2].plot(t, neg_avg[1, :], color = color_opt[1], label=case_names[1])
ax[2].plot(t, neg_avg[2, :], color = color_opt[2], label=case_names[2])
ax[2].set_title('Average of negative S values', fontsize = 12)
ax[2].set_xlabel('Time (days)', fontsize = 12)
ax[2].set_ylabel('Average negative S value', fontsize = 12)

ax[3].plot(t, S_avg[0, :], color = color_opt[0], label=case_names[0])
ax[3].plot(t, S_avg[1, :], color = color_opt[1], label=case_names[1])
ax[3].plot(t, S_avg[2, :], color = color_opt[2], label=case_names[2])
ax[3].set_title('Average S value', fontsize = 12)
ax[3].set_xlabel('Time (days)', fontsize = 12)
ax[3].set_ylabel('Average S value', fontsize = 12)

outdir = os.path.join(universal_folder, 'callback comparisons')
os.makedirs(outdir, exist_ok=True)
plt.savefig(os.path.join(outdir, 'neg_tracer_comparison.svg'))