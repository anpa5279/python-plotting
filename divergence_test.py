import os
import numpy as np
import math
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

from plotting_functions import plot_format, comparison_plot_opt
from diagnostics import comparison_info
from interpolation import velocities_to_center
from reader import OceananigansData

# Set up folder and simulation parameters
universal_folder = '/Users/annapauls/Documents/TESLa /Simulations/Oceananigans/dense plume/salinity and temperature/no noise circle inlet'

cases_info = comparison_info('WENO')
num_cases = cases_info['num_cases']
case_names = cases_info['case_names']
readers = []
for name in cases_info["folder_names"]:
    folder = os.path.join(universal_folder, name)
    readers.append(OceananigansData(folder))
    readers[-1].load_grid()
    readers[-1].load_time()

x = readers[0].x
y = readers[0].y
z = readers[0].z
nx = readers[0].nx
nt = readers[0].nt

S_div_sum = np.empty((3, nt))
dSdt_sum = np.empty((3, nt))

for it in range(nt):
    for n, reader in enumerate(readers):
        S = reader.lazy_field('S', reader.t_save[it])
        u = reader.lazy_field('u', reader.t_save[it])
        v = reader.lazy_field('v', reader.t_save[it])
        w = reader.lazy_field('w', reader.t_save[it])
        u, v, w = velocities_to_center(u, v, w)
        # Load S at three levels
        S_prev = reader.lazy_field('S', reader.t_save[it - 1]) if it > 0 else None
        S_next = reader.lazy_field('S', reader.t_save[it + 1]) if it < nt - 1 else None

        if S_prev is not None and S_next is not None:
            dt = (reader.time[it + 1] - reader.time[it - 1])
            dSdt = (S_next - S_prev) / dt
        elif S_next is None:
            dt = (reader.time[it] - reader.time[it - 1])
            dSdt = (S - S_prev) / dt
        else:
            dSdt = np.zeros_like(S)

        div_uS = (np.gradient(u * S, x, axis=0) +
                np.gradient(v * S, y, axis=1) +
                np.gradient(w * S, z, axis=2))
        dSdt_sum[n, it] = np.sum(dSdt)
        S_div_sum[n, it] = np.sum(div_uS)

t = readers[0].time / 3600 / 24
color_opt, line_opt = comparison_plot_opt(num_cases)
plot_format()
scale = [1, 0.02]
gridspec_kw={'height_ratios': scale}
fig, ax = plt.subplots(2, 3, figsize=(12, 4), dpi = 300, gridspec_kw = gridspec_kw)
for a in ax[-1, :]:
        a.remove()
ax = ax.ravel()
plt.subplots_adjust(top=0.9)
case_handles = [Line2D([0], [0], color=color_opt[i], linestyle='solid', label=case_names[i]) for i in range(num_cases)]
fig.legend(handles=case_handles,
        loc='lower center',
        ncol=num_cases,
        bbox_to_anchor=(0.52, 0.005), fontsize = 12)

ax[0].set_title(r'dSu$_i$/dx$_i$', fontsize = 12)
ax[0].set_xlabel('Time (days)', fontsize = 12)
ax[0].set_ylabel('[g/kg/s]', fontsize = 12)

ax[1].set_title('dS/dt', fontsize = 12)
ax[1].set_xlabel('Time (days)', fontsize = 12)
ax[1].set_ylabel('[g/kg/s]', fontsize = 12)

ax[2].set_title('Residual', fontsize = 12)
ax[2].set_xlabel('Time (days)', fontsize = 12)
ax[2].set_ylabel('[g/kg/s]', fontsize = 12)

for n in range(num_cases):
    ax[0].plot(t, dSdt_sum[n, :], color = color_opt[n], label=case_names[n])
    ax[1].plot(t, S_div_sum[n, :], color = color_opt[n], label=case_names[n])
    ax[2].plot(t, dSdt_sum[n, :] + S_div_sum[n, :], color = color_opt[n], label=case_names[n])

outdir = os.path.join(universal_folder, 'callback comparisons')
os.makedirs(outdir, exist_ok=True)
plt.savefig(os.path.join(outdir, 'divergence testing dSdtupdate.svg'))