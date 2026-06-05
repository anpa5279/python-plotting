import os
import numpy as np
import math
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

from plotting_general import plot_format, comparison_plot_opt
from diagnostics import comparison_info
from reader import OceananigansData
# flags for plotting
area_scaling = True
neg_tracer = False
tracer_integral = True

salinity = True
# Set up folder and simulation parameters
universal_folder = '/glade/derecho/scratch/apauls/outputs/version109/max-MLD/horizontal-domain/'
#'/Users/annapauls/Documents/TESLa /Simulations/Oceananigans/dense plume/salinity and temperature/no noise circle inlet/version109/default/horizontal domain/'
variations = 'horizontal resolution'
cases_info = comparison_info(variations, universal_folder = universal_folder)
num_cases = cases_info['num_cases']
case_names = cases_info['case_names']
readers = []
for name in cases_info["folder_names"]:
    folder = os.path.join(universal_folder, name)
    readers.append(OceananigansData(folder, salinity = salinity))

if area_scaling:
    def area_scale(r, dx):
        kmin = (np.floor(-r/dx + 0.5)).astype(int)
        kmax = (np.floor(r/dx - 0.5)).astype(int)
        x = np.arange(kmin, kmax + 1) * dx + dx/2
        y = x
        X, Y = np.meshgrid(x, y)
        dist_squared = X**2 + Y**2
        return np.sum(dist_squared <= (r)**2)
    rp = 5.0 #m
    area = np.pi*rp**2
    factor = []
rho0 = 1026 # kg/m^3
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

S_int = []
dS_intdt = []
t = []
if tracer_integral:
    dmdt = []
    S_mass = []
if neg_tracer:
    S_sum = []
    neg_avg = []
    S_neg_sum = []
    S_min = []
    percents = []
    dSdt = []

# Load data from files
for n, reader in enumerate(readers):
    t.append(reader.time / 3600 / 24)
    # load tracer
    if "/glade" in universal_folder:
        domain = math.prod(reader.nx)
        vol = math.prod(reader.lx)
        dims = (1, 2, 3)
        S = np.array(reader.lazy_field('S'))
        # volume integral of S value in domain
        S_int.append(np.mean(S, axis = dims)*vol)
    else:
        domain = nx[0, n]//2*nx[2, n]
        vol = np.pi*(lx[0, n]/2)**2*lx[2, n]
        dims = (0, 1)
        S = reader.load_binning_var('S')
        r_bins = reader.loading_bin_radius()
        # volume integral of S value in domain
        S_int.append(np.mean(S*r_bins[:, None, None], axis = dims)*vol)
    dS_intdt.append(np.gradient(S_int[n], t[n]))
    if tracer_integral:
        S_mass.append(S_int[n]*rho0)
        dmdt.append(np.gradient(S_mass[n], t[n]))
    if neg_tracer:
        S_neg = S
        S_neg[S>=0] = None
        # sum of S values in domain
        S_sum.append(np.sum(S, axis = dims))
        dSdt.append(np.gradient(S_sum[n], t[n]))
        # minimum S value in domain
        S_min.append(np.min(S, axis = dims))
        # negative number of negative values appearing in domain 
        neg_avg.append(np.nanmean(S_neg, axis = dims))
        S_neg_count = np.sum(S<0, axis = dims)
        S_neg_sum.append(np.nansum(S_neg, axis = dims))
        percents.append(np.nansum(S_neg, axis = dims)/domain*100)
    if area_scaling:
        Nr = area_scale(rp, reader.dx[0])
        grid_area = Nr*reader.dx[0]*reader.dx[1]
        factor.append(area/grid_area)
        S_int[n] = S_int[n]*factor[n]
        dS_intdt[n] = dS_intdt[n]*factor[n]
        if tracer_integral:
            S_mass[n] = S_int[n]*rho0
            dmdt[n] = np.gradient(S_mass[n], t[n])
        print(f"case: {case_names[n]}\nNr: {Nr}\ngrid area: {grid_area} vs circle area: {area}")
        print(f"\ttime: {t[n]}")
        print(f"\tS_int after scaling:\n {S_int[n]}")
        print(f"\tdS_int/dt before scaling:\n {dS_intdt[n]/factor[n]}")
        print(f"\tdS_int/dt after scaling:\n {dS_intdt[n]}\n")
        if neg_tracer:
            neg_avg[n] = neg_avg[n]*factor[n]
            S_neg_sum[n] = S_neg_sum[n]*factor[n]
            S_sum[n] = S_sum[n]*factor[n]
            dSdt[n] = dSdt[n]*factor[n]

color_opt, line_opt = comparison_plot_opt(num_cases)
plot_format()
if tracer_integral:
    scale = [1, 0.02]
    gridspec_kw={'height_ratios': scale}
    fig, ax = plt.subplots(2, 4, figsize=(16, 5), dpi = 300, gridspec_kw = gridspec_kw, sharex = True)
    for a in ax[-1, :]:
            a.remove()
    ax = ax.ravel()
    plt.subplots_adjust(top=0.9)
    case_handles = [Line2D([0], [0], color=color_opt[i], linestyle='solid', label=case_names[i]) for i in range(num_cases)]
    fig.legend(handles=case_handles,
            loc='lower center',
            ncol=num_cases,
            bbox_to_anchor=(0.52, 0.005), fontsize = 12)
    if area_scaling:
        fig.suptitle(f"Tracer statistics with area scaling (r = {rp}m)")

    ax[0].set_title('Volume Integral of S', fontsize = 12)
    ax[0].set_xlabel('Time (days)', fontsize = 12)
    ax[0].set_ylabel(r'[m$^3$g/kg]', fontsize = 12)

    ax[1].set_title(r'dS$_{vol}$/dt', fontsize = 12)
    ax[1].set_xlabel('Time (days)', fontsize = 12)
    ax[1].set_ylabel(r'[m$^3$g/kg/days]', fontsize = 12)

    ax[2].set_title('Mass of S', fontsize = 12)
    ax[2].set_xlabel('Time (days)', fontsize = 12)
    ax[2].set_ylabel('[g]', fontsize = 12)

    ax[3].set_title(r'dm$_S$/dt', fontsize = 12)
    ax[3].set_xlabel('Time (days)', fontsize = 12)
    ax[3].set_ylabel('[g/days]', fontsize = 12)

    for n in range(num_cases):
        ax[0].plot(t[n], S_int[n], color = color_opt[n], label=case_names[n])
        ax[1].plot(t[n], dS_intdt[n], color = color_opt[n], label=case_names[n])
        ax[2].plot(t[n], S_mass[n], color = color_opt[n], label=case_names[n])
        ax[3].plot(t[n], dmdt[n], color = color_opt[n], label=case_names[n])

    for a in ax[:8]:
        if a.get_yscale() != 'log':
            a.ticklabel_format(axis='y', style='sci', scilimits=(0, 3))
    fig.tight_layout(pad=1.5)

    outdir = os.path.join(universal_folder, 'callback comparisons')
    os.makedirs(outdir, exist_ok=True)
    if "/glade" in universal_folder:
        variations += ' with fields'
    else:
        variations += ' with binned data'
    if area_scaling:
        variations += f' and area scaling'
    plt.savefig(os.path.join(outdir, variations + ' comparisons.svg'))

if neg_tracer:
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
    if area_scaling:
        fig.suptitle(f"Tracer statistics with area scaling (r = {rp}m)")

    ax[0].set_title(r'-S$_{avg}$', fontsize = 12)
    ax[0].set_ylabel('[g/kg]', fontsize = 12)

    ax[1].set_title('Maximum magnitude of -S values', fontsize = 12)
    ax[1].set_yscale('log')
    ax[1].set_ylabel('[g/kg]', fontsize = 12)

    ax[2].set_title('Sum of -S in domain', fontsize = 12)
    ax[2].set_ylabel('[g/kg]', fontsize = 12)

    ax[3].set_title('Percent of -S values', fontsize = 12)
    ax[3].set_ylabel('% of domain', fontsize = 12)

    ax[0].set_title(r'S$_{avg}$', fontsize = 12)
    ax[0].set_xlabel('Time (days)', fontsize = 12)
    ax[0].set_ylabel('[g/kg]', fontsize = 12)

    ax[1].set_title(r'dS$_{avg}$/dt', fontsize = 12)
    ax[1].set_xlabel('Time (days)', fontsize = 12)
    ax[1].set_ylabel('[g/kg/days]', fontsize = 12)

    ax[2].set_title('Sum of S in domain', fontsize = 12)
    ax[2].set_xlabel('Time (days)', fontsize = 12)
    ax[2].set_ylabel('[g/kg]', fontsize = 12)

    ax[3].set_title(r'dS$_{sum}$/dt', fontsize = 12)
    ax[3].set_xlabel('Time (days)', fontsize = 12)
    ax[3].set_ylabel('[g/kg/days]', fontsize = 12)

    for n in range(num_cases):
        ax[0].plot(t[n], np.abs(neg_avg[n]), color = color_opt[n], label=case_names[n])
        ax[1].plot(t[n], np.abs(S_min[n]), color = color_opt[n], label=case_names[n])
        ax[3].plot(t[n], percents[n], color = color_opt[n], label=case_names[n])
        ax[2].plot(t[n], S_neg_sum[n], color = color_opt[n], label=case_names[n])
        ax[0].plot(t[n], S_avg[n], color = color_opt[n], label=case_names[n])
        ax[1].plot(t[n], dS_intdt[n], color = color_opt[n], label=case_names[n])
        ax[2].plot(t[n], S_sum[n], color = color_opt[n], label=case_names[n])
        ax[3].plot(t[n], dSdt[n], color = color_opt[n], label=case_names[n])

    for a in ax[:8]:
        if a.get_yscale() != 'log':
            a.ticklabel_format(axis='y', style='sci', scilimits=(0, 3))
    fig.tight_layout(pad=1.5)

    outdir = os.path.join(universal_folder, 'callback comparisons')
    os.makedirs(outdir, exist_ok=True)
    if "/glade" in universal_folder:
        variations += ' with fields'
    else:
        variations += ' with binned data'
    if area_scaling:
        variations += f' and area scaling'
    plt.savefig(os.path.join(outdir, variations + ' neg_tracer.svg'))