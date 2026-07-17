import os
import numpy as np
import h5py
import math
import matplotlib.pyplot as plt
import itertools

from matplotlib.lines import Line2D

from plotting_general import plot_format, comparison_plot_opt
from diagnostics import comparison_info
from reader import OceananigansData
from interpolation import velocities_to_center
# flags for plotting
area_scaling = False
tracer_integral = False
mass_divergence = True
neg_tracer = False

salinity = True
# Set up folder and simulation parameters
universal_folder = '/Users/annapauls/Documents/TESLa /Simulations/Oceananigans/dense plume/salinity and temperature/version109/w BC testing/'
#'/Users/annapauls/Documents/TESLa /Simulations/Oceananigans/dense plume/salinity and temperature/no noise circle inlet/version109/default/horizontal domain/'
variations = 'w BC'
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
#grid_specs[2] = True # flag for whether to plot grid specs in title
for i, reader in enumerate(readers):
    reader.load_grid(grid_specs = grid_specs[i])
    time.append(reader.t)
    nx[:, i] = reader.nx
    lx[:, i] = reader.lx
    nt[i] = reader.nt
t = []

if tracer_integral or mass_divergence:
    dmdt = []
    S_mass = []
    if mass_divergence:
        div_east = []
        div_west = []
        div_north = []
        div_south = []
        div_top = []
        div_bottom = []
        div_faces = []
        div_vol = []
if neg_tracer:
    S_sum = []
    neg_avg = []
    S_neg_sum = []
    S_min = []
    percents = []
    dSdt = []

# Load data from files
for n, reader in enumerate(readers):
    t.append(reader.t / 3600 / 24)
    # load tracer
    if "/glade" in universal_folder:
        domain = math.prod(reader.nx)
        vol = math.prod(reader.lx)
        dims = (1, 2, 3)
        S = reader.lazy_field('S').compute()
        # volume integral of S value in domain
        S_int = np.mean(S, axis = dims)*vol*rho0
    else:
        file_path = os.path.join(reader.folder, 'binning_rtz.h5')
        with h5py.File(file_path, 'r') as f:
            S_int = f["S mass"][:]
            dmdt_reader = f["time gradient of S mass"][:]
    if tracer_integral or mass_divergence:
        S_mass.append(S_int)
        if "glade" in universal_folder:
            dmdt.append(np.gradient(S_mass[n], t[n]))
        else:
            dmdt.append(dmdt_reader)
        if mass_divergence:
            w = reader.lazy_field('w').compute()
            reader.load_equation_of_state()
            dwdz = np.gradient(w, reader.zf, axis=-1)

            dmw_top = np.sum(dwdz[:, :, :, 0].squeeze(), axis = (1, 2))
            dmw_bottom = -1*np.sum(dwdz[:, :, :, -1].squeeze(), axis = (1, 2))
    
            div_top.append(dmw_top)
            div_bottom.append(dmw_bottom)

            div_faces.append(div_top[-1] + div_bottom[-1]) 

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
        if tracer_integral:
            S_mass[n] = S_int*factor[n]
            dmdt[n] = np.gradient(S_mass[n], t[n])

        if neg_tracer:
            neg_avg[n] = neg_avg[n]*factor[n]
            S_neg_sum[n] = S_neg_sum[n]*factor[n]
            S_sum[n] = S_sum[n]*factor[n]
            dSdt[n] = dSdt[n]*factor[n]

color_opt, line_opt = comparison_plot_opt(num_cases)
plot_format()
if tracer_integral:
    scale = [1, 0.1]
    gridspec_kw={'height_ratios': scale}
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), dpi = 300, gridspec_kw = gridspec_kw)
    for a in axes[-1, :]:
            a.remove()
    axes = axes.ravel()
    plt.subplots_adjust(top=0.9)
    case_handles = [Line2D([0], [0], color=color_opt[i], linestyle='solid', label=case_names[i]) for i in range(num_cases)]
    leg_col = num_cases//2 if num_cases >= 4 else num_cases
    fig.legend(handles=case_handles,
            loc='lower center',
            ncol=leg_col,
            bbox_to_anchor=(0.5, 0.005), fontsize = 10)
    if area_scaling:
        fig.suptitle(f"Tracer statistics with area scaling (r = {rp}m)")
        mass_label = r'$\rho_{0}L_{x}L_{y}L_{z}\frac{\pi r_{p}^2}{N_{r}d_{x}d_{y}}\langle\text{C}\rangle_{\text{xyz}}$[g]'
        mass_rate_label = r'$\rho_{0}L_{x}L_{y}L_{z}\frac{\pi r_{p}^2}{N_{r}d_{x}d_{y}}\frac{\text{d}\langle\text{C}\rangle_{\text{xyz}}}{\text{dt}}$[g/days]'
    else:
        mass_label = r'$\rho_{0}L_{x}L_{y}L_{z}\langle\text{C}\rangle_{\text{xyz}}$[g]'
        mass_rate_label = r'$\rho_{0}L_{x}L_{y}L_{z}\frac{\text{d}\langle\text{C}\rangle_{\text{xyz}}}{\text{dt}}$[g/days]'

    axes[0].set_title('Mass', fontsize = 12)
    axes[0].set_xlabel('Time (days)', fontsize = 12)
    axes[0].set_ylabel(mass_label, fontsize = 12)

    axes[1].set_title(r'Temporal rate of Mass', fontsize = 12)
    axes[1].set_xlabel('Time (days)', fontsize = 12)
    axes[1].set_ylabel(mass_rate_label, fontsize = 12)
    dmdt_flat = list(itertools.chain.from_iterable(dmdt))
    mass_rate = [min(dmdt_flat), max(dmdt_flat)]
    axes[1].set_ylim(mass_rate[0]*0.9, mass_rate[1]*1.1)

    for n in range(num_cases):
        axes[0].plot(t[n], S_mass[n], color = color_opt[n], label=case_names[n])
        axes[1].plot(t[n], dmdt[n], color = color_opt[n], label=case_names[n])

    for a in axes[:4]:
        if a.get_yscale() != 'log':
            a.ticklabel_format(axis='y', style='sci', scilimits=(0, 3), useOffset=False)
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
if mass_divergence:
    scale = [1, 0.1]
    gridspec_kw={'height_ratios': scale}
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), dpi = 300, gridspec_kw = gridspec_kw)
    axes = axes.ravel()
    axes[-1].remove()
    #plt.subplots_adjust(top=0.9, right=0.8)
    case_handles = [Line2D([0], [0], color=color_opt[i], linestyle='solid', label=case_names[i]) for i in range(num_cases)]
    leg_col = num_cases//2 if num_cases >= 4 else num_cases
    fig.legend(handles=case_handles,
            loc='lower center',
            ncol=leg_col,
            bbox_to_anchor=(0.5, 0.005), fontsize = 10)

    axes[0].set_xlabel('Time (days)', fontsize = 12)
    #axes[0].set_ylabel(r'$\frac{\partial\text{m}}{\partial t} + \nabla\cdot$ ($\text{m}$u$_i$)', fontsize = 12)
    axes[0].set_ylabel(r'$\frac{\partial\text{w}}{\partial z}_{bottom} - \frac{\partial\text{w}}{\partial z}_{top}$', fontsize = 12)

    for n in range(num_cases):
        if n == 0: #, marker = 'o', markersize=3

            axes[0].plot(t[n], div_faces[n], color = color_opt[n], linestyle = 'solid', label=r'$\frac{\partial\text{w}}{\partial z}$')
            axes[0].plot(t[n], div_bottom[n], color = color_opt[n], linestyle = 'dashed', label=r'$\frac{\partial\text{w}}{\partial z}_{bottom}$')
            axes[0].plot(t[n], div_top[n], color = color_opt[n], linestyle = 'dotted', label=r'$\frac{\partial\text{w}}{\partial z}_{top}$')
        else:
            #axes[0].plot(t[n], div_east[n], color = color_opt[n], linestyle = 'dashed')
            #axes[0].plot(t[n], div_west[n], color = color_opt[n], linestyle = 'dotted')
            #axes[0].plot(t[n], div_north[n], color = color_opt[n], linestyle = 'dashdot')
            #axes[0].plot(t[n], div_south[n], color = color_opt[n], linestyle = 'dashdot')
            #axes[0].plot(t[n], div_top[n], color = color_opt[n], linestyle = 'dashed')
            #axes[0].plot(t[n], div_bottom[n], color = color_opt[n], linestyle = 'dotted')
            axes[0].plot(t[n], div_faces[n], color = color_opt[n], linestyle = 'solid')
            axes[0].plot(t[n], div_bottom[n], color = color_opt[n], linestyle = 'dashed')
            axes[0].plot(t[n], div_top[n], color = color_opt[n], linestyle = 'dotted')

    axes[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    for a in axes[:4]:
        if a.get_yscale() != 'log':
            a.ticklabel_format(axis='y', style='sci', scilimits=(0, 3), useOffset=False)

    outdir = os.path.join(universal_folder, 'callback comparisons')
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, variations + ' divergence_edges_all.svg'))

if neg_tracer:
    scale = [1, 1, 0.02]
    gridspec_kw={'height_ratios': scale}
    fig, axes = plt.subplots(3, 4, figsize=(16, 8), dpi = 300, gridspec_kw = gridspec_kw, sharex = True)
    for a in axes[-1, :]:
            a.remove()
    axes = axes.ravel()
    plt.subplots_adjust(top=0.9)
    case_handles = [Line2D([0], [0], color=color_opt[i], linestyle='solid', label=case_names[i]) for i in range(num_cases)]
    fig.legend(handles=case_handles,
            loc='lower center',
            ncol=num_cases,
            bbox_to_anchor=(0.52, 0.005), fontsize = 12)
    if area_scaling:
        fig.suptitle(f"Tracer statistics with area scaling (r = {rp}m)")

    axes[0].set_title(r'-S$_{avg}$', fontsize = 12)
    axes[0].set_ylabel('[g/kg]', fontsize = 12)

    axes[1].set_title('Maximum magnitude of -S values', fontsize = 12)
    axes[1].set_yscale('log')
    axes[1].set_ylabel('[g/kg]', fontsize = 12)

    axes[2].set_title('Sum of -S in domain', fontsize = 12)
    axes[2].set_ylabel('[g/kg]', fontsize = 12)

    axes[3].set_title('Percent of -S values', fontsize = 12)
    axes[3].set_ylabel('% of domain', fontsize = 12)

    axes[0].set_title(r'S$_{avg}$', fontsize = 12)
    axes[0].set_xlabel('Time (days)', fontsize = 12)
    axes[0].set_ylabel('[g/kg]', fontsize = 12)

    axes[1].set_title(r'dS$_{avg}$/dt', fontsize = 12)
    axes[1].set_xlabel('Time (days)', fontsize = 12)
    axes[1].set_ylabel('[g/kg/days]', fontsize = 12)

    axes[2].set_title('Sum of S in domain', fontsize = 12)
    axes[2].set_xlabel('Time (days)', fontsize = 12)
    axes[2].set_ylabel('[g/kg]', fontsize = 12)

    axes[3].set_title(r'dS$_{sum}$/dt', fontsize = 12)
    axes[3].set_xlabel('Time (days)', fontsize = 12)
    axes[3].set_ylabel('[g/kg/days]', fontsize = 12)

    for n in range(num_cases):
        axes[0].plot(t[n], np.abs(neg_avg[n]), color = color_opt[n], label=case_names[n])
        axes[1].plot(t[n], np.abs(S_min[n]), color = color_opt[n], label=case_names[n])
        axes[3].plot(t[n], percents[n], color = color_opt[n], label=case_names[n])
        axes[2].plot(t[n], S_neg_sum[n], color = color_opt[n], label=case_names[n])
        axes[0].plot(t[n], S_avg[n], color = color_opt[n], label=case_names[n])
        axes[1].plot(t[n], dS_intdt[n], color = color_opt[n], label=case_names[n])
        axes[2].plot(t[n], S_sum[n], color = color_opt[n], label=case_names[n])
        axes[3].plot(t[n], dSdt[n], color = color_opt[n], label=case_names[n])

    for a in axes[:8]:
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