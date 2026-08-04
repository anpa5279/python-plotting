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

# ==========================================================
# FLAGS
# ==========================================================
area_scaling = False
tracer_mass = False
mass_divergence = False
neg_tracer = True
w_surface = True

salinity = True

# ==========================================================
# COMPARISON CASES
# ==========================================================
universal_folder = '/Users/annapauls/Documents/TESLa /Simulations/Oceananigans/dense plume/salinity and temperature/version109/square inlet/open BC/no SGS/default WENO/bottom PA'

variations = "else"
if variations != "else":
    cases_info = comparison_info(variations, universal_folder=universal_folder)

    case_names = cases_info["case_names"]
    folder_names = cases_info["folder_names"]
    num_cases = cases_info["num_cases"]
    fig_folder = os.path.join(cases_info["fig_folder"], "convergence")
    F_s = cases_info["F_s"]
    mld = cases_info["mld"]
    dTdz = cases_info["dTdz"]
else:

    folder_names = ['dx2', 'dx1', 'dx05', 'dx025', 'dx0125']#['dx2', 'horizontal resolution/dx1', 'horizontal resolution/dx05']#

    case_names = [r'$\Delta x = 2.0$', r'$\Delta x = 1.0$', r'$\Delta x = 0.5$', r'$\Delta x = 0.25$', r'$\Delta x = 0.125$']#, r'$\Delta x = 0.25$']#[r'$\Delta x = \Delta y = \Delta z = 2.0$', r'$\Delta x = \Delta y = 1.0$ $ \Delta z = 2.0$', r'$\Delta x = \Delta y = 0.5$ $ \Delta z = 2.0$']#[r'$\Delta x = \Delta y = \Delta z = 2.0$', r'$\Delta x = \Delta y = 2.0$ $ \Delta z = 1.0$', r'$\Delta x = \Delta y = 2.0$ $ \Delta z = 0.5$']#
    
    num_cases = len(folder_names)
    fig_folder = os.path.join(universal_folder, 'callback comparisons')
    F_s = 0.1 * np.ones(num_cases)
    mld = 60 * np.ones(num_cases)
    dTdz = 0.01 * np.ones(num_cases)

os.makedirs(fig_folder, exist_ok=True)
# ==========================================================
# READERS
# ==========================================================
readers = []
with_halos = [True, True, True, False, False]
for i, name in enumerate(folder_names):
    folder = os.path.join(universal_folder, name)
    readers.append(OceananigansData(folder, salinity = salinity, with_halos = with_halos[i]))

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
for n, reader in enumerate(readers):
    time.append(reader.t)
    nx[:, n] = reader.nx
    lx[:, n] = reader.lx
    nt[n] = reader.nt

# ==========================================================
# DATA STORAGE
# ==========================================================
t = []

dmdt = []
S_mass = []
div_top = []
div_bottom = []
div_faces = []
S_neg_percent = []
neg_avg = []
S_max = []
S_min = []
dSdt = []
w_min = []
w_max = []
w_sum = []

# ==========================================================
# LOAD DATA
# ==========================================================
for n, reader in enumerate(readers):
    t.append(reader.t / 3600 / 24)
    domain = math.prod(reader.nx)
    vol = math.prod(reader.lx)

    file_path = os.path.join(reader.folder, 'binning_rtz.h5')
    with h5py.File(file_path, 'r') as f:
        S_int = f["S mass"][:]
        dmdt_loc = f["time gradient of S mass"][:]

    S_mass.append(S_int)
    if tracer_mass or mass_divergence:
        if "glade" in universal_folder:
            dmdt.append(np.gradient(S_mass[n], t[n]))
        else:
            dmdt.append(dmdt_loc)
        if mass_divergence:
            w = reader.lazy_field('w').compute()
            
            dwdz = np.gradient(w, reader.zf, axis=-1)

            dmw_top = np.sum(dwdz[:, :, :, 0].squeeze(), axis = (1, 2))
            dmw_bottom = -1*np.sum(dwdz[:, :, :, -1].squeeze(), axis = (1, 2))
    
            div_top.append(dmw_top)
            div_bottom.append(dmw_bottom)

            div_faces.append(div_top[-1] + div_bottom[-1]) 
    if w_surface:
        w_loc = reader.load_plane_var('w', plane = 'XY')
        w_min.append(np.min(w_loc, axis = (1, 2)))
        w_max.append(np.max(w_loc, axis = (1, 2)))
        w_sum.append(np.sum(w_loc, axis = (1, 2)))
    if neg_tracer:
        with h5py.File(file_path, 'r') as f:
            S_min_loc = f["min of S"][:]
            S_max_loc = f["max of S"][:]
            S_neg_count_loc = f["negative S count"][:]
            S_neg_avg_loc = f["negative S average"][:]
        # minimum S value in domain
        S_min.append(S_min_loc)
        S_max.append(S_max_loc)
        # negative number of negative values appearing in domain 
        neg_avg.append(S_neg_avg_loc/S_neg_count_loc)
        S_neg_percent.append(S_neg_count_loc/np.prod(reader.nx)*100)
    if area_scaling:
        Nr = area_scale(rp, reader.dx[0])
        grid_area = Nr*reader.dx[0]*reader.dx[1]
        factor.append(area/grid_area)
        if tracer_mass:
            S_mass[n] = S_int*factor[n]
            dmdt[n] = np.gradient(S_mass[n], t[n])

        if neg_tracer:
            neg_avg[n] = neg_avg[n]*factor[n]
            S_mass[n] = S_mass[n]*factor[n]
            dSdt[n] = dSdt[n]*factor[n]

# ==========================================================
# PLOTTING
# ==========================================================
color_opt, line_opt = comparison_plot_opt(num_cases)
plot_format()
if tracer_mass:
    scale = [1, 0.1]
    gridspec_kw={'height_ratios': scale}
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), gridspec_kw = gridspec_kw)
    for a in axes[-1, :]:
            a.remove()
    axes = axes.ravel()
    plt.subplots_adjust(top=0.9)
    case_handles = [Line2D([0], [0], color=color_opt[i], linestyle='solid', label=case_names[i]) for i in range(num_cases)]
    leg_col = num_cases//2 if num_cases >= 4 else num_cases
    fig.legend(handles=case_handles,
            loc='lower center',
            ncol=leg_col,
            bbox_to_anchor=(0.5, 0.005))
    if area_scaling:
        fig.suptitle(f"Tracer statistics with area scaling (r = {rp}m)")
        mass_label = r'$\rho_{0}L_{x}L_{y}L_{z}\frac{\pi r_{p}^2}{N_{r}d_{x}d_{y}}\langle\text{C}\rangle_{\text{xyz}}$[g]'
        mass_rate_label = r'$\rho_{0}L_{x}L_{y}L_{z}\frac{\pi r_{p}^2}{N_{r}d_{x}d_{y}}\frac{\text{d}\langle\text{C}\rangle_{\text{xyz}}}{\text{dt}}$[g/days]'
    else:
        mass_label = r'$\rho_{0}L_{x}L_{y}L_{z}\langle\text{C}\rangle_{\text{xyz}}$[g]'
        mass_rate_label = r'$\rho_{0}L_{x}L_{y}L_{z}\frac{\text{d}\langle\text{C}\rangle_{\text{xyz}}}{\text{dt}}$[g/days]'

    axes[0].set_title('Mass')
    axes[0].set_xlabel('Time (days)')
    axes[0].set_ylabel(mass_label)

    axes[1].set_title(r'Temporal rate of Mass')
    axes[1].set_xlabel('Time (days)')
    axes[1].set_ylabel(mass_rate_label)
    dmdt_flat = list(itertools.chain.from_iterable(dmdt))
    mass_rate = [min(dmdt_flat), max(dmdt_flat)]
    axes[1].set_ylim(mass_rate[0]*0.9, mass_rate[1]*1.1)

    axes[2].set_title('Percent difference in mass\nfrom control case')
    axes[2].set_xlabel('Time (days)')
    axes[2].set_ylabel(r'$\frac{(\text{S} - \text{S}_{\text{control}})}{\text{S}_{\text{control}}} $[%]')

    for n in range(num_cases):
        axes[0].plot(t[n], S_mass[n], color = color_opt[n], label=case_names[n])
        axes[1].plot(t[n], dmdt[n], color = color_opt[n], label=case_names[n])
        if n>0:
            if len(S_mass[n]) == len(S_mass[0]):
                axes[2].plot(t[n], (S_mass[n] - S_mass[0])/S_mass[0]*100, color = color_opt[n], label=case_names[n])
            else:
                min_len = min(len(S_mass[n]), len(S_mass[0]))
                axes[2].plot(t[n][:min_len], (S_mass[n][:min_len] - S_mass[0][:min_len])/S_mass[0][:min_len]*100, color = color_opt[n], label=case_names[n])

    for a in axes[:4]:
        if a.get_yscale() != 'log':
            a.ticklabel_format(axis='y', style='sci', scilimits=(0, 3), useOffset=False)
    fig.tight_layout(pad=1.5)

    if "/glade" in universal_folder:
        variations += ' with fields'
    if area_scaling:
        variations += f' and area scaling'
    plt.savefig(os.path.join(fig_folder, variations + ' comparisons mass.svg'))

if mass_divergence:
    scale = [1, 0.1]
    gridspec_kw={'height_ratios': scale}
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw = gridspec_kw)
    axes = axes.ravel()
    axes[-1].remove()
    #plt.subplots_adjust(top=0.9, right=0.8)
    case_handles = [Line2D([0], [0], color=color_opt[i], linestyle='solid', label=case_names[i]) for i in range(num_cases)]
    leg_col = num_cases//2 if num_cases >= 4 else num_cases
    fig.legend(handles=case_handles,
            loc='lower center',
            ncol=leg_col,
            bbox_to_anchor=(0.5, 0.005))

    axes[0].set_xlabel('Time (days)')
    #axes[0].set_ylabel(r'$\frac{\partial\text{m}}{\partial t} + \nabla\cdot$ ($\text{m}$u$_i$)')
    axes[0].set_ylabel(r'$\frac{\partial\text{w}}{\partial z}_{bottom} - \frac{\partial\text{w}}{\partial z}_{top}$')

    for n in range(num_cases):
        if n == 0: #, marker = 'o', markersize=3

            axes[0].plot(t[n], div_faces[n], color = color_opt[n], linestyle = 'solid', label=r'$\frac{\partial\text{w}}{\partial z}$')
            axes[0].plot(t[n], div_bottom[n], color = color_opt[n], linestyle = 'dashed', label=r'$\frac{\partial\text{w}}{\partial z}_{bottom}$')
            axes[0].plot(t[n], div_top[n], color = color_opt[n], linestyle = 'dotted', label=r'$\frac{\partial\text{w}}{\partial z}_{top}$')
        else:
            axes[0].plot(t[n], div_faces[n], color = color_opt[n], linestyle = 'solid')
            axes[0].plot(t[n], div_bottom[n], color = color_opt[n], linestyle = 'dashed')
            axes[0].plot(t[n], div_top[n], color = color_opt[n], linestyle = 'dotted')

    axes[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    for a in axes[:4]:
        if a.get_yscale() != 'log':
            a.ticklabel_format(axis='y', style='sci', scilimits=(0, 3), useOffset=False)

    plt.savefig(os.path.join(fig_folder, variations + ' divergence_edges_all.svg'))

if neg_tracer:
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharex = True)
    axes = axes.ravel()
    if area_scaling:
        fig.suptitle(f"Tracer statistics with area scaling (r = {rp}m)")

    for n in range(num_cases):
        axes[0].plot(t[n], neg_avg[n], color = color_opt[n], label=case_names[n])
        axes[1].plot(t[n], S_neg_percent[n], color = color_opt[n], label=case_names[n])
        axes[2].plot(t[n], S_min[n], color = color_opt[n], label=case_names[n])
        axes[3].plot(t[n], S_max[n], color = color_opt[n], label=case_names[n])

    axes[0].set_title(r'-S$_{avg}$/N$_{\text{negative}}$')
    axes[0].set_ylabel('[g/kg]')
    axes[0].legend(loc='lower left', handlelength = 0.55)

    axes[1].set_title('Percent of cells with negative S')
    axes[1].set_ylabel(r'N$_{\text{negative}}$/(N$_{x}\cdot$N$_{y}\cdot$N$_{z}$) [%]')

    axes[2].set_title('Minimum of S')
    axes[2].set_xlabel('Time (days)')
    axes[2].set_yscale('symlog', linthresh=1e-12)
    axes[2].set_ylim(-10**-1, -10**-8)
    axes[2].set_ylabel('[g/kg]')

    axes[3].set_title('Maximum of S')
    axes[3].set_xlabel('Time (days)')
    axes[3].set_ylabel('[g/kg]')

    for a in axes:
        if a.get_yscale() != 'symlog':
            a.ticklabel_format(axis='y', style='sci', scilimits=(0, 3))
    fig.tight_layout(pad=1.5)

    if "/glade" in universal_folder:
        variations += ' with fields on Derecho'
    if area_scaling:
        variations += f' and area scaling'
    plt.savefig(os.path.join(fig_folder, variations + ' neg_tracer.svg'))

if w_surface:
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    axes = axes.ravel()


    axes[0].set_xlabel('Time (days)')
    axes[0].set_ylabel(r'$\text{w}_{min}$')
    axes[1].set_xlabel('Time (days)')
    axes[1].set_ylabel(r'$\text{w}_{max}$')
    axes[2].set_xlabel('Time (days)')
    axes[2].set_ylabel(r'$\text{w}_{sum}$')

    for n in range(num_cases):
        axes[0].plot(t[n], w_min[n], color = color_opt[n], label=case_names[n])
        axes[1].plot(t[n], w_max[n], color = color_opt[n], label=case_names[n])
        axes[2].plot(t[n], w_sum[n], color = color_opt[n], label=case_names[n])

    axes[0].legend(loc='upper left', handlelength = 0.55)

    plt.savefig(os.path.join(fig_folder, variations + ' w_surface.svg'))