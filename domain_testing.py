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
from interpolation import horizontal_line
from physics import buoyancy

# ==========================================================
# FUNCTIONS
# ==========================================================
def compute_radial_error(coord, profiles, error_norm, r_targets):
    """
    Normalized error at specified r locations between each case's profile
    and the finest-resolution reference profile (assumed to be the last
    case, matching the convention in plot_lines_h).

    Parameters
    ----------
    coord : list of ndarray
        Radial coordinate array (e.g. r) per case.
    profiles : list of ndarray
        1D profile per case (same length as the corresponding coord entry),
        e.g. [np.mean(w_bin_hor[n][it-1:it+1, :, k], axis=0) for n in range(num_cases)].
    error_norm : float
        Normalization constant for this variable (as in error_norm dict
        already used in plot_lines_h).
    r_targets : list of float
        Radial locations at which to evaluate error.

    Returns
    -------
    errors : ndarray, shape (num_cases - 1, len(r_targets))
        Normalized error for each non-reference case at each r location.
    """
    num_cases = len(coord)
    comparison_coord = coord[-1]
    comparison_profile = profiles[-1]

    errors = np.full((num_cases - 1, len(r_targets)), np.nan)
    for n in range(num_cases - 1):
        for j, r0 in enumerate(r_targets):
            idx_n = np.argmin(np.abs(coord[n] - r0))
            idx_ref = np.argmin(np.abs(comparison_coord - r0))
            errors[n, j] = (profiles[n][idx_n] - comparison_profile[idx_ref]) / error_norm
    return errors

def save_bin_error(h5_error_path, case_types, var_name, loc_z, errors):
    """
    errors : ndarray, shape (len(case_types), nt, len(loc_z))
    Layout: <case>/<var_name>/z<loc> = 1D array over time
    """
    with h5py.File(h5_error_path, 'a') as f:
        for n, case in enumerate(case_types):
            for k, z0 in enumerate(loc_z):
                grp = f.require_group(f"{case}/{var_name}")
                dset_name = f"z{z0:.2f}"
                if dset_name in grp:
                    del grp[dset_name]
                grp.create_dataset(dset_name, data=errors[n, :, k])

# ==========================================================
# FLAGS
# ==========================================================
area_scaling = False
tracer_mass = False
mass_divergence = False
neg_tracer = False
w_surface = False
internal_gravity_waves = False
scaling_analysis_text = False
error_analysis = True

salinity = True

# ==========================================================
# COMPARISON CASES
# ==========================================================
universal_folder = '/Users/annapauls/Documents/Github repositories/3d_langmuir_gpu/localoutputs/'

variations = 'else'
if variations != 'else':
    cases_info = comparison_info(variations, universal_folder = universal_folder)
    case_names = cases_info['case_names']
    num_cases = cases_info['num_cases']
    folder_names = cases_info['folder_names']
    fig_folder = cases_info['fig_folder']
else:
    folder_names = [#'scaled S double to match both gauss/WENO5/dx2.0', 'scaled S double to match both gauss/WENO5/dx1.0', 'scaled S double to match both gauss/WENO5/dx0.5', 
                    'scaled S double to match both gauss/WENO5/dx0.25', 'scaled S double to match both gauss/WENO5/dx0.125', 
                    'scheme-tests/longer/WENO3/dx2.0', 'scheme-tests/longer/WENO3/dx1.0', 'scheme-tests/longer/WENO3/dx0.5', 'scheme-tests/longer/WENO3/dx0.25', 'scheme-tests/longer/WENO3/dx0.125',
                    'scheme-tests/longer/WENO5/dx2.0', 'scheme-tests/longer/WENO5/dx1.0', 'scheme-tests/longer/WENO5/dx0.5', 'scheme-tests/longer/WENO5/dx0.25', 'scheme-tests/longer/WENO5/dx0.125',
                    'scheme-tests/longer/WENO9/dx2.0', 'scheme-tests/longer/WENO9/dx1.0', 'scheme-tests/longer/WENO9/dx0.5', 'scheme-tests/longer/WENO9/dx0.25', #'scheme-tests/longer/WENO9/dx0.125',
                    'scheme-tests/longer/WENO5/dx0.0625']
    #['dx2.0', 'dx1.0', 'dx0.5', 'dx0.25', 'dx0.125']#, 'dx0.0625']#['dx2', 'dx1', 'dx05']#, 'dx025', 'dx0125']#, 'dx00625']
   
    case_names = [#r'Gaussian WENO 5, $\Delta x = 2.0$', r'Gaussian WENO 5, $\Delta x = 1.0$', r'Gaussian WENO 5, $\Delta x = 0.5$', 
                    r'Gaussian WENO 5, $\Delta x = 0.25$', r'Gaussian WENO 5, $\Delta x = 0.125$', 
                  r'WENO 3, $\Delta x = 2.0$', r'WENO 3, $\Delta x = 1.0$', r'WENO 3, $\Delta x = 0.5$', r'WENO 3, $\Delta x = 0.25$', r'WENO 3, $\Delta x = 0.125$', 
                  r'WENO 5, $\Delta x = 2.0$', r'WENO 5, $\Delta x = 1.0$', r'WENO 5, $\Delta x = 0.5$', r'WENO 5, $\Delta x = 0.25$', r'WENO 5, $\Delta x = 0.125$', 
                  r'WENO 9, $\Delta x = 2.0$', r'WENO 9, $\Delta x = 1.0$', r'WENO 9, $\Delta x = 0.5$', r'WENO 9, $\Delta x = 0.25$', #r'WENO 9, $\Delta x = 0.125$', 
                  r'$\Delta x = 0.0625$']
    #[r'$\Delta x = 2.0$', r'$\Delta x = 1.0$', r'$\Delta x = 0.5$', r'$\Delta x = 0.25$', r'$\Delta x = 0.125$', r'$\Delta x = 0.0625$']#, r'$\Delta x = 0.25$']#[r'$\Delta x = \Delta y = \Delta z = 2.0$', r'$\Delta x = \Delta y = 1.0$ $ \Delta z = 2.0$', r'$\Delta x = \Delta y = 0.5$ $ \Delta z = 2.0$']#[r'$\Delta x = \Delta y = \Delta z = 2.0$', r'$\Delta x = \Delta y = 2.0$ $ \Delta z = 1.0$', r'$\Delta x = \Delta y = 2.0$ $ \Delta z = 0.5$']#
    if error_analysis:
        case_types = [#'/Gaussian WENO 5/delta x = 2.0', '/Gaussian WENO 5/delta x = 1.0', '/Gaussian WENO 5/delta x = 0.5', 
                    '/Gaussian WENO 5/delta x = 0.25', '/Gaussian WENO 5/delta x = 0.125', 
                    '/WENO 3/delta x = 2.0', '/WENO 3/delta x = 1.0', '/WENO 3/delta x = 0.5', '/WENO 3/delta x = 0.25', '/WENO 3/delta x = 0.125', 
                    '/WENO 5/delta x = 2.0', '/WENO 5/delta x = 1.0', '/WENO 5/delta x = 0.5', '/WENO 5/delta x = 0.25', '/WENO 5/delta x = 0.125', 
                    '/WENO 9/delta x = 2.0', '/WENO 9/delta x = 1.0', '/WENO 9/delta x = 0.5', '/WENO 9/delta x = 0.25'#, '/WENO 9/delta x = 0.125'
                    ]
    num_cases = len(folder_names)
    fig_folder = os.path.join(universal_folder, 'callback comparisons')
    F_s = 0.1 * np.ones(num_cases)
    mld = 60 * np.ones(num_cases)
    dTdz = 0.01 * np.ones(num_cases)

os.makedirs(fig_folder, exist_ok=True)
# ==========================================================
# PARAMETERS
# ==========================================================
g = 9.80665 # [m/s^2]
T0 = 25 # [°C]
rho0 = 1026 # [kg/m^3]
w0 = -0.001 # [m/s]
Sval = 0.1 # [g/kg]
# ==========================================================
# READERS
# ==========================================================
readers = []
with_halos = [True, True, True, False, False, 
              True, True, True, False, False, 
              True, True, True, False, False,
              True, True, True, False, False,
              True]
for n, name in enumerate(folder_names):
    folder = os.path.join(universal_folder, name)
    readers.append(OceananigansData(folder, salinity = salinity, with_halos = with_halos[n], Sval = 0.1))

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
# collecting model information for all cases
nx = np.empty((3, num_cases), dtype=object)
lx = np.empty((3, num_cases), dtype=object)
nt = np.empty(num_cases, dtype=int)
time  = []
if error_analysis:
    r = []
    nt_min = np.inf
for n, reader in enumerate(readers):
    time.append(reader.t)
    nx[:, n] = reader.nx
    lx[:, n] = reader.lx
    nt[n] = reader.nt
    if error_analysis:
        r.append(reader.r)
        nt_min = int(np.min([nt_min, reader.nt]))
dx = lx/nx
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

u_fluc_yz = []
v_fluc_yz = []
w_fluc_yz = []
T_fluc_yz = []
S_fluc_yz = []
b_fluc_yz = []

w_fluc_xy = []

y_edge = []
x_edge = []

brunt_freq = []
omega = []
power = []

alpha = []
c_delta = []

w_bin = []
S_bin = []
b_bin = []

w_bin_hor = []
S_bin_hor = []
b_bin_hor = []

omega_len = np.inf

# ==========================================================
# LOAD DATA
# ==========================================================
for n, reader in enumerate(readers):
    t.append(reader.t / 3600 / 24)
    domain = math.prod(reader.nx)
    vol = math.prod(reader.lx)

    file_path = os.path.join(reader.folder, 'binning_rtz.h5')

    #with h5py.File(file_path, 'r') as f:
    #    S_int = f["S mass"][:]
    #    dmdt_loc = f["time gradient of S mass"][:]

    #S_mass.append(S_int)
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
    if internal_gravity_waves:
        w = reader.load_plane_var("w'", plane="XY", loc=-mld)
        window = np.hanning(w.shape[0])
        w_windowed = w * window[:, None, None]

        # FFT
        w_hat = np.fft.rfft(w_windowed, axis=0)
        freq = np.fft.rfftfreq(w.shape[0], d=reader.dt)
        omega = 2 * np.pi * freq
        power = np.abs(w_hat)**2

        # Spatially averaged spectrum
        P_omega = np.mean(power, axis=(1, 2))

        # Find dominant frequencies
        P_search = P_omega.copy()
        P_search[0] = 0

        peaks = np.argsort(P_search)[-5:]
    if scaling_analysis_text:
        opt = 'fft convolve w_c*10**-5'
        with h5py.File(file_path, 'r') as f:
            c_delta_loc = f["scaling analysis/outer length scale/"+opt+"/c_delta"][:]
            alpha_loc = f["scaling analysis/outer length scale/"+opt+"/alpha"][:]
        alpha.append(alpha_loc)
        c_delta.append(c_delta_loc)
    if error_analysis:
        loc_z = [-mld[n], -40, -20, -10]
        r_targets = [0.0]  # r locations of interest — extend as needed
        h5_error_path = os.path.join(fig_folder, 'bin_errors.h5')

        w_rz = reader.load_binning_var('w')
        if salinity:
            S_rz = reader.load_binning_var('S')
        b_rz = buoyancy(reader, type = 'bin')
        w_bin.append(w_rz)
        if salinity:
            S_bin.append(S_rz)
        b_bin.append(b_rz)
        w_bin_line_loc = np.empty((reader.nt, len(loc_z)))
        S_bin_line_loc = np.empty((reader.nt, len(loc_z)))
        b_bin_line_loc = np.empty((reader.nt, len(loc_z)))
        for k, loc in enumerate(loc_z):
            k_opt = np.where(reader.z == loc)[0]
            w_bin_line_loc[:, k] = horizontal_line(w_bin[n][:, 0, :].squeeze(), z = reader.z, z0 = loc, axis=-1)
            S_bin_line_loc[:, k] = horizontal_line(S_bin[n][:, 0, :].squeeze(), z = reader.z, z0 = loc, axis=-1)
            b_bin_line_loc[:, k] = horizontal_line(b_bin[n][:, 0, :].squeeze(), z = reader.z, z0 = loc, axis=-1)
        w_bin_hor.append(w_bin_line_loc)
        S_bin_hor.append(S_bin_line_loc)
        b_bin_hor.append(b_bin_line_loc)
# AFTER the reader loop — error computation, once all cases are loaded:
if error_analysis:
    error_norm = {'w': w0*100, 'S': Sval, 'b': -Sval*reader.beta*g}
    for var_name, var_hor in zip(['w', 'S', 'b'], [w_bin_hor, S_bin_hor, b_bin_hor]):
        ref = var_hor[-1]  # finest-resolution reference case, shape (nt_ref, len(loc_z))
        errors_var = np.full((num_cases - 1, nt_min, len(loc_z)), np.nan)
        for n in range(num_cases - 1):
            nt_case = min(nt_min, var_hor[n].shape[0], ref.shape[0])
            errors_var[n, :nt_case, :] = (var_hor[n][:nt_case, :] - ref[:nt_case, :]) / error_norm[var_name]
        save_bin_error(h5_error_path, case_types, var_name, loc_z, errors_var)
# ==========================================================
# PLOTTING
# ==========================================================
color_opt, line_opt = comparison_plot_opt(num_cases)
plot_format(fontsize = 18)
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

if internal_gravity_waves:
    outdir =  os.path.join(fig_folder, "testing/")
    os.makedirs(outdir, exist_ok=True)
    for i in range(omega_len):
        ncols = 3
        if num_cases < ncols*2:
            ncols = num_cases
        nrows = int(math.ceil(num_cases/ncols))
        hor_len = 12.0
        vert_len = hor_len * nrows / (ncols) + 0.5 * nrows + 1.1

        fig, axes = plt.subplots(nrows, ncols, figsize=(hor_len, vert_len), sharey = True, sharex = True, constrained_layout=True)
        axes = [axes,]#axes.ravel()
        for n, reader in enumerate(readers):
            im =  axes[n].imshow(power[n][i, :, :].T, origin = "lower", interpolation = "none", cmap = 'RdBu_r', extent = [reader.y[0], reader.y[-1], reader.z[0], reader.z[-1]], aspect = 'auto')

            axes[n].set_xlabel(r"$\omega$ [rad s$^{-1}$]")
            axes[n].set_ylabel(r"$|\hat{w}|^2$")
            axes[n].set_title(f"{omega[n][i]}")
        plt.colorbar(im, ax = axes, anchor = (0.5, 0.0), orientation='horizontal', shrink=0.75, aspect=80)
        frame_path = os.path.join(outdir, f"oc_plane_slices_{i:04d}.png")
        plt.savefig(frame_path)
        plt.close()

if scaling_analysis_text:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex = True)
    axes = axes.ravel()

    axes[0].set_xlabel(r'dx$_{i}$ [m]')
    axes[0].set_ylabel(r'$\alpha$')
    axes[1].set_xlabel(r'dx$_{i}$ [m]')
    axes[1].set_ylabel(r'$c_{\delta}$')

    for n in range(num_cases):
        alpha_err = [np.min(alpha[n]), np.max(alpha[n])]
        c_err = [np.min(c_delta[n]), np.max(c_delta[n])]
        alpha_err = np.array([[np.median(alpha[n]) - alpha_err[0]], [alpha_err[1] - np.median(alpha[n])]])
        c_err = np.array([[np.median(c_delta[n]) - c_err[0]], [c_err[1] - np.median(c_delta[n])]])
        axes[0].errorbar(dx[0, n], np.median(alpha[n]), yerr=alpha_err, color = color_opt[n], label=case_names[n], fmt="--o", capsize=3)
        axes[1].errorbar(dx[0, n], np.median(c_delta[n]), yerr=c_err, color = color_opt[n], label=case_names[n], fmt="--o", capsize=3)
    def _func(x):
        return np.log2(x)
    def _inverse(x):
        return 2**x
    axes[0].set_xscale('log', base=2)
    axes[1].set_xscale('log', base=2)
    
    axes[0].legend(loc='upper left', handlelength = 0.55)

    plt.savefig(os.path.join(fig_folder, variations + ' scaling_analysis_text.svg'))

if error_analysis:
    for it in range(nt_min):
        title = f'{reader.t[it]/3600:.2f} hours'
        lines_var_it = {}
        lines_var_it['w'] = {'var':[w_bin_hor[n][it, k] for n in range(num_cases)], 'title': f'w(r, {loc})', 'label': '[m/s]'}
        lines_var_it['S'] = {'var':[S_bin_hor[n][it, k] for n in range(num_cases)], 'title': f'S(r, {loc})', 'label': '[g/kg]'}
        lines_var_it['b'] = {'var':[b_bin_hor[n][it, k] for n in range(num_cases)], 'title': f'b(r, {loc})', 'label': r"[m/s$^2$]"}
        with h5py.File(h5_error_path, 'r') as f:
            case_types = list(f.keys())
            if color_opt is None:
                color_opt, _, marker_opt = comparison_plot_opt(len(case_types), markers = True)
            var_names = list(lines_var_it.keys())
            nrows = len(loc_z) * len(r_targets)+1
            ncols = len(var_names)
            gridspec_kw = np.ones(nrows)
            gridspec_kw[-1] = 0.1
            fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), constrained_layout=True, gridspec_kw = {'height_ratios': gridspec_kw})
            for ax in axes[-1, :]:
                    ax.remove()
            row = 0
            markers = {}
            colors = {}
            for z0 in loc_z:
                c_opt = 0
                for r0 in r_targets:
                    for vcol, var_name in enumerate(var_names):
                        ax = axes[row, vcol]
                        for c, case in enumerate(case_types):
                            print(c)
                            group_path = f"{case}/{var_name}/z{z0:.2f}/r{r0:.2f}"
                            if group_path not in f:
                                continue
                            grp = f[group_path]
                            dx_vals = np.array([float(k.replace('dx', '')) for k in grp.keys()])
                            err_vals = np.array([grp[k][()] for k in grp.keys()])
                            order = np.argsort(dx_vals)
                            if c >= num_cases:
                                c_opt += 1
                                m_opt = c_opt +1
                                colors.append(color_opt[c_opt])
                                markers.append(marker_opt[m_opt])
                            else:
                                c_opt = c
                                m_opt = c
                                colors.append(color_opt[c_opt])
                                markers.append(marker_opt[m_opt])
                            ax.scatter(dx_vals[order], err_vals[order], color=color_opt[c_opt], marker=marker_opt[m_opt], label=case)
                        ax.set_title(f"{var_name}, z={z0:.1f} m, r={r0:.1f} m")
                        ax.set_xlabel(r'$\Delta x$ [m]')
                        ax.set_ylabel('Normalized error')
                    row += 1
            handles, labels = axes[0, 0].get_legend_handles_labels()
            case_handles = [Line2D([0], [0], color=colors[n], markerstyle=markers[n], linestyle = None, label=case_names[n]) for n in range(num_cases)]
            fig.legend(handles=case_handles, loc='lower center', ncol=num_cases, bbox_to_anchor=(0.5, 0.0))