import os
import numpy as np
import matplotlib.pyplot as plt

from reader import OceananigansData
from dense_plume import PlumeAnalysis
from diagnostics import comparison_info
from physics import rms, a_fluc_b, buoyancy
from plotting_functions import plot_format, plot_ranges, create_video, comparison_plot_opt, plume_vertical_spatial_plot, plume_horizontal_spatial_plot, plot_variable_vert_slice, plot_variable_xy_slice, plot_combo_exponents, plot_rig_exponents, plot_Fr_exponents, plot_mld_exponents
from interpolation import velocities_to_center, vertical_line, horizontal_line, yz_plane, xy_plane, xz_plane

# flags for what to plot
plot_variables = True
plume_analysis_plot = False
plot_1d_z = False
transient_mld = False
video = True

# flags for how to read data
with_halos = False
closure = False
salinity = True
stokes = False

contour_bound = 0.05
name_uni = f'contour-{contour_bound:.2f}'
universal_folder = '/Users/annapauls/Library/CloudStorage/OneDrive-UCB-O365/CU-Boulder/TESLa/Carbon Sequestration/Simulations/Oceananigans/NBP/salinity and temperature/no noise circle inlet/'#vertical domain increase/dTdz = 0.01'

# selecting cases to compare
variations = 'all' # 'MLD', 'flux', 'strat', 'all', 'length', 'else'
cases_info = comparison_info(variations, universal_folder)
mld = cases_info['mld']
dTdz = cases_info['dTdz']
F_s = cases_info['F_s']
case_names = cases_info['case_names']
num_cases = cases_info['num_cases']
fig_folder = cases_info['fig_folder']

readers = []
for name in cases_info["folder_names"]:
    folder = os.path.join(universal_folder, name)
    readers.append(OceananigansData(folder))

# collecting model information for all cases
t_save = []
mld_idx = []
if variations == 'length':
    z = []
    nx = []
    lx = []
else:
    z = readers[0].z
    nx = readers[0].nx
    lx = readers[0].lx

nz = np.max(nx[:][2])
x = readers[0].x
y = readers[0].y

# physical parameters
rj = 5 # m, radius of salinity flux circle at the surface
g = 9.80665  # gravity in m/s^2
rho0 = 1026
T0 = 25
S0 = 0 
coeffs = readers[0].load_equation_of_state(salinity)
alpha = coeffs['alpha']
if salinity:
    beta = coeffs['beta']

# video or not setup
if video:
    nt = np.arange(0, readers[0].nt)
    time = readers[0].time

# plotting prep
plot_format()
if plot_variables:
    if salinity:
        var_names = ['Tracer', 'Temperature', 'Perturbed Temperature'] #['Tracer', 'Temperature', 'Density', 'u', 'v', 'w', 'Perturbed Vertical Buoyancy Flux', 'Perturbed Density']
        range_names = ['Tracer', 'T', 'T_fluc'] #['Tracer', 'T', 'rho', 'u', 'v', 'w', 'bw_fluc', 'T_fluc']
    else:
        var_names = ['Temperature', 'Density', 'u', 'v', 'w', 'Perturbed Vertical Buoyancy Flux', 'Perturbed Density']
        range_names = ['T', 'rho', 'u', 'v', 'w', 'bw_fluc', 'T_fluc']
    variable_dir = {}

S_tol = 10**(-5)
ranges = plot_ranges(lz = 96, mld = np.max(mld), rho0 = rho0, T0 = T0, dTdz = np.max(dTdz), C_tol = S_tol)
ranges['rho'] = [rho0, rho0+0.15]
ranges['T_fluc'] = [-0.025, 0.025]
ranges['Tracer'] = [S_tol, 0.2]
ranges['T'] = [T0-1.0, T0 + 0.01]
ranges['w'] = [-1.5*10**(-1), 1.5*10**(-1)]
ranges['u'] = [-1.2*10**(-2), 1.2*10**(-2)]
ranges['v'] = [-2*10**(-2), 2*10**(-2)]
ranges['vel_rms'] = [0, 4*10**-3]
ranges['bw_fluc'] = [-1.5*10**(-5), 1.5*10**(-5)]
ranges['T_fluc'] = [-0.2, 0.2]

if salinity:
    S_avg = []
    S_fluc_center = []
    S_hor = []
    S_plane = []
if plot_1d_z:
    T_avg = []
    b_avg = []
    u_rms = []
    v_rms = []
    w_rms = []
    u_fluc_avg = []
    v_fluc_avg = []
    w_fluc_avg = []
    bu_fluc_avg = []
    bv_fluc_avg = []
    bw_fluc_avg = []
if plume_analysis_plot:
    r_profile = []
    b_center = []
    T_fluc_center = []
if plot_variables:
    T_plane = []
    u_plane = []
    v_plane = []
    w_plane = []
    rho_plane = []
    bw_plane = []
    T_fluc_plane = []
for i, reader in enumerate(readers):
    # Load binning from files
    r, z, time, S_rz, T_fluc_rz, T_rz, u_rz, v_rz, w_rz = reader.load_binning()

    # plane slices to save for plotting
    if plot_variables:
        T_plane.append(T_rz)
        T_fluc_plane.append(T_fluc_rz)
        if salinity:
            S_plane.append(S_rz)

############ PLOTTING ############
for it, t in enumerate(time):
    if plot_variables:
        if salinity: #'Tracer', 'T', 'Density', 'u', 'v', 'w', 'Perturbed Vertical Buoyancy Flux'
            variables = [S_plane, T_plane, T_fluc_plane] #[S_plane, T_plane, rho_plane, u_plane, v_plane, w_plane, bw_plane, T_fluc_plane] 
            colorbar_labels = [r"g/kg", r"$^\circ$C", r"$^\circ$C"]#[r"g/kg", r"$^\circ$C", r"kg/m$^3$", r"m/s", r"m/s", r"m/s", r"m$^2$/s$^3$", r"kg/m$^3$"]
            cmaps = ['viridis', 'viridis', 'RdBu_r']#, 'RdBu_r', 'RdBu_r', 'RdBu_r']
        else: #'T', 'Density', 'u', 'v', 'w', 'Perturbed Vertical Buoyancy Flux'
            variables = [T_plane, rho_plane, u_plane, v_plane, w_plane, bw_plane, T_fluc_plane] 
            colorbar_labels = [r"$^\circ$C", r"kg/m$^3$", r"m/s", r"m/s", r"m/s", r"m$^2$/s$^3$", r"kg/m$^3$"]
            cmaps = ['viridis', 'viridis', 'RdBu_r', 'RdBu_r', 'RdBu_r', 'RdBu_r', 'RdBu_r']
        for n, var in enumerate(variables):
            variable_dir[var_names[n]] = plot_variable_vert_slice(time[it], it, ranges, fig_folder, lx, r, z, var, case_names, var_names[n], range_names[n], colorbar_label = colorbar_labels[n], cmap = cmaps[n])
            variable_dir[var_names[n]] = os.path.join(variable_dir[var_names[n]], 'binned results')
print("All frames created.")
# creating videos
if video:
    if plot_variables:
        for n, name in enumerate(var_names):
            create_video(variable_dir[var_names[n]], fig_folder, '', name)
