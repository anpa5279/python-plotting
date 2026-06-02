import os
import numpy as np
import matplotlib.pyplot as plt

from reader import OceananigansData
from diagnostics import comparison_info
from interpolation import point
from plotting_general import plot_format, comparison_plot_opt
from plotting_lines import plot_plume_depths #, plot_plume_horizontal_spatial

# flags for what to plot
plot_depths = True

# flags for how to read data
with_halos = False
closure = False
salinity = True
stokes = False

contour_bound = 0.001
name_uni = f'contour-{contour_bound:.4f}'
universal_folder = '/Users/annapauls/Documents/TESLa /Simulations/Oceananigans/dense plume/salinity and temperature/no noise circle inlet/domain testing'
#'/glade/derecho/scratch/apauls/outputs/'
#harddrive: '/Volumes/Anna External/Oceananigans/dense plume with stratification/salinity and temperature /no noise circle inlet/resolution testing'#

# selecting cases to compare
variations = 'Lz160m' # 'MLD', 'flux', 'strat', 'all', 'vertical length', 'Lz160m','WENO', 'vertical resolution', 'horizontal resolution', 'else'
if variations != 'else':
    cases_info = comparison_info(variations, universal_folder = universal_folder)
    dTdz = cases_info['dTdz']
    case_names = cases_info['case_names']
    num_cases = cases_info['num_cases']
    folder_names = cases_info['folder_names']
    fig_folder = cases_info['fig_folder']
    mld = cases_info['mld']
else:
    folder_names = ['proposed resolution/S0 = 0.1 dTdz = 0.01 MLD = 60', 'Lz = 160m/S0 = 0.1 dTdz = 0.01 MLD = 60']
    num_cases = len(folder_names)
    fig_folder = os.path.join(universal_folder, 'comparison figures', '96m vs 160m' + ' comparison figures')
    case_names =[r'L$_z = 96$m', r'L$_z = 160$m']#r'$\Delta z = 0.5$m', r'$\Delta z = 0.375$m'#[r'F$_{\text{C}} = -1.0\cdot 10^{-4}$, MLD = 60m, dTdz = 0.01', r'F$_{\text{C}} = -1.0\cdot 10^{-4}$, MLD = 70m, dTdz = 0.01', r'F$_{\text{C}} = - 2.0\cdot 10^{-4}$, MLD = 60m, dTdz = 0.01']
    dTdz = 0.01*np.ones(num_cases)
    mld = np.array([60, 60])

readers = []
for folder in folder_names:
    folder = os.path.join(universal_folder, folder)
    readers.append(OceananigansData(folder, salinity = salinity))
# physical parameters
x0 = 0.0
y0 = 0.0
rj = 5 # m, radius of salinity flux circle at the surface
g = 9.80665  # gravity in m/s^2
T0 = 25
# collecting model information for all cases
mld_idx = []
x = []
r_bin = []
y = []
z = []
bS = []
nx = np.empty((3, num_cases), dtype=object)
lx = np.empty((3, num_cases), dtype=object)
nt = np.empty(num_cases, dtype=int)
time  = []
grid_specs = False*np.ones(num_cases)
grid_specs[2] = True # flag for whether to plot grid specs in title
for i, reader in enumerate(readers):
    reader.load_time()
    reader.load_grid(grid_specs = grid_specs[i])
    r_bin.append(reader.loading_bin_radius())
    x.append(reader.x)
    y.append(reader.y)
    z.append(reader.z)
    time.append(reader.time)
    nx[:, i] = reader.nx
    lx[:, i] = reader.lx
    nt[i] = reader.nt
    if i == 0:
        nz = reader.nx[2]
        ny = reader.nx[1]
    else:
        nz = np.max([nz, reader.nx[2]])
        ny = np.max([ny, reader.nx[1]])
    if salinity and plot_depths:
        S_value = reader.load_S_temporal_avg('binning_rtz.h5')
        reader.load_equation_of_state()
        bS.append(-g*reader.beta*S_value)

# plotting prep
plot_format()
if plot_depths:
    color_opt, line_opt = comparison_plot_opt(num_cases)

    zp = []
    zneutral = []
    zc = []

for i, reader in enumerate(readers):# Load binning from files
    print(rf"Processing case: {case_names[i]}")
    if plot_depths:
        T_rz = reader.load_binning_var('T')
        S_rz = reader.load_binning_var('S')
        r = reader.loading_bin_contours()
        # calculate buoyancy differences
        S_value = reader.load_S_temporal_avg('binning_rtz.h5')
        reader.load_equation_of_state()
        bS = -g*reader.beta*S_value
        bT = g*reader.alpha*np.mean(T_rz, axis = 0) - g*reader.alpha*reader.T0
        zneutral.append(point(bT-bS, z, f0 = 0, nt = nt[i]))
        # calculate where w = 0 on the centerline
        w_rz = reader.load_binning_var('w')
        w_centerline = w_rz[0, :, :]
        zp.append(point(w_centerline, z, f0 = 0, nt = nt[i]))
        zc.append(point(S_rz[0, :, :], z, f0 = S_value*contour_bound, nt = nt[i]))

############ PLOTTING ############
if plot_depths:
    depth_dir = plot_plume_depths(time, color_opt, fig_folder, case_names, name_uni, lx, zp, zneutral, zc, contour_bound, trend = True)

