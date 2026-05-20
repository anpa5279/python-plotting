import os
import numpy as np
import h5py

from reader import OceananigansData
from dense_plume import PlumeAnalysis
from diagnostics import comparison_info
from interpolation import point
from physics import buoyancy
from plotting_general import plot_format, comparison_plot_opt
from plotting_analysis import plot_r_at_depth_in_time

# flags for how to read data
with_halos = False
closure = False
salinity = True
stokes = False

contours = np.array([0.001, 0.005, 0.01, 0.05])
universal_folder = '/Users/annapauls/Documents/TESLa /Simulations/Oceananigans/dense plume/salinity and temperature/no noise circle inlet/'#/Lz = 160m'#resolution testing'#vertical domain increase/dTdz = 0.01'
#harddrive: '/Volumes/Anna External/Oceananigans/dense plume with stratification/salinity and temperature /no noise circle inlet/'#

# selecting cases to compare
variations = 'one' # 'MLD', 'flux', 'strat', 'all', 'length', 'WENO', 'resolution', 'one case', 'else'
if variations != 'else' and variations != 'one':
    cases_info = comparison_info(variations, universal_folder = universal_folder)
elif variations == 'one':
    folder_names = ['Lz = 160m/S0 = 0.1 dTdz = 0.01 MLD = 70',]
    fig_folder =os.path.join(universal_folder, folder_names[0], 'plotting outputs')
    os.makedirs(fig_folder, exist_ok=True)
    num_cases = len(folder_names)
    cases_info = {
            "folder_names": folder_names,
            "fig_folder": fig_folder,
            "case_names": '',
            "num_cases": num_cases,
        }
elif variations == 'else':
    folder_names = ['proposed resolution/S0 = 0.1 dTdz = 0.01 MLD = 70', 'Lz = 160m/S0 = 0.1 dTdz = 0.01 MLD = 70']
    num_cases = len(folder_names)
    fig_folder = os.path.join(universal_folder, 'comparison figures', '96m vs 160m' + ' comparison figures', 'interpolated', 'MLD = 70m')
    case_names =[r'L$_z = 96$m', r'L$_z = 160$m']#r'$\Delta z = 0.5$m', r'$\Delta z = 0.375$m'#[r'F$_{\text{C}} = -1.0\cdot 10^{-4}$, MLD = 60m, dTdz = 0.01', r'F$_{\text{C}} = -1.0\cdot 10^{-4}$, MLD = 70m, dTdz = 0.01', r'F$_{\text{C}} = - 2.0\cdot 10^{-4}$, MLD = 60m, dTdz = 0.01']
    cases_info = {
            "folder_names": folder_names,
            "fig_folder": fig_folder,
            "case_names": case_names,
            "num_cases": num_cases,
        }

case_names = cases_info['case_names']
num_cases = cases_info['num_cases']
fig_folder = cases_info['fig_folder']

if num_cases > 1:
    color_opt = comparison_plot_opt(num_cases)
else:
    color_opt = 'black'
readers = []
for folder in cases_info["folder_names"]:
    folder = os.path.join(universal_folder, folder)
    readers.append(OceananigansData(folder, salinity = salinity))

# collecting model information for all cases
nx = []
S0 = []
dense_plume = []
time = []

for i, reader in enumerate(readers):

    reader.load_time()
    time.append(reader.time)

    S_value, w_value = reader.load_contour_temporal_averages('interp_temporal_averages.h5')
    S0.append(S_value)

# physical parameters
x0 = 0.0
y0 = 0.0
rj = 5 # m, radius of salinity flux circle at the surface

neutral_depths = []
r_contour = []
r_maximum = []
where_max = []

for f, contour in enumerate(contours):
    for i, reader in enumerate(readers):
        # Load data from files
        fname = os.path.join(reader.folder, 'binning', 'binning_rtz.h5')
        with h5py.File(fname, 'r') as f:
            r_profile = f[f'r given contour/contour = {contour}'][()]
            z = f['ccc/dimensions/z'][()]
            S_rz = f['ccc/S_rz'][()]
            T_rz = f['ccc/T_rz'][()]
            r_bin = f['ccc/dimensions/r_bin'][()]
        # convert temperature and salinity to buoyancy 
        bs = buoyancy(reader, T = T_rz, S = S_rz)
        cylinder = 15
        b_T = bs['b_T'][:cylinder, :, :]
        b_S = bs['b_C'][:cylinder, :, :]
        b_fluc_avg = np.mean(b_S, axis=-3) - np.mean(b_T, axis=-3)

        # calculate neutral buoyancy depth and relative r
        start = 10
        neutral = np.empty(len(time[i]) - start)
        r_c = np.empty(len(time[i]) - start)
        r_max = np.empty(len(time[i]) - start)
        z_max = np.empty(len(time[i]) - start)
        for it in range(start, len(reader.t_save)):
            b_fluc_avg_it = b_fluc_avg[:, it]
            neutral[it-start] = point(b_fluc_avg_it, z, f0 = 0)
            r_c[it-start] = point(r_profile[:, it], z, z0 = neutral[it-start])
            r_max[it-start] = r_profile[:, it].max()
            z_max[it-start] = z[np.where(r_profile[:, it] == r_max[it-start])[0][0]]

        neutral_depths.append(neutral)
        r_contour.append(r_c)
        r_maximum.append(r_max)
        where_max.append(z_max)
neutral_depths = neutral_depths[0]
where_max = where_max[0]
r_maximum = r_maximum[0]

############ PLOTTING ############
plot_format()
outdir = plot_r_at_depth_in_time(color_opt, fig_folder, time[0][start:], r_contour, contours, neutral_depths, r_maximum, where_max)