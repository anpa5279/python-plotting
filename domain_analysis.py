import os
import numpy as np
import h5py
from scipy.optimize import curve_fit

from reader import OceananigansData
from dense_plume import PlumeAnalysis
from diagnostics import comparison_info
from interpolation import point
from physics import buoyancy
from plotting_general import plot_format, comparison_plot_opt
from plotting_analysis import plot_r_at_depth_in_time

def func(x, a, b):
    return (x - a)**b
# flags for how to read data
with_halos = False
closure = False
salinity = True
stokes = False

contours = np.array([0.001, 0.005, 0.01, 0.05])
universal_folder = '/Users/annapauls/Documents/TESLa /Simulations/Oceananigans/dense plume/salinity and temperature/no noise circle inlet/'#/Lz = 160m'#resolution testing'#vertical domain increase/dTdz = 0.01'
#harddrive: '/Volumes/Anna External/Oceananigans/dense plume with stratification/salinity and temperature /no noise circle inlet/'#

# selecting cases to compare
variations = 'else' # 'MLD', 'flux', 'strat', 'all', 'length', 'WENO', 'resolution', 'one case', 'else'
if variations != 'else' and variations != 'one':
    cases_info = comparison_info(variations, universal_folder = universal_folder)
    case_names = cases_info['case_names']
    num_cases = cases_info['num_cases']
    fig_folder = cases_info['fig_folder']
    folder_names = cases_info['folder_names']
elif variations == 'one':
    folder_names = ['Lz = 160m/S0 = 0.1 dTdz = 0.01 MLD = 70',]
    fig_folder =os.path.join(universal_folder, folder_names[0], 'plotting outputs')
    os.makedirs(fig_folder, exist_ok=True)
    num_cases = len(folder_names)
    case_names = [r'L$_z = 160$m and MLD  = 70m']
    num_cases = 1
elif variations == 'else':
    folder_names = ['Lz = 160m/S0 = 0.1 dTdz = 0.01 MLD = 60', 'Lz = 160m/S0 = 0.1 dTdz = 0.01 MLD = 70', 'Lz = 160m/S0 = 0.2 dTdz = 0.01 MLD = 60']
    num_cases = len(folder_names)
    fig_folder = os.path.join(universal_folder, 'comparison figures', 'Lz = 160m' + ' comparison figures', 'binning')
    os.makedirs(fig_folder, exist_ok=True)
    case_names =[r'F$_{\text{C}} = -1.0\cdot 10^{-4}$, MLD = 60m, dTdz = 0.01', r'F$_{\text{C}} = -1.0\cdot 10^{-4}$, MLD = 70m, dTdz = 0.01', r'F$_{\text{C}} = - 2.0\cdot 10^{-4}$, MLD = 60m, dTdz = 0.01']#[r'L$_z = 96$m', r'L$_z = 160$m']#r'$\Delta z = 0.5$m', r'$\Delta z = 0.375$m'#

color_opt, _ = comparison_plot_opt(len(contours))

readers = []
for folder in folder_names:
    folder = os.path.join(universal_folder, folder)
    readers.append(OceananigansData(folder, salinity = salinity))

# physical parameters
x0 = 0.0
y0 = 0.0
rj = 5 # m, radius of salinity flux circle at the surface
g = 9.80665  # gravity in m/s^2

# collecting model information for all cases
lz = 0
nx = []
bS = []
dense_plume = []
time = []

for i, reader in enumerate(readers):

    reader.load_grid()
    lz = np.max((lz, reader.lx[-1]))
    reader.load_time()
    time.append(reader.time)

    S_value, w_value = reader.load_contour_temporal_averages('interp_temporal_averages.h5')
    reader.load_equation_of_state()
    bS.append(-g*reader.beta*S_value)
lz = []
r_scale = []
neutral_depths = []
r_contour = []
r_maximum = []
where_max = []
best_fit = []
best_fit_max = []
fit_exp = []
for i, reader in enumerate(readers):
    start = 7
    r_max = np.zeros([len(time[i]) - start, len(contours)])
    z_max = np.zeros([len(time[i]) - start, len(contours)])
    neutral = np.zeros([len(time[i]) - start, len(contours)])
    r_c = np.zeros([len(time[i]) - start, len(contours)])
    params_neutral = np.zeros([2, len(contours)])
    params_max = np.zeros([2, len(contours)])
    r_n_calc = np.zeros([len(time[i]) - start, len(contours)])
    r_max_calc = np.zeros([len(time[i]) - start, len(contours)])
    for j, contour in enumerate(contours):
        # Load data from files
        fname = os.path.join(reader.folder, 'binning', 'binning_rtz.h5')
        with h5py.File(fname, 'r') as f:
            r_profile = f[f'r given contour/contour = {contour}'][()]
            z = f['ccc/dimensions/z'][()]
            S_rz = f['ccc/S_rz'][()]
            T_rz = f['ccc/T_rz'][()]
            r_bin = f['ccc/dimensions/r_bin'][()]
        # calculate buoyancy differences
        bT = np.mean(g*reader.alpha*np.mean(T_rz, axis = 2) - g*reader.alpha*reader.T0, axis = 0)

        # calculate neutral buoyancy depth and relative r
        for it in range(start, len(reader.t_save)):
            if it == 0:
                r_max[it-start, j] = 0
                z_max[it-start, j] = 0
                neutral[it-start, j] = point(bT-bS[i], z, f0 = 0)
                r_c[it-start, j] = 0
            else:
                r_max[it-start, j] = r_profile[:, it].max()
                z_temp = z[r_profile[:, it] == r_max[it-start, j]]
                nmax = np.size(z_temp)
                if np.size(z_temp) > 1:
                    z_temp = z_temp[nmax // 2]
                z_max[it-start, j] = z_temp
                neutral[it-start, j] = point(bT-bS[i], z, f0 = 0)
                r_c[it-start, j] = point(r_profile[:, it], z, z0 = neutral[it-start, j])
        params_neutral[:, j], _ = curve_fit(func, time[i][start:], r_c[:, j], p0=[0, 0.5], bounds=([-np.inf, 0], [time[i][start], 2]))
        params_max[:, j], _ = curve_fit(func, time[i][start:], r_max[:, j], p0=[0, 0.5], bounds=([-np.inf, 0], [time[i][start], 2]))
        r_n_calc[:, j] = func(time[i][start:], *params_neutral[:, j]) #- (params_neutral[0, j])**params_neutral[1, j]
        r_max_calc[:, j] = func(time[i][start:], *params_max[:, j]) #- (params_max[0, j])**params_max[1, j]
    lz.append(np.min(z_max))
    lz.append(np.max(z_max))
    r_scale.append(np.min(r_c))
    r_scale.append(np.max(r_c))
    r_scale.append(np.min(r_max))
    r_scale.append(np.max(r_max))
    time[i] = time[i][start:]
    best_fit.append(r_n_calc)
    best_fit_max.append(r_max_calc)
    fit_exp.append([params_neutral[1, :], params_max[1, :]])
    neutral_depths.append(neutral)
    r_contour.append(r_c)
    r_maximum.append(r_max)
    where_max.append(z_max)
    lz = [np.min(lz), np.max(lz)]
    r_scale = [np.min(r_scale), np.max(r_scale)]
############ PLOTTING ############
plot_format(fontsize = 10)
plot_r_at_depth_in_time(color_opt, fig_folder, case_names, time, r_contour, contours, neutral_depths, r_maximum, where_max, lz, r_scale, [best_fit, best_fit_max], fit_exp)