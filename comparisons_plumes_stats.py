import os
import numpy as np
import matplotlib.pyplot as plt

from reader import OceananigansData
from diagnostics import comparison_info
from interpolation import point
from plotting_general import plot_format, plot_ranges, comparison_plot_opt, create_video
from plotting_lines import plot_plume_depths, plot_plume_vertical_spatial

# flags for what to plot
plot_depths = False
plot_z_1d = True
video = True

# flags for how to read data
with_halos = False
salinity = True

contour = 0.001
name_uni = f'contour-{contour:.4f}'
universal_folder = '/Users/annapauls/Documents/TESLa /Simulations/Oceananigans/dense plume/salinity and temperature/version109/res testing/square inlet/open BC '

# selecting cases to compare
variations = 'vertical resolution' # 'MLD', 'flux', 'strat', 'all', 'vertical length', 'Lz160m','WENO', 'vertical resolution', 'horizontal resolution', 'AR=1', 'else'
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
grid_specs = False*np.ones(num_cases)
grid_specs[2] = True # flag for whether to plot grid specs in title
for i, folder in enumerate(folder_names):
    folder = os.path.join(universal_folder, folder)
    readers.append(OceananigansData(folder, salinity = salinity, grid_specs = grid_specs[i]))
# physical parameters
x0 = 0.0
y0 = 0.0
rj = 5 # m, radius of salinity flux circle at the surface
g = 9.80665  # gravity in m/s^2
T0 = 25
# collecting model information for all cases
x = []
y = []
z = []
nx = np.empty((3, num_cases), dtype=object)
lx = np.empty((3, num_cases), dtype=object)
nt = np.empty(num_cases, dtype=int)
time  = []
r_bin = []

if plot_depths:
    bS = []
    zp = []
    zneutral = []
    zc = []
if plot_z_1d:
    b_avg = []
    u_rms = []
    v_rms = []
    w_rms = []
    b_center = []
    bur_fluc_avg = []
    bw_fluc_avg = []
    T_avg = []
    S_avg = []
    T_fluc_center = []
    S_fluc_center = []


for i, reader in enumerate(readers):# Load binning from files
    print(rf"Processing case: {case_names[i]}")
    x.append(reader.x)
    y.append(reader.y)
    z.append(reader.z)
    time.append(reader.t)
    nx[:, i] = reader.nx
    lx[:, i] = reader.lx
    nt[i] = reader.nt

    if i == 0:
        nz = reader.nx[2]
        ny = reader.nx[1]
    else:
        nz = np.max([nz, reader.nx[2]])
        ny = np.max([ny, reader.nx[1]])

    r_bin.append(reader.loading_bin_contours(contour = contour))
    if salinity and plot_depths:
        S_value = reader.load_S_temporal_avg()
        reader.load_equation_of_state()
        bS.append(-g*reader.beta*S_value)
    if plot_depths:
        T_rz = reader.load_binning_var('T')
        S_rz = reader.load_binning_var('S')
        # calculate buoyancy differences
        S_value = reader.load_S_temporal_avg()
        reader.load_equation_of_state()
        bS = -g*reader.beta*S_value
        bT = g*reader.alpha*np.mean(T_rz, axis = 0) - g*reader.alpha*reader.T0
        zneutral.append(point(bT-bS, z, f0 = 0, nt = nt[i]))
        # calculate where w = 0 on the centerline
        w_rz = reader.load_binning_var('w')
        w_centerline = w_rz[0, :, :]
        zp.append(point(w_centerline, z, f0 = 0, nt = nt[i]))
        zc.append(point(S_rz[0, :, :], z, f0 = S_value*contour, nt = nt[i]))
    if plot_z_1d:
        u_rms.append(reader.load_rms('u'))
        v_rms.append(reader.load_rms('v'))
        w_rms.append(reader.load_rms('w'))
        b_avg_loc, b_rms_loc, b_centerline_loc, b_fluc_centerline_loc = reader.load_buoyancy()
        b_avg.append(b_avg_loc)
        b_center.append(b_centerline_loc)
        del b_avg_loc, b_rms_loc, b_centerline_loc, b_fluc_centerline_loc
        bur_fluc_avg.append(reader.load_fluc('bur'))
        bw_fluc_avg.append(reader.load_fluc('bw'))
        T_avg.append(reader.load_averages('T', steps=reader.t_save))
        S_avg.append(reader.load_averages('S', steps=reader.t_save))

        T_center = reader.field_centerline('T')[::100, :]
        T_fluc_center.append(T_center-T_avg[i])
        S_center = reader.field_centerline('S')[::100, :]
        S_fluc_center.append(S_center-S_avg[i])
        del T_center, S_center
############ PLOTTING ############
# plotting prep
plot_format()
color_opt, line_opt = comparison_plot_opt(num_cases)
ranges = plot_ranges(lz = 96, mld = np.max(mld), rho0 = 1026, T0 = T0, dTdz = np.max(dTdz), C_tol = 0)
ranges['Tracer'] =[0, 0.15]
ranges['Tracer_fluc'] = [-0.2, 0.2]
ranges['Tracer_avg'] = [0, 1.2*10**(-3)]
ranges['T'] = [T0-0.7, T0 + 0.05]
ranges['w'] = [-1.5*10**(-1), 1.5*10**(-1)]
ranges['vel_rms'] = [0, 7*10**-3]
ranges['bw_fluc'] = [-5*10**(-8), 5*10**(-8)]

if plot_depths:
    depth_dir = plot_plume_depths(time, color_opt, fig_folder, case_names, name_uni, lx, zp, zneutral, zc, contour, trend = True)
if plot_z_1d:
    buoyancy_dir_z = plot_plume_vertical_spatial(min(time, key=len), ranges, color_opt, fig_folder, case_names, name_uni, -min(lx[-1, :]), z, S_avg, u_rms, v_rms, w_rms, b_avg, b_center, r_bin, bur_fluc_avg, bw_fluc_avg, T_avg, T_fluc_center, S_fluc_center)
# creating videos
if video:
    if plot_z_1d:
        create_video(buoyancy_dir_z, fig_folder, 'binning', 'turb_stats')