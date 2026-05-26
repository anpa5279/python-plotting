import os
import numpy as np

from reader import OceananigansData
from physics import buoyancy, rms
from diagnostics import comparison_info
from plotting_general import plot_format, plot_ranges, create_video, plot_variable_vert_slice, plot_turb_stats_bin, comparison_plot_opt

# flags for what to plot
video = True
plot_rz_plane = True
plot_turb_stats = True

# flags for how to read data
with_halos = False
closure = False
salinity = True
stokes = False

contour_bound = 0.05
name_uni = f'contour-{contour_bound:.2f}'
universal_folder = '/Users/annapauls/Documents/TESLa /Simulations/Oceananigans/dense plume/salinity and temperature/no noise circle inlet/'

# selecting cases to compare
variations = 'else' # 'MLD', 'flux', 'strat', 'all', 'length', 'WENO', 'vertical resolution', 'else'
if variations != 'else':
    cases_info = comparison_info(variations, universal_folder = universal_folder)
else:
    folder_names = ['proposed resolution/S0 = 0.1 dTdz = 0.01 MLD = 70', 'Lz = 160m/S0 = 0.1 dTdz = 0.01 MLD = 70']
    num_cases = len(folder_names)
    fig_folder = os.path.join(universal_folder, 'comparison figures', '96m vs 160m' + ' comparison figures', 'binning', 'MLD = 70m')
    case_names =[r'L$_z = 96$m', r'L$_z = 160$m']#r'$\Delta z = 0.5$m', r'$\Delta z = 0.375$m'#[r'F$_{\text{C}} = -1.0\cdot 10^{-4}$, MLD = 60m, dTdz = 0.01', r'F$_{\text{C}} = -1.0\cdot 10^{-4}$, MLD = 70m, dTdz = 0.01', r'F$_{\text{C}} = - 2.0\cdot 10^{-4}$, MLD = 60m, dTdz = 0.01']
    cases_info = {
            "folder_names": folder_names,
            "fig_folder": fig_folder,
            "case_names": case_names,
            "num_cases": num_cases,
            "dTdz": 0.01*np.ones(num_cases),
            "mld": np.array([70, 70]),
        }

dTdz = cases_info['dTdz']
case_names = cases_info['case_names']
num_cases = cases_info['num_cases']
fig_folder = cases_info['fig_folder']
mld = cases_info['mld']

readers = []
z = []
nx = []
lx = []
for i, name in enumerate(cases_info["folder_names"]):
    folder = os.path.join(universal_folder, name)
    readers.append(OceananigansData(folder, salinity = salinity))
    readers[-1].load_grid()
    readers[-1].load_time()
    z.append(readers[-1].z)
    nx.append(readers[-1].nx)
    lx.append(readers[-1].lx)
    if i == 0:
        nt = readers[-1].nt
        nz = readers[-1].nx[2]
    else:
        nt = np.min([nt, readers[-1].nt])
        nz = np.max([nz, readers[-1].nx[2]])

# physical parameters
rho0 = 1026
T0 = 25
S0 = 0 
# video or not setup
if video:
    time = readers[0].time

# plotting prep
if plot_turb_stats:
    color_opt, line_opt = comparison_plot_opt(num_cases)
plot_format()
var_names = ['Tracer', 'Temperature', 'Perturbed Temperature', 'Horizontal Velocity', 'Vertical Velocity', 'Perturbed Buoyancy'] 
range_names = ['Tracer', 'T', 'T_fluc', 'u', 'w', 'b_fluc'] 
variable_dir = {}

S_tol = 10**(-6)
ranges = plot_ranges(lz = 96, mld = np.max(mld), rho0 = rho0, T0 = T0, dTdz = np.max(dTdz), C_tol = S_tol)
ranges['Tracer'] = [S_tol, 0.1]
ranges['T'] = [T0-1.0, T0 + 0.01]
ranges['w'] = [-1.5*10**(-1), 1.5*10**(-1)]
ranges['u'] = [-1.0*10**(-2), 1.0*10**(-2)]
ranges['b_fluc'] = [-5.0*10**(-4), 5.0*10**(-4)]
ranges['T_fluc'] = [-0.3, 0.3]
ranges['bw_fluc'] = [-2.0*10**(-7), 2.0*10**(-7)]
ranges['restress'] = [-2*10**(-5), 2*10**(-5)]
ranges['U_rms'] = [0, 4*10**(-3)]
ranges['Tw_fluc'] = [-1.6*10**(-4), 1.6*10**(-4)]
ranges['Cw'] = [-3.5*10**(-5), 3.5*10**(-5)]
ranges['b_avg'] = [-1.0*10**(-3), 1.0*10**(-5)]
for i, reader in enumerate(readers):
    if plot_rz_plane:
        S_n = np.empty((nt, num_cases, nx[i][0]//2, nx[i][-1]))
        T_n = np.empty((nt, num_cases, nx[i][0]//2, nx[i][-1]))
        ur_n = np.empty((nt, num_cases, nx[i][0]//2, nx[i][-1]))
        w_n = np.empty((nt, num_cases, nx[i][0]//2, nx[i][-1]))
        b_fluc_n = np.empty((nt, num_cases, nx[i][0]//2, nx[i][-1]))
        T_fluc_n = np.empty((nt, num_cases, nx[i][0]//2, nx[i][-1]))

    if plot_turb_stats:
        u_rms = np.empty((nt, num_cases, nx[i][-1]))
        w_rms = np.empty((nt, num_cases, nx[i][-1]))
        uw = np.empty((nt, num_cases, nx[i][-1]))
        b_avg = np.empty((nt, num_cases, nx[i][-1]))
        bu_fluc_avg = np.empty((nt, num_cases, nx[i][-1]))
        bw_fluc_avg = np.empty((nt, num_cases, nx[i][-1]))
        Tu = np.empty((nt, num_cases, nx[i][-1]))
        Tw = np.empty((nt, num_cases, nx[i][-1]))
        Cu = np.empty((nt, num_cases, nx[i][-1]))
        Cw = np.empty((nt, num_cases, nx[i][-1]))
    # Load binning from files
    r, z, time, S_rz, T_rz, ur_rz, w_rz = reader.load_binning()
    if plot_rz_plane:
        # plane slices to save for plotting
        S_rz[S_rz < S_tol] = S_tol
        S_n[:, i, :, :] = S_rz.transpose(2, 0, 1)
        T_n[:, i, :, :] = T_rz.transpose(2, 0, 1)
        ur_n[:, i, :, :] = ur_rz.transpose(2, 0, 1)
        w_n[:, i, :, :] = w_rz.transpose(2, 0, 1)
    if plot_turb_stats:
        bs = buoyancy(reader, T_rz, S = S_rz)
        b = bs['b']
        # Average over the radial dimension (axis=0), keeping time and z
        u_avg = np.mean(ur_rz, axis=0)  # shape: (nz, nt) or (nr_bins, nt) etc.
        w_avg = np.mean(w_rz, axis=0)
        b_avg_temp = np.mean(b, axis=0)
        b_fluc = b - b_avg_temp

        u_rms[:, i, :] = np.mean(np.sqrt((ur_rz - u_avg)**2), axis=0).T
        w_rms[:, i, :] = np.mean(np.sqrt((w_rz - w_avg)**2), axis=0).T
        uw[:, i, :]     = np.mean((ur_rz - u_avg) * (w_rz - w_avg), axis=0).T
        bu_fluc_avg[:, i, :] = np.mean(b_fluc * ur_rz, axis=0).T
        bw_fluc_avg[:, i, :] = np.mean(b_fluc * w_rz, axis=0).T
        b_avg[:, i, :] = b_avg_temp.T
        Cu[:, i, :] = np.mean(S_rz * ur_rz, axis=0).T
        Cw[:, i, :] = np.mean(S_rz * w_rz, axis=0).T

############ PLOTTING ############
if plot_rz_plane:
    for it, t in enumerate(time):
        variables = [S_n[it, :, :, :], T_n[it, :, :, :], ur_n[it, :, :, :], w_n[it, :, :, :]] 
        colorbar_labels = [r"g/kg", r"$^\circ$C", r"m/s", r"m/s", r"m/s$^2$"]
        cmaps = ['viridis', 'viridis', 'RdBu_r', 'RdBu_r']
        for n, var in enumerate(variables): #time, it, ranges, fig_folder, lx, hor, z
            variable_dir[var_names[n]] = plot_variable_vert_slice(t, it, ranges, fig_folder, lx[-1], r, z, var, case_names, var_names[n], range_names[n], colorbar_label = colorbar_labels[n], cmap = cmaps[n], plane='binning')
if plot_turb_stats:
    for it, t in enumerate(time):
        turb_plot = plot_turb_stats_bin(t, it, ranges, color_opt, fig_folder, case_names, z, u_rms[it], w_rms[it], uw[it], b_avg[it], bu_fluc_avg[it], bw_fluc_avg[it], Tu[it], Tw[it], Cu[it], Cw[it])
print("All frames created.")
# creating videos
if video:
    if plot_rz_plane:
        for n, name in enumerate(var_names):
            create_video(variable_dir[var_names[n]], fig_folder, 'binning', name)
    if plot_turb_stats:
        create_video(turb_plot, fig_folder, 'binning', 'turb_stats')
