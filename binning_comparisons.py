import os
import numpy as np

from reader import OceananigansData
from diagnostics import comparison_info
from plotting_functions import plot_format, plot_ranges, create_video, plot_variable_vert_slice

# flags for what to plot
video = True

# flags for how to read data
with_halos = False
closure = False
salinity = True
stokes = False

contour_bound = 0.05
name_uni = f'contour-{contour_bound:.2f}'
universal_folder = '/Users/annapauls/Documents/TESLa /Simulations/Oceananigans/dense plume/salinity and temperature/no noise circle inlet/'

# selecting cases to compare
variations = 'all' # 'MLD', 'flux', 'strat', 'all', 'length', 'else'
cases_info = comparison_info(variations, universal_folder)
mld = cases_info['mld']
dTdz = cases_info['dTdz']
F_s = cases_info['F_s']
case_names = cases_info['case_names']
num_cases = cases_info['num_cases']
fig_folder = cases_info['fig_folder']
os.makedirs(fig_folder, exist_ok=True)

readers = []
for name in cases_info["folder_names"]:
    folder = os.path.join(universal_folder, name)
    readers.append(OceananigansData(folder))

readers[0].load_grid()
readers[0].load_time()
z = readers[0].z
nx = readers[0].nx
lx = readers[0].lx
lx = [lx[0]/2, lx[-1]]

nz = np.max(nx[:][2])
x = readers[0].x
y = readers[0].y

# physical parameters
rho0 = 1026
T0 = 25
S0 = 0 

# video or not setup
if video:
    nt = np.arange(0, readers[0].nt)
    time = readers[0].time

# plotting prep
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

S_plane = np.empty((readers[0].nt, num_cases, nx[0]//2, nx[2]))
T_plane = np.empty((readers[0].nt, num_cases, nx[0]//2, nx[2]))
ur_plane = np.empty((readers[0].nt, num_cases, nx[0]//2, nx[2]))
w_plane = np.empty((readers[0].nt, num_cases, nx[0]//2, nx[2]))
b_fluc_plane = np.empty((readers[0].nt, num_cases, nx[0]//2, nx[2]))
T_fluc_plane = np.empty((readers[0].nt, num_cases, nx[0]//2, nx[2]))
for i, reader in enumerate(readers):
    # Load binning from files
    r, z, time, S_rz, T_fluc_rz, T_rz, ur_rz, w_rz, b_fluc_rz = reader.load_binning()

    # plane slices to save for plotting
    S_rz[S_rz < S_tol] = S_tol
    S_plane[:, i, :, :] = S_rz.transpose(2, 0, 1)
    #T_plane[:, i, :, :] = T_rz.transpose(2, 0, 1)
    #T_fluc_plane[:, i, :, :] = T_fluc_rz.transpose(2, 0, 1)
    #ur_plane[:, i, :, :] = ur_rz.transpose(2, 0, 1)
    #w_plane[:, i, :, :] = w_rz.transpose(2, 0, 1)
    #b_fluc_plane[:, i, :, :] = b_fluc_rz.transpose(2, 0, 1)

############ PLOTTING ############
for it, t in enumerate(time):
    variables = [S_plane[it, :, :, :], ]#T_plane[it, :, :, :], T_fluc_plane[it, :, :, :], ur_plane[it, :, :, :], w_plane[it, :, :, :], b_fluc_plane[it, :, :, :]] 
    colorbar_labels = [r"g/kg", r"$^\circ$C", r"$^\circ$C", r"m/s", r"m/s", r"m/s$^2$"]
    cmaps = ['viridis', 'viridis', 'RdBu_r', 'RdBu_r', 'RdBu_r', 'RdBu_r']
    for n, var in enumerate(variables):
        variable_dir[var_names[n]] = plot_variable_vert_slice(time[it], it, ranges, fig_folder, lx, r, z, var, case_names, var_names[n], range_names[n], colorbar_label = colorbar_labels[n], cmap = cmaps[n], plane='binning')
print("All frames created.")
# creating videos
if video:
    for n, name in enumerate(var_names):
        create_video(variable_dir[var_names[n]], fig_folder, 'binning', name)
