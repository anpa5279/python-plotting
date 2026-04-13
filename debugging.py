import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import imageio.v2 as imageio
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

from plotting_functions import plot_ranges, create_video
from general_analysis_functions import a2_fluc_mean, ab_fluc_mean
from plotting_comparisons import plot_format, plume_horizontal_spatial_plot, plume_vertical_spatial_plot
from data_collection_functions import collect_time_outputs, collect_fields_distributed, collect_temp_and_sal
from dense_plume_analysis import plume_tracer_radius

# Set up folder and simulation parameters
universal_folder = '/Users/annapauls/Library/CloudStorage/OneDrive-UCB-O365/CU-Boulder/TESLa/Carbon Sequestration/Simulations/Oceananigans/NBP/salinity and temperature/no noise small square inlet'
folder_names =['beta = default S0 = 0.1 dTdz = 0.005 MLD = 60', 'beta = default S0 = 0.1 dTdz = 0.01 MLD = 60', 'beta = default S0 = 0.1 dTdz = 0.05 MLD = 60', 'beta = default S0 = 0.1 dTdz = 0.1 MLD = 60'] 
case_names =[r'dT/dz = 0.005', r'dT/dz = 0.01', r'dT/dz = 0.05', r'dT/dz = 0.10'] 

num_cases = len(case_names)

# flags for how to read data
with_halos = False
closure = False
salinity = True

# physical parameters
rj = 10 # m, radius of salinity flux circle at the surface
g = 9.80665  # gravity in m/s^2
dTdz = np.array([0.005, 0.01, 0.05, 0.1]) # background temperature gradient in K/m
rho0 = 1026
mld = 60 * np.ones(num_cases) # 
T0 = 25.0
S0 = 0 
wp = 0.001
Sj = 0.1 * np.ones(num_cases) # np.array([0.05, 0.1, 0.15, 0.2])# 
F_s = np.dot(Sj, wp)

color_opt, line_opt = plot_format(num_cases)
# font for plotting 
plt.rcParams['font.family'] = 'serif' # or 'sans-serif' or 'monospace'
plt.rcParams['font.serif'] = 'cmr10'
plt.rcParams['font.sans-serif'] = 'cmss10'
plt.rcParams['font.monospace'] = 'cmtt10'
plt.rcParams["axes.formatter.use_mathtext"] = True 
plt.rcParams['font.size'] = 10
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'DejaVu Serif'
plt.rcParams['mathtext.it'] = 'DejaVu Serif:italic'
plt.rcParams['mathtext.bf'] = 'DejaVu Serif:bold'

# collecting model informations for all cases
t_save = []

folders = []
for name in folder_names:
    folders.append(os.path.join(universal_folder, name))

for i, folder in enumerate(folders):
    # List JLD2 files
    dtn = [f for f in os.listdir(folder) if (f.endswith('.jld2') and f.startswith('fields'))]
    Nranks = len(dtn)
    if Nranks > 1:
        dtn = []
        for file in np.arange(Nranks):
            dtn.append(f'fields_rank{file}.jld2')
    # Read model information
    fid = os.path.join(folder, dtn[0])
    time, t_save_temp, nx, hx, lx, x, y, z, xf, yf, zf, dx, visc, diff = collect_time_outputs(fid, Nranks, stokes=False, closure=closure)
    if salinity:
        alpha, beta = collect_temp_and_sal(fid, salinity)
    else:
        alpha = collect_temp_and_sal(fid, salinity)
    t_save.append(t_save_temp)

N2 = g*alpha*dTdz

centerline_index = np.zeros((3, nx[2])).astype(int)
center_xy_loc = np.zeros((3, nx[2]))
center_xy_loc[0, :] = lx[0]/2
center_xy_loc[1, :] = lx[1]/2
center_xy_loc[2, :] = z
centerline_index[0, :] = nx[0]//2 - 1
centerline_index[1, :] = nx[1]//2 - 1
centerline_index[2, :] = np.arange(nx[2]).astype(int)

nt = len(t_save_temp)

dbdz_fixed = np.zeros((nt, num_cases))
dbdz_N2_half = np.zeros((nt, num_cases))
dbdz_N2_tenth = np.zeros((nt, num_cases))
depth_fixed = np.zeros((nt, num_cases))
depth_N2_half = np.zeros((nt, num_cases))
depth_N2_tenth = np.zeros((nt, num_cases))
for it in range(nt):
    for i, folder in enumerate(folders):
        # Load data from files
        u, v, w, T, S, Pdynamic, Pstatic = collect_fields_distributed(Nranks, folder, dtn, t_save[i][it], hx, nx, True, salinity, with_halos)

        b = g*alpha*(T - T0) - g*beta*(S - S0)
        b_T = g*alpha*(T - T0)
        b_S = - g*beta*(S - S0)

        b_avg = np.mean(b, axis=(-3, -2))

        dbdz = np.gradient(b_avg, z)
        fixed_tol = dbdz <= (5.0*10**(-7))
        if np.any(fixed_tol):
            fixed_idx = np.min(np.where(fixed_tol))
        else:
            fixed_idx = nx[2] - 1
        dbdz_fixed[it, i] = dbdz[fixed_idx]
        depth_fixed[it, i] = z[fixed_idx]

        N2_half_tol = dbdz <= N2[i]*0.5
        if np.any(N2_half_tol):
            N2_half_idx = np.min(np.where(N2_half_tol))
        else:
            N2_half_idx = nx[2] - 1
        dbdz_N2_half[it, i] = dbdz[N2_half_idx]
        depth_N2_half[it, i] = z[N2_half_idx]

        N2_tenth_tol = dbdz <= N2[i]*0.1
        if np.any(N2_tenth_tol):
            N2_tenth_idx = np.min(np.where(N2_tenth_tol))
        else:
            N2_tenth_idx = nx[2] - 1
        dbdz_N2_tenth[it, i] = dbdz[N2_tenth_idx]
        depth_N2_tenth[it, i] = z[N2_tenth_idx]
    print(f"Completed time step {it+1}/{nt}")

tols = [r'5 $\times$ 10$^{-7}$', '1/2 N2 Threshold', '1/10 N2 Threshold'] 
labels = case_names + tols
line_opt = ['solid'] * num_cases + line_opt[0:len(tols)]
color_opt += ['black'] * len(tols)
case_handles = [Line2D([0], [0], color=color_opt[i], linestyle=line_opt[i], label=labels[i]) for i in range(len(labels))]
empty = Line2D([], [], linestyle='none', label='')
case_handles.append(empty)  
nrow = 2
grid = np.array(case_handles, dtype=object).reshape(nrow, -1)
case_handles = grid.T.flatten().tolist()
gridspec_kw={'height_ratios': [1, 0.05]}
fig, ax = plt.subplots(2, 2, dpi=120, figsize=(20, 12), gridspec_kw=gridspec_kw)
for a in ax[1, :]:
    a.remove()
fig.legend(handles=case_handles,
        loc='lower center',
        ncols=num_cases,
        bbox_to_anchor=(0.52, 0.005))
ax1 = ax[0, 0] # depth through time
ax2 = ax[0, 1] # dbdz through time
ax1.set_title("MLD estimates", fontsize=18)
ax2.set_title("Buoyancy Gradient at Depth", fontsize=18)
ax1.set_ylabel("Depth [m]", fontsize=16)
ax2.set_ylabel(r"$\partial b / \partial z$ [s$^{-2}$]", fontsize=16)
ax1.set_xlabel("Time [days]", fontsize=16)
ax2.set_xlabel("Time [days]", fontsize=16)
for i in range(num_cases):
    ax1.plot(time/60/60/24, depth_fixed[:, i], color = color_opt[i])
    ax1.plot(time/60/60/24, depth_N2_half[:, i], color = color_opt[i], linestyle='--')
    ax1.plot(time/60/60/24, depth_N2_tenth[:, i], color = color_opt[i], linestyle=':')
    ax2.semilogy(time/60/60/24, dbdz_fixed[:, i], color = color_opt[i])
    ax2.semilogy(time/60/60/24, dbdz_N2_half[:, i], color = color_opt[i], linestyle='--')
    ax2.semilogy(time/60/60/24, dbdz_N2_tenth[:, i], color = color_opt[i], linestyle=':')
ax2.set_ylim(1e-8, 1e-4)
plt.show()


