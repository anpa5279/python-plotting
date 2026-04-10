import os
import numpy as np
import matplotlib.pyplot as plt
from plotting_functions import plot_ranges
from general_analysis_functions import a2_fluc_mean, ab_fluc_mean
from plotting_comparisons import plot_format, plume_temporal_analysis, mld_temporal_analysis
from data_collection_functions import collect_time_outputs, collect_fields_distributed, collect_temp_and_sal
from dense_plume_analysis import plume_tracer_radius, neutral_buoyancy_loc

# selecting cases to comparse
variations = 'strat' # 'MLD', 'flux', 'strat'
if variations == 'strat':
    folder_names =['beta = default S0 = 0.1 dTdz = 0.005', 'beta = default S0 = 0.1', 'beta = default S0 = 0.1 dTdz = 0.05', 'beta = default S0 = 0.1 dTdz = 0.1'] 
    case_names =[r'dTdz = 0.005', r'dTdz = 0.01', r'dTdz = 0.05', r'dTdz = 0.10']  
    num_cases = len(case_names)
    dTdz = np.array([0.005, 0.01, 0.05, 0.1]) # background temperature gradient in K/m
    mld = 30 * np.ones(num_cases) 
    Sj = 0.1 * np.ones(num_cases) 
elif variations == 'MLD':
    folder_names =['beta = default S0 = 0.1 MLD = 20m', 'beta = default S0 = 0.1', 'beta = default S0 = 0.1 MLD = 40m']
    case_names =[r'MLD = 20m', r'MLD = 30m', r'MLD = 40m']
    num_cases = len(case_names)
    dTdz = 0.01 * np.ones(num_cases) # background temperature gradient in K/m
    mld = np.array([20, 30, 40])
    Sj = 0.1 * np.ones(num_cases) 
elif variations == 'flux':
    folder_names =['beta = default S0 = 0.05', 'beta = default S0 = 0.1', 'beta = default S0 = 0.15', 'beta = default S0 = 0.2']
    case_names =[r'F$^{\text{C}} = -5.0*10^{-5}$', r'F$^{\text{C}} = -1.0*10^{-4}$', r'F$^{\text{C}} = -1.5*10^{-4}$', r'F$^{\text{C}} = - 2.0*10^{-4}$']
    num_cases = len(case_names)
    dTdz = 0.01 * np.ones(num_cases) # background temperature gradient in K/m
    mld = 30 * np.ones(num_cases) 
    Sj = np.array([0.05, 0.1, 0.15, 0.2]) # 
# Set up folder and simulation parameters
universal_folder = '/Users/annapauls/Library/CloudStorage/OneDrive-UCB-O365/CU-Boulder/TESLa/Carbon Sequestration/Simulations/Oceananigans/NBP/salinity and temperature/with noise'
fig_folder = os.path.join(universal_folder, 'comparison figures/contour 0.15/')
name_uni ='contour-0.15-dTdz'
folders = []
for name in folder_names:
    folders.append(os.path.join(universal_folder, name))
output_folder = universal_folder

# flags for what to plot
plume_analysis_plot = True
mld_analysis_plot = False
ND = True

# flags for how to read data
with_halos = False
salinity = True


# physical parameters
rj = 10 # m, radius of salinity flux circle at the surface
g = 9.80665  # gravity in m/s^2
rho0 = 1026
T0 = 25
S0 = 0 
wp = 0.001
F_s = np.dot(Sj, wp)

# collecting model informations for all cases
t_save = []
mld_idx = []
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
    time, t_save_temp, nx, hx, lx, x, y, z, xf, yf, zf, dx, visc, diff = collect_time_outputs(fid, Nranks, False)

    if salinity:
        alpha, beta = collect_temp_and_sal(fid, salinity)
    else:
        alpha = collect_temp_and_sal(fid, salinity)
    t_save.append(t_save_temp)

centerline_index = np.zeros((3, nx[2])).astype(int)
centerline_index[0, :] = nx[0]//2 - 1
centerline_index[1, :] = nx[1]//2 - 1
centerline_index[2, :] = np.arange(nx[2]).astype(int)

S_contour = np.zeros(num_cases)
w_contour = np.zeros(num_cases)
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
    time, t_save_temp, nx, hx, lx, x, y, z, xf, yf, zf, dx, visc, diff = collect_time_outputs(fid, Nranks, False)
    
    if salinity:
        alpha, beta = collect_temp_and_sal(fid, salinity)
    else:
        alpha = collect_temp_and_sal(fid, salinity)
    t_save.append(t_save_temp)
    centerline_index = np.zeros((3, nx[2])).astype(int)
    center_xy_loc = np.zeros((3, nx[2]))
    center_xy_loc[0, :] = lx[0]/2
    center_xy_loc[1, :] = lx[1]/2
    center_xy_loc[2, :] = z
    centerline_index[0, :] = nx[0]//2 - 1
    centerline_index[1, :] = nx[1]//2 - 1
    centerline_index[2, :] = np.arange(nx[2]).astype(int)
    nt = len(t_save_temp)
    n = 0.0
    S_sum = 0.0
    w_sum = 0.0
    for it in range(10, nt):

        # Load data from files
        u, v, w, T, S, Pdynamic, Pstatic = collect_fields_distributed(Nranks, folder, dtn, t_save[i][it], hx, nx, True, salinity, with_halos)
        wc = 0.5 * (w[..., :-1] + w[..., 1:])
        w_sum += np.mean(w[centerline_index[0, :], centerline_index[1, :], :])
        b = g*alpha*(T - T0) - g*beta*(S - S0)
        b_avg = np.mean(b, axis=(-3, -2))
        wc_avg = np.mean(wc, axis=(-3, -2))
        bw_fluc, bw_fluc_avg = ab_fluc_mean(b, wc, b_avg, wc_avg)
        bw_idx = np.where(bw_fluc_avg==np.max(bw_fluc_avg))[0][0]
        S_sum += np.mean(S[centerline_index[0, :], centerline_index[1, :], bw_idx])
        n += 1
    S_contour[i] = S_sum/n
    w_contour[i] = w_sum/n
    print(f"Case {case_names[i]}: w_contour = {w_contour[i]}, S_contour = {S_contour[i]}")