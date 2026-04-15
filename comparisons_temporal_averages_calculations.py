import os
import numpy as np
import h5py
from general_analysis_functions import ab_fluc_mean, a2_fluc_mean
from data_collection_functions import collect_time_outputs, collect_fields_distributed, collect_temp_and_sal, collect_contour_val
from dense_plume_analysis import plume_tracer_radius

# Set up folder and simulation parameters
universal_folder = '/Users/annapauls/Library/CloudStorage/OneDrive-UCB-O365/CU-Boulder/TESLa/Carbon Sequestration/Simulations/Oceananigans/NBP/salinity and temperature/no noise circle inlet'
fig_folder = os.path.join(universal_folder, 'comparison figures/contour 0.15/')
name_uni ='temporal_averages'

# selecting cases to comparse
variations = 'all' # 'MLD', 'flux', 'strat', 'all'
if variations == 'strat':
    folder_names =['S0 = 0.1 dTdz = 0.005 MLD = 60', 'S0 = 0.1 dTdz = 0.01 MLD = 60', 'S0 = 0.1 dTdz = 0.05 MLD = 60', 'S0 = 0.1 dTdz = 0.1 MLD = 60'] 
    case_names =[r'dTdz = 0.005', r'dTdz = 0.01', r'dTdz = 0.05', r'dTdz = 0.10']  
    num_cases = len(case_names)
    dTdz = np.array([0.005, 0.01, 0.05, 0.1]) # background temperature gradient in K/m
    mld = 60 * np.ones(num_cases) 
    Sj = 0.1 * np.ones(num_cases) 
elif variations == 'MLD':
    folder_names =['S0 = 0.1 dTdz = 0.01 MLD = 50', 'S0 = 0.1 dTdz = 0.01 MLD = 60', 'S0 = 0.1 dTdz = 0.01 MLD = 70']
    case_names =[r'MLD = 50m', r'MLD = 60m', r'MLD = 70m']
    num_cases = len(case_names)
    dTdz = 0.01 * np.ones(num_cases) # background temperature gradient in K/m
    mld = np.array([50, 60, 70])
    Sj = 0.1 * np.ones(num_cases) 
elif variations == 'flux':
    folder_names =['S0 = 0.05 dTdz = 0.01 MLD = 60', 'S0 = 0.1 dTdz = 0.01 MLD = 60', 'S0 = 0.15 dTdz = 0.01 MLD = 60', 'S0 = 0.2 dTdz = 0.01 MLD = 60']
    case_names =[r'F$_{\text{C}} = -5.0*10^{-5}$', r'F$_{\text{C}} = -1.0*10^{-4}$', r'F$_{\text{C}} = -1.5*10^{-4}$', r'F$_{\text{C}} = - 2.0*10^{-4}$']
    num_cases = len(case_names)
    dTdz = 0.01 * np.ones(num_cases) # background temperature gradient in K/m
    mld = 60 * np.ones(num_cases) 
    Sj = np.array([0.05, 0.1, 0.15, 0.2]) 
elif variations == 'all':
    folder_names =['S0 = 0.1 dTdz = 0.01 MLD = 60', 
                   'S0 = 0.1 dTdz = 0.01 MLD = 50', 'S0 = 0.1 dTdz = 0.01 MLD = 70', 
                   'S0 = 0.1 dTdz = 0.005 MLD = 60', 'S0 = 0.1 dTdz = 0.05 MLD = 60', 'S0 = 0.1 dTdz = 0.1 MLD = 60',
                   'S0 = 0.05 dTdz = 0.01 MLD = 60', 'S0 = 0.15 dTdz = 0.01 MLD = 60', 'S0 = 0.2 dTdz = 0.01 MLD = 60']
    case_names =[r'F$_{\text{C}} = -1.0*10^{-4}$, MLD = 60m, dTdz = 0.01', 
                 r'F$_{\text{C}} = -1.0*10^{-4}$, MLD = 50m, dTdz = 0.01', r'F$_{\text{C}} = -1.0*10^{-4}$, MLD = 70m, dTdz = 0.01', 
                 r'F$_{\text{C}} = -1.0*10^{-4}$, MLD = 60m, dTdz = 0.005', r'F$_{\text{C}} = -1.0*10^{-4}$, MLD = 60m, dTdz = 0.05', r'F$_{\text{C}} = -1.0*10^{-4}$, MLD = 60m, dTdz = 0.1', 
                 r'F$_{\text{C}} = -5.0*10^{-5}$, MLD = 60m, dTdz = 0.01', r'F$_{\text{C}} = -1.5*10^{-4}$, MLD = 60m, dTdz = 0.01', r'F$_{\text{C}} = - 2.0*10^{-4}$, MLD = 60m, dTdz = 0.01']
    num_cases = len(case_names)
    dTdz = np.array([0.01, 
                     0.01, 0.01, 
                     0.005, 0.05, 0.1,
                     0.01, 0.01, 0.01]) # background temperature gradient in K/m
    mld = np.array([60, 
                    50, 70, 
                    60, 60, 60,
                    60, 60, 60])
    Sj = np.array([0.1, 
                   0.1, 0.1, 
                   0.1, 0.1, 0.1,
                   0.05, 0.15, 0.2]) 

folders = []
for name in folder_names:
    folders.append(os.path.join(universal_folder, name))
output_folder = universal_folder

# flags for what to plot
plume_analysis_plot = True
mld_analysis_plot = False
ND = True

# flags for how to write data
with_halos = False
closure = False
salinity = True
plume_stats_only = True

file_name = 'temporal_averages.h5'
percent_contour = 0.05
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
    time, t_save_temp, nx, hx, lx, x, y, z, xf, yf, zf, dx, visc, diff = collect_time_outputs(fid, Nranks, stokes=False, closure=closure)

    if salinity:
        alpha, beta = collect_temp_and_sal(fid, salinity)
    else:
        alpha = collect_temp_and_sal(fid, salinity)
    t_save.append(t_save_temp)

centerline_index = np.zeros((3, nx[2])).astype(int)
centerline_index[0, :] = nx[0]//2 - 1
centerline_index[1, :] = nx[1]//2 - 1
centerline_index[2, :] = np.arange(nx[2]).astype(int)

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
    centerline_index = np.zeros((3, nx[2])).astype(int)
    center_xy_loc = np.zeros((3, nx[2]))
    center_xy_loc[0, :] = lx[0]/2
    center_xy_loc[1, :] = lx[1]/2
    center_xy_loc[2, :] = z
    centerline_index[0, :] = nx[0]//2 - 1
    centerline_index[1, :] = nx[1]//2 - 1
    centerline_index[2, :] = np.arange(nx[2]).astype(int)

    nt = len(t_save_temp)
    if not plume_stats_only:
        n = 0.0
        # initializing arrays to collect temporal averages
        S_sum = 0.0
        w_sum = 0.0
        S_centerline_avg = np.zeros(nx[2])
        T_centerline_avg = np.zeros(nx[2])
        b_centerline_avg = np.zeros(nx[2])
        S_fluc_centerline_avg = np.zeros(nx[2])
        T_fluc_centerline_avg = np.zeros(nx[2])
        b_fluc_centerline_avg = np.zeros(nx[2])
        S_avg = np.zeros(nx[2])
        T_avg = np.zeros(nx[2])
        b_avg = np.zeros(nx[2])
        w_avg = np.zeros(nx[2]+1)
        S_fluc_avg = np.zeros(nx[2])
        T_fluc_avg = np.zeros(nx[2])
        b_fluc_avg = np.zeros(nx[2])
        w_fluc_avg = np.zeros(nx[2]+1)
        Tw_fluc_avg = np.zeros(nx[2])
        Sw_fluc_avg = np.zeros(nx[2])
        bw_fluc_avg = np.zeros(nx[2])
        u_rms = np.zeros(nx[2])
        v_rms = np.zeros(nx[2])
        w_rms = np.zeros(nx[2]+1)
        for it in range(10, nt):
            # Load data from files
            u, v, w, T, S, Pdynamic, Pstatic = collect_fields_distributed(Nranks, folder, dtn, t_save[i][it], hx, nx, True, salinity, with_halos)
            wc = 0.5 * (w[..., :-1] + w[..., 1:])
            b = g*alpha*(T - T0) - g*beta*(S - S0)
            # horizontal averages
            S_avg_temp = np.mean(S, axis=(-3, -2))
            T_avg_temp = np.mean(T, axis=(-3, -2))
            b_avg_temp = np.mean(b, axis=(-3, -2))
            w_avg_temp = np.mean(w, axis=(-3, -2))
            wc_avg_temp = np.mean(wc, axis=(-3, -2))
            bw_fluc, bw_fluc_avg_temp = ab_fluc_mean(b, wc, b_avg_temp, wc_avg_temp)
            Tw_fluc, Tw_fluc_avg_temp = ab_fluc_mean(T, wc, T_avg_temp, wc_avg_temp)
            Sw_fluc, Sw_fluc_avg_temp = ab_fluc_mean(S, wc, S_avg_temp, wc_avg_temp)
            bw_idx = np.where(bw_fluc_avg_temp==np.max(bw_fluc_avg_temp))[0][0]
            w_sum += np.mean(w[centerline_index[0, :], centerline_index[1, :], bw_idx])
            S_sum += np.mean(S[centerline_index[0, :], centerline_index[1, :], bw_idx])
            # calculating rms 
            u_fluc = u - np.mean(u, axis=(-3, -2))
            v_fluc = v - np.mean(v, axis=(-3, -2))
            w_fluc = w - w_avg_temp
            w_fluc_avg_temp, w2_fluc, w2_fluc_avg = a2_fluc_mean(w_fluc)
            u_fluc_avg_temp, u2_fluc, u2_fluc_avg = a2_fluc_mean(u_fluc)
            v_fluc_avg_temp, v2_fluc, v2_fluc_avg = a2_fluc_mean(v_fluc)
            u_rms_temp= u2_fluc_avg**0.5
            v_rms_temp= v2_fluc_avg**0.5
            w_rms_temp= w2_fluc_avg**0.5
            # collecting desired averages
            S_centerline_avg += S[centerline_index[0, :], centerline_index[1, :], centerline_index[2, :]]
            T_centerline_avg += T[centerline_index[0, :], centerline_index[1, :], centerline_index[2, :]]
            b_centerline_avg += b[centerline_index[0, :], centerline_index[1, :], centerline_index[2, :]]
            S_avg += S_avg_temp
            T_avg += T_avg_temp
            b_avg += b_avg_temp
            w_avg += w_avg_temp
            S_fluc_avg += np.mean(S - S_avg_temp, axis=(-3, -2))
            T_fluc_avg += np.mean(T - T_avg_temp, axis=(-3, -2))
            b_fluc_avg += np.mean(b - b_avg_temp, axis=(-3, -2))
            w_fluc_avg += w_fluc_avg_temp
            Tw_fluc_avg += Tw_fluc_avg_temp
            Sw_fluc_avg += Sw_fluc_avg_temp
            bw_fluc_avg += bw_fluc_avg_temp
            S_fluc_centerline_avg += S[centerline_index[0, :], centerline_index[1, :], centerline_index[2, :]] - S_avg_temp
            T_fluc_centerline_avg += T[centerline_index[0, :], centerline_index[1, :], centerline_index[2, :]] - T_avg_temp
            b_fluc_centerline_avg += b[centerline_index[0, :], centerline_index[1, :], centerline_index[2, :]] - b_avg_temp
            u_rms += u_rms_temp
            v_rms += v_rms_temp
            w_rms += w_rms_temp

            n += 1
        S_contour = S_sum/n
        w_contour = w_sum/n
        S_centerline_avg = S_centerline_avg/n
        T_centerline_avg = T_centerline_avg/n
        b_centerline_avg = b_centerline_avg/n
        S_fluc_centerline_avg = S_fluc_centerline_avg/n
        T_fluc_centerline_avg = T_fluc_centerline_avg/n
        b_fluc_centerline_avg = b_fluc_centerline_avg/n
        S_avg = S_avg/n
        T_avg = T_avg/n
        b_avg = b_avg/n
        w_avg = w_avg/n
        S_fluc_avg = S_fluc_avg/n
        T_fluc_avg = T_fluc_avg/n
        b_fluc_avg = b_fluc_avg/n
        w_fluc_avg = w_fluc_avg/n
        Tw_fluc_avg = Tw_fluc_avg/n
        Sw_fluc_avg = Sw_fluc_avg/n
        bw_fluc_avg = bw_fluc_avg/n
        u_rms = u_rms/n
        v_rms = v_rms/n
        w_rms = w_rms/n
        print(f"writing to {folder}")
        # saving temporal averages per case
        file = h5py.File(os.path.join(folder, file_name), 'w')
        file.create_dataset("contour temporal averages/S", data=S_contour)
        file.create_dataset("contour temporal averages/w", data=w_contour)
        file.create_dataset("centerline temporal averages/S", data=S_centerline_avg)
        file.create_dataset("centerline temporal averages/T", data=T_centerline_avg)
        file.create_dataset("centerline temporal averages/b", data=b_centerline_avg)
        file.create_dataset("centerline temporal averages/S'", data=S_fluc_centerline_avg)
        file.create_dataset("centerline temporal averages/T'", data=T_fluc_centerline_avg)
        file.create_dataset("centerline temporal averages/b'", data=b_fluc_centerline_avg)
        file.create_dataset("1D temporal averages/S", data=S_avg)
        file.create_dataset("1D temporal averages/T", data=T_avg)
        file.create_dataset("1D temporal averages/b", data=b_avg)
        file.create_dataset("1D temporal averages/w", data=w_avg)
        file.create_dataset("1D temporal averages/S'", data=S_fluc_avg)
        file.create_dataset("1D temporal averages/T'", data=T_fluc_avg)
        file.create_dataset("1D temporal averages/b'", data=b_fluc_avg)
        file.create_dataset("1D temporal averages/w'", data=w_fluc_avg)
        file.create_dataset("1D temporal averages/T'w'", data=Tw_fluc_avg)
        file.create_dataset("1D temporal averages/S'w'", data=Sw_fluc_avg)
        file.create_dataset("1D temporal averages/b'w'", data=bw_fluc_avg)
        file.create_dataset("1D temporal averages/urms", data=u_rms)
        file.create_dataset("1D temporal averages/vrms", data=v_rms)
        file.create_dataset("1D temporal averages/wrms", data=w_rms)
        file.close()

# calculating plume statistics from temporal averages for all cases
if plume_stats_only:
    for i, folder in enumerate(folders):
        S_value, w_value = collect_contour_val(folder, file_name)
        S_contour = S_value * percent_contour
        rp_profile = np.zeros(nx[2])
        n = 0
        for it in range(10, nt):
            # Load data from files
            u, v, w, T, S, Pdynamic, Pstatic = collect_fields_distributed(Nranks, folder, dtn, t_save[i][it], hx, nx, True, salinity, with_halos)
            rp_profile_temp, temp, temp = plume_tracer_radius(x, y, nx, centerline_index, S, S_contour) # plume analysis
            n += 1
            rp_profile += rp_profile_temp
        rp_profile = rp_profile/n
        print(f"writing to {folder}")
        with h5py.File(os.path.join(folder, file_name), 'a') as file:
            dset_path = f"plume statistics/contour {percent_contour}/plume tracer radius with depth"
            
            if dset_path in file:
                del file[dset_path]   # remove existing dataset
            
            file.create_dataset(dset_path, data=rp_profile)
