import os
import numpy as np
import h5py
from general_analysis_functions import ab_fluc_mean, a2_fluc_mean
from data_collection_functions import collect_time_outputs, collect_fields_distributed, collect_temp_and_sal, collect_contour_val, collect_grid
from dense_plume_analysis import plume_tracer_radius

# Set up folder and simulation parameters
folder = '/Users/annapauls/Library/CloudStorage/OneDrive-UCB-O365/CU-Boulder/TESLa/Carbon Sequestration/Simulations/Oceananigans/NBP/salinity and temperature/no noise circle inlet/vertical domain increase/dTdz = 0.01/nz = 192 z = 240 m'
output_folder = os.path.join(folder, "plotting outputs") 
name = ""

# flags for how to write data
with_halos = False
closure = False
stokes = False
salinity = True
plume_stats_only = False

file_name = 'temporal_averages.h5'
percent_contour = 0.05

# physical parameters
#nums = re.findall(r' -?\d*\.?\d+', folder)
g = 9.80665  # gravity in m/s^2
T0 = 25 
S0 = 0 
Sj = 0.1#float(nums[-3]) # salinity of the source 
wp = 0.001
F_s = Sj*wp

# List JLD2 files
dtn = [f for f in os.listdir(folder) if (f.endswith('.jld2') and f.startswith('fields'))]
Nranks = len(dtn)
if Nranks > 1: 
    dtn = []
    for file in np.arange(Nranks):
        dtn.append(f'fields_rank{file}.jld2')
nx, hx, lx, x, y, z, zf = collect_grid(folder, dtn, Nranks)
# Read model information
fid = os.path.join(folder, dtn[0])
time, t_save, visc, diff, u_f, u_s = collect_time_outputs(fid, stokes, closure)

if salinity:
    alpha, beta = collect_temp_and_sal(fid, salinity)
else:
    alpha = collect_temp_and_sal(fid, salinity)

nt = len(t_save)

centerline_index = np.zeros((3, nx[2])).astype(int)
center_xy_loc = np.zeros((3, nx[2]))
center_xy_loc[0, :] = lx[0]/2
center_xy_loc[1, :] = lx[1]/2
center_xy_loc[2, :] = z
centerline_index[0, :] = nx[0]//2 - 1
centerline_index[1, :] = nx[1]//2 - 1
centerline_index[2, :] = np.arange(nx[2]).astype(int)

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
        u, v, w, T, S, Pdynamic, Pstatic = collect_fields_distributed(Nranks, folder, dtn, t_save[it], hx, nx, True, salinity, with_halos)
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
S_value, w_value = collect_contour_val(folder, file_name)
S_contour = S_value * percent_contour
rp_profile = np.zeros(nx[2])
n = 0
for it in range(10, nt):
    # Load data from files
    u, v, w, T, S, Pdynamic, Pstatic = collect_fields_distributed(Nranks, folder, dtn, t_save[it], hx, nx, True, salinity, with_halos)
    centerx = 0.0
    centery = 0.0
    rp_profile_temp, temp = plume_tracer_radius(x, y, nx, S, S_contour) # plume analysis
    n += 1
    rp_profile += rp_profile_temp
rp_profile = rp_profile/n
print(f"writing to {folder}")
with h5py.File(os.path.join(folder, file_name), 'a') as file:
    dset_path = f"plume statistics/contour {percent_contour}/plume tracer radius with depth"
    
    if dset_path in file:
        del file[dset_path]   # remove existing dataset
    
    file.create_dataset(dset_path, data=rp_profile)
