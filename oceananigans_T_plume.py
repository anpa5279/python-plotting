import os
import re
import numpy as np
import matplotlib.pyplot as plt

from plotting_functions import plot_ranges, turb_stats, plot_3d_fields, vert_plane_slices, xy_plane_slices, create_video
from general_analysis_functions import a2_fluc_mean, ab_fluc_mean
from dense_plume_analysis import mld_info, plume_momentum_analysis, plume_tracer_analysis
from plotting_dense_plume import buoyancy_analysis, plot_tracer_plume, plot_momentum_plume
from data_collection_functions import collect_time_outputs, collect_fields_distributed, collect_temp_and_sal, writing_grid, collect_grid, collect_contour_val
from data_manipulation_functions import fcc_ccc, cfc_ccc, ccf_ccc
# Set up folder and simulation parameters
folder = '/Users/annapauls/Library/CloudStorage/OneDrive-UCB-O365/CU-Boulder/TESLa/Carbon Sequestration/Simulations/Oceananigans/NBP/salinity and temperature/no noise circle inlet/vertical domain increase/dTdz = 0.01/nz = 77 z = 96.25 m'
output_folder = os.path.join(folder, "plotting outputs") 
name = ""

# flags for how to read data
with_halos = False
closure = False
stokes = False
salinity = True
write_grid = False

# flags for what to plot
video = True

video_3d_flag = False
turb_stats_plot = False
vert_slice_plot = True
xy_plot = False
buoyancy_analysis_plot = False
buoyancy_momentum_analysis = False

# physical parameters
#nums = re.findall(r' -?\d*\.?\d+', folder)
mld = 60.0 #float(nums[-1]) # mixed layer depth in meters
g = 9.80665  # gravity in m/s^2
dTdz = 0.01#float(nums[-2]) # background temperature gradient in K/m
rho0 = 1026
T0 = 25 
S0 = 0 
Sj = 0.1#float(nums[-3]) # salinity of the source 
wp = 0.001
F_s = Sj*wp
S_value, w_value = collect_contour_val(folder, 'temporal_averages.h5')
# plotting prep
# font for plotting 
plt.rcParams['font.family'] = 'serif' # or 'sans-serif' or 'monospace'
plt.rcParams['font.serif'] = 'cmr10'
plt.rcParams['font.sans-serif'] = 'cmss10'
plt.rcParams['font.monospace'] = 'cmtt10'
plt.rcParams["axes.formatter.use_mathtext"] = True # to fix the minus signs
plt.rcParams['font.size'] = 12
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'DejaVu Serif'
plt.rcParams['mathtext.it'] = 'DejaVu Serif:italic'
plt.rcParams['mathtext.bf'] = 'DejaVu Serif:bold'

ranges = plot_ranges(lz = 96, rho0 = rho0, T0 = T0, dTdz = dTdz, Sj = Sj)
# plot ranges
ranges['w'] = [-2*10**(-2), 2*10**(-2)]
ranges['w_fluc'] = [-2*10**(-2), 2*10**(-2)]
ranges['vel'] = [-1e-5, 1e-5]
ranges['b'] = [-6.0*10**(-4), 6.0*10**(-4)]
ranges['rho'] = [rho0-0.01, rho0+0.1] # <--for stratification [rho0-0.01, rho0+0.1] # 
ranges['rho_fluc'] = [-0.01, 0.01]
ranges['S'] = [0.0, 0.04]
ranges['T'] = [T0-0.4, T0 + 0.005] # <--for stratification [T0-0.4, T0 + 0.005] # 
ranges['u'] = [-6*10**(-3), 6*10**(-3)]
ranges['v'] = [-6*10**(-3), 6*10**(-3)]
ranges['u_fluc'] = ranges['u']
ranges['v_fluc'] = ranges['v']
ranges['Q'] = [-2*10**(1), 2*10**(1)]
ranges['M'] = [-2*10**(-1), 2*10**(-1)]
ranges['F'] = [-1*10**(-3), 1*10**(-3)]
ranges['B'] = [-1*10**(-1), 1*10**(-1)]
ranges['Ri'] = [0, 1]

if xy_plot and salinity:
    xy_ranges = ranges.copy()
    xy_ranges['b_fluc'] = [-4*10**(-5), 4*10**(-5)]
    xy_ranges['rho_fluc'] = [-5*10**-3, 5*10**-3]
    xy_ranges['Pdynamic'] = [-1*10**(-4), 1*10**(-4)]
    xy_ranges['T'] = [T0-0.05, T0 + 0.005]
    xy_ranges['S'] = [0.0, 0.012]
    xy_ranges['u'] = [-6*10**(-3), 6*10**(-3)]
    xy_ranges['v'] = xy_ranges['u']

# List JLD2 files
dtn = [f for f in os.listdir(folder) if (f.endswith('.jld2') and f.startswith('fields'))]
Nranks = len(dtn)
if write_grid:
    if Nranks > 1: 
        nx = np.array([48, 48, 48])
        lx = np.array([320.0, 320.0, 96.0])
        hx = np.array([3, 3, 3]) 
        x, y, z, zf = writing_grid(folder, dtn[0], nx, lx, hx)
    else:
        nx = np.array([48, 48, 48])
        lx = np.array([320.0, 320.0, 96.0])
        hx = np.array([3, 3, 3]) 
        x, y, z, zf = writing_grid(folder, dtn[0], nx, lx, hx)
    dtn = [f for f in os.listdir(folder) if (f.endswith('.jld2') and f.startswith('fields'))]
    x = x[hx[0]:-hx[0]]
    y = y[hx[1]:-hx[1]]
    z = z[hx[2]:-hx[2]]
    zf = zf[hx[2]:-hx[2]]
elif not write_grid:
    nx, hx, lx, x, y, z, zf = collect_grid(folder, dtn[0], Nranks)
if Nranks > 1:
    dtn = []
    for file in np.arange(Nranks):
        dtn.append(f'fields_rank{file}.jld2')
# Read model information
fid = os.path.join(folder, dtn[0])
time, t_save, visc, diff, u_f, u_s = collect_time_outputs(fid, stokes, closure)

if salinity:
    alpha, beta = collect_temp_and_sal(fid, salinity)
else:
    alpha = collect_temp_and_sal(fid, salinity)

w_mag_tol = np.floor(np.log10(np.abs(w_value)))
dbdz_tol = (5.0*10**(-7)) #dTdz*alpha*g

name+=f'Nx{nx[0]}_Ny{nx[1]}_Nz{nx[2]}'

# getting mld index location 
dz_ml = np.abs(z + mld)/mld
mld_idx = np.where(dz_ml==dz_ml.min())[0][-1]

if video:
    nt = np.arange(len(t_save))
else:
    nt = [-1, ]  # only last time step
X, Y, Z = np.meshgrid(x, y, z)
Xf, Yf, Zf = np.meshgrid(x, y, zf)

depth_intrusion_list = []
depth_neutral_list = []
w_neutral_list = []

if buoyancy_analysis_plot or turb_stats_plot or buoyancy_momentum_analysis:
    w_intrusion_list = []
    w_mld_list = []
    bwfluc_neutral_list = []
    bwfluc_intrusion_list = []
    bwfluc_mld_list = []
    rho_perturbed_neutral_list = []
    rho_perturbed_intrusion_list = []
    rho_perturbed_mld_list = []
    l_scale_list = []
    rp_list = []
for it in nt:
    # Load data from files
    if not salinity:
        u, v, w, T, Pdynamic, Pstatic = collect_fields_distributed(Nranks, folder, dtn, t_save[it], hx, nx, True, salinity, with_halos)
    else:
        u, v, w, T, S, Pdynamic, Pstatic = collect_fields_distributed(Nranks, folder, dtn, t_save[it], hx, nx, True, salinity, with_halos)
    # interpolate velocities to cell centers
    u = fcc_ccc(u)
    v = cfc_ccc(v)
    w = ccf_ccc(w)
    # convert temperature and salinity to buoyancy 
    if not salinity:
        rho = rho0 - rho0 * alpha * (T - T0)
        #drhodz = np.gradient(rho, z, axis=-1)
        b = -g*alpha*(T - T0)
    else:
        rhoS = rho0 * beta * (S - S0)
        rhoT = - rho0 * alpha * (T - T0)
        rho = rho0 + rhoS + rhoT
        #drhodz = np.gradient(rho, z, axis=-1)
        b = g*alpha*(T - T0) - g*beta*(S - S0)
        b_T = g*alpha*(T - T0)
        b_S = - g*beta*(S - S0)
        bT_avg = np.mean(b_T, axis=(-3, -2))
        bS_avg = np.mean(b_S, axis=(-3, -2))

    if stokes:
        u = u - u_s

    # calculate means
    u_avg = np.mean(u, axis=(-3, -2))
    v_avg = np.mean(v, axis=(-3, -2))
    w_avg = np.mean(w, axis=(-3, -2))
    b_avg = np.mean(b, axis=(-3, -2))
    rho_avg = np.mean(rho, axis=(-3, -2))
    S_avg = np.mean(S, axis=(-3, -2))
    T_avg = np.mean(T, axis=(-3, -2))

    # calculate fluctuations
    u_fluc = u-u_avg
    v_fluc = v-v_avg
    w_fluc = w-w_avg
    rho_fluc = rho - rho_avg
    S_fluc = S - S_avg
    T_fluc = T - T_avg

    # calcualte reynolds stresses
    uw_fluc, uw_fluc_avg = ab_fluc_mean(u, w, u_avg, w_avg)
    vw_fluc, vw_fluc_avg = ab_fluc_mean(v, w, v_avg, w_avg)

    bu_fluc, bu_fluc_avg = ab_fluc_mean(b, u, b_avg, u_avg)
    bv_fluc, bv_fluc_avg = ab_fluc_mean(b, v, b_avg, v_avg)
    bw_fluc, bw_fluc_avg = ab_fluc_mean(b, w, b_avg, w_avg)
    
    if turb_stats_plot or buoyancy_momentum_analysis:
        u_fluc_avg, u2_fluc, u2_fluc_avg = a2_fluc_mean(u_fluc)
        v_fluc_avg, v2_fluc, v2_fluc_avg = a2_fluc_mean(v_fluc)
        w_fluc_avg, w2_fluc, w2_fluc_avg = a2_fluc_mean(w_fluc)
        uv_fluc, uv_fluc_avg = ab_fluc_mean(u, v, u_avg, v_avg)
        b2_fluc, b2_fluc_avg = ab_fluc_mean(b, b, b_avg, b_avg)
        # rms fluctuations
        u_rms = u2_fluc_avg**0.5
        v_rms = v2_fluc_avg**0.5
        w_rms = w2_fluc_avg**0.5
        b_rms = b2_fluc_avg**0.5

    # calculating density 
    b_fluc = b - b_avg
    rho_perturbed = ((b_fluc)*rho0)/(-g)
    # prepping variables for plume statistics
    dbdz = np.gradient(b, z, axis=-1)

    if salinity:
        centerline_index, rp_profile, plume_index = plume_tracer_analysis(x, y, nx, S, tracer_contour = S_value*0.05)
    
        dbdx_avg = np.mean(np.gradient(b, x, axis=0), axis=(-3, -2))
        dbdy_avg = np.mean(np.gradient(b, y, axis=1), axis=(-3, -2))
        dbdz_avg = np.mean(np.gradient(b, z, axis=2), axis=(-3, -2))
        S_fluc_center = S_fluc[centerline_index[0], centerline_index[1], centerline_index[2]]
        T_fluc_center = T_fluc[centerline_index[0], centerline_index[1], centerline_index[2]]
    i_idx, j_idx, k_idx = plume_index
    values = bw_fluc[i_idx, j_idx, k_idx]

    # sum per k
    sum_per_k = np.bincount(k_idx, weights=values)

    # count per k
    count_per_k = np.bincount(k_idx)

    # average per k
    bw_fluc_plume_avg = sum_per_k / count_per_k
    bw_fluc_plume_avg[np.isnan(bw_fluc_plume_avg)] = 0


    # buoyancy analysis 
    if buoyancy_analysis_plot or turb_stats_plot or buoyancy_momentum_analysis:
        Q, M, F, B, wm, dm, bm, Ri, area_idx, max_index, neutral_index = plume_momentum_analysis(centerline_index, nx, w, b, b_fluc, rho_fluc, X, Y, w_mag_tol)
        z_neutral = z[neutral_index]

        w_center = w[centerline_index[0], centerline_index[1], centerline_index[2]]
        bw_fluc_center = bw_fluc[centerline_index[0], centerline_index[1], centerline_index[2]]
        rho_perturbed_center = rho_perturbed[centerline_index[0], centerline_index[1], centerline_index[2]]
        b_center = b[centerline_index[0], centerline_index[1], centerline_index[2]]
        z_intrusion = z[max_index]
        w_intrusion = w_center[max_index]
        w_neutral = w_center[neutral_index]
        bw_intrusion = bw_fluc_center[max_index]
        bw_neutral = bw_fluc_center[neutral_index]
        rho_intrusion = rho_perturbed_center[max_index]
        rho_neutral = rho_perturbed_center[neutral_index]
        mld_idx, w_mld, mld_bw_fluc, rho_mld = mld_info(w_center, bw_fluc_center, rho_perturbed_center, z, mld)
        # appending plume statistics to lists
        depth_intrusion_list.append(z_intrusion)
        depth_neutral_list.append(z_neutral)
        
        w_intrusion_list.append(w_intrusion)
        w_neutral_list.append(w_neutral)
        w_mld_list.append(w_mld)

        bwfluc_neutral_list.append(bw_neutral)
        bwfluc_intrusion_list.append(bw_intrusion)
        bwfluc_mld_list.append(mld_bw_fluc)

        rho_perturbed_neutral_list.append(rho_neutral)
        rho_perturbed_intrusion_list.append(rho_intrusion)
        rho_perturbed_mld_list.append(rho_mld)

        plume_depths = [depth_intrusion_list, depth_neutral_list]
        ws = [w_intrusion_list, w_neutral_list, w_mld_list]
        rhos = [rho_perturbed_intrusion_list, rho_perturbed_neutral_list, rho_perturbed_mld_list]
        bw_flucs = [bwfluc_intrusion_list, bwfluc_neutral_list, bwfluc_mld_list]
        
    ############ PLOTTING ############
    # --- Create Video ---
    if turb_stats_plot:
        if salinity:
            plume_info = []
            plume_info.append(z_intrusion)
            plume_info.append(z_neutral)
            plume_info.append(rho_tracer_center)
            turb_stat_dir = turb_stats(time, it, ranges, output_folder, lx, nx, z, zf, mld, u_avg, v_avg, w_avg, u_rms, v_rms, w_rms, uv_fluc_avg, uw_fluc_avg, vw_fluc_avg, bu_fluc_avg, bv_fluc_avg, bw_fluc_avg, b_rms, rho_avg, plume_info)
        else:
            turb_stat_dir = turb_stats(time, it, ranges, output_folder, lx, nx, z, zf, mld, u_avg, v_avg, w_avg, u_rms, v_rms, w_rms, uv_fluc_avg, uw_fluc_avg, vw_fluc_avg, bu_fluc_avg, bv_fluc_avg, bw_fluc_avg, b_rms, rho_avg)
    if video_3d_flag:
        video_3d_dir = plot_3d_fields(time, it, ranges, output_folder, lx, X, Y, Z, Xf, Yf, Zf, u, v, w, T, S)
    if vert_slice_plot:
        if it < 10:
            depths = np.array([-mld, ])
        else:
            neutral_test = z[np.where(np.diff(np.sign(bw_fluc_plume_avg))>0)][-1]
            depths = np.array([-mld, neutral_test])
        plane_slices_dir = vert_plane_slices(time, it, ranges, output_folder, lx, nx, X, Y, Z, u, v, w, rho, rho_perturbed, T = T, S = S, depths = depths)
    if xy_plot and salinity:
        loc = "mld"#"n = 230, z = " + str(np.round(z[230], 2)) + " m"
        loc_idx = mld_idx
        if loc_idx >(nx[2]-1):
            loc_idx = nx[2] - 1
        surface_dir = xy_plane_slices(time, it, xy_ranges, output_folder, lx, X, Y, u, v, w, b, b_fluc, Pdynamic, rho, rho_perturbed, loc_idx, loc, T, S)
    elif xy_plot and not salinity:
        loc = ""
        surface_dir = xy_plane_slices(time, it, xy_ranges, output_folder, lx, X, Y, u, v, w, b, b_fluc, Pdynamic, rho, rho_perturbed, max_index, loc, T)
    if buoyancy_analysis_plot and not salinity:
        buoyancy_dir = buoyancy_analysis(time, it, ranges, output_folder, lx, nx, z, zf, X, Z, mld, b_avg, w_avg, b_center, w_center, b_rms, bu_fluc_avg, bv_fluc_avg, bw_fluc_avg, b_fluc, rho_perturbed, Ri_avg, Ri_strat, Ri_plume, intrusion, neutral, w_neutral, w_intrusion, w_mld, rho_neutral, rho_intrusion, rho_perturbed_mld, bwfluc_neutral, bwfluc_intrusion, bwfluc_mld, alpha_vel, alpha_length, salinity)
    if buoyancy_analysis_plot and salinity:
        buoyancy_dir = plot_tracer_plume(time, it, ranges, output_folder, lx, nx, z, zf, Y, Z, mld, u_avg, v_avg, w_avg, uv_fluc_avg, uw_fluc_avg, vw_fluc_avg, u_rms, v_rms, w_rms, dbdx, dbdy, dbdz, b_avg, b_avg, b_center, w_center, b_rms, bu_fluc_avg, bv_fluc_avg, bw_fluc_avg, b_fluc, rho_perturbed, S_avg, rp_list, plume_depths, ws, rhos, bw_flucs, l_scale_list)
    if buoyancy_momentum_analysis:
        momentum_dir = plot_momentum_plume(time, it, ranges, output_folder, lx, z, zf, mld, b_avg, S_avg, u_rms, v_rms, w_rms, b_rms, bu_fluc_avg, bv_fluc_avg, bw_fluc_avg, S_fluc_center, T_fluc_center, Q, M, F, B, wm, dm, bm, Ri, rp_profile, b_center, plume_depths)
print("All frames created.")
# creating videos
if video:
    if turb_stats_plot:
        create_video(turb_stat_dir, output_folder, name, 'turbulence_statistics')
    if video_3d_flag:
        create_video(video_3d_dir, output_folder, name, '3D_fields')
    if vert_slice_plot:
        create_video(plane_slices_dir, output_folder, name, 'vert_plane_slices')
    if xy_plot:
        name_xy = loc + '-xy-plane-slices'
        create_video(surface_dir, output_folder, name, name_xy)
    if buoyancy_analysis_plot:
        create_video(buoyancy_dir, output_folder, name, 'buoyancy_analysis')
    if buoyancy_momentum_analysis:
        create_video(momentum_dir, output_folder, name, 'buoyancy_momentum_analysis')