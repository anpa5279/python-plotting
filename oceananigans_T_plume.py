import os
import re
import numpy as np
import matplotlib.pyplot as plt

from plotting_functions import plot_ranges, plot_momentum_plume, plot_tracer_plume, plume_momentum_analysis, plume_tracer_radius, z_line_interpolation, z_plane_interpolation, xy_plane_interpolation, vert_plane_slices, xy_plane_slices, create_video

from .reader import OceananigansData
from .general_physics import velocities_to_center, ab_fluc, buoyancy 
from .interpolation import xy_plane, velocities_to_center
# Set up folder and simulation parameters
folder = ''
output_folder = os.path.join(folder, "plotting outputs") 
name = "interp"

# flags for how to read data
with_halos = False
closure = False
stokes = False
salinity = True
write_grid = False

# flags for what to plot
video = True

turb_stats_plot = False
vert_slice_plot = True
xy_plot = True
buoyancy_analysis_plot = False
buoyancy_momentum_analysis = False

# physical parameters
nums = re.findall(r' -?\d*\.?\d+', folder)
mld = float(nums[-1]) # mixed layer depth in meters #60.0 #
g = 9.80665  # gravity in m/s^2
dTdz = float(nums[-2]) # background temperature gradient in K/m #0.01#
rho0 = 1026
T0 = 25 
S0 = 0 
wp = 0.001
contour = 0.05 
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

ranges = plot_ranges(lz = 96, rho0 = rho0, T0 = T0, dTdz = dTdz)
# plot ranges
ranges['w'] = [-2*10**(-2), 2*10**(-2)]
ranges['w_fluc'] = [-2*10**(-2), 2*10**(-2)]
ranges['vel'] = [-1e-5, 1e-5]
ranges['b'] = [-6.0*10**(-4), 6.0*10**(-4)]
ranges['rho'] = [rho0-0.01, rho0+0.1]#0.1] # <--for stratification [rho0-0.01, rho0+0.1] # 
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
    xy_ranges['rho_fluc'] = [-5*10**-3, 5*10**-3]
    xy_ranges['rho'] = [rho0-0.01, rho0+0.01]
    xy_ranges['Pdynamic'] = [-1*10**(-4), 1*10**(-4)]
    xy_ranges['T'] = [T0-0.05, T0+0.01]
    xy_ranges['S'] = [0.0, 0.01]
    xy_ranges['u'] = [-6*10**(-3), 6*10**(-3)]
    xy_ranges['v'] = xy_ranges['u']
# ------------------------- GENERAL MODEL INFORMATION ------------------------- #
# List JLD2 files
files = [f for f in os.listdir(folder) if (f.endswith('.jld2') and f.startswith('fields'))]
Nranks = len(files)
if Nranks > 1:
    files = []
    for file in np.arange(Nranks):
        files.append(f'fields_rank{file}.jld2')

reader = OceananigansData(folder, files, Nranks)
# grid info
reader.load_grid()
x, y, z = reader.x, reader.y, reader.z
nx = reader.nx
dx = reader.dx
hx = reader.hx
# load time and equation of state info
nt, time, t_save, visc, diff, u_f, u_s = reader.load_time(files[0])
coeffs = reader.load_equation_of_state(files[0], salinity)
alpha = coeffs['alpha']
if salinity:
    beta = coeffs['beta']
    S_value, w_value = reader.load_contour_temporal_averages('interp_temporal_averages.h5')
    S_contour = S_value*contour

name+=f'Nx{nx[0]}_Ny{nx[1]}_Nz{nx[2]}'

# getting mld index location 
dz_ml = np.abs(z + mld)/mld
mld_idx = np.where(dz_ml==dz_ml.min())[0][-1]

if video:
    nt = np.arange(len(t_save))
else:
    nt = [-1, ]  # only last time step
X, Y, Z = np.meshgrid(x, y, z)

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
for it, t in enumerate(t_save):
    # Load data from files
    T = reader.lazy_field('T', t)
    u = reader.lazy_field('u', t)
    v = reader.lazy_field('v', t)
    w = reader.lazy_field('w', t)
    if salinity:
        S = reader.lazy_field('S', t)
    # interpolate velocities to cell centers
    u, v, w = velocities_to_center(u, v, w)
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
    uw_fluc, uw_fluc_avg = ab_fluc(u, w, u_avg, w_avg)
    vw_fluc, vw_fluc_avg = ab_fluc(v, w, v_avg, w_avg)

    bu_fluc, bu_fluc_avg = ab_fluc(b, u, b_avg, u_avg)
    bv_fluc, bv_fluc_avg = ab_fluc(b, v, b_avg, v_avg)
    bw_fluc, bw_fluc_avg = ab_fluc(b, w, b_avg, w_avg)
    
    if turb_stats_plot or buoyancy_momentum_analysis:
        u_fluc_avg, u2_fluc, u2_fluc_avg = a2_fluc_mean(u_fluc)
        v_fluc_avg, v2_fluc, v2_fluc_avg = a2_fluc_mean(v_fluc)
        w_fluc_avg, w2_fluc, w2_fluc_avg = a2_fluc_mean(w_fluc)
        uv_fluc, uv_fluc_avg = ab_fluc(u, v, u_avg, v_avg)
        b2_fluc, b2_fluc_avg = ab_fluc(b, b, b_avg, b_avg)
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
        centerx = 0.0
        centery = 0.0
        rp_profile, plume_index = plume_tracer_radius(x, y, nx, S, S_contour)
        S_fluc_center = z_line_interpolation(S_fluc, x, y, centerx, centery)
        T_fluc_center = z_line_interpolation(T_fluc, x, y, centerx, centery)

    # buoyancy analysis 
    if buoyancy_analysis_plot or turb_stats_plot or buoyancy_momentum_analysis:
        Q, M, F, B, wm, dm, bm, Ri, area_idx, max_index, neutral_index = plume_momentum_analysis(nx, w, b, b_fluc, rho_fluc, X, Y, w_mag_tol)
        z_neutral = z[neutral_index]

        w_center = z_line_interpolation(w, x, y, centerx, centery)
        bw_fluc_center = z_line_interpolation(bw_fluc, x, y, centerx, centery)
        rho_perturbed_center = z_line_interpolation(rho_perturbed, x, y, centerx, centery)
        b_center = z_line_interpolation(b, x, y, centerx, centery)
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
    if vert_slice_plot:
        x_loc = 0.0
        u_yz = z_plane_interpolation(u, x, x_loc)
        v_yz = z_plane_interpolation(v, x, x_loc)
        w_yz = z_plane_interpolation(w, x, x_loc)
        rho_yz = z_plane_interpolation(rho, x, x_loc)
        rho_perturbed_yz = z_plane_interpolation(rho_perturbed, x, x_loc)
        T_yz = z_plane_interpolation(T, x, x_loc)
        S_yz = z_plane_interpolation(S, x, x_loc)
        if it < 10:
            depths = np.array([-mld, ])
        else:
            neutral_depth = neutral_layer(z, bw_fluc, plume_index)
            depths = np.array([-mld, neutral_depth])
        plane_slices_dir = vert_plane_slices(time[it], ranges, output_folder, lx, x, y, z, u_yz, v_yz, w_yz, rho_yz, rho_perturbed_yz, T = T_yz, S = S_yz, depths = depths)
    if xy_plot and salinity:
        loc = "z = MLD"#"n = 230, z = " + str(np.round(z[230], 2)) + " m"
        loc_z = -mld
        u_xy = xy_plane_interpolation(u, z, loc_z)
        v_xy = xy_plane_interpolation(v, z, loc_z)
        w_xy = xy_plane_interpolation(w, z, loc_z)
        rho_xy = xy_plane_interpolation(rho, z, loc_z)
        rho_perturbed_xy = xy_plane_interpolation(rho_perturbed, z, loc_z)
        Pdynamic_xy = xy_plane_interpolation(Pdynamic, z, loc_z)
        T_xy = xy_plane_interpolation(T, z, loc_z)
        S_xy = xy_plane_interpolation(S, z, loc_z)
        surface_dir = xy_plane_slices(time[it], xy_ranges, output_folder, x, y, u_xy, v_xy, w_xy, Pdynamic_xy, rho_xy, rho_perturbed_xy, loc, T = T_xy, S = S_xy)
    if buoyancy_analysis_plot and not salinity:
        buoyancy_dir = buoyancy_analysis(time, it, ranges, output_folder, lx, nx, z, zf, x, z, mld, b_avg, w_avg, b_center, w_center, b_rms, bu_fluc_avg, bv_fluc_avg, bw_fluc_avg, b_fluc, rho_perturbed, Ri_avg, Ri_strat, Ri_plume, intrusion, neutral, w_neutral, w_intrusion, w_mld, rho_neutral, rho_intrusion, rho_perturbed_mld, bwfluc_neutral, bwfluc_intrusion, bwfluc_mld, alpha_vel, alpha_length, salinity)
    if buoyancy_analysis_plot and salinity:
        buoyancy_dir = plot_tracer_plume(time, it, ranges, output_folder, lx, nx, z, zf, y, z, mld, u_avg, v_avg, w_avg, uv_fluc_avg, uw_fluc_avg, vw_fluc_avg, u_rms, v_rms, w_rms, dbdx, dbdy, dbdz, b_avg, b_avg, b_center, w_center, b_rms, bu_fluc_avg, bv_fluc_avg, bw_fluc_avg, b_fluc, rho_perturbed, S_avg, rp_list, plume_depths, ws, rhos, bw_flucs, l_scale_list)
    if buoyancy_momentum_analysis:
        momentum_dir = plot_momentum_plume(time, it, ranges, output_folder, lx, z, zf, mld, b_avg, S_avg, u_rms, v_rms, w_rms, b_rms, bu_fluc_avg, bv_fluc_avg, bw_fluc_avg, S_fluc_center, T_fluc_center, Q, M, F, B, wm, dm, bm, Ri, rp_profile, b_center, plume_depths)
print("All frames created.")
# creating videos
if video:
    if turb_stats_plot:
        create_video(turb_stat_dir, output_folder, name, 'turbulence_statistics')
    if vert_slice_plot:
        create_video(plane_slices_dir, output_folder, name, 'vert_plane_slices')
    if xy_plot:
        name_xy = loc + '-xy-plane-slices'
        create_video(surface_dir, output_folder, name, name_xy)
    if buoyancy_analysis_plot:
        create_video(buoyancy_dir, output_folder, name, 'buoyancy_analysis')
    if buoyancy_momentum_analysis:
        create_video(momentum_dir, output_folder, name, 'buoyancy_momentum_analysis')