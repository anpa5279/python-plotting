import os
import re
import numpy as np
import matplotlib.pyplot as plt

from plotting_functions import plot_format, plot_ranges, create_video, plot_momentum_plume, plot_tracer_plume, vert_plane_slices, xy_plane_slices, buoyancy_analysis_plot, turb_stats_plot

from .reader import OceananigansData
from .physics import reynolds_stress, buoyancy , rms, a_fluc_b
from .interpolation import velocities_to_center, yz_plane, xy_plane, vertical_line
from .dense_plume import PlumeAnalysis
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

turb_stat_flag = False
vert_slice_plot = True
xy_plot = True
buoyancy_analysis_flag = False
buoyancy_momentum_flag = False

# ------------------------- GENERAL MODEL INFORMATION ------------------------- #
# physical parameters
nums = re.findall(r' -?\d*\.?\d+', folder)
mld = float(nums[-1]) # mixed layer depth in meters #60.0 #
g = 9.80665  # gravity in m/s^2
dTdz = float(nums[-2]) # background temperature gradient in K/m #0.01#
rho0 = 1026
T0 = 25 
S0 = 0 
contour = 0.05 

reader = OceananigansData(folder)
# grid info
reader.load_grid()
x, y, z = reader.x, reader.y, reader.z
nx = reader.nx
dx = reader.dx
hx = reader.hx
# load time and equation of state info
reader.load_time(reader)
coeffs = reader.load_equation_of_state(reader, salinity)
alpha = coeffs['alpha']
if salinity:
    beta = coeffs['beta']
    S_value, w_value = reader.load_contour_temporal_averages('interp_temporal_averages.h5')
    S_contour = S_value*contour
    dense_plume = PlumeAnalysis(S_value*contour)

name+=f'Nx{nx[0]}_Ny{nx[1]}_Nz{nx[2]}'

# getting mld index location 
dz_ml = np.abs(z + mld)/mld
mld_idx = np.where(dz_ml==dz_ml.min())[0][-1]

# plotting setup
plot_format()
ranges = plot_ranges(lz = 96, mld = mld, rho0 = rho0, T0 = T0, dTdz = dTdz)
# plot ranges
ranges['w'] = [-2*10**(-2), 2*10**(-2)]
ranges['w_fluc'] = [-2*10**(-2), 2*10**(-2)]
ranges['vel'] = [-1e-5, 1e-5]
ranges['b'] = [-6.0*10**(-4), 6.0*10**(-4)]
ranges['rho'] = [rho0-0.01, rho0+0.1]#0.1] # <--for stratification [rho0-0.01, rho0+0.1] # 
ranges['rho_fluc'] = [-0.01, 0.01]
ranges['Tracer'] = [0.0, 0.04]
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
    xy_ranges['Tracer'] = [0.0, 0.01]
    xy_ranges['u'] = [-6*10**(-3), 6*10**(-3)]
    xy_ranges['v'] = xy_ranges['u']

if video:
    nt = np.arange(0, reader.nt)
    time = reader.time
else:
    nt = [reader.nt - 1,] # last time step
    time = reader.time[-1]

X, Y, Z = np.meshgrid(x, y, z)

if buoyancy_analysis_flag or turb_stat_flag or buoyancy_momentum_flag:
    depth_intrusion_list = []
    depth_neutral_list = []
    w_neutral_list = []
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
for it, t in enumerate(reader.t_save):
    # Load data from files
    T = reader.lazy_field('T', t)
    u = reader.lazy_field('u', t)
    v = reader.lazy_field('v', t)
    w = reader.lazy_field('w', t)
    if salinity:
        S = reader.lazy_field('S', t)
    if xy_plot:
        Pdynamic = reader.lazy_field('Pdynamic', t)
    # interpolate velocities to cell centers
    u, v, w = velocities_to_center(u, v, w)
    # convert temperature and salinity to buoyancy 
    bs = buoyancy(T, rho0, coeffs, T0, g, tracer = S if salinity else None)
    b = bs['b_total']
    rho = bs['rho']

    if stokes:
        u = u - reader.u_s

    # calculate means
    u_avg = np.mean(u, axis=(-3, -2))
    v_avg = np.mean(v, axis=(-3, -2))
    w_avg = np.mean(w, axis=(-3, -2))
    b_avg = np.mean(b, axis=(-3, -2))
    rho_avg = np.mean(rho, axis=(-3, -2))
    T_avg = np.mean(T, axis=(-3, -2))
    T_fluc = T - T_avg
    if salinity:
        S_avg = np.mean(S, axis=(-3, -2))
        S_fluc = S - S_avg

    # calcualte reynolds stresses
    uw_fluc, uw_fluc_avg = reynolds_stress(u, w, u_avg, w_avg)
    vw_fluc, vw_fluc_avg = reynolds_stress(v, w, v_avg, w_avg)

    bu_fluc, bu_fluc_avg = reynolds_stress(b, u, b_avg, u_avg)
    bv_fluc, bv_fluc_avg = reynolds_stress(b, v, b_avg, v_avg)
    bw_fluc, bw_fluc_avg = reynolds_stress(b, w, b_avg, w_avg)
    
    if turb_stat_flag or buoyancy_momentum_flag:
        uv_fluc_avg = np.mean(reynolds_stress(u, v, u_avg, v_avg), axis=(-3, -2))
        # rms fluctuations
        u_rms = rms(u)
        v_rms = rms(v)
        w_rms = rms(w)
        b_rms = rms(b)

    # calculating density 
    b_fluc = b - b_avg
    rho_perturbed = ((b_fluc)*rho0)/(-g)
    # prepping variables for plume statistics
    dbdz = np.gradient(b, z, axis=-1)

    if salinity:
        centerx = 0.0
        centery = 0.0
        rp_profile = dense_plume.plume_tracer_radius(x, y)
        S_fluc_center = vertical_line(S_fluc, x, y, centerx, centery)
        T_fluc_center = vertical_line(T_fluc, x, y, centerx, centery)

    # buoyancy analysis 
    if buoyancy_analysis_flag or turb_stat_flag or buoyancy_momentum_flag:
        dense_plume.plume_momentum_area(x, y, w, w_mag_tol)
        dense_plume.plume_momentum_analysis(w, b)
        Q = dense_plume.Q
        M = dense_plume.M
        F = dense_plume.F
        B = dense_plume.B
        wm = dense_plume.wm
        dm = dense_plume.dm
        bm = dense_plume.bm
        Ri = dense_plume.Ri
        z_neutral = dense_plume.neutral_layer(z)

        w_center = vertical_line(w, x, y, centerx, centery)
        bw_fluc_center = vertical_line(bw_fluc, x, y, centerx, centery)
        rho_perturbed_center = vertical_line(rho_perturbed, x, y, centerx, centery)
        b_center = vertical_line(b, x, y, centerx, centery)
        z_intrusion = dense_plume.max_penetration(z)
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
    if turb_stat_flag:
        if salinity:
            plume_info = []
            plume_info.append(z_intrusion)
            plume_info.append(z_neutral)
            plume_info.append(rho_tracer_center)
            turb_stat_dir = turb_stats_plot(time, it, ranges, output_folder, lx, nx, z, mld, u_avg, v_avg, w_avg, u_rms, v_rms, w_rms, uv_fluc_avg, uw_fluc_avg, vw_fluc_avg, bu_fluc_avg, bv_fluc_avg, bw_fluc_avg, b_rms, rho_avg, plume_info)
        else:
            turb_stat_dir = turb_stats_plot(time, it, ranges, output_folder, lx, nx, z, mld, u_avg, v_avg, w_avg, u_rms, v_rms, w_rms, uv_fluc_avg, uw_fluc_avg, vw_fluc_avg, bu_fluc_avg, bv_fluc_avg, bw_fluc_avg, b_rms, rho_avg)
    if vert_slice_plot:
        u_yz = yz_plane(u, x, centerx)
        v_yz = yz_plane(v, x, centerx)
        w_yz = yz_plane(w, x, centerx)
        rho_yz = yz_plane(rho, x, centerx)
        rho_perturbed_yz = yz_plane(rho_perturbed, x, centerx)
        T_yz = yz_plane(T, x, centerx)
        S_yz = yz_plane(S, x, centerx)
        if it < 10:
            depths = np.array([-mld, ])
        else:
            neutral_depth = dense_plume.neutral_layer(z)
            depths = np.array([-mld, neutral_depth])
        plane_slices_dir = vert_plane_slices(time[it], ranges, output_folder, lx, x, y, z, u_yz, v_yz, w_yz, rho_yz, rho_perturbed_yz, T = T_yz, S = S_yz, depths = depths)
    if xy_plot and salinity:
        loc = "z = MLD"#"n = 230, z = " + str(np.round(z[230], 2)) + " m"
        loc_z = -mld
        u_xy = xy_plane(u, z, loc_z)
        v_xy = xy_plane(v, z, loc_z)
        w_xy = xy_plane(w, z, loc_z)
        rho_xy = xy_plane(rho, z, loc_z)
        rho_perturbed_xy = xy_plane(rho_perturbed, z, loc_z)
        Pdynamic_xy = xy_plane(Pdynamic, z, loc_z)
        T_xy = xy_plane(T, z, loc_z)
        S_xy = xy_plane(S, z, loc_z)
        surface_dir = xy_plane_slices(time[it], xy_ranges, output_folder, x, y, u_xy, v_xy, w_xy, Pdynamic_xy, rho_xy, rho_perturbed_xy, loc, T = T_xy, S = S_xy)
    if buoyancy_analysis_flag and not salinity:
        buoyancy_dir = buoyancy_analysis_plot(time[it], it, ranges, output_folder, lx, nx, z, x, z, mld, b_avg, w_avg, b_center, w_center, b_rms, bu_fluc_avg, bv_fluc_avg, bw_fluc_avg, b_fluc, rho_perturbed, Ri_avg, Ri_strat, Ri_plume, intrusion, neutral, w_neutral, w_intrusion, w_mld, rho_neutral, rho_intrusion, rho_perturbed_mld, bwfluc_neutral, bwfluc_intrusion, bwfluc_mld, alpha_vel, alpha_length, salinity)
    if buoyancy_analysis_flag and salinity:
        buoyancy_dir = plot_tracer_plume(time[it], it, ranges, output_folder, lx, nx, z, y, z, mld, u_avg, v_avg, w_avg, uv_fluc_avg, uw_fluc_avg, vw_fluc_avg, u_rms, v_rms, w_rms, dbdx, dbdy, dbdz, b_avg, b_avg, b_center, w_center, b_rms, bu_fluc_avg, bv_fluc_avg, bw_fluc_avg, b_fluc, rho_perturbed, S_avg, rp_list, plume_depths, ws, rhos, bw_flucs, l_scale_list)
    if buoyancy_momentum_flag:
        momentum_dir = plot_momentum_plume(time[it], it, ranges, output_folder, lx, z, mld, b_avg, S_avg, u_rms, v_rms, w_rms, b_rms, bu_fluc_avg, bv_fluc_avg, bw_fluc_avg, S_fluc_center, T_fluc_center, Q, M, F, B, wm, dm, bm, Ri, rp_profile, b_center, plume_depths)
print("All frames created.")
# creating videos
if video:
    if turb_stat_flag:
        create_video(turb_stat_dir, output_folder, name, 'turbulence_statistics')
    if vert_slice_plot:
        create_video(plane_slices_dir, output_folder, name, 'vert_plane_slices')
    if xy_plot:
        name_xy = loc + '-xy-plane-slices'
        create_video(surface_dir, output_folder, name, name_xy)
    if buoyancy_analysis_flag:
        create_video(buoyancy_dir, output_folder, name, 'buoyancy_analysis_plot')
    if buoyancy_momentum_flag:
        create_video(momentum_dir, output_folder, name, 'buoyancy_momentum_flag')