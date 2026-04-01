import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

from plotting_functions import stratification_profile, plot_ranges, turb_stats, plot_3d_fields, vert_plane_slices, xy_plane_slices, create_video
from general_analysis_functions import a2_fluc_mean, ab_fluc_mean, richardson_number
from dense_plume_analysis import mld_info, centerline_analysis_buoyancy, plume_momentum_analysis, plume_tracer_analysis
from plotting_dense_plume import buoyancy_analysis, plot_tracer_plume, plot_momentum_plume
from data_collection_functions import collect_time_outputs, collect_fields, collect_fields_distributed, collect_temp_and_sal
from plotting_comparisons import plume_spatial_analysis
def stokes_exp(z):
    g_Earth = 9.80665
    wavelength = 60.0 #m
    amplitude = 0.8 #m
    wavenumber = 2 * np.pi / wavelength
    frequency = np.sqrt(g_Earth * wavenumber)
    vert_scale = wavelength / (4 * np.pi)
    us = amplitude**2* wavenumber* frequency #0.05501259798225732#
    return us*np.exp(z/vert_scale)
# Set up folder and simulation parameters
folder = '/Users/annapauls/Library/CloudStorage/OneDrive-UCB-O365/CU-Boulder/TESLa/Carbon Sequestration/Simulations/Oceananigans/NBP/salinity and temperature/beta = default S0 = 0.2/'
output_folder = folder #'figures and videos/'
name = 'hydrosymposium-'#'SvsT'

# flags to analyze data 
rho_IC_perturb = False

# flags for what to plot
video = True

video_3d_flag = False
turb_stats_plot = False
vert_slice_plot = True
xy_plot = False
buoyancy_analysis_plot = False
buoyancy_momentum_analysis = False
plume_plot = False

# flags for how to read data
with_halos = False
stokes = False
salinity = True

# physical parameters
mld = 30.0  # mixed layer depth in meters
g = 9.80665  # gravity in m/s^2
dTdz = 0.01 # background temperature gradient in K/m
rho0 = 1026
T0 = 25 
S0 = 0 
Sj = 0.2
wp = 0.001
F_s = Sj*wp
rj = 10
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
if xy_plot and salinity:
    xy_ranges = ranges.copy()
    #xy_ranges['rho'] = [rho0-0.01, rho0+0.1]
    xy_ranges['b_fluc'] = [-8*10**(-5), 8*10**(-5)]
    xy_ranges['rho_fluc'] = [-1*10**-3, 1*10**-3]
    #xy_ranges['b'] = [-2.0*10**(-4), 2.0*10**(-4)]
    xy_ranges['Pdynamic'] = [-1*10**(-3), 1*10**(-3)]
    xy_ranges['T'] = [T0-0.06, T0 + 0.06]
    xy_ranges['S'] = [0.0, 0.012]
    #xy_ranges['w'] = [-5*10**(-3), 5*10**(-3)]

ranges['w'] = [-2*10**(-2), 2*10**(-2)]
ranges['w_fluc'] = [-0.003, 0.003]
ranges['restress'] = [-4*10**(-8), 4*10**(-8)]
ranges['vel'] = [-1e-5, 1e-5]
ranges['b'] = [-2.0*10**(-3), 2.0*10**(-3)]#[-1.0*10**(-2), 1.0*10**(-2)]#
ranges['rho'] = [rho0-0.02, rho0+0.14]#[rho0-0.02, rho0+0.9] #
ranges['S'] = [0.0, 0.05]
ranges['T'] = [T0-0.65, T0 + 0.02]# [T0-3.4, T0 + 0.05]#
ranges['Q'] = [-3.5*10**(0), 3.5*10**(0)]
ranges['M'] = [-2*10**(-1), 2*10**(-1)]
ranges['F'] = [-2.5*10**(-4), 2.5*10**(-4)]
ranges['B'] = [-8*10**(-2), 8*10**(-2)]
ranges['richardson'] = [-1*10**5, 1*10**5]
ranges['u'] = [-2*10**(-2), 2*10**(-2)]
ranges['u_fluc'] = ranges['u']
ranges['v'] = [-4*10**(-3), 4*10**(-3)]

# List JLD2 files
dtn = [f for f in os.listdir(folder) if (f.endswith('.jld2') and f.startswith('fields'))]
Nranks = len(dtn)
if Nranks > 1:
    dtn = []
    for file in np.arange(Nranks):
        dtn.append(f'fields_rank{file}.jld2')
# Read model information
fid = os.path.join(folder, dtn[0])
if Nranks == 1 and not stokes:
    time, t_save, nx, hx, lx, x, y, z, xf, yf, zf, dx, visc, diff = collect_time_outputs(fid, Nranks, stokes)
elif Nranks == 1 and stokes:
    time, t_save, nx, hx, lx, x, y, z, xf, yf, zf, dx, visc, diff, u_f, u_s = collect_time_outputs(fid, Nranks, stokes)
elif Nranks > 1 and not stokes:
    time, t_save, nx, hx, lx, x, y, z, xf, yf, zf, dx, visc, diff = collect_time_outputs(fid, Nranks, stokes)
elif Nranks > 1 and stokes:
    time, t_save, nx, hx, lx, x, y, z, xf, yf, zf, dx, visc, diff, u_f, u_s = collect_time_outputs(fid, Nranks, stokes)
    u_s = stokes_exp(z)
if salinity:
    alpha, beta = collect_temp_and_sal(fid, salinity)
else:
    alpha = collect_temp_and_sal(fid, salinity)

if buoyancy_momentum_analysis:
    w_mag_tol = np.floor(np.log10(wp))
    dbdz_tol = dTdz*alpha*g

if rho_IC_perturb:
    name+='-rhoICperturbation-'
name+=f'Nx{nx[0]}_Ny{nx[1]}_Nz{nx[2]}'
print(name)
# getting mld index location 

dz_ml = np.abs(z + mld)/mld
mld_index = np.where(dz_ml==dz_ml.min())[0][-1]

if video:
    nt = np.arange(len(t_save))
else:
    nt = [-1, ]  # only last time step
X, Y, Z = np.meshgrid(x, y, z)
X_zf, Y_zf, Z_zf = np.meshgrid(x, y, zf)
if buoyancy_analysis_plot or turb_stats_plot or buoyancy_momentum_analysis:
    L_ozmidov_average_list = []
    L_ozmidov_background_list = []
    L_ozmidov_plume_list = []
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
for it in nt:
    # Load data from files
    if not salinity:
        u, v, w, T, Pdynamic, Pstatic = collect_fields_distributed(Nranks, folder, dtn, t_save[it], hx, nx, True, salinity, with_halos)
    else:
        u, v, w, T, S, Pdynamic, Pstatic = collect_fields_distributed(Nranks, folder, dtn, t_save[it], hx, nx, True, salinity, with_halos)
    # convert temperature and salinity to buoyancy 
    if not salinity:
        rho = rho0 - rho0 * alpha * (T - T0)
        drhodz = np.gradient(rho, z, axis=-1)
        b = -g*alpha*(T - T0)
    else:
        rhoS = rho0 * beta * (S - S0)
        rhoT = - rho0 * alpha * (T - T0)
        rho = rho0 + rhoS + rhoT
        drhodz = np.gradient(rho, z, axis=-1)
        b = g*alpha*(T - T0) - g*beta*(S - S0)
    # interpolate so all values are from the center, center, center of the grid cell
    w_face = make_interp_spline(zf, w, axis=-1, k=1)
    wc = w_face(z)

    if stokes:
        u = u - u_s
    
    b_background = stratification_profile(z, alpha*g*dTdz, mld)
    # calculate means
    u_avg = np.mean(u, axis=(-3, -2))
    v_avg = np.mean(v, axis=(-3, -2))
    w_avg = np.mean(w, axis=(-3, -2))
    wc_avg = np.mean(wc, axis=(-3, -2))
    b_avg = np.mean(b, axis=(-3, -2))
    rho_avg = np.mean(rho, axis=(-3, -2))
    S_avg = np.mean(S, axis=(-3, -2))
    T_avg = np.mean(T, axis=(-3, -2))

    # calculate fluctuations
    u_fluc = u-u_avg
    v_fluc = v-v_avg
    w_fluc = w-w_avg
    wc_fluc = wc-wc_avg
    rho_fluc = rho - rho_avg
    S_fluc = S - S_avg
    T_fluc = T - T_avg

    # calcualte reynolds stresses
    u_fluc_avg, u2_fluc, u2_fluc_avg = a2_fluc_mean(u_fluc)
    v_fluc_avg, v2_fluc, v2_fluc_avg = a2_fluc_mean(v_fluc)
    w_fluc_avg, w2_fluc, w2_fluc_avg = a2_fluc_mean(w_fluc)
    wc_fluc_avg, wc2_fluc, wc2_fluc_avg = a2_fluc_mean(wc_fluc)
    rho_fluc_avg, rho2_fluc, rho2_fluc_avg = a2_fluc_mean(rho_fluc)
    uv_fluc, uv_fluc_avg = ab_fluc_mean(u, v, u_avg, v_avg)
    uw_fluc, uw_fluc_avg = ab_fluc_mean(u, wc, u_avg, wc_avg)
    vw_fluc, vw_fluc_avg = ab_fluc_mean(v, wc, v_avg, wc_avg)

    b2_fluc, b2_fluc_avg = ab_fluc_mean(b, b, b_avg, b_avg)
    bu_fluc, bu_fluc_avg = ab_fluc_mean(b, u, b_avg, u_avg)
    bv_fluc, bv_fluc_avg = ab_fluc_mean(b, v, b_avg, v_avg)
    bw_fluc, bw_fluc_avg = ab_fluc_mean(b, wc, b_avg, wc_avg)
    
    # rms fluctuations
    u_rms = u2_fluc_avg**0.5
    v_rms = v2_fluc_avg**0.5
    w_rms = w2_fluc_avg**0.5
    wc_rms = wc2_fluc_avg**0.5
    b_rms = b2_fluc_avg**0.5

    if (vert_slice_plot or buoyancy_analysis_plot or xy_plot or turb_stats_plot or buoyancy_momentum_analysis) and rho_IC_perturb:
        # calculating density 
        rho_perturbed = ((b - b_background)*rho0)/(-g)
        b_fluc = b - b_background
    elif (vert_slice_plot or buoyancy_analysis_plot or xy_plot or turb_stats_plot or buoyancy_momentum_analysis) and not rho_IC_perturb:
        # calculating density 
        b_fluc = b - b_avg
        rho_perturbed = ((b_fluc)*rho0)/(-g)

    # buoyancy analysis 
    if buoyancy_analysis_plot or turb_stats_plot or buoyancy_momentum_analysis:
        # prepping variables for plume statistics
        dbdz = np.gradient(b, z, axis=-1)
        dbdz_avg = np.mean(dbdz, axis=(-3, -2))
        dbdz_vol_avg = np.mean(dbdz, axis=(0, 1, 2))

        b_fluc_avg = np.mean(b_fluc, axis=(-3, -2))
        db_flucdz = np.gradient(b_fluc, z, axis=-1)
        db_flucdz_avg = np.mean(db_flucdz, axis=(-3, -2))

        
        if salinity:
            bw_idx = np.where(bw_fluc_avg==np.max(bw_fluc_avg))[0][0]
            center_xy_loc, centerline_index, rp_profile, plume_index, S_contour = plume_tracer_analysis(x, y, z, lx, nx, S, idx = bw_idx, contour = 0.05)
            dbdx = np.mean(np.gradient(b, x, axis=0), axis=(-3, -2))
            dbdy = np.mean(np.gradient(b, y, axis=1), axis=(-3, -2))
            dbdz =  np.mean(np.gradient(b, z, axis=2), axis=(-3, -2))
            S_fluc_center = S_fluc[centerline_index[0], centerline_index[1], centerline_index[2]]
            T_fluc_center = T_fluc[centerline_index[0], centerline_index[1], centerline_index[2]]
        else:
            # centerline analysis
            u_center = u[int(nx[0]/2), int(nx[1]/2), :]
            v_center = v[int(nx[0]/2), int(nx[1]/2), :]
            w_center = w[int(nx[0]/2), int(nx[1]/2), :]
            wc_center = wc[int(nx[0]/2), int(nx[1]/2), :]
            u_fluc_center = u_fluc[int(nx[0]/2), int(nx[1]/2), :]
            v_fluc_center = v_fluc[int(nx[0]/2), int(nx[1]/2), :]
            w_fluc_center = w_fluc[int(nx[0]/2), int(nx[1]/2), :]
            wc_fluc_center = wc_fluc[int(nx[0]/2), int(nx[1]/2), :]

            b_center = b[int(nx[0]/2), int(nx[1]/2), :]
            b_fluc_center = b_fluc[int(nx[0]/2), int(nx[1]/2), :]
            dbdz_center = dbdz[int(nx[0]/2), int(nx[1]/2), :]
            db_flucdz_center = db_flucdz[int(nx[0]/2), int(nx[1]/2), :]
            rho_center = rho[int(nx[0]/2), int(nx[1]/2), :]
            rho_perturbed_center = rho_perturbed[int(nx[0]/2), int(nx[1]/2), :]
            bw_fluc_center = b_fluc_center*wc_center
            mld_index, w_mld, mld_bw_fluc, rho_mld = mld_info(w_center, bw_fluc_center, rho_perturbed_center, z, mld)
        
            neutral_index, max_index, dbdz_plume_avg = centerline_analysis_buoyancy(bw_fluc_center, dbdz_center, z, nx)
            z_intrusion = z[max_index]
            z_neutral = z[neutral_index]
            w_intrusion = wc_center[max_index]
            w_neutral = wc_center[neutral_index]
            bw_intrusion = bw_fluc_center[max_index]
            bw_neutral = bw_fluc_center[neutral_index]
            rho_intrusion = rho_perturbed_center[max_index]
            rho_neutral = rho_perturbed_center[neutral_index]

            #richardson number
            Ri_avg = richardson_number(dbdz_avg, z, u_avg, v_avg)
            Ri_strat = richardson_number(np.gradient(b_background, z, axis=-1), z, u_avg, v_avg)
            Ri_plume = richardson_number(dbdz_center, z, u_center, v_center)
        
            intrusion = np.array(z_intrusion)
            neutral = np.array(z_neutral)
        #b_difference = g*(rho_avg-rhoS-rho0)/rho0 #g*alpha*(T_fluc) - g*beta*(S - S0)
        Q, M, F, B, wm, dm, bm, Ri, area_idx, max_index, neutral_index = plume_momentum_analysis(centerline_index, center_xy_loc, nx, x, y, z, wc, b, b_fluc, rho_fluc, X, Y, dbdz_tol, g*beta*S_contour, w_mag_tol)

        wc_center = wc[centerline_index[0], centerline_index[1], centerline_index[2]]
        bw_fluc_center = b_fluc[centerline_index[0], centerline_index[1], centerline_index[2]]
        rho_perturbed_center = rho_perturbed[centerline_index[0], centerline_index[1], centerline_index[2]]
        b_center = b[centerline_index[0], centerline_index[1], centerline_index[2]]
        z_intrusion = z[max_index]
        z_neutral = z[neutral_index]
        w_intrusion = wc_center[max_index]
        w_neutral = wc_center[neutral_index]
        bw_intrusion = bw_fluc_center[max_index]
        bw_neutral = bw_fluc_center[neutral_index]
        rho_intrusion = rho_perturbed_center[max_index]
        rho_neutral = rho_perturbed_center[neutral_index]
        mld_index, w_mld, mld_bw_fluc, rho_mld = mld_info(wc_center, bw_fluc_center, rho_perturbed_center, z, mld)
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
        video_3d_dir = plot_3d_fields(time, it, ranges, output_folder, lx, X, Y, Z, X_zf, Y_zf, Z_zf, u, v, w, T, S)
    if vert_slice_plot:
        plane_slices_dir = vert_plane_slices(time, it, ranges, output_folder, lx, nx, X, X_zf, Y, Y_zf, Z, Z_zf, u, v, w, u_fluc, v_fluc, w_fluc, b_fluc, Pstatic, Pdynamic, rho, rho_perturbed, b, T, S)
    if xy_plot and salinity:
        loc = "max height +1"
        loc_idx = max_index + 1
        if loc_idx >(nx[2]-1):
            loc_idx = nx[2] - 1
        surface_dir = xy_plane_slices(time, it, xy_ranges, output_folder, lx, X, Y, u, v, w, b, b_fluc, Pdynamic, rho, rho_perturbed, loc_idx, loc, T, S)
    elif xy_plot and not salinity:
        loc = "max height + 5"
        surface_dir = xy_plane_slices(time, it, xy_ranges, output_folder, lx, X, Y, u, v, w, b, b_fluc, Pdynamic, rho, rho_perturbed, max_index, loc, T)
    if buoyancy_analysis_plot and not salinity:
        b_ranges = ranges.copy()
        buoyancy_dir = buoyancy_analysis(time, it, b_ranges, output_folder, lx, nx, z, zf, X, Z, mld, b_avg, b_background, w_avg, b_center, w_center, b_rms, bu_fluc_avg, bv_fluc_avg, bw_fluc_avg, b_fluc, rho_perturbed, Ri_avg, Ri_strat, Ri_plume, intrusion, neutral, w_neutral, w_intrusion, w_mld, rho_neutral, rho_intrusion, rho_perturbed_mld, bwfluc_neutral, bwfluc_intrusion, bwfluc_mld, alpha_vel, alpha_length, salinity)
    if buoyancy_analysis_plot and salinity:
        buoyancy_dir = plot_tracer_plume(time, it, ranges, output_folder, lx, nx, z, zf, Y, Z, mld, u_avg, v_avg, w_avg, uv_fluc_avg, uw_fluc_avg, vw_fluc_avg, u_rms, v_rms, w_rms, dbdx, dbdy, dbdz, b_avg, b_background, b_center, w_center, b_rms, bu_fluc_avg, bv_fluc_avg, bw_fluc_avg, b_fluc, rho_perturbed, S_avg, rp_list, plume_depths, ws, rhos, bw_flucs, l_scale_list)
    if plume_plot:
        plume_dir = plume_spatial_analysis(time, it, ranges, line_opt, output_folder, case_names, name, lx, z, zf, S_avg, u_rms, v_rms, w_rms, b_avg, b_center, rp_profile, bu_fluc_avg, bv_fluc_avg, bw_fluc_avg, b_rms)
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