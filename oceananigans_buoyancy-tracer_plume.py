import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm

import imageio.v2 as imageio
import matplotlib.ticker as mticker

from plotting_functions import stratification_profile, plot_ranges, turb_stats, plot_3d_fields, vert_plane_slices, buoyancy_analysis, xy_plane_slices, create_video
from general_analysis_functions import a2_fluc_mean, centerline_analysis_buoyancy, ab_fluc_mean, richardson_number, froude_number, reynolds_buoyancy_number, richardson_number_ratio, lamb_vector
from data_collection_functions import collect_time_outputs, collect_fields, collect_fields_distributed
# Set up folder and simulation parameters
folder = '/Users/annapauls/Library/CloudStorage/OneDrive-UCB-O365/CU-Boulder/TESLa/Carbon Sequestration/Simulations/Oceananigans/NBP/b tracer for NBP/with closure Re 3000/continuous source/varied buoyancy flux/b0 = -4E-1/'
output_folder = os.path.join(folder, "plotting outputs") 
name = 'Ri-NBP-'

# flags to analyze data 
rho_IC_perturb = False

# flags for what to plot
video_3d_flag = False
turb_stats_plot = False
vert_slice_plot = True
buoyancy_analysis_plot = True
xy_plot = False

# flags for how to read data
with_halos = False
closure = False
stokes = False

# physical parameters
mld = 30.0  # mixed layer depth in meters
g = 9.80665  # gravity in m/s^2
alpha = 2.0e-4 
dTdz = 0.01 # background temperature gradient in K/m
rho0 = 1026
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

# plot ranges
ranges = plot_ranges(lz = 96, rho0 = rho0, T0 = 25, dTdz = dTdz)

# List JLD2 files
dtn = sorted([f for f in os.listdir(folder) if f.endswith('.jld2')])
Nranks = len(dtn)

# Read model information
fid = os.path.join(folder, dtn[-1])
if Nranks == 1 and not stokes:
    time, t_save, nx, hx, lx, x, y, z, xf, yf, zf, visc = collect_time_outputs(fid, Nranks, stokes, closure)
elif Nranks == 1 and stokes:
    time, t_save, nx, hx, lx, x, y, z, xf, yf, zf, visc, u_f, u_s = collect_time_outputs(fid, Nranks, stokes, closure)
elif Nranks > 1 and not stokes:
    time, t_save, nx, hx, lx, x, y, z, xf, yf, zf, visc = collect_time_outputs(fid, Nranks, stokes, closure)
else:
    time, t_save, nx, hx, lx, x, y, z, xf, yf, zf, visc, u_f, u_s = collect_time_outputs(fid, Nranks, stokes, closure)
if rho_IC_perturb:
    name+='-rhoICperturbation-'
name+=f'Nx{nx[0]}_Ny{nx[1]}_Nz{nx[2]}'
print(name)

if video_3d_flag or turb_stats_plot or vert_slice_plot or buoyancy_analysis:
    nt = len(t_save)
else:
    nt = 1  # only last time step
X, Y, Z = np.meshgrid(x, y, z)
Xf, Yf, Zf = np.meshgrid(xf, yf, zf)
if buoyancy_analysis:
    L_ozmidov_average = []
    L_ozmidov_background = []
    L_ozmidov_plume = []
    plume_depth_intrusion = []
    plume_depth_neutral = []
    w_neutral = []
    w_intrusion = []
    w_mld = []
    b_neutral = []
    b_intrusion = []
    b_mld = []
    rho_perturbed_neutral = []
    rho_perturbed_intrusion = []
    rho_perturbed_mld = []
for it in range(nt):
    # Load data from files
    if Nranks == 1:
        u, v, w, b, Pdynamic, Pstatic = collect_fields(folder, dtn[0], t_save, it, hx, False, False, with_halos)
    else:
        u, v, w, b, Pdynamic, Pstatic = collect_fields_distributed(Nranks, folder, dtn, t_save, it, hx, nx, False, False, with_halos)
    #interpolate so all values are from the center, center, center of the grid cell
    wc = 0.5 * (w[..., :-1] + w[..., 1:])

    if stokes:
        u = u - u_s

    if buoyancy_analysis or turb_stats_plot:
        rho = ((b)*rho0)/(-g) + rho0
        # calculate means
        u_avg = np.mean(u, axis=(-3, -2))
        v_avg = np.mean(v, axis=(-3, -2))
        w_avg = np.mean(w, axis=(-3, -2))
        wc_avg = np.mean(wc, axis=(-3, -2))
        b_avg = np.mean(b, axis=(-3, -2))
        rho_avg = np.mean(rho, axis=(-3, -2))

        # calculate fluctuations
        u_fluc = u-u_avg
        v_fluc = v-v_avg
        w_fluc = w-w_avg
        wc_fluc = wc-wc_avg
        rho_fluc = rho - rho_avg

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

    if (vert_slice_plot or buoyancy_analysis_plot or xy_plot) and rho_IC_perturb:
        # calculating density 
        rho_perturbed = ((b - b_background)*rho0)/(-g)
        b_fluc = b - b_background
    elif (vert_slice_plot or buoyancy_analysis_plot or xy_plot) and not rho_IC_perturb:
        # calculating density 
        rho_perturbed = ((b - b_avg)*rho0)/(-g)
        b_fluc = b - b_avg

    # buoyancy analysis 
    if buoyancy_analysis_plot:
        # prepping variables for plume statistics
        dbdz = np.gradient(b, z, axis=-1)
        dbdz_avg = np.mean(dbdz, axis=(-3, -2))
        dbdz_vol_avg = np.mean(dbdz, axis=(0, 1, 2))

        b_fluc_avg = np.mean(b_fluc, axis=(-3, -2))
        db_flucdz = np.gradient(b_fluc, z, axis=-1)
        db_flucdz_avg = np.mean(db_flucdz, axis=(-3, -2))
        

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
        
        neutral_index, max_index, b_center, w_center, dbdz_plume_avg, z_max, z_neutral, max_w, max_b_fluc, max_rho_perturbed, neutral_w, neutral_b_fluc, neutral_rho_perturbed, mld_w, mld_b_fluc, mld_rho_perturbed = centerline_analysis_buoyancy(wc_center, b_center, b_fluc_center, dbdz_center, rho_perturbed_center, nx, mld)
        
        # ozmidov length scale
        #epsilon =visc_dissipation_rate(visc, u, v, wc, dx)
        #epsilon_avg = np.mean(epsilon, axis=(0, 1, 2))
        #Lo_plume = ozmidov_length(epsilon, dbdz_plume_avg)
        #L_ozmidov_background.append(ozmidov_length(epsilon_avg, g*dTdz*alpha))
        #L_ozmidov_average.append(ozmidov_length(epsilon_avg, dbdz_vol_avg))
        #L_ozmidov_plume.append(Lo_plume)

        #richardson number
        Ri_avg = richardson_number(dbdz_avg, z, u_avg, v_avg)
        Ri_strat = richardson_number(np.gradient(b_background, z, axis=-1), z, u_avg, v_avg)
        Ri_plume = richardson_number(dbdz_center, z, u_center, v_center)
        #Ri_avg_h = richardson_number_ratio(b_avg, rho_avg, rho0, z, u_fluc_avg, v_fluc_avg, wc_fluc_avg, lx)
        #Ri_strat_h = richardson_number_ratio(g*alpha*dTdz, rho_avg, rho0, z, u_fluc_avg, v_fluc_avg, wc_fluc_avg, lx)
        #Ri_plume_h = richardson_number_ratio(dbdz_center, rho_avg, rho0, z, u_fluc_center, v_fluc_center, wc_fluc_center, lx)
        #Ri_avg = [Ri_avg, Ri_avg_h]
        #Ri_strat = [Ri_strat, Ri_strat_h]
        #Ri_plume = [Ri_plume, Ri_plume_h]

        # appending plume statistics to lists
        plume_depth_intrusion.append(z_max)
        plume_depth_neutral.append(z_neutral)
        
        w_intrusion.append(max_w)
        w_neutral.append(neutral_w)
        w_mld.append(mld_w)

        b_neutral.append(neutral_b_fluc)
        b_intrusion.append(max_b_fluc)
        b_mld.append(mld_b_fluc)

        rho_perturbed_neutral.append(neutral_rho_perturbed)
        rho_perturbed_intrusion.append(max_rho_perturbed)
        rho_perturbed_mld.append(mld_rho_perturbed)
        
        intrusion = np.array(plume_depth_intrusion)
        neutral = np.array(plume_depth_neutral)
        
    ############ PLOTTING ############
    # --- Create Video ---
    if turb_stats_plot:
        turb_stat_dir = turb_stats(time, it, ranges, output_folder, lx, nx, z, zf, X, Z, u_avg, v_avg, w_avg, u_rms, v_rms, w_rms, uv_fluc_avg, uw_fluc_avg, vw_fluc_avg, b_rms, lamb_x_avg, lamb_y_avg, lamb_z_avg, b, Pdynamic, Pstatic, b_avg = b_avg)
    if video_3d_flag:
        video_3d_dir = plot_3d_fields(time, it, ranges, output_folder, lx, X, Y, Z, Xf, Yf, Zf, u, v, w, b = b)
    if vert_slice_plot:
        p_ranges = ranges.copy()
        #p_ranges['u'] = [-2*10**(-4), 2*10**(-4)]
        #p_ranges['u_fluc'] = [-2*10**(-4), 2*10**(-4)]
        #p_ranges['v'] = [-5*10**(-3), 5*10**(-3)]
        #p_ranges['v_fluc'] = [-5*10**(-3), 5*10**(-3)]
        #p_ranges['w'] = [-1*10**(-2), 1*10**(-2)]
        #p_ranges['w_fluc'] = [-1*10**(-2), 1*10**(-2)]
        plane_slices_dir = vert_plane_slices(time, it, p_ranges, output_folder, lx, nx, X, Xf, Y, Yf, Z, Zf, u, v, w, b, u_fluc, v_fluc, w_fluc, b_fluc, Pstatic, Pdynamic, rho, rho_perturbed)
    if xy_plot:
        surface_dir = xy_plane_slices(time, it, ranges, output_folder, lx, X, Y, u, v, w, b, u_fluc, v_fluc, w_fluc, b_fluc, Pstatic, Pdynamic, rho, rho_perturbed)
    if buoyancy_analysis_plot:
        b_ranges = ranges.copy()
        b_ranges['w'] = [-1.2*10**(-1), 1.2*10**(-1)]
        b_ranges['lengthscale'] = [0, 0.3]
        buoyancy_dir = buoyancy_analysis(time, it, b_ranges, output_folder, lx, nx, z, zf, X, Z, mld, b_avg, b_background, w_avg, b_center, w_center, b_rms, bu_fluc_avg, bv_fluc_avg, bw_fluc_avg, b_fluc, rho_perturbed, Ri_avg, Ri_strat, Ri_plume, intrusion, neutral, w_neutral, w_intrusion, w_mld, rho_perturbed_neutral, rho_perturbed_intrusion, rho_perturbed_mld, b_neutral, b_intrusion, b_mld)
print("All frames created.")
# creating videos
if turb_stats_plot:
    create_video(turb_stat_dir, output_folder, name, 'turbulence_statistics')
if video_3d_flag:
    create_video(video_3d_dir, output_folder, name, '3D_fields')
if vert_slice_plot:
    create_video(plane_slices_dir, output_folder, name, 'vert_plane_slices')
if xy_plot:
    create_video(surface_dir, output_folder, name, 'xy_plane_slices')
if buoyancy_analysis_plot:
    create_video(buoyancy_dir, output_folder, name, 'buoyancy_analysis')