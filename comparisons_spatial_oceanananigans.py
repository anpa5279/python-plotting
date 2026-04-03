import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

from plotting_functions import plot_ranges, create_video
from general_analysis_functions import a2_fluc_mean, ab_fluc_mean
from plotting_comparisons import plot_format, turb_stats_multi, plume_vertical_spatial_plot
from data_collection_functions import collect_time_outputs, collect_fields, collect_fields_distributed, collect_temp_and_sal
from dense_plume_analysis import plume_tracer_radius

# Set up folder and simulation parameters
universal_folder = '/Users/annapauls/Library/CloudStorage/OneDrive-UCB-O365/CU-Boulder/TESLa/Carbon Sequestration/Simulations/Oceananigans/NBP/salinity and temperature/'
folder_names =['beta = default S0 = 0.1 MLD = 20m', 'beta = default S0 = 0.1', 'beta = default S0 = 0.1 MLD = 40m']
#['beta = default S0 = 0.05', 'beta = default S0 = 0.1', 'beta = default S0 = 0.15', 'beta = default S0 = 0.2']
#['beta = default S0 = 0.1', 'beta = default S0 = 0.1 with Langmuir']
#['beta = default S0 = 0.1 with wind stress', 'beta = default S0 = 0.1']
#['beta = default S0 = 0.1', 'beta = default S0 = 0.1 dTdz = 0.05', 'beta = default S0 = 0.1 dTdz = 0.1'] 
#['beta = default S0 = 0.1 MLD = 20m', 'beta = default S0 = 0.1', 'beta = default S0 = 0.1 MLD = 40m']
fig_folder = os.path.join(universal_folder, 'comparison figures')
case_names = [r'MLD = 20m', r'MLD = 30m', r'MLD = 40m']  
#[r'F$^{\text{C}} = -5.0*10^{-5}$', r'F$^{\text{C}} = -1.0*10^{-4}$', r'F$^{\text{C}} = -1.5*10^{-4}$', r'F$^{\text{C}} = - 2.0*10^{-4}$']
#[r'dTdz = 0.01', r'dTdz = 0.05', r'dTdz = 0.10'] 
#[r'MLD = 20m', r'MLD = 30m', r'MLD = 40m'] 
name_uni = "contour-0.15-MLD-test"

num_cases = len(case_names)
folders = []
for name in folder_names:
    folders.append(os.path.join(universal_folder, name))
output_folder = universal_folder

# flags for what to plot
plume_analysis_plot = True
turb_stats_plot = False
ND = False

# flags for how to read data
with_halos = False
stokes = False * np.ones(num_cases) 
salinity = True

video = False
if num_cases > 1:
    fig_folder = os.path.join(universal_folder, 'comparison figures', 'comparison plume analysis')
else:
    fig_folder = os.path.join(universal_folder, 'plume analysis')
if video:
    fig_folder = os.path.join(fig_folder, name_uni)
    os.makedirs(fig_folder, exist_ok=True)

# physical parameters
rj = 10 # m, radius of salinity flux circle at the surface
g = 9.80665  # gravity in m/s^2
dTdz = 0.01 * np.ones(num_cases) #  np.array([0.01, 0.05, 0.1]) # background temperature gradient in K/m
rho0 = 1026
mld = np.array([20, 30, 40]) # 30 * np.ones(num_cases) # 
T0 = 25.0
S0 = 0 
wp = 0.001
Sj = 0.1 * np.ones(num_cases) # np.array([0.05, 0.1, 0.15, 0.2])# 
F_s = np.dot(Sj, wp)
if np.size(Sj) == 1:
    contour = np.dot(Sj * np.ones(num_cases), 0.05)
else:
    contour = np.dot(Sj, 0.05)

S_value = np.array([0.03995705848735615, 0.03602588163919859, 0.032189606877616704]) # for MLD variations
#np.array([0.03995705848735615, 0.03602588163919859, 0.032189606877616704]) # for MLD variations
#np.array([0.03602588163919859, 0.03995705848735615, 0.042189206877616705]) # for dTdz variations
#np.dot([0.0010948250136870168, 0.0018012940819599295, 0.0024005411329652226, 0.0029359463404349034], 20) # for Sj variations 
S_contour = S_value*0.15 

# plotting prep
ranges = plot_ranges(lz = 96, rho0 = rho0, T0 = T0, dTdz = np.max(dTdz), Sj = np.max(Sj))
ranges['S'] = [0, 1.8*10**-3]
ranges['vel_rms'] = [0, 4*10**-3]
ranges['bw_fluc'] = [-8*10**(-9), 8*10**(-9)]
ranges['b_rms'] = [0, 1.5*10**(-5)]
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
    if not stokes[i]:
        time, t_save_temp, nx, hx, lx, x, y, z, xf, yf, zf, dx, visc, diff = collect_time_outputs(fid, Nranks, stokes[i])
    else:
        time, t_save_temp, nx, hx, lx, x, y, z, xf, yf, zf, dx, visc, diff, u_f, u_s = collect_time_outputs(fid, Nranks, stokes[i])
        #u_s = stokes_exp(z)
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

if video:
    nt = len(t_save[0])
    nt = np.arange(0, nt)
else:
    nt = len(t_save[0]) -1 # only last time step
    nt = [nt,]

z = (z*np.ones([num_cases, nx[2]])).T
zf = (zf*np.ones([num_cases, nx[2] + 1])).T
Sj = F_s / (np.sqrt(g  * rj))

############ NONDIMENSIONALIZATION ############
if ND: 
    name_nd = 'ND_' + name_uni

    vel_scale = np.zeros(num_cases)
    b_scale = np.zeros(num_cases)
    N2 = np.zeros(num_cases)
    z_nd = np.zeros((nx[2], num_cases))
    zf_nd = np.zeros((nx[2]+1, num_cases))
    for i in range(num_cases):
        N2[i] = g  * dTdz[i] / T0
        b_scale[i] = mld[i] * N2[i]
        vel_scale[i] = mld[i] * np.sqrt(N2[i])
        Sj[i] = F_s[i]/ (vel_scale[i]) # / (np.sqrt(g  * rj)) #
        z_nd[:, i] = z[:, i] / mld[i]
        zf_nd[:, i] = zf[:, i] / mld[i]
        bflux_scale = b_scale[i] * vel_scale[i]

    lx_nd= np.array(lx) / np.min(mld)
    bflux_scale = b_scale * vel_scale
    nd_ranges = ranges.copy()
    nd_ranges['vel_rms'] = nd_ranges['vel_rms'] / np.min(vel_scale)
    nd_ranges['b_avg'] = nd_ranges['b_avg'] / np.min(b_scale)
    nd_ranges['bw_fluc'] = nd_ranges['bw_fluc'] / np.min(bflux_scale)
    nd_ranges['b_rms'] = nd_ranges['b_rms'] / np.min(b_scale)
    nd_ranges['S'] = nd_ranges['S'] / np.min(Sj)
    nd_ranges['S_fluc'] = nd_ranges['S_fluc'] / np.min(Sj)
    nd_ranges['T_fluc'] = nd_ranges['T_fluc'] / T0

start_neutral = np.zeros(num_cases).astype(int)
for it in nt:
    u_avg = np.zeros((nx[2], num_cases))
    v_avg = np.zeros((nx[2], num_cases))
    w_avg = np.zeros((nx[2] + 1, num_cases))
    wc_avg = np.zeros((nx[2], num_cases))
    T_avg = np.zeros((nx[2], num_cases))
    b_avg = np.zeros((nx[2], num_cases))
    rho_avg = np.zeros((nx[2], num_cases))
    u_rms = np.zeros((nx[2], num_cases))
    v_rms = np.zeros((nx[2], num_cases))
    w_rms = np.zeros((nx[2] + 1, num_cases))
    wc_rms = np.zeros((nx[2], num_cases))
    b_rms = np.zeros((nx[2], num_cases))
    u_fluc_avg = np.zeros((nx[2], num_cases))
    v_fluc_avg = np.zeros((nx[2], num_cases))
    w_fluc_avg = np.zeros((nx[2] + 1, num_cases))
    uv_fluc_avg = np.zeros((nx[2], num_cases))
    vw_fluc_avg = np.zeros((nx[2], num_cases))
    uw_fluc_avg = np.zeros((nx[2], num_cases))
    wc_fluc_avg = np.zeros((nx[2], num_cases))
    bu_fluc_avg = np.zeros((nx[2], num_cases))
    bv_fluc_avg = np.zeros((nx[2], num_cases))
    bw_fluc_avg = np.zeros((nx[2], num_cases))
    if plume_analysis_plot:
        r_profile = np.zeros((nx[2], num_cases))
        b_center = np.zeros((nx[2], num_cases))
        T_fluc_center = np.zeros((nx[2], num_cases))
        S_fluc_center = np.zeros((nx[2], num_cases))
    if salinity:
        S_avg = np.zeros((nx[2], num_cases))
    for i, folder in enumerate(folders):
        # Load data from files
        if Nranks == 1 and not salinity:
            u, v, w, T, Pdynamic, Pstatic = collect_fields(folder, dtn, t_save[i][it], hx, nx, True, salinity, with_halos)
        elif Nranks == 1 and salinity:
            u, v, w, T, S, Pdynamic, Pstatic = collect_fields(folder, dtn, t_save[i][it], hx, nx, True, salinity, with_halos)
        elif Nranks > 1 and not salinity:
            u, v, w, T, Pdynamic, Pstatic = collect_fields_distributed(Nranks, folder, dtn, t_save[i][it], hx, nx, True, salinity, with_halos)
        else:
            u, v, w, T, S, Pdynamic, Pstatic = collect_fields_distributed(Nranks, folder, dtn, t_save[i][it], hx, nx, True, salinity, with_halos)
        if stokes[i]:
            u = u - u_s
        # convert temperature and salinity to buoyancy 
        if not salinity:
            rho_total = rho0 - rho0 * alpha * (T - T0)
            b = -g*alpha*(T - T0)
        else:
            rho_total = rho0 - rho0 * alpha * (T - T0)+ rho0 * beta * (S - S0)
            b = g*alpha*(T - T0) - g*beta*(S - S0)
            b_T = g*alpha*(T - T0)
            b_S = - g*beta*(S - S0)
        # interpolate so all values are from the center, center, center of the grid cell
        w_face = make_interp_spline(zf[:, i], w, axis=-1, k=1)
        wc = w_face(z[:, i])

        # calculate means
        u_avg[:, i] = np.mean(u, axis=(-3, -2))
        v_avg[:, i] = np.mean(v, axis=(-3, -2))
        w_avg[:, i] = np.mean(w, axis=(-3, -2))
        wc_avg[:, i] = np.mean(wc, axis=(-3, -2))
        b_avg[:, i] = np.mean(b, axis=(-3, -2))
        rho_avg[:, i] = np.mean(rho_total, axis=(-3, -2))
        S_avg[:, i] = np.mean(S, axis=(-3, -2))
        T_avg[:, i] = np.mean(T, axis=(-3, -2))

        # calculate fluctuations
        u_fluc = u-u_avg[:, i]
        v_fluc = v-v_avg[:, i]
        w_fluc = w-w_avg[:, i]
        wc_fluc = wc-wc_avg[:, i]
        T_fluc = T - T_avg[:, i]
        S_fluc = S - S_avg[:, i]

        # calcualte reynolds stresses
        u_fluc_avg[:, i], u2_fluc, u2_fluc_avg = a2_fluc_mean(u_fluc)
        v_fluc_avg[:, i], v2_fluc, v2_fluc_avg = a2_fluc_mean(v_fluc)
        w_fluc_avg[:, i], w2_fluc, w2_fluc_avg = a2_fluc_mean(w_fluc)
        wc_fluc_avg[:, i], wc2_fluc, wc2_fluc_avg = a2_fluc_mean(wc_fluc)
        uv_fluc, uv_fluc_avg[:, i] = ab_fluc_mean(u, v, u_avg[:, i], v_avg[:, i])
        uw_fluc, uw_fluc_avg[:, i] = ab_fluc_mean(u, wc, u_avg[:, i], wc_avg[:, i])
        vw_fluc, vw_fluc_avg[:, i] = ab_fluc_mean(v, wc, v_avg[:, i], wc_avg[:, i])

        b2_fluc, b2_fluc_avg = ab_fluc_mean(b, b, b_avg[:, i], b_avg[:, i])
        bu_fluc, bu_fluc_avg[:, i] = ab_fluc_mean(b, u, b_avg[:, i], u_avg[:, i])
        bv_fluc, bv_fluc_avg[:, i] = ab_fluc_mean(b, v, b_avg[:, i], v_avg[:, i])
        bw_fluc, bw_fluc_avg[:, i] = ab_fluc_mean(b, wc, b_avg[:, i], wc_avg[:, i])
  
        # rms fluctuations
        u_rms[:, i] = u2_fluc_avg**0.5
        v_rms[:, i] = v2_fluc_avg**0.5
        w_rms[:, i] = w2_fluc_avg**0.5
        wc_rms[:, i] = wc2_fluc_avg**0.5
        b_rms[:, i] = b2_fluc_avg**0.5
        
        # dense plume analysis
        if salinity:
            bw_idx = np.where(bw_fluc_avg==np.max(bw_fluc_avg))[0][0]
            rp_profile, plume_index, S_contour_temp = plume_tracer_radius(x, y, nx, centerline_index, S, S_contour[i])
            r_profile[:, i] = rp_profile
            b_center[:, i] = b[centerline_index[0], centerline_index[1], centerline_index[2]]
            T_fluc_center[:, i] = T_fluc[centerline_index[0], centerline_index[1], centerline_index[2]]
            S_fluc_center[:, i] = S_fluc[centerline_index[0], centerline_index[1], centerline_index[2]]

    ############ PLOTTING ############
    # --- Create Video ---
    if turb_stats_plot:
        turb_stat_dir = turb_stats_multi(time, it, ranges, color_opt, fig_folder, case_names, name_uni, lx, z, zf, u_avg, v_avg, w_avg, u_rms, v_rms, w_rms, uv_fluc_avg, uw_fluc_avg, vw_fluc_avg, bu_fluc_avg, bv_fluc_avg, bw_fluc_avg, b_rms, rho_avg)
    if plume_analysis_plot:
        buoyancy_dir = plume_vertical_spatial_plot(time, it, ranges, color_opt, fig_folder, case_names, name_uni, lx, z, zf, S_avg, u_rms, v_rms, w_rms, b_avg, b_center, r_profile, bu_fluc_avg, bv_fluc_avg, bw_fluc_avg, b_rms, T_fluc_center, S_fluc_center)
    if ND:
        ############ NONDIMENSIONALIZATION ############
        for i in range(num_cases):
            S_avg[:, i] = S_avg[:, i]/Sj[i]
            u_rms[:, i] = u_rms[:, i]/vel_scale[i]
            v_rms[:, i] = v_rms[:, i]/vel_scale[i]
            w_rms[:, i] = w_rms[:, i]/vel_scale[i]
            b_avg[:, i] = b_avg[:, i]/b_scale[i]
            bv_fluc_avg[:, i] = bv_fluc_avg[:, i]/bflux_scale[i]
            bu_fluc_avg[:, i] = bu_fluc_avg[:, i]/bflux_scale[i]
            bw_fluc_avg[:, i] = bw_fluc_avg[:, i]/bflux_scale[i]
            b_rms[:, i] = b_rms[:, i]/b_scale[i]
            b_center[:, i] = b_center[:, i]/b_scale[i]
            T_fluc_center[:, i] = T_fluc_center[:, i]/(T0)
            S_fluc_center[:, i] = S_fluc_center[:, i]/Sj[i]
            r_profile[:, i] = r_profile[:, i]/mld[i] #/rj
        if turb_stats_plot:
            turb_stat_dir_nd = turb_stats_multi(time, it, nd_ranges, color_opt, fig_folder, case_names, name_nd, lx_nd, z_nd, zf_nd, u_avg, v_avg, w_avg, u_rms, v_rms, w_rms, uv_fluc_avg, uw_fluc_avg, vw_fluc_avg, bu_fluc_avg, bv_fluc_avg, bw_fluc_avg, b_rms, rho_avg, ND)
        if plume_analysis_plot:
            buoyancy_dir_nd = plume_vertical_spatial_plot(time, it, nd_ranges, color_opt, fig_folder, case_names, name_nd, lx_nd, z_nd, zf_nd, S_avg, u_rms, v_rms, w_rms, b_avg, b_center, r_profile, bu_fluc_avg, bv_fluc_avg, bw_fluc_avg, b_rms, T_fluc_center, S_fluc_center, ND)
print("All frames created.")
# creating videos
if video:
    if turb_stats_plot:
        create_video(turb_stat_dir, fig_folder, name_uni, 'turbulence_statistics')
    if plume_analysis_plot:
        create_video(buoyancy_dir, fig_folder, name_uni, 'buoyancy_analysis')
if video and ND:
    if turb_stats_plot:
        create_video(turb_stat_dir_nd, fig_folder, name_nd, 'turbulence_statistics')
    if plume_analysis_plot:
        create_video(buoyancy_dir_nd, fig_folder, name_nd, 'buoyancy_analysis')
