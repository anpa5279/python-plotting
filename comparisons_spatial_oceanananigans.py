import os
import numpy as np
import matplotlib.pyplot as plt

from plotting_functions import plot_ranges, create_video
from general_analysis_functions import a2_fluc_mean, ab_fluc_mean
from plotting_comparisons import plot_format, plume_vertical_spatial_plot, plume_horizontal_spatial_plot
from data_collection_functions import collect_time_outputs, collect_fields, collect_fields_distributed, collect_temp_and_sal, collect_grid, collect_contour_val
from dense_plume_analysis import plume_tracer_radius
from data_manipulation_functions import fcc_ccc, cfc_ccc, ccf_ccc, z_line_interpolation, xy_line_interpolation

# flags for what to plot
plume_analysis_plot = True
plot_1d_z = True
plot_1d_y = False
transient_mld = False
video = True

# flags for how to read data
with_halos = False
closure = False
salinity = True
stokes = False

contour_bound = 0.05
name_uni = f'contour-{contour_bound:.2f}'
universal_folder = '/Users/annapauls/Library/CloudStorage/OneDrive-UCB-O365/CU-Boulder/TESLa/Carbon Sequestration/Simulations/Oceananigans/NBP/salinity and temperature/no noise circle inlet/'#vertical domain increase/dTdz = 0.01'

# selecting cases to compare
variations = 'flux' # 'MLD', 'flux', 'strat', 'length', 'else'
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
    Sj = np.array([0.05, 0.1, 0.15, 0.2]) # 
elif variations == 'length':
    folder_names =['nz = 77 z = 96.25 m', 'nz = 128 z = 160 m', 'nz = 192 z = 240 m']
    case_names =[r'L$_{\text{z}}$ = 96.25 m', r'L$_{\text{z}}$ = 160 m', r'L$_{\text{z}}$ = 240 m']
    num_cases = len(case_names)
    dTdz = 0.01 * np.ones(num_cases) # background temperature gradient in K/m
    mld = 60 * np.ones(num_cases) 
    Sj = np.array([0.1, 0.1, 0.1]) #
else:
    folder_names =['vertical domain increase/dTdz = 0.01/nz = 77 z = 96.25 m', 'S0 = 0.1 dTdz = 0.01 MLD = 60']
    case_names =[r'$\delta z$ = 1.25 m', r'$\delta z$ = 0.375 m']
    num_cases = len(case_names)
    dTdz = 0.01 * np.ones(num_cases) # background temperature gradient in K/m
    mld = 60 * np.ones(num_cases) 
    Sj = np.array([0.1, 0.1, 0]) #
# Set up folder and simulation parameters
if variations == 'length':
    universal_folder = '/Users/annapauls/Library/CloudStorage/OneDrive-UCB-O365/CU-Boulder/TESLa/Carbon Sequestration/Simulations/Oceananigans/NBP/salinity and temperature/no noise circle inlet/vertical domain increase/dTdz = 0.01'
else:
    universal_folder = '/Users/annapauls/Library/CloudStorage/OneDrive-UCB-O365/CU-Boulder/TESLa/Carbon Sequestration/Simulations/Oceananigans/NBP/salinity and temperature/no noise circle inlet'
fig_folder = os.path.join(universal_folder, 'comparison figures', variations + ' comparison figures', 'interpolated')

folders = []
for name in folder_names:
    folders.append(os.path.join(universal_folder, name))
output_folder = universal_folder

# physical parameters
rj = 5 # m, radius of salinity flux circle at the surface
g = 9.80665  # gravity in m/s^2
rho0 = 1026
T0 = 25
S0 = 0 

# plotting prep
ranges = plot_ranges(lz = 96, rho0 = rho0, T0 = T0, dTdz = np.max(dTdz), Sj = np.max(Sj))
ranges['S'] = [0, 1.8*10**-3]
ranges['vel_rms'] = [0, 4*10**-3]
ranges['bw_fluc'] = [-8*10**(-9), 8*10**(-9)]
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

# collecting model information for all cases

S_value = Sj
for n, folder in enumerate(folders):
    S_value[n], w_value = collect_contour_val(folder, 'interp_temporal_averages.h5')
S_contour = S_value*contour_bound
t_save = []
mld_idx = []

z = []
nx = []
lx = []
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
    time, t_save_temp, visc, diff, u_f, u_s = collect_time_outputs(fid, stokes, closure)
    if salinity:
        alpha, beta = collect_temp_and_sal(fid, salinity)
    else:
        alpha = collect_temp_and_sal(fid, salinity)
    t_save.append(t_save_temp)
    nx_temp, hx, lx_temp, x, y, z_temp, zf = collect_grid(folder, dtn, Nranks)
    z.append(z_temp)
    nx.append(nx_temp)
    lx.append(lx_temp)
nz = np.max(nx[:][2])

if video:
    nt = len(t_save[0])
    nt = np.arange(0, nt)
else:
    nt = len(t_save[0]) -1 # only last time step
    nt = [nt,]

if plot_1d_y:
    ranges_hor = ranges.copy()
    ranges_hor['S'] = [0, 3*10**(-2)]
    ranges_hor['vel_rms'] = [0, 4*10**-3]
    ranges_hor['bw_fluc'] = [-2*10**(-5), 2*10**(-5)]
    ranges_hor['b_flux'] = [-4*10**(-6), 4*10**(-6)]
    ranges_hor['b_fluc'] = [-2*10**(-4), 2*10**(-4)]
    ranges_hor['w'] = [-0.15, 0.15]
    ranges_hor['T'] = [T0 - 0.2, T0 + 0.2]
    loc_z = -mld
    hor_str = ' '.join([f"{depth} m" for depth in loc_z])
    name_xy = name_uni + f"at z = {hor_str}"

for it in nt:
    u_avg = []
    v_avg = []
    w_avg = []
    T_avg = []
    b_avg = []
    rho_avg = []
    u_rms = []
    v_rms = []
    w_rms = []
    u_fluc_avg = []
    v_fluc_avg = []
    w_fluc_avg = []
    uv_fluc_avg = []
    vw_fluc_avg = []
    uw_fluc_avg = []
    bu_fluc_avg = []
    bv_fluc_avg = []
    bw_fluc_avg = []
    if plume_analysis_plot:
        r_profile = []
        b_center = []
        T_fluc_center = []
        S_fluc_center = []
    if salinity:
        S_avg = []
    if plot_1d_y:
        centery = 0.0
        u_hor = []
        v_hor = []
        w_hor = []
        b_fluc_hor = []
        bu_fluc_hor = []
        bv_fluc_hor = []
        bw_fluc_hor = []
        T_hor = []
        S_hor = []
    for i, folder in enumerate(folders):
        # Load data from files
        if Nranks == 1 and not salinity:
            u, v, w, T, Pdynamic, Pstatic = collect_fields(folder, dtn, t_save[i][it], hx, nx, True, salinity, with_halos)
        elif Nranks == 1 and salinity:
            u, v, w, T, S, Pdynamic, Pstatic = collect_fields(folder, dtn, t_save[i][it], hx, nx, True, salinity, with_halos)
        elif Nranks > 1 and not salinity:
            u, v, w, T, Pdynamic, Pstatic = collect_fields_distributed(Nranks, folder, dtn, t_save[i][it], hx, nx, True, salinity, with_halos)
        else:
            u, v, w, T, S, Pdynamic, Pstatic = collect_fields_distributed(Nranks, folder, dtn, t_save[i][it], hx, nx[i], True, salinity, with_halos)
        if stokes:
            u = u - u_s
        # interpolate velocities to cell centers
        u = fcc_ccc(u)
        v = cfc_ccc(v)
        w = ccf_ccc(w)
        # convert temperature and salinity to buoyancy 
        if not salinity:
            rho_total = rho0 - rho0 * alpha * (T - T0)
            b = -g*alpha*(T - T0)
        else:
            rho_total = rho0 - rho0 * alpha * (T - T0)+ rho0 * beta * (S - S0)
            b = g*alpha*(T - T0) - g*beta*(S - S0)
            b_T = g*alpha*(T - T0)
            b_S = - g*beta*(S - S0)

        # calculate means
        u_avg.append(np.mean(u, axis=(-3, -2)))
        v_avg.append(np.mean(v, axis=(-3, -2)))
        w_avg.append(np.mean(w, axis=(-3, -2)))
        b_avg.append(np.mean(b, axis=(-3, -2)))
        rho_avg.append(np.mean(rho_total, axis=(-3, -2)))
        S_avg.append(np.mean(S, axis=(-3, -2)))
        T_avg.append(np.mean(T, axis=(-3, -2)))

        # calculate fluctuations
        u_fluc = u-u_avg[i]
        v_fluc = v-v_avg[i]
        w_fluc = w-w_avg[i]
        T_fluc = T - T_avg[i]
        S_fluc = S - S_avg[i]

        # calcualte reynolds stresses
        u_fluc_avg_temp, u2_fluc, u2_fluc_avg = a2_fluc_mean(u_fluc)
        v_fluc_avg_temp, v2_fluc, v2_fluc_avg = a2_fluc_mean(v_fluc)
        w_fluc_avg_temp, w2_fluc, w2_fluc_avg = a2_fluc_mean(w_fluc)
        uv_fluc, uv_fluc_avg_temp = ab_fluc_mean(u, v, u_avg[i], v_avg[i])
        uw_fluc, uw_fluc_avg = ab_fluc_mean(u, w, u_avg[i], w_avg[i])
        vw_fluc, vw_fluc_avg = ab_fluc_mean(v, w, v_avg[i], w_avg[i])

        b2_fluc, b2_fluc_avg = ab_fluc_mean(b, b, b_avg[i], b_avg[i])
        bu_fluc, bu_fluc_avg_temp = ab_fluc_mean(b, u, b_avg[i], u_avg[i])
        bv_fluc, bv_fluc_avg_temp = ab_fluc_mean(b, v, b_avg[i], v_avg[i])
        bw_fluc, bw_fluc_avg_temp = ab_fluc_mean(b, w, b_avg[i], w_avg[i])

        bu_fluc_avg.append(bu_fluc_avg_temp)
        bv_fluc_avg.append(bv_fluc_avg_temp)
        bw_fluc_avg.append(bw_fluc_avg_temp)

        if plot_1d_z:
            # rms fluctuations
            u_rms.append(u2_fluc_avg**0.5)
            v_rms.append(v2_fluc_avg**0.5)
            w_rms.append(w2_fluc_avg**0.5)

        # dense plume analysis
        if salinity:
            centerx = 0.0
            centery = 0.0
            rp_profile, plume_index = plume_tracer_radius(x, y, nx[i], S, S_contour[i])
            r_profile.append(rp_profile)
            b_center.append(z_line_interpolation(b, x, y, centerx, centery))
            T_fluc_center.append(z_line_interpolation(T_fluc, x, y, centerx, centery))
            S_fluc_center.append(z_line_interpolation(S_fluc, x, y, centerx, centery))
        if plot_1d_y:
            u_hor.append(xy_line_interpolation(u, y, z[i], centery, loc_z[i]))
            v_hor.append(xy_line_interpolation(v, y, z[i], centery, loc_z[i]))
            w_hor.append(xy_line_interpolation(w, y, z[i], centery, loc_z[i]))
            b_fluc_hor.append(xy_line_interpolation(b-b_avg[i], y, z[i], centery, loc_z[i]))
            bu_fluc_hor.append(xy_line_interpolation(bu_fluc, y, z[i], centery, loc_z[i]))
            bv_fluc_hor.append(xy_line_interpolation(bv_fluc, y, z[i], centery, loc_z[i]))
            bw_fluc_hor.append(xy_line_interpolation(bw_fluc, y, z[i], centery, loc_z[i]))
            T_hor.append(xy_line_interpolation(T, y, z[i], centery, loc_z[i]))
            S_hor.append(xy_line_interpolation(S, y, z[i], centery, loc_z[i]))
    ############ PLOTTING ############
    if plot_1d_z:
        buoyancy_dir_z = plume_vertical_spatial_plot(time, it, ranges, color_opt, fig_folder, case_names, name_uni, lx, z, S_avg, u_rms, v_rms, w_rms, b_avg, b_center, r_profile, bu_fluc_avg, bv_fluc_avg, bw_fluc_avg, T_avg, T_fluc_center, S_fluc_center)
    if plot_1d_y:
        buoyancy_dir_y = plume_horizontal_spatial_plot(time, it, ranges_hor, color_opt, fig_folder, case_names, name_xy, lx, y, u_hor, v_hor, w_hor, b_fluc_hor, bu_fluc_hor, bv_fluc_hor, bw_fluc_hor, T_hor, S_hor)
print("All frames created.")
# creating videos
if video:
    if plot_1d_z:
        create_video(buoyancy_dir_z, fig_folder, name_uni, 'vertical profile')
    if plot_1d_y:
        create_video(buoyancy_dir_y, fig_folder, name_uni, 'horizontal profile')
