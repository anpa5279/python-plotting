import os
import numpy as np
import matplotlib.pyplot as plt

from plotting_functions import plot_ranges, create_video
from general_analysis_functions import a2_fluc_mean, ab_fluc_mean
from plotting_comparisons import plot_format, plume_vertical_spatial_plot, plume_horizontal_spatial_plot
from data_collection_functions import collect_time_outputs, collect_fields, collect_fields_distributed, collect_temp_and_sal, collect_grid, collect_contour_val
from dense_plume_analysis import plume_tracer_radius
from data_manipulation_functions import fcc_ccc, cfc_ccc, ccf_ccc, vertical_line_interpolation

# flags for what to plot
plume_analysis_plot = True
plot_1d_z = False
plot_1d_y = True
ND = False
transient_mld = False
video = True

# flags for how to read data
with_halos = False
closure = False
salinity = True
stokes = False

contour_bound = 0.05
name_uni = f'contour-{contour_bound:.2f}'

# selecting cases to compare
variations = 'strat' # 'MLD', 'flux', 'strat'
if variations == 'strat':
    folder_names =['S0 = 0.1 dTdz = 0.005 MLD = 60', 'S0 = 0.1 dTdz = 0.01 MLD = 60', 'S0 = 0.1 dTdz = 0.05 MLD = 60', 'S0 = 0.1 dTdz = 0.1 MLD = 60'] 
    case_names =[r'dTdz = 0.005', r'dTdz = 0.01', r'dTdz = 0.05', r'dTdz = 0.10']  
    num_cases = len(case_names)
    dTdz = np.array([0.005, 0.01, 0.05, 0.1]) # background temperature gradient in K/m
    mld = 60 * np.ones(num_cases) 
    Sj = 0.1 * np.ones(num_cases) 
    S_value = Sj
    for n, folder in enumerate(folders):
        S_value[n], w_value = collect_contour_val(folder, 'temporal_averages.h5')
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
# Set up folder and simulation parameters
universal_folder = '/Users/annapauls/Library/CloudStorage/OneDrive-UCB-O365/CU-Boulder/TESLa/Carbon Sequestration/Simulations/Oceananigans/NBP/salinity and temperature/no noise circle inlet'
fig_folder = os.path.join(universal_folder, 'comparison figures/contour 0.15/')
name_uni ='contour-0.15-dTdz'
folders = []
for name in folder_names:
    folders.append(os.path.join(universal_folder, name))
output_folder = universal_folder

# physical parameters
rj = 10 # m, radius of salinity flux circle at the surface
g = 9.80665  # gravity in m/s^2
rho0 = 1026
T0 = 25
S0 = 0 
wp = 0.001
F_s = np.dot(Sj, wp)

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

# collecting model information for all cases

S_value = Sj
for n, folder in enumerate(folders):
    S_value[n], w_value = collect_contour_val(folder, 'temporal_averages.h5')
S_contour = S_value*contour_bound
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
    time, t_save_temp, visc, diff, u_f, u_s = collect_time_outputs(fid, Nranks, stokes, closure)
    if salinity:
        alpha, beta = collect_temp_and_sal(fid, salinity)
    else:
        alpha = collect_temp_and_sal(fid, salinity)
    t_save.append(t_save_temp)
nx, hx, lx, x, y, z, zf = collect_grid(folder, dtn[0], Nranks)

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

    area = (2*rj)**2 
    l_area = np.sqrt(area)
    N2 = g * alpha * dTdz 
    Ri_g = (N2/g*l_area)**(1/3)
    Fr_flux = F_s * beta / np.sqrt(l_area * g)
    vel_scale = np.sqrt(l_area * g) / Ri_g #Fr_flux * 
    b_scale = Ri_g / Fr_flux * g
    F_b_scale = b_scale * vel_scale
    T_scale = Ri_g * Fr_flux / alpha
    S_scale = Ri_g * Fr_flux / beta
    F_T_scale = beta * F_s / alpha
    F_S_scale = F_s * np.sqrt(l_area * dTdz * alpha)
    hor_scale = l_area * Fr_flux

    F0 = area * beta * g * F_s
    Ln =(F0/N2**(3/2))**(1/4)
    z_nd = (z+mld)*(mld)**(1/3)/(Ln**(4/3))
    zf_nd = (zf+mld)*(mld)**(1/3)/(Ln**(4/3))

    y_nd = y / l_area
    lx_nd = np.zeros(3)
    lx_nd[0:2]= np.array(lx[0:2])/ np.min(hor_scale)
    lx_nd[-1] = np.max((lx[-1] - mld) * dTdz * alpha)


    nd_ranges = ranges.copy()
    nd_ranges['vel_rms'] = nd_ranges['vel_rms'] / np.min(vel_scale)
    nd_ranges['w'] = nd_ranges['w'] / np.min(vel_scale)
    nd_ranges['b_avg'] = nd_ranges['b_avg'] / np.min(b_scale)
    nd_ranges['bw_fluc'] = nd_ranges['bw_fluc'] / np.min(F_b_scale)
    nd_ranges['b_rms'] = nd_ranges['b_rms'] / np.min(b_scale)
    nd_ranges['b_fluc'] = nd_ranges['b_fluc'] / np.min(b_scale)
    nd_ranges['S'] = nd_ranges['S'] / np.min(S_scale)
    nd_ranges['S_fluc'] = nd_ranges['S_fluc'] / np.min(S_scale)
    nd_ranges['T_fluc'] = nd_ranges['T_fluc'] / np.min(T_scale)
    nd_ranges['T'] = nd_ranges['T'] / np.min(T_scale)

if plot_1d_y:
    ranges_hor = ranges.copy()
    ranges_hor['S'] = [0, 3*10**(-2)]
    ranges_hor['vel_rms'] = [0, 4*10**-3]
    ranges_hor['bw_fluc'] = [-2*10**(-6), 2*10**(-6)]
    ranges_hor['b_flux'] = [-1*10**(-5), 1*10**(-5)]
    ranges_hor['b_fluc'] = [-1*10**(-4), 1*10**(-4)]
    ranges_hor['w'] = [-0.1, 0.1]
    ranges_hor['T'] = [T0 - 0.25, T0 + 0.25]
    loc_z = -mld
    hor_str = ' '.join([f"{depth} m" for depth in loc_z])
    name_xy = name_uni + f"at z = {hor_str}"

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
        # interpolate so all values are from the center, center, center of the grid cell
        wc = 0.5 * (w[..., :-1] + w[..., 1:])

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
            centerx = 0.0
            centery = 0.0
            rp_profile, plume_index, S_contour_temp = plume_tracer_radius(x, y, centerx, centery, nx, S, S_contour[i])
            r_profile[:, i] = rp_profile
            b_center[:, i] = vertical_line_interpolation(b, x, y, centerx, centery)
            T_fluc_center[:, i] = vertical_line_interpolation(T_fluc, x, y, centerx, centery)
            S_fluc_center[:, i] = vertical_line_interpolation(S_fluc, x, y, centerx, centery)

    ############ PLOTTING ############
    ############ NONDIMENSIONALIZATION ############
    if ND:
        if transient_mld:
            Ln =(F0/N2**(3/2))**(1/4)
            z_nd = (z+mld)*(mld)**(1/3)/(Ln**(4/3))
            zf_nd = (zf+mld)*(mld)**(1/3)/(Ln**(4/3))
            lx_nd[-1] = np.max((lx[-1] - mld) * dTdz * alpha)
        ############ PLOTTING ############
        if plot_1d_z and salinity:
            bv_fluc_avg = bv_fluc_avg/F_b_scale
            bu_fluc_avg = bu_fluc_avg/F_b_scale
            bw_fluc_avg = bw_fluc_avg/F_b_scale
            S_avg = S_avg/S_scale
            b_avg = b_avg/b_scale
            b_rms = b_rms/b_scale
            b_center = b_center/b_scale
            T_fluc_center = T_fluc_center / T_scale
            S_fluc_center = S_fluc_center / S_scale
            r_profile = r_profile / hor_scale
            u_rms = u_rms/vel_scale
            v_rms = v_rms/vel_scale
            w_rms = w_rms/vel_scale
            #print(f"it = {it}, mld_idx = {mld_idx}, mld = {mld}, dbdz at mld = {dbdz[mld_idx, np.arange(num_cases)]}")
            buoyancy_dir_z_nd = plume_vertical_spatial_plot(time, it, nd_ranges, color_opt, fig_folder, case_names, name_nd, lx_nd, z_nd, zf_nd, S_avg, u_rms, v_rms, w_rms, b_avg, b_center, r_profile, bu_fluc_avg, bv_fluc_avg, bw_fluc_avg, b_rms, T_fluc_center, S_fluc_center, ND, z_nd = r"(z - h$_{MLD}$)h$_{MLD}^{1/3}$/L$_N^{4/3}$")
        if plot_1d_y and salinity:
            u_hor = u_hor/vel_scale
            v_hor = v_hor/vel_scale
            w_hor = w_hor/vel_scale
            b_fluc_hor = b_fluc_hor/b_scale
            bu_fluc_hor = bu_fluc_hor/F_b_scale
            bv_fluc_hor = bv_fluc_hor/F_b_scale
            bw_fluc_hor = bw_fluc_hor/F_b_scale
            T_hor = T_hor / T_scale
            S_hor = S_hor / S_scale
            buoyancy_dir_y_nd = plume_horizontal_spatial_plot(time, it, nd_ranges, color_opt, fig_folder, case_names, name_nd, lx_nd, y_nd, u_hor, v_hor, w_hor, b_fluc_hor, bu_fluc_hor, bv_fluc_hor, bw_fluc_hor, T_hor, S_hor, ND)
    else:
        if plot_1d_z:
            buoyancy_dir_z = plume_vertical_spatial_plot(time, it, ranges, color_opt, fig_folder, case_names, name_uni, lx, z, zf, S_avg, u_rms, v_rms, w_rms, b_avg, b_center, r_profile, bu_fluc_avg, bv_fluc_avg, bw_fluc_avg, b_rms, T_fluc_center, S_fluc_center)
        if plot_1d_y:
            buoyancy_dir_y = plume_horizontal_spatial_plot(time, it, ranges_hor, color_opt, fig_folder, case_names, name_xy, lx, y, u_hor, v_hor, w_hor, b_fluc_hor, bu_fluc_hor, bv_fluc_hor, bw_fluc_hor, T_hor, S_hor)
print("All frames created.")
# creating videos
if video:
    if plot_1d_z:
        create_video(buoyancy_dir_z, fig_folder, name_uni, 'vertical profile')
    if plot_1d_y:
        create_video(buoyancy_dir_y, fig_folder, name_uni, 'horizontal profile')
if video and ND:
    if plot_1d_z:
        create_video(buoyancy_dir_z_nd, fig_folder, name_nd, 'vertical profile')
    if plot_1d_y:
        create_video(buoyancy_dir_y_nd, fig_folder, name_nd, 'horizontal profile')
