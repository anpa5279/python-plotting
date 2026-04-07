import os
import numpy as np
import matplotlib.pyplot as plt

from plotting_functions import plot_ranges, create_video
from general_analysis_functions import a2_fluc_mean, ab_fluc_mean
from plotting_comparisons import plot_format, plume_vertical_spatial_plot, plume_horizontal_spatial_plot
from data_collection_functions import collect_time_outputs, collect_fields_distributed, collect_temp_and_sal
from dense_plume_analysis import plume_tracer_radius

# Set up folder and simulation parameters
universal_folder = '/Users/annapauls/Library/CloudStorage/OneDrive-UCB-O365/CU-Boulder/TESLa/Carbon Sequestration/Simulations/Oceananigans/NBP/salinity and temperature/'
folder_names =['beta = default S0 = 0.1 MLD = 20m', 'beta = default S0 = 0.1', 'beta = default S0 = 0.1 MLD = 40m']
case_names = [r'MLD = 20m', r'MLD = 30m', r'MLD = 40m']  
name_uni = ""
fig_folder = os.path.join(universal_folder, 'comparison figures', 'MLD comparison figures')

num_cases = len(case_names)

# flags for what to plot
plot_1d_z = True
plot_1d_y = False
ND = True

# flags for how to read data
with_halos = False
stokes = False * np.ones(num_cases) 
salinity = True

video = True

# physical parameters
rj = 10 # m, radius of salinity flux circle at the surface
g = 9.80665  # gravity in m/s^2
dTdz = 0.01 * np.ones(num_cases) 
rho0 = 1026
mld = np.array([20, 30, 40]) 
T0 = 25.0
S0 = 0 
wp = 0.001
Sj = 0.1 * np.ones(num_cases) 
F_s = np.dot(Sj, wp)

S_value = np.array([0.03995705848735615, 0.03602588163919859, 0.032189606877616704]) # for MLD variations
S_contour = S_value*0.15 
w_avg_centerline = np.array([-0.025053252620373258, -0.03394752674800345, -0.04425781328270483]) # for MLD centerline w values thorughout time

# plotting prep
ranges = plot_ranges(lz = 96, rho0 = rho0, T0 = T0, dTdz = np.max(dTdz), Sj = np.max(Sj))
ranges['S'] = [0, 1*10**(-3)]#[0, 9*10**(-2)]
ranges['vel_rms'] = [0, 4*10**-3]
ranges['bw_fluc'] = [-9*10**(-9), 9*10**(-9)] #[-2*10**(-5), 2*10**(-5)]
ranges['b_rms'] = [0, 1.5*10**(-5)]
ranges['b_fluc'] = [-2*10**(-4), 2*10**(-4)]
ranges['w'] = [-0.1, 0.1]
#ranges['T'] = [24.9, 25.02]
ranges['S_fluc'] = [-5*10**(-2), 5*10**(-2)]
ranges['T_fluc'] = [-4*10**(-1), 4*10**(-1)]

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
t_save = []

folders = []
for name in folder_names:
    folders.append(os.path.join(universal_folder, name))

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

mld_idx = []
for i in range(num_cases):
    mld_idx.append(np.argmin(np.abs(z[:, i]+mld[i])))

if plot_1d_y:
    hor_idx = mld_idx
    name_uni = name_uni + f"at z = {z[hor_idx, np.arange(num_cases)]} m"

############ NONDIMENSIONALIZATION ############
if ND: 
    name_nd = 'ND_' + name_uni

    N2 = g * dTdz / T0
    area = (2*rj)**2
    F0 = area * beta * g * F_s
    Ln =(F0/N2**(3/2))**(1/4)
    z_nd = (z+mld)*mld**(1/3)/(Ln**(4/3))
    zf_nd = (zf+mld)*mld**(1/3)/(Ln**(4/3))#
    y_nd = y / rj
    vel_scale = F_s * beta
    b_scale = N2 * rj
    b_perturbed_scale = F_s * beta * np.sqrt(rj * g) / rj
    F_b_scale = F_s * beta * g
    T_scale = T0 * F_s * beta / np.sqrt(rj*g)
    S_scale = F_s / np.sqrt(rj*g)
    F_T_scale = beta * F_s * T0
    F_S_scale = F_s * np.sqrt(rj * dTdz / T0)
    y_nd = y / rj
    lx_nd = np.zeros(3)
    lx_nd[0:2]= np.array(lx[0:2])/ rj
    lx_nd[-1] = np.max((lx[-1] - mld) * dTdz / T0)


    nd_ranges = ranges.copy()
    nd_ranges['vel_rms'] = nd_ranges['vel_rms'] / np.min(vel_scale)
    nd_ranges['w'] = nd_ranges['w'] / np.min(vel_scale)
    nd_ranges['b_avg'] = nd_ranges['b_avg'] / np.min(b_scale)
    nd_ranges['bw_fluc'] = nd_ranges['bw_fluc'] / np.min(F_b_scale)
    nd_ranges['b_rms'] = nd_ranges['b_rms'] / np.min(b_perturbed_scale)
    nd_ranges['b_fluc'] = nd_ranges['b_fluc'] / np.min(b_perturbed_scale)
    nd_ranges['S'] = nd_ranges['S'] / np.min(S_scale)
    nd_ranges['S_fluc'] = nd_ranges['S_fluc'] / np.min(S_scale)
    nd_ranges['T_fluc'] = nd_ranges['T_fluc'] / np.min(T_scale)
    nd_ranges['T'] = nd_ranges['T'] / np.min(T_scale)


start_neutral = np.zeros(num_cases).astype(int)
for it in nt:
    u_avg = np.zeros((nx[2], num_cases))
    v_avg = np.zeros((nx[2], num_cases))
    w_avg = np.zeros((nx[2] + 1, num_cases))
    wc_avg = np.zeros((nx[2], num_cases))
    T_avg = np.zeros((nx[2], num_cases))
    b_avg = np.zeros((nx[2], num_cases))
    rho_avg = np.zeros((nx[2], num_cases))
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
    if plot_1d_z:
        u_rms = np.zeros((nx[2], num_cases))
        v_rms = np.zeros((nx[2], num_cases))
        w_rms = np.zeros((nx[2] + 1, num_cases))
        r_profile = np.zeros((nx[2], num_cases))
        b_center = np.zeros((nx[2], num_cases))
        T_fluc_center = np.zeros((nx[2], num_cases))
        S_fluc_center = np.zeros((nx[2], num_cases))
    if plot_1d_y:
        u_hor = np.zeros((nx[1], num_cases))
        v_hor = np.zeros((nx[1], num_cases))
        w_hor = np.zeros((nx[1], num_cases))
        b_fluc_hor = np.zeros((nx[1], num_cases))
        bu_fluc_hor = np.zeros((nx[1], num_cases))
        bv_fluc_hor = np.zeros((nx[1], num_cases))
        bw_fluc_hor = np.zeros((nx[1], num_cases))
        T_hor = np.zeros((nx[1], num_cases))
        S_hor = np.zeros((nx[1], num_cases))
    if salinity:
        S_avg = np.zeros((nx[2], num_cases))
    for i, folder in enumerate(folders):
        # Load data from files
        if not salinity:
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
        b_fluc = b - b_avg[:, i]

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
        if plot_1d_z:
            # rms fluctuations
            u_rms[:, i] = u2_fluc_avg**0.5
            v_rms[:, i] = v2_fluc_avg**0.5
            w_rms[:, i] = w2_fluc_avg**0.5
            b_rms[:, i] = b2_fluc_avg**0.5
        
        # collecting data for plotting
        if plot_1d_z and salinity:
            bw_idx = np.where(bw_fluc_avg==np.max(bw_fluc_avg))[0][0]
            rp_profile, plume_index, S_contour_temp = plume_tracer_radius(x, y, nx, centerline_index, S, S_contour[i]) # plume analysis
            r_profile[:, i] = rp_profile
            b_center[:, i] = b[centerline_index[0], centerline_index[1], centerline_index[2]]
            T_fluc_center[:, i] = T_fluc[centerline_index[0], centerline_index[1], centerline_index[2]]
            S_fluc_center[:, i] = S_fluc[centerline_index[0], centerline_index[1], centerline_index[2]]
        if plot_1d_y:
            u_hor[:, i] = u[centerline_index[0, hor_idx[i]], :, hor_idx[i]]
            v_hor[:, i] = v[centerline_index[0, hor_idx[i]], :, hor_idx[i]]
            w_hor[:, i] = w[centerline_index[0, hor_idx[i]], :, hor_idx[i]]
            b_fluc_hor[:, i] = b_fluc[centerline_index[0, hor_idx[i]], :, hor_idx[i]]
            bu_fluc_hor[:, i] = bu_fluc[centerline_index[0, hor_idx[i]], :, hor_idx[i]]
            bv_fluc_hor[:, i] = bv_fluc[centerline_index[0, hor_idx[i]], :, hor_idx[i]]
            bw_fluc_hor[:, i] = bw_fluc[centerline_index[0, hor_idx[i]], :, hor_idx[i]]
            T_hor[:, i] = T[centerline_index[0, hor_idx[i]], :, hor_idx[i]]
            S_hor[:, i] = S[centerline_index[0, hor_idx[i]], :, hor_idx[i]]
    ############ NONDIMENSIONALIZATION ############
    if ND:
        ############ PLOTTING ############
        if plot_1d_z:
            # rms fluctuations
            u_rms[:, i] = u2_fluc_avg**0.5
            v_rms[:, i] = v2_fluc_avg**0.5
            w_rms[:, i] = w2_fluc_avg**0.5
            b_rms[:, i] = b2_fluc_avg**0.5
        
        # collecting data for plotting
        if plot_1d_z and salinity:
            bw_idx = np.where(bw_fluc_avg==np.max(bw_fluc_avg))[0][0]
            rp_profile, plume_index, S_contour_temp = plume_tracer_radius(x, y, nx, centerline_index, S, S_contour[i]) # plume analysis
            r_profile[:, i] = rp_profile
            b_center[:, i] = b[centerline_index[0], centerline_index[1], centerline_index[2]]
            T_fluc_center[:, i] = T_fluc[centerline_index[0], centerline_index[1], centerline_index[2]]
            S_fluc_center[:, i] = S_fluc[centerline_index[0], centerline_index[1], centerline_index[2]]
        if plot_1d_y:
            u_hor[:, i] = u[centerline_index[0, hor_idx[i]], :, hor_idx[i]]
            v_hor[:, i] = v[centerline_index[0, hor_idx[i]], :, hor_idx[i]]
            w_hor[:, i] = w[centerline_index[0, hor_idx[i]], :, hor_idx[i]]
            b_fluc_hor[:, i] = b_fluc[centerline_index[0, hor_idx[i]], :, hor_idx[i]]
            bu_fluc_hor[:, i] = bu_fluc[centerline_index[0, hor_idx[i]], :, hor_idx[i]]
            bv_fluc_hor[:, i] = bv_fluc[centerline_index[0, hor_idx[i]], :, hor_idx[i]]
            bw_fluc_hor[:, i] = bw_fluc[centerline_index[0, hor_idx[i]], :, hor_idx[i]]
            T_hor[:, i] = T[centerline_index[0, hor_idx[i]], :, hor_idx[i]]
            S_hor[:, i] = S[centerline_index[0, hor_idx[i]], :, hor_idx[i]]
    ############ NONDIMENSIONALIZATION ############
    if ND:
        ############ PLOTTING ############
        if plot_1d_z:
            bv_fluc_avg = bv_fluc_avg/F_b_scale
            bu_fluc_avg = bu_fluc_avg/F_b_scale
            bw_fluc_avg = bw_fluc_avg/F_b_scale
            S_avg = S_avg/S_scale
            b_avg = b_avg/b_scale
            b_rms = b_rms/b_perturbed_scale
            b_center = b_center/b_scale
            T_fluc_center = T_fluc_center / T_scale
            S_fluc_center = S_fluc_center / S_scale
            r_profile = r_profile / rj
            u_rms = u_rms/vel_scale
            v_rms = v_rms/vel_scale
            w_rms = w_rms/vel_scale
            buoyancy_dir_z_nd = plume_vertical_spatial_plot(time, it, nd_ranges, color_opt, fig_folder, case_names, name_nd, lx_nd, z_nd, zf_nd, S_avg, u_rms, v_rms, w_rms, b_avg, b_center, r_profile, bu_fluc_avg, bv_fluc_avg, bw_fluc_avg, b_rms, T_fluc_center, S_fluc_center, ND, z_nd = r"(z - h$_{\mathrm{MLD}_0}$)h$_{\mathrm{MLD}_0}^{1/3}$/L$_N^{4/3}$")
        if plot_1d_y:
            u_hor = u_hor/vel_scale
            v_hor = v_hor/vel_scale
            w_hor = w_hor/vel_scale
            b_fluc_hor = b_fluc_hor/b_perturbed_scale
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
            buoyancy_dir_y = plume_horizontal_spatial_plot(time, it, ranges, color_opt, fig_folder, case_names, name_uni, lx, y, u_hor, v_hor, w_hor, b_fluc_hor, bu_fluc_hor, bv_fluc_hor, bw_fluc_hor, T_hor, S_hor)

print("All frames created.")
# creating videos
if video and ND:
    if plot_1d_z:
        create_video(buoyancy_dir_z_nd, fig_folder, name_nd, 'buoyancy_analysis')
    if plot_1d_y:
        create_video(buoyancy_dir_y_nd, fig_folder, name_nd, 'buoyancy_analysis')
elif video:
    if plot_1d_z:
        create_video(buoyancy_dir_z, fig_folder, name_uni, 'buoyancy_analysis')
    if plot_1d_y:
        create_video(buoyancy_dir_y, fig_folder, name_uni, 'buoyancy_analysis')