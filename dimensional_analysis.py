import os
import numpy as np

from general_analysis_functions import a2_fluc_mean, ab_fluc_mean
from plotting_comparisons import plot_format
from data_collection_functions import collect_time_outputs, collect_fields_distributed, collect_temp_and_sal, collect_contour_val, collect_temporal_averages, collect_plume_stats
from dense_plume_analysis import plume_tracer_radius
from dimensional_analysis_functions import plot_rig_exponents, plot_Fr_exponents, plot_mld_exponents, plot_combo_exponents

# output name 
contour_bound = 0.05
name_uni = f'contour-{contour_bound:.2f}'
# flags for what to plot
all_steps = False
transient_mld = True
# manually select which plotting flags if not the component being varied by default
strat_flag = False
flux_flag = False
mld_flag = False
combo_flag = True
temporal_averages_flag = True

exponents = [] # for plotting reference lines with different exponents, set to empty array to not plot any

# selecting cases to compare
variations = 'MLD' # 'MLD', 'flux', 'strat', 'all'
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
    Sj = np.array([0.05, 0.1, 0.15, 0.2]) 
elif variations == 'all':
    folder_names =['S0 = 0.1 dTdz = 0.01 MLD = 60', 
                   'S0 = 0.1 dTdz = 0.01 MLD = 50', 'S0 = 0.1 dTdz = 0.01 MLD = 70', 
                   'S0 = 0.1 dTdz = 0.005 MLD = 60', 'S0 = 0.1 dTdz = 0.05 MLD = 60', 'S0 = 0.1 dTdz = 0.1 MLD = 60',
                   'S0 = 0.05 dTdz = 0.01 MLD = 60', 'S0 = 0.15 dTdz = 0.01 MLD = 60', 'S0 = 0.2 dTdz = 0.01 MLD = 60']
    case_names =[r'F$_{\text{C}} = -1.0*10^{-4}$, MLD = 60m, dTdz = 0.01', 
                 r'F$_{\text{C}} = -1.0*10^{-4}$, MLD = 50m, dTdz = 0.01', r'F$_{\text{C}} = -1.0*10^{-4}$, MLD = 70m, dTdz = 0.01', 
                 r'F$_{\text{C}} = -1.0*10^{-4}$, MLD = 60m, dTdz = 0.005', r'F$_{\text{C}} = -1.0*10^{-4}$, MLD = 60m, dTdz = 0.05', r'F$_{\text{C}} = -1.0*10^{-4}$, MLD = 60m, dTdz = 0.1', 
                 r'F$_{\text{C}} = -5.0*10^{-5}$, MLD = 60m, dTdz = 0.01', r'F$_{\text{C}} = -1.5*10^{-4}$, MLD = 60m, dTdz = 0.01', r'F$_{\text{C}} = - 2.0*10^{-4}$, MLD = 60m, dTdz = 0.01']
    num_cases = len(case_names)
    dTdz = np.array([0.01, 
                     0.01, 0.01, 
                     0.005, 0.05, 0.1,
                     0.01, 0.01, 0.01]) # background temperature gradient in K/m
    mld = np.array([60, 
                    50, 70, 
                    60, 60, 60,
                    60, 60, 60])
    Sj = np.array([0.1, 
                   0.1, 0.1, 
                   0.1, 0.1, 0.1,
                   0.05, 0.15, 0.2]) 
if variations == 'all' or combo_flag:
    vars_exps = np.array([ # columns: Ri, Fr, MLD
        [0, -1/3, -1/2], # w_rms
        [-1/2, -1/3, 1/3], # b_center
        [-1/2, -3/8, 0], # bw_fluc_avg
        [-1/4, 1/4, 0], # r_profile
        [-3/8, -1/4, 1], # T_fluc_center
        [-1/4, -7/8, 1/4] # S_avg
    ]) # manually manipulate

T0 = 25.0
g = 9.80665  # gravity in m/s^2
wp = 0.001
F_s = np.dot(Sj, wp)
rj = 5.0

# Set up folder and simulation parameters
universal_folder = '/Users/annapauls/Library/CloudStorage/OneDrive-UCB-O365/CU-Boulder/TESLa/Carbon Sequestration/Simulations/Oceananigans/NBP/salinity and temperature/no noise circle inlet'
fig_folder = os.path.join(universal_folder, 'ND analysis', variations, name_uni)
os.makedirs(fig_folder, exist_ok=True)
file_name = 'temporal_averages.h5'
folders = []
for name in folder_names:
    folders.append(os.path.join(universal_folder, name))

num_cases = len(case_names)

# flags for how to read data
with_halos = False
closure = False * np.ones(num_cases) 
stokes = False * np.ones(num_cases) 
salinity = True

color_opt, line_opt = plot_format(num_cases)

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
        time, t_save_temp, nx, hx, lx, x, y, z, xf, yf, zf, dx, visc, diff = collect_time_outputs(fid, Nranks, stokes[i], closure[i])
    else:
        time, t_save_temp, nx, hx, lx, x, y, z, xf, yf, zf, dx, visc, diff, u_f, u_s = collect_time_outputs(fid, Nranks, stokes[i], closure[i])
    if salinity:
        alpha, beta = collect_temp_and_sal(fid, salinity)
    else:
        alpha = collect_temp_and_sal(fid, salinity)
    t_save.append(t_save_temp)

z = (z*np.ones([num_cases, nx[2]])).T
zf = (zf*np.ones([num_cases, nx[2] + 1])).T
mld_idx = []
for i in range(num_cases):
    mld_idx.append(np.argmin(np.abs(z[:, i]+mld[i])))

if temporal_averages_flag:
    title = 'Temporal averages'
    name_uni += '_temporal_averages'
else:
    centerline_index = np.zeros((3, nx[2])).astype(int)
    center_xy_loc = np.zeros((3, nx[2]))
    center_xy_loc[0, :] = lx[0]/2
    center_xy_loc[1, :] = lx[1]/2
    center_xy_loc[2, :] = z
    centerline_index[0, :] = nx[0]//2 - 1
    centerline_index[1, :] = nx[1]//2 - 1
    centerline_index[2, :] = np.arange(nx[2]).astype(int)
    if all_steps:
        nt = len(t_save[0])
        nt = np.arange(0, nt)
    else:
        nt = len(t_save[0]) -1 # only last time step
        nt = [nt,]

############ NONDIMENSIONALIZATION ############
area = (2*rj)**2 
N2 = g * alpha * dTdz 
Ri_g = N2/(g/rj)
Fr_flux = F_s * beta / np.sqrt(rj * g)
vel_scale = np.sqrt(rj * g)
b_scale = g
F_b_scale = b_scale * vel_scale
T_scale = 1/alpha
S_scale =  1/beta
F_T_scale = beta * F_s / alpha
F_S_scale = F_s
hor_scale = rj

F0 = area * beta * g * F_s
Ln =(F0/N2**(3/2))**(1/4)
z_nd = (z+mld)*(mld)**(1/3)/(Ln**(4/3))#(z+mld)/(Ln)#
zf_nd = (zf+mld)*(mld)**(1/3)/(Ln**(4/3))#(zf+mld)/(Ln)#

y_nd = y / rj

if temporal_averages_flag:
    T_avg = np.zeros((nx[2], num_cases))
    b_avg = np.zeros((nx[2], num_cases))
    w_fluc_avg = np.zeros((nx[2] + 1, num_cases))
    bw_fluc_avg = np.zeros((nx[2], num_cases))
    w_rms = np.zeros((nx[2] + 1, num_cases))
    r_profile = np.zeros((nx[2], num_cases))
    b_center = np.zeros((nx[2], num_cases))
    T_fluc_center = np.zeros((nx[2], num_cases))
    S_fluc_center = np.zeros((nx[2], num_cases))
    S_avg = np.zeros((nx[2], num_cases))
    for i, folder in enumerate(folders):
        rms_list, b_and_w_list, T_list, S_list = collect_temporal_averages(folder, file_name, True, salinity)
        w_rms[:, i] = rms_list['w_rms']
        b_avg[:, i] = b_and_w_list['b_avg']
        bw_fluc_avg[:, i] = b_and_w_list['bw_fluc_avg']
        b_center[:, i] = b_and_w_list['b_centerline_avg']
        T_fluc_center[:, i] = T_list['T_fluc_centerline_avg']
        S_avg[:, i] = S_list['S_avg']
        r_profile[:, i] = collect_plume_stats(folder, file_name, contour_bound)
        if transient_mld:
                dbdz = np.gradient(b_avg[:, i], z[:, i])
                dbdz_tol = dbdz <= (5.0*10**(-7))
                if np.any(dbdz_tol):
                    mld_idx[i] = np.min(np.where(dbdz_tol))
                else:
                    mld_idx[i] = nx[2] - 1
                mld[i] = -z[mld_idx[i], i]
    if transient_mld:
        Ln =(F0/N2**(3/2))**(1/4)
        z_nd = (z+mld)*(mld)**(1/3)/(Ln**(4/3))
        zf_nd = (zf+mld)*(mld)**(1/3)/(Ln**(4/3))
    bw_fluc_avg = bw_fluc_avg/F_b_scale
    S_avg = S_avg/S_scale
    b_avg = b_avg/b_scale
    b_center = b_center/b_scale
    T_fluc_center = T_fluc_center / T_scale
    S_fluc_center = S_fluc_center / S_scale
    r_profile = r_profile / hor_scale
    w_rms = w_rms/vel_scale
    ############ PLOTTING ############
    if variations == 'all' or combo_flag:
        plot_combo_exponents(color_opt, title, fig_folder, w_rms, b_center, bw_fluc_avg, r_profile, T_fluc_center, S_avg, z_nd, zf_nd, vars_exps, Ri_g, Fr_flux, mld/rj, case_names)
    if np.size(exponents)==0 and (variations == 'strat' or strat_flag or variations == 'flux' or flux_flag or variations == 'MLD' or mld_flag):
        if variations == 'strat' or strat_flag:
            plot_rig_exponents(color_opt, title, fig_folder, w_rms, b_center, bw_fluc_avg, r_profile, T_fluc_center, S_avg, z_nd, zf_nd, Ri_g, case_names)
        if variations == 'flux' or flux_flag:
            plot_Fr_exponents(color_opt, title, fig_folder, w_rms, b_center, bw_fluc_avg, r_profile, T_fluc_center, S_avg, z_nd, zf_nd, Fr_flux, case_names)
        if variations == 'MLD' or mld_flag:
            plot_mld_exponents(color_opt, title, fig_folder, w_rms, b_center, bw_fluc_avg, r_profile, T_fluc_center, S_avg, z_nd, zf_nd, mld/rj, case_names)
    elif np.size(exponents)!=0 and (variations == 'strat' or strat_flag or variations == 'flux' or flux_flag or variations == 'MLD' or mld_flag):
        if variations == 'strat' or strat_flag:
            plot_rig_exponents(color_opt, title, fig_folder, w_rms, b_center, bw_fluc_avg, r_profile, T_fluc_center, S_avg, z_nd, zf_nd, Ri_g, case_names, exponents = exponents)
        if variations == 'flux' or flux_flag:
            plot_Fr_exponents(color_opt, title, fig_folder, w_rms, b_center, bw_fluc_avg, r_profile, T_fluc_center, S_avg, z_nd, zf_nd, Fr_flux, case_names, exponents = exponents)
        if variations == 'MLD' or mld_flag:
            plot_mld_exponents(color_opt, title, fig_folder, w_rms, b_center, bw_fluc_avg, r_profile, T_fluc_center, S_avg, z_nd, zf_nd, mld/rj, case_names, exponents = exponents)
else:
    for it in nt:
        u_avg = np.zeros((nx[2], num_cases))
        v_avg = np.zeros((nx[2], num_cases))
        w_avg = np.zeros((nx[2] + 1, num_cases))
        wc_avg = np.zeros((nx[2], num_cases))
        T_avg = np.zeros((nx[2], num_cases))
        b_avg = np.zeros((nx[2], num_cases))
        w_fluc_avg = np.zeros((nx[2] + 1, num_cases))
        bw_fluc_avg = np.zeros((nx[2], num_cases))
        u_rms = np.zeros((nx[2], num_cases))
        v_rms = np.zeros((nx[2], num_cases))
        w_rms = np.zeros((nx[2] + 1, num_cases))
        r_profile = np.zeros((nx[2], num_cases))
        b_center = np.zeros((nx[2], num_cases))
        T_fluc_center = np.zeros((nx[2], num_cases))
        S_fluc_center = np.zeros((nx[2], num_cases))
        S_avg = np.zeros((nx[2], num_cases))
        for i, folder in enumerate(folders):
            # Load data from files
            u, v, w, T, S, Pdynamic, Pstatic = collect_fields_distributed(Nranks, folder, dtn, t_save[i][it], hx, nx, True, salinity, with_halos)
            # convert temperature and salinity to buoyancy 
            b = g*alpha*(T - T0) - g*beta*(S)

            wc = 0.5 * (w[..., :-1] + w[..., 1:])
            # calculate means
            u_avg[:, i] = np.mean(u, axis=(-3, -2))
            v_avg[:, i] = np.mean(v, axis=(-3, -2))
            w_avg[:, i] = np.mean(w, axis=(-3, -2))
            wc_avg[:, i] = np.mean(wc, axis=(-3, -2))
            b_avg[:, i] = np.mean(b, axis=(-3, -2))
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
            w_fluc_avg[:, i], w2_fluc, w2_fluc_avg = a2_fluc_mean(w_fluc)
            b2_fluc, b2_fluc_avg = ab_fluc_mean(b, b, b_avg[:, i], b_avg[:, i])
            bw_fluc, bw_fluc_avg[:, i] = ab_fluc_mean(b, wc, b_avg[:, i], wc_avg[:, i])

            if transient_mld and it != 0:
                dbdz = np.gradient(b_avg[:, i], z[:, i])
                dbdz_tol = dbdz <= (5.0*10**(-7))
                if np.any(dbdz_tol):
                    mld_idx[i] = np.min(np.where(dbdz_tol))
                else:
                    mld_idx[i] = nx[2] - 1
                mld[i] = -z[mld_idx[i], i]
            # finding radius of plume based on tracer contour
            S_value, w_value = collect_contour_val(folder, file_name)
            S_contour = S_value*contour_bound
            rp_profile, plume_index, S_contour_temp = plume_tracer_radius(x, y, nx, centerline_index, S, S_contour) # plume analysis
            r_profile[:, i] = rp_profile
            b_center[:, i] = b[centerline_index[0], centerline_index[1], centerline_index[2]]
            T_fluc_center[:, i] = T_fluc[centerline_index[0], centerline_index[1], centerline_index[2]]
            S_fluc_center[:, i] = S_fluc[centerline_index[0], centerline_index[1], centerline_index[2]]
            # rms fluctuations
            w_rms[:, i] = w2_fluc_avg**0.5
        if transient_mld:
            Ln =(F0/N2**(3/2))**(1/4)
            z_nd = (z+mld)*(mld)**(1/3)/(Ln**(4/3))
            zf_nd = (zf+mld)*(mld)**(1/3)/(Ln**(4/3))
        bw_fluc_avg = bw_fluc_avg/F_b_scale
        S_avg = S_avg/S_scale
        b_avg = b_avg/b_scale
        b_center = b_center/b_scale
        T_fluc_center = T_fluc_center / T_scale
        S_fluc_center = S_fluc_center / S_scale
        r_profile = r_profile / hor_scale
        w_rms = w_rms/vel_scale
        ############ PLOTTING ############
        td = title/3600/24
        title = f'{td:.2f} days'
        if variations == 'all' or combo_flag:
            plot_combo_exponents(color_opt, title, fig_folder, w_rms, b_center, bw_fluc_avg, r_profile, T_fluc_center, S_avg, z_nd, zf_nd, vars_exps, Ri_g, Fr_flux, mld/rj, case_names)
        if np.size(exponents)==0 and (variations == 'strat' or strat_flag or variations == 'flux' or flux_flag or variations == 'MLD' or mld_flag):
            if variations == 'strat' or strat_flag:
                plot_rig_exponents(color_opt, title, fig_folder, w_rms, b_center, bw_fluc_avg, r_profile, T_fluc_center, S_avg, z_nd, zf_nd, Ri_g, case_names)
            if variations == 'flux' or flux_flag:
                plot_Fr_exponents(color_opt, title, fig_folder, w_rms, b_center, bw_fluc_avg, r_profile, T_fluc_center, S_avg, z_nd, zf_nd, Fr_flux, case_names)
            if variations == 'MLD' or mld_flag:
                plot_mld_exponents(color_opt, title, fig_folder, w_rms, b_center, bw_fluc_avg, r_profile, T_fluc_center, S_avg, z_nd, zf_nd, mld/rj, case_names)
        elif np.size(exponents)!=0 and (variations == 'strat' or strat_flag or variations == 'flux' or flux_flag or variations == 'MLD' or mld_flag):
            if variations == 'strat' or strat_flag:
                plot_rig_exponents(color_opt, title, fig_folder, w_rms, b_center, bw_fluc_avg, r_profile, T_fluc_center, S_avg, z_nd, zf_nd, Ri_g, case_names, exponents = exponents)
            if variations == 'flux' or flux_flag:
                plot_Fr_exponents(color_opt, title, fig_folder, w_rms, b_center, bw_fluc_avg, r_profile, T_fluc_center, S_avg, z_nd, zf_nd, Fr_flux, case_names, exponents = exponents)
            if variations == 'MLD' or mld_flag:
                plot_mld_exponents(color_opt, title, fig_folder, w_rms, b_center, bw_fluc_avg, r_profile, T_fluc_center, S_avg, z_nd, zf_nd, mld/rj, case_names, exponents = exponents)