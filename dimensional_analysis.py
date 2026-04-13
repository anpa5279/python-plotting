import os
import numpy as np
import matplotlib.pyplot as plt

from plotting_functions import plot_ranges, create_video
from general_analysis_functions import a2_fluc_mean, ab_fluc_mean
from plotting_comparisons import plot_format
from data_collection_functions import collect_time_outputs, collect_fields_distributed, collect_temp_and_sal
from dense_plume_analysis import plume_tracer_radius
from dimensional_analysis_functions import plot_rig_exponents, plot_Fr_exponents, plot_mld_exponents

# output name 
contour_bound = 0.05
name_uni = f'contour-{contour_bound:.2f}'
# flags for what to plot
ND = True
video = False
mld_transient = True
# manually select which plotting flags if not the component being varied by default
strat_flag = False
flux_flag = False
mld_flag = False

exponents = [-1/4, -1/8, 0, 1/8, 1/4] # for plotting reference lines with different exponents, set to empty array to not plot any

# selecting cases to compare
variations = 'MLD' # 'MLD', 'flux', 'strat'
if variations == 'strat':
    folder_names =['beta = default S0 = 0.1 dTdz = 0.005 MLD = 60', 'beta = default S0 = 0.1 dTdz = 0.01 MLD = 60', 'beta = default S0 = 0.1 dTdz = 0.05 MLD = 60', 'beta = default S0 = 0.1 dTdz = 0.1 MLD = 60'] 
    case_names =[r'dTdz = 0.005', r'dTdz = 0.01', r'dTdz = 0.05', r'dTdz = 0.10']  
    num_cases = len(case_names)
    dTdz = np.array([0.005, 0.01, 0.05, 0.1]) # background temperature gradient in K/m
    mld = 60 * np.ones(num_cases) 
    Sj = 0.1 * np.ones(num_cases) 
    S_value = [0.005202084553938343, 0.005010973297310681, 0.005283282089196569, 0.005173444793047881]# with noise and closure cases: S_value = np.array([0.034487168519906714, 0.03602588163919859, 0.03995705848735615, 0.042189206877616705]) # for dTdz variations at max bw index
    S_contour = S_value*contour_bound
    # with noise and closure cases: w_avg_centerline = np.array([-0.043499393099289844, -0.03394752674800345, -0.018453789243636633, -0.01406895477434289]) # for strat centerline w values thorughout time
elif variations == 'MLD':
    folder_names =['beta = default S0 = 0.1 dTdz = 0.01 MLD = 50', 'beta = default S0 = 0.1 dTdz = 0.01 MLD = 60', 'beta = default S0 = 0.1 dTdz = 0.01 MLD = 70']
    case_names =[r'MLD = 50m', r'MLD = 60m', r'MLD = 70m']
    num_cases = len(case_names)
    dTdz = 0.01 * np.ones(num_cases) # background temperature gradient in K/m
    mld = np.array([50, 60, 70])
    Sj = 0.1 * np.ones(num_cases) 
    S_value = [0.005574689291054047, 0.005010973297310681, 0.004682995795255769]# with noise and closure cases: S_value = np.array([0.03995705848735615, 0.03602588163919859, 0.032189606877616704]) # for MLD variations
    S_contour = S_value*contour_bound
    # with noise and closure cases: w_avg_centerline = np.array([-0.025053252620373258, -0.03394752674800345, -0.04425781328270483]) # for MLD centerline w values thorughout time
elif variations == 'flux':
    folder_names =['beta = default S0 = 0.05 dTdz = 0.01 MLD = 60', 'beta = default S0 = 0.1 dTdz = 0.01 MLD = 60', 'beta = default S0 = 0.15 dTdz = 0.01 MLD = 60', 'beta = default S0 = 0.2 dTdz = 0.01 MLD = 60']
    case_names =[r'F$^{\text{C}} = -5.0*10^{-5}$', r'F$^{\text{C}} = -1.0*10^{-4}$', r'F$^{\text{C}} = -1.5*10^{-4}$', r'F$^{\text{C}} = - 2.0*10^{-4}$']
    num_cases = len(case_names)
    dTdz = 0.01 * np.ones(num_cases) # background temperature gradient in K/m
    mld = 60 * np.ones(num_cases) 
    Sj = np.array([0.05, 0.1, 0.15, 0.2]) 
    S_value = [0.0031834305308354204, 0.005010973297310681, 0.006362182592469852, 0.008358286212836368] # with noise and closure cases: S_value = np.dot([0.0010948250136870168, 0.0018012940819599295, 0.0024005411329652226, 0.0029359463404349034], 20) # for Sj variations 
    S_contour = S_value*contour_bound
    # with noise and closure cases: w_avg_centerline = np.array([-0.02020130913788876, -0.03394752674800345, -0.044740617760247015,  -0.05271218084132068]) # for Sj centerline w_avg values thorughout time
rho0 = 1026
T0 = 25.0
g = 9.80665  # gravity in m/s^2
wp = 0.001
F_s = np.dot(Sj, wp)
rj = 10.0

# Set up folder and simulation parameters
universal_folder = '/Users/annapauls/Library/CloudStorage/OneDrive-UCB-O365/CU-Boulder/TESLa/Carbon Sequestration/Simulations/Oceananigans/NBP/salinity and temperature/no noise small square inlet'
fig_folder = os.path.join(universal_folder, 'ND analysis', variations, name_uni)
folders = []
for name in folder_names:
    folders.append(os.path.join(universal_folder, name))

num_cases = len(case_names)
transient_mld = True

# flags for how to read data
with_halos = False
closure = False
stokes = False * np.ones(num_cases) 
salinity = True

# plotting prep
ranges = plot_ranges(lz = 96, rho0 = rho0, T0 = T0, dTdz = np.max(dTdz), Sj = np.max(Sj))
ranges['S'] = [0, 1*10**(-3)]
ranges['vel_rms'] = [0, 4*10**-3]
ranges['bw_fluc'] = [-9*10**(-9), 9*10**(-9)]
ranges['b_rms'] = [0, 1.5*10**(-5)]
ranges['b_fluc'] = [-2*10**(-4), 2*10**(-4)]
ranges['w'] = [-0.1, 0.1]
ranges['S_fluc'] = [-5*10**(-2), 5*10**(-2)]
ranges['T_fluc'] = [-4*10**(-1), 4*10**(-1)]
ranges['b_avg'] = [-1.5*10**(-2), 1.0*10**(-5)]

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
        time, t_save_temp, nx, hx, lx, x, y, z, xf, yf, zf, dx, visc, diff = collect_time_outputs(fid, Nranks, stokes[i])
    else:
        time, t_save_temp, nx, hx, lx, x, y, z, xf, yf, zf, dx, visc, diff, u_f, u_s = collect_time_outputs(fid, Nranks, stokes[i])
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

############ NONDIMENSIONALIZATION ############
if ND: 
    name_nd = 'ND_' + name_uni

    area = (2*rj)**2 
    lj = np.sqrt(area)
    N2 = g * alpha * dTdz 
    Ri_g = (N2/g*lj)
    Fr_flux = F_s * beta / np.sqrt(lj * g)
    vel_scale = np.sqrt(lj * g)
    b_scale = g
    F_b_scale = b_scale * vel_scale
    T_scale = 1/alpha
    S_scale =  1/beta
    F_T_scale = beta * F_s / alpha
    F_S_scale = F_s
    hor_scale = lj

    F0 = area * beta * g * F_s
    Ln =(F0/N2**(3/2))**(1/4)
    z_nd = (z+mld)/(Ln)#(z+mld)*(mld)**(1/3)/(Ln**(4/3))
    zf_nd = (zf+mld)/(Ln)#(zf+mld)*(mld)**(1/3)/(Ln**(4/3))

    y_nd = y / lj


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

start_neutral = np.zeros(num_cases).astype(int)
for it in nt:
    u_avg = np.zeros((nx[2], num_cases))
    v_avg = np.zeros((nx[2], num_cases))
    w_avg = np.zeros((nx[2] + 1, num_cases))
    wc_avg = np.zeros((nx[2], num_cases))
    T_avg = np.zeros((nx[2], num_cases))
    b_avg = np.zeros((nx[2], num_cases))
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
    dbdz = np.zeros((nx[2], num_cases))
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

        if transient_mld and it != 0:
            dbdz[:, i] = np.gradient(b_avg[:, i], z[:, i])
            dbdz_tol = dbdz[:, i] <= (5.0*10**(-7))
            if np.any(dbdz_tol):
                mld_idx[i] = np.min(np.where(dbdz_tol))
            else:
                mld_idx[i] = nx[2] - 1
            mld[i] = -z[mld_idx[i], i]
        # finding radius of plume based on tracer contour
        bw_idx = np.where(bw_fluc_avg==np.max(bw_fluc_avg))[0][0]
        rp_profile, plume_index, S_contour_temp = plume_tracer_radius(x, y, nx, centerline_index, S, S_contour[i]) # plume analysis
        r_profile[:, i] = rp_profile
        b_center[:, i] = b[centerline_index[0], centerline_index[1], centerline_index[2]]
        T_fluc_center[:, i] = T_fluc[centerline_index[0], centerline_index[1], centerline_index[2]]
        S_fluc_center[:, i] = S_fluc[centerline_index[0], centerline_index[1], centerline_index[2]]
        # rms fluctuations
        u_rms[:, i] = u2_fluc_avg**0.5
        v_rms[:, i] = v2_fluc_avg**0.5
        w_rms[:, i] = w2_fluc_avg**0.5
        b_rms[:, i] = b2_fluc_avg**0.5
    if ND:
        if transient_mld:
            Ln =(F0/N2**(3/2))**(1/4)
            z_nd = (z+mld)*(mld)**(1/3)/(Ln**(4/3))
            zf_nd = (zf+mld)*(mld)**(1/3)/(Ln**(4/3))
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
    ############ PLOTTING ############
    if np.size(exponents)==0:
        if variations == 'strat' or strat_flag:
            Rig_outdir = plot_rig_exponents(color_opt, time, it, fig_folder, w_rms, b_center, bw_fluc_avg, r_profile, T_fluc_center, S_avg, z_nd, zf_nd, Ri_g, case_names)
        if variations == 'flux' or flux_flag:
            Fr_outdir = plot_Fr_exponents(color_opt, time, it, fig_folder, w_rms, b_center, bw_fluc_avg, r_profile, T_fluc_center, S_avg, z_nd, zf_nd, Fr_flux, case_names)
        if variations == 'MLD' or mld_flag:
            mld_outdir = plot_mld_exponents(color_opt, time, it, fig_folder, w_rms, b_center, bw_fluc_avg, r_profile, T_fluc_center, S_avg, z_nd, zf_nd, mld/lj, case_names)
    else:
        if variations == 'strat' or strat_flag:
            Rig_outdir = plot_rig_exponents(color_opt, time, it, fig_folder, w_rms, b_center, bw_fluc_avg, r_profile, T_fluc_center, S_avg, z_nd, zf_nd, Ri_g, case_names, exponents = exponents)
        if variations == 'flux' or flux_flag:
            Fr_outdir = plot_Fr_exponents(color_opt, time, it, fig_folder, w_rms, b_center, bw_fluc_avg, r_profile, T_fluc_center, S_avg, z_nd, zf_nd, Fr_flux, case_names, exponents = exponents)
        if variations == 'MLD' or mld_flag:
            mld_outdir = plot_mld_exponents(color_opt, time, it, fig_folder, w_rms, b_center, bw_fluc_avg, r_profile, T_fluc_center, S_avg, z_nd, zf_nd, mld/lj, case_names, exponents = exponents)
# creating video
if video:
    create_video(Rig_outdir, fig_folder, name_uni, 'Ri_g_NDanalysis')