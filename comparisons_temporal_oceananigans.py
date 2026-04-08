import os
import numpy as np
import matplotlib.pyplot as plt
from plotting_functions import plot_ranges
from general_analysis_functions import a2_fluc_mean, ab_fluc_mean
from plotting_comparisons import plot_format, plume_temporal_analysis, mld_temporal_analysis
from data_collection_functions import collect_time_outputs, collect_fields_distributed, collect_temp_and_sal
from dense_plume_analysis import plume_tracer_radius, neutral_buoyancy_loc
# output name 
name_uni ='contour-0.15-dTdz'
# flags for what to plot
plume_analysis_plot = True
mld_analysis_plot = False
ND = True
if mld_analysis_plot:
    mld_transient = True

# selecting cases to compare
variations = 'strat' # 'MLD', 'flux', 'strat'
if variations == 'strat':
    folder_names =['beta = default S0 = 0.1 dTdz = 0.005', 'beta = default S0 = 0.1', 'beta = default S0 = 0.1 dTdz = 0.05', 'beta = default S0 = 0.1 dTdz = 0.1'] 
    case_names =[r'dTdz = 0.005', r'dTdz = 0.01', r'dTdz = 0.05', r'dTdz = 0.10']  
    num_cases = len(case_names)
    dTdz = np.array([0.005, 0.01, 0.05, 0.1]) # background temperature gradient in K/m
    mld = 30 * np.ones(num_cases) 
    Sj = 0.1 * np.ones(num_cases) 
elif variations == 'MLD':
    folder_names =['beta = default S0 = 0.1 MLD = 20m', 'beta = default S0 = 0.1', 'beta = default S0 = 0.1 MLD = 40m']
    case_names =[r'MLD = 20m', r'MLD = 30m', r'MLD = 40m']
    num_cases = len(case_names)
    dTdz = 0.01 * np.ones(num_cases) # background temperature gradient in K/m
    mld = np.array([20, 30, 40])
    Sj = 0.1 * np.ones(num_cases) 
elif variations == 'flux':
    folder_names =['beta = default S0 = 0.05', 'beta = default S0 = 0.1', 'beta = default S0 = 0.15', 'beta = default S0 = 0.2']
    case_names =[r'F$^{\text{C}} = -5.0*10^{-5}$', r'F$^{\text{C}} = -1.0*10^{-4}$', r'F$^{\text{C}} = -1.5*10^{-4}$', r'F$^{\text{C}} = - 2.0*10^{-4}$']
    num_cases = len(case_names)
    dTdz = 0.01 * np.ones(num_cases) # background temperature gradient in K/m
    mld = 30 * np.ones(num_cases) 
    Sj = np.array([0.05, 0.1, 0.15, 0.2]) # 


# flags for how to read data
with_halos = False
stokes = False * np.ones(num_cases) 
salinity = True
temporal_avg = True
mld_transient = False

# Set up folder and simulation parameters
universal_folder = '/Users/annapauls/Library/CloudStorage/OneDrive-UCB-O365/CU-Boulder/TESLa/Carbon Sequestration/Simulations/Oceananigans/NBP/salinity and temperature/'
fig_folder = os.path.join(universal_folder, 'comparison figures/contour 0.15/')
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

num_cases = len(case_names)
folders = []
for name in folder_names:
    folders.append(os.path.join(universal_folder, name))
output_folder = universal_folder

# plotting prep
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
centerline_index[0, :] = nx[0]//2 - 1
centerline_index[1, :] = nx[1]//2 - 1
centerline_index[2, :] = np.arange(nx[2]).astype(int)
if temporal_avg and salinity:
    S_contour = np.zeros(num_cases)
    w_contour = np.zeros(num_cases)
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
        if Nranks == 1 and not stokes[i]:
            time, t_save_temp, nx, hx, lx, x, y, z, xf, yf, zf, dx, visc, diff = collect_time_outputs(fid, Nranks, stokes[i])
        elif Nranks == 1 and stokes[i]:
            time, t_save_temp, nx, hx, lx, x, y, z, xf, yf, zf, dx, visc, diff, u_f, u_s = collect_time_outputs(fid, Nranks, stokes[i])
        elif Nranks > 1 and not stokes[i]:
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
        nt = len(t_save_temp)
        n = 0.0
        S_sum = 0.0
        w_sum = 0.0
        for it in range(10, nt):

            # Load data from files
            u, v, w, T, S, Pdynamic, Pstatic = collect_fields_distributed(Nranks, folder, dtn, t_save[i][it], hx, nx, True, salinity, with_halos)
            wc = 0.5 * (w[..., :-1] + w[..., 1:])
            w_sum += np.mean(w[centerline_index[0, :], centerline_index[1, :], :])
            b = g*alpha*(T - T0) - g*beta*(S - S0)
            b_avg = np.mean(b, axis=(-3, -2))
            wc_avg = np.mean(wc, axis=(-3, -2))
            bw_fluc, bw_fluc_avg = ab_fluc_mean(b, wc, b_avg, wc_avg)
            bw_idx = np.where(bw_fluc_avg==np.max(bw_fluc_avg))[0][0]
            S_sum += np.mean(S[centerline_index[0, :], centerline_index[1, :], bw_idx])
            n += 1
        S_contour[i] = S_sum/n
        w_contour[i] = w_sum/n
        print(f"Case {case_names[i]}: w_contour = {w_contour[i]}, S_contour = {S_contour[i]}")
nt = len(t_save_temp)

z = z*np.ones([num_cases, nx[2]])
zf = zf*np.ones([num_cases, nx[2]+1])
lx_plot = lx

for i in range(num_cases):
    mld_idx.append(np.argmin(np.abs(z[i, :]+mld[i])))

start_neutral = np.zeros(num_cases).astype(int)
# preallocate arrays for temporal analysis
h_neutral = np.zeros((nt, num_cases))
h_max = np.zeros((nt, num_cases))
r_neutral = np.zeros((nt, num_cases))
r_mld = np.zeros((nt, num_cases))
r_hmax = np.zeros((nt, num_cases))
w_neutral = np.zeros((nt, num_cases))
w_mld = np.zeros((nt, num_cases))
w_hmax = np.zeros((nt, num_cases))
b_neutral = np.zeros((nt, num_cases))
b_mld = np.zeros((nt, num_cases))
b_hmax = np.zeros((nt, num_cases))
T_neutral = np.zeros((nt, num_cases))
T_mld = np.zeros((nt, num_cases))
T_hmax = np.zeros((nt, num_cases))
S_neutral = np.zeros((nt, num_cases))
S_mld = np.zeros((nt, num_cases))
S_hmax = np.zeros((nt, num_cases))
Tw_mld_avg = np.zeros((nt, num_cases))
Sw_mld_avg = np.zeros((nt, num_cases))
mld0 = mld.copy()
if mld_analysis_plot:
    mld = mld*np.ones((nt, num_cases))
    mass_ratio = np.zeros((nt, num_cases))
    bw_mld_avg = np.zeros((nt, num_cases))
    w_mld_rms = np.zeros((nt, num_cases))

for it in range(nt):
    for i, folder in enumerate(folders):
        # Load data from files
        if salinity:
            u, v, w, T, S, Pdynamic, Pstatic = collect_fields_distributed(Nranks, folder, dtn, t_save[i][it], hx, nx, True, salinity, with_halos)
        else:
            u, v, w, T, Pdynamic, Pstatic = collect_fields_distributed(Nranks, folder, dtn, t_save[i][it], hx, nx, True, salinity, with_halos)

        #if stokes[i]:
        #    u = u - u_s
        # convert temperature and salinity to buoyancy 
        if not salinity:
            rho_total = rho0 - rho0 * alpha * (T - T0)
            b = -g*alpha*(T - T0)
        else:
            S[np.where(S<10**(-16))] = 0
            rho_total = rho0 - rho0 * alpha * (T - T0)+ rho0 * beta * (S - S0)
            b = g*alpha*(T - T0) - g*beta*(S - S0)
            #b_T = g*alpha*(T - T0)
            #b_S = - g*beta*(S - S0)

        # interpolate so all values are from the center, center, center of the grid cell
        wc = 0.5 * (w[..., :-1] + w[..., 1:])


        # calculate means
        #u_avg = np.mean(u, axis=(-3, -2))
        #v_avg = np.mean(v, axis=(-3, -2))
        w_avg = np.mean(w, axis=(-3, -2))
        wc_avg = np.mean(wc, axis=(-3, -2))
        b_avg = np.mean(b, axis=(-3, -2))
        rho_avg = np.mean(rho_total, axis=(-3, -2))
        S_avg = np.mean(S, axis=(-3, -2))
        T_avg = np.mean(T, axis=(-3, -2))

        # calculate fluctuations
        w_fluc = w-w_avg
        #wc_fluc = wc-wc_avg
        #rho_fluc = rho_total - rho_avg
        b_fluc = b - b_avg
        T_fluc = T - T_avg
        S_fluc = S - S_avg

        T_fluc_w_avg = np.mean(T_fluc * wc, axis=(-3, -2))
        S_fluc_w_avg = np.mean(S_fluc * wc, axis=(-3, -2))

        w_fluc_avg, w2_fluc, w2_fluc_avg = a2_fluc_mean(w_fluc)

        Sw_mld_avg[it,  i] = S_fluc_w_avg[mld_idx[i]]
        Tw_mld_avg[it,  i] = T_fluc_w_avg[mld_idx[i]]

                
        if mld_transient or mld_analysis_plot:
            if it == 0:
                mass_ratio[it, i] = 0
            else:
                dbdz = np.gradient(b_avg, z[i, :])
                dbdz_far = np.gradient(b[0, 0, :], z[i, :])
                reduction_factor = 0.1
                mld_idx[i] = np.min([np.min(np.where(dbdz_far < dTdz[i]*alpha*g*reduction_factor)[0]), np.min(np.where(dbdz < dTdz[i]*alpha*g*reduction_factor)[0])])
                mld[it, i] = -z[i, mld_idx[i]]
                mass_ratio[it,  i] = np.sum(S[:, :, mld_idx[i]::])/np.sum(S)
            w_rms = np.sqrt(w2_fluc_avg)
            w_mld_rms[it, i] = w_rms[mld_idx[i]]

        # plume analysis 
        r_profile, plume_index, S_contour_temp = plume_tracer_radius(x, y, nx, centerline_index, S, S_contour[i])
        if np.size(plume_index)==0:
            plume_index = [nx[0]//2, nx[1]//2, nx[2]-1]
            r_profile = np.zeros(nx[2])
        neutral_index = neutral_buoyancy_loc(b_fluc, plume_index, centerline_index)
        max_index = np.min(plume_index[2])

        # find vertical and horizontal lengths and correlated values
        if plume_analysis_plot:
            if np.size(neutral_index)!=0: # neutral plume in current case, apply data to matrices
                h_neutral[it,  i] = z[i, neutral_index]
                r_neutral[it,  i] = r_profile[neutral_index]
                w_neutral[it,  i] = w[centerline_index[0][neutral_index], centerline_index[1][neutral_index], neutral_index]
                b_neutral[it,  i] = b_fluc[centerline_index[0][neutral_index], centerline_index[1][neutral_index], neutral_index]
                T_neutral[it,  i] = T_fluc[centerline_index[0][neutral_index], centerline_index[1][neutral_index], neutral_index]
                S_neutral[it,  i] = S_fluc[centerline_index[0][neutral_index], centerline_index[1][neutral_index], neutral_index]
            else: # neutral plume does not exist in current case, update neutral starting point and leave matrices be
                start_neutral[i] += 1 

        h_max[it,  i] = z[i, max_index]
        r_hmax[it,  i] = r_profile[max_index]
        w_hmax[it,  i] = w[centerline_index[0][max_index], centerline_index[1][max_index], max_index]
        b_hmax[it,  i] = b_fluc[centerline_index[0][max_index], centerline_index[1][max_index], max_index] 
        T_hmax[it,  i] = T_fluc[centerline_index[0][max_index], centerline_index[1][max_index], max_index]
        S_hmax[it,  i] = S_fluc[centerline_index[0][max_index], centerline_index[1][max_index], max_index]

        r_mld[it,  i] = r_profile[mld_idx[i]]
        w_mld[it,  i] = w[centerline_index[0][mld_idx[i]], centerline_index[1][mld_idx[i]], mld_idx[i]]
        b_mld[it,  i] = b_fluc[centerline_index[0][mld_idx[i]], centerline_index[1][mld_idx[i]], mld_idx[i]]
        if mld_analysis_plot:
            bw_fluc_avg = np.mean(b_fluc * wc, axis=(-3, -2))
            bw_mld_avg[it,  i] = bw_fluc_avg[mld_idx[i]]

        T_mld[it,  i] = T_fluc[centerline_index[0][mld_idx[i]], centerline_index[1][mld_idx[i]], mld_idx[i]]
        S_mld[it,  i] = S_fluc[centerline_index[0][mld_idx[i]], centerline_index[1][mld_idx[i]], mld_idx[i]]


    print(f"Time step {it} out of {nt} complete.")

############ NONDIMENSIONALIZATION ############
if ND: 
    name_uni = 'ND_' + name_uni
    
    vel_scale = np.zeros(num_cases)
    b_scale = np.zeros(num_cases)
    N2 = np.zeros(num_cases)
    for i in range(num_cases):
        N2[i] = g  * dTdz[i] / T0
        b_scale[i] = mld0[i] * N2[i]
        vel_scale[i] = mld0[i] * np.sqrt(N2[i])
        Sj[i] = F_s[i] / (vel_scale[i]) #/ (np.sqrt(g  * rj)) #
        z[i, :] = z[i, :] / mld0[i]
        zf[i, :] = zf[i, :] / mld0[i]
    bflux_scale = b_scale * vel_scale
    
    # plotting dimensional first for sanity check
    # plot ranges
    ranges = plot_ranges(lz = 96, rho0 = rho0, T0 = T0, dTdz = np.max(dTdz), Sj = np.max(Sj))
    tol = 1.1
    wmax = np.max(np.abs([w_mld, w_hmax, w_neutral])) * tol
    ranges['w'] = [-wmax, wmax]
    bflucmax = np.max(np.abs([b_mld, b_hmax, b_neutral])) * tol
    ranges['b_fluc'] = [-bflucmax, bflucmax]
    Tw_max = np.max(np.abs(Tw_mld_avg)) * tol
    ranges['Tw_fluc'] = [-Tw_max, Tw_max]
    Sw_max = np.max(np.abs(Sw_mld_avg)) * tol
    ranges['Sw_fluc'] = [-Sw_max, Sw_max]
    rmax = np.max(np.abs([r_mld, r_hmax, r_neutral])) * tol
    ranges['radius'] = [0, rmax]
    Sfluc_max = np.max(np.abs([S_mld, S_hmax, S_neutral])) * tol
    ranges['S_fluc'] = [-Sfluc_max, Sfluc_max]
    Tfluc_max = np.max(np.abs([T_mld, T_hmax, T_neutral])) * tol
    ranges['T_fluc'] = [-Tfluc_max, Tfluc_max]
    if not mld_transient:
        mld = mld0*np.ones((nt, num_cases))
  

    ############ PLOTTING ############
    if plume_analysis_plot:
        plume_temporal_analysis(time, ranges, color_opt, fig_folder, case_names, name_uni[3::], lx_plot, start_neutral, mld, h_neutral, h_max, r_mld, r_neutral, r_hmax, w_mld, w_neutral, w_hmax, b_mld, b_neutral, b_hmax, T_mld, T_neutral, T_hmax, S_mld, S_neutral, S_hmax, Sw_mld_avg, Tw_mld_avg)
    if mld_analysis_plot:
        bw_max = np.max(np.abs(bw_mld_avg)) * tol
        ranges['bw_fluc'] = [-bw_max, bw_max] 
        ranges['mass_ratio'] = [0, 1.05]
        wrms_max = np.max(w_mld_rms) * tol
        ranges['vel_rms'] = [0, wrms_max]
        mld_temporal_analysis(time, ranges, color_opt, fig_folder, case_names, name_uni[3::], lx_plot, mld, mld0, bw_mld_avg, Tw_mld_avg, Sw_mld_avg, mass_ratio, w_mld_rms)
        bw_mld_avg = bw_mld_avg / b_scale / vel_scale
    for i in range(num_cases):
        Sw_mld_avg[:, i] = Sw_mld_avg[:, i] / F_s[i]
        Tw_mld_avg[:, i] = Tw_mld_avg[:, i] / vel_scale[i] / T0

        h_neutral[:, i] = h_neutral[:, i] / mld0[i]
        h_max[:, i] = h_max[:, i] / mld0[i]

        w_mld[:, i] = w_mld[:, i] / vel_scale[i]
        w_neutral[:, i] = w_neutral[:, i] / vel_scale[i]
        w_hmax[:, i] = w_hmax[:, i] / vel_scale[i]

        b_mld[:, i] = b_mld[:, i] / b_scale[i]
        b_neutral[:, i] = b_neutral[:, i] / b_scale[i]
        b_hmax[:, i] = b_hmax[:, i] / b_scale[i]

        S_mld[:, i] = S_mld[:, i] / S_contour[i] #* vel_scale[i] / F_s[i]
        S_neutral[:, i] = S_neutral[:, i] / S_contour[i] #* vel_scale[i] / F_s[i]
        S_hmax[:, i] = S_hmax[:, i] / S_contour[i] #* vel_scale[i] / F_s[i]

        z[:, i] = z[:, i] / mld0[i]
        zf[:, i] = zf[:, i] / mld0[i]

        r_mld[:, i] = r_mld[:, i] / rj # / mld0[i] # 
        r_neutral[:, i] = r_neutral[:, i] / rj # / mld0[i] # 
        r_hmax[:, i] = r_hmax[:, i] / rj #/ mld0[i] # 
        if mld_transient:
            mld[:, i] = mld[:, i] / mld0[i]


    lx_plot= np.array(lx) / np.min(mld0)
    mld_plot = np.ones((nt, num_cases))

    T_mld = T_mld / T0
    T_neutral = T_neutral / T0
    T_hmax = T_hmax / T0
else:
    if not mld_transient:
        mld_plot = mld0*np.ones((nt, num_cases))
        mld0 = np.ones((nt, num_cases))
    else:
        mld_plot = mld
# plot ranges
ranges = plot_ranges(lz = 96, rho0 = rho0, T0 = T0, dTdz = np.max(dTdz), Sj = np.max(Sj))
tol = 1.1
wmax = np.max(np.abs([w_mld, w_hmax, w_neutral])) * tol
ranges['w'] = [-wmax, wmax]
bflucmax = np.max(np.abs([b_mld, b_hmax, b_neutral])) * tol
ranges['b_fluc'] = [-bflucmax, bflucmax]
Tw_max = np.max(np.abs(Tw_mld_avg)) * tol
ranges['Tw_fluc'] = [-Tw_max, Tw_max]
Sw_max = np.max(np.abs(Sw_mld_avg)) * tol
ranges['Sw_fluc'] = [-Sw_max, Sw_max]
rmax = np.max(np.abs([r_mld, r_hmax, r_neutral])) * tol
ranges['radius'] = [0, rmax]
Sfluc_max = np.max(np.abs([S_mld, S_hmax, S_neutral])) * tol
ranges['S_fluc'] = [-Sfluc_max, Sfluc_max]
Tfluc_max = np.max(np.abs([T_mld, T_hmax, T_neutral])) * tol
ranges['T_fluc'] = [-Tfluc_max, Tfluc_max]
ranges['mass_ratio'] = [0, 1.05]

if plume_analysis_plot:
    plume_temporal_analysis(time, ranges, color_opt, fig_folder, case_names, name_uni, lx_plot, start_neutral, mld_plot, h_neutral, h_max, r_mld, r_neutral, r_hmax, w_mld, w_neutral, w_hmax, b_mld, b_neutral, b_hmax, T_mld, T_neutral, T_hmax, S_mld, S_neutral, S_hmax, Sw_mld_avg, Tw_mld_avg, ND)
if mld_analysis_plot:
    bw_max = np.max(np.abs(bw_mld_avg)) * tol
    ranges['bw_fluc'] = [-bw_max, bw_max] 
    wrms_max = np.max(w_mld_rms) * tol
    ranges['vel_rms'] = [0, wrms_max]
    mld_temporal_analysis(time, ranges, color_opt, fig_folder, case_names, name_uni, lx_plot, mld, mld0, bw_mld_avg, Tw_mld_avg, Sw_mld_avg, mass_ratio, w_mld_rms, ND)