import os
import numpy as np
import matplotlib.pyplot as plt

from plotting_functions import plot_ranges, create_video
from general_analysis_functions import a2_fluc_mean, ab_fluc_mean
from plotting_comparisons import plot_format, plume_horizontal_spatial_plot, plume_vertical_spatial_plot
from data_collection_functions import collect_time_outputs, collect_fields_distributed, collect_temp_and_sal
from dense_plume_analysis import plume_tracer_radius

# Set up folder and simulation parameters
universal_folder = '/Users/annapauls/Library/CloudStorage/OneDrive-UCB-O365/CU-Boulder/TESLa/Carbon Sequestration/Simulations/Oceananigans/NBP/salinity and temperature/'
folder_names =['beta = default S0 = 0.1 dTdz = 0.005', 'beta = default S0 = 0.1', 'beta = default S0 = 0.1 dTdz = 0.05', 'beta = default S0 = 0.1 dTdz = 0.1'] 
case_names =[r'dT/dz = 0.005', r'dT/dz = 0.01', r'dT/dz = 0.05', r'dT/dz = 0.10'] 

name_uni = "transient mld -l_area - fix dbdz-dTdz"
fig_folder = os.path.join(universal_folder, 'comparison figures', 'strat comparison figures')

num_cases = len(case_names)

# flags for what to plot
plot_1d_z = True
plot_1d_y = False
ND = True
transient_mld = True

# flags for how to read data
with_halos = False
stokes = False * np.ones(num_cases) 
salinity = True

video = True

# physical parameters
rj = 10 # m, radius of salinity flux circle at the surface
g = 9.80665  # gravity in m/s^2
dTdz = np.array([0.005, 0.01, 0.05, 0.1]) # background temperature gradient in K/m
rho0 = 1026
mld = 30 * np.ones(num_cases) # 
T0 = 25.0
S0 = 0 
wp = 0.001
Sj = 0.1 * np.ones(num_cases) # np.array([0.05, 0.1, 0.15, 0.2])# 
F_s = np.dot(Sj, wp)

S_value = np.array([0.034487168519906714, 0.03602588163919859, 0.03995705848735615, 0.042189206877616705]) # for dTdz variations at max bw index
S_contour = S_value*0.15 
w_avg_centerline = np.array([-0.043499393099289844, -0.03394752674800345, -0.018453789243636633, -0.01406895477434289]) # for strat centerline w values thorughout time


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
    ranges_hor = ranges.copy()
    ranges_hor['S'] = [0, 9*10**(-2)]
    ranges_hor['vel_rms'] = [0, 4*10**-3]
    ranges_hor['bw_fluc'] = [-2*10**(-5), 2*10**(-5)]
    ranges_hor['b_flux'] = [-1*10**(-5), 1*10**(-5)]
    ranges_hor['b_rms'] = [0, 1.5*10**(-5)]
    ranges_hor['b_fluc'] = [-5*10**(-4), 5*10**(-4)]
    ranges_hor['w'] = [-0.1, 0.1]
    ranges_hor['T'] = [23.5, 25.5]
    hor_idx = np.array(mld_idx)
    name_uni = name_uni + f"at z = {z[hor_idx, np.arange(num_cases)]} m"

if plot_1d_z:
    name_uni +="_centerline or average"

############ NONDIMENSIONALIZATION ############
if ND: 
    name_nd = 'ND_' + name_uni

    area = (2*rj)**2 
    l_area = np.sqrt(area)
    N2 = g * alpha * dTdz 
    N2_scale = (N2/g*l_area)**(1/2)
    Fr_flux = F_s * beta / np.sqrt(l_area * g)
    vel_scale = Fr_flux * np.sqrt(l_area * g) / N2_scale
    b_scale = N2_scale * Fr_flux * g
    F_b_scale = Fr_flux * g**(3/2) * l_area**(1/2)
    T_scale = N2_scale * Fr_flux / alpha
    S_scale = N2_scale * Fr_flux / beta
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
        S_avg[:, i] = np.mean(S, axis=(-3, -2))
        T_avg[:, i] = np.mean(T, axis=(-3, -2))

variables = ['b', 'T', 'S', 'u', 'v', 'wc']

# --------------------------
# Set up plot
# --------------------------
plt.rcParams.update({'font.size': 16})
fig, ax = plt.subplots(2, 3, dpi=150, figsize=(30, 12))
ax = ax.ravel()

# --------------------------
# Loop over variables
# --------------------------
importance_scores = {}

for i, var_name in enumerate(variables):
    nd_profiles_all_runs = []

    for run in runs:
        var = run[var_name]
        z = run['z']  # shape (nz, ny) or (nz,)
        dx = run['dx']
        nx_run = run['nx']

        # Remove horizontal mean
        var_prime = var - np.mean(var, axis=(0,1))[..., np.newaxis]

        nz = var.shape[2]
        rms_list = []
        L_list = []

        for k in range(nz):
            var_hat = np.fft.fft2(var_prime[:,:,k])
            E = 0.5 * np.abs(var_hat)**2
            var_rms = np.sqrt(E.sum())
            rms_list.append(var_rms)

            # FFT wavenumbers
            kx = np.fft.fftfreq(nx_run[0], dx[0]) * 2*np.pi
            ky = np.fft.fftfreq(nx_run[1], dx[1]) * 2*np.pi
            k_h = np.sqrt(kx[:,None]**2 + ky[None,:]**2)
            k_mean = (E.ravel() * k_h.ravel()).sum() / E.sum()
            L_list.append(2*np.pi / k_mean)

        var_rms = np.array(rms_list)
        L_dom = np.array(L_list)

        # Nondimensionalize
        z_nd = z[:, 0] / L_dom       # pick first horizontal slice
        var_nd = np.mean(var, axis=(0,1)) / var_rms
        nd_profiles_all_runs.append(var_nd)

        # Plot
        ax[i].plot(var_nd, z_nd, linewidth=2, alpha=0.7)

    # Compute simple importance score: mean spread across runs
    nd_profiles_all_runs = np.array(nd_profiles_all_runs)
    spread = np.std(nd_profiles_all_runs, axis=0)
    importance_scores[var_name] = np.mean(spread)

    ax[i].set_xlabel(f'{var_name} / RMS', fontsize=18)
    ax[i].set_ylabel('z / L_dom', fontsize=18)
    ax[i].grid(True)
    ax[i].set_title(f'{var_name}', fontsize=20)

plt.tight_layout()
plt.show()

# --------------------------
# Print importance scores
# --------------------------
print("Variable importance (higher = more sensitive to input changes):")
for var, score in importance_scores.items():
    print(f"{var}: {score:.3f}")
