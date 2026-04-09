import os
import numpy as np
import matplotlib.pyplot as plt

from plotting_functions import create_video
from plotting_comparisons import plot_format
from general_analysis_functions import ab_fluc_mean
from data_collection_functions import collect_time_outputs, collect_fields_distributed, collect_temp_and_sal
from dense_plume_analysis import plume_tracer_radius
from matplotlib.lines import Line2D

def compute_importance_scores(var, nx, dx, z, fig_folder):
    var_rms = np.zeros(nx[2])
    L_dom = np.zeros(nx[2])
    var_fluc = var - np.mean(var, axis=(0,1))
    for k in range(nx[2]):
        var_hat = np.fft.fft2(var_fluc[:,:,k])
        E = 0.5 * np.abs(var_hat)**2
        var_rms_temp = np.sqrt(E.sum())
        var_rms[k] = var_rms_temp

        # FFT wavenumbers
        kx = np.fft.fftfreq(nx[0], dx[0]) * 2*np.pi
        ky = np.fft.fftfreq(nx[1], dx[1]) * 2*np.pi
        k_h = np.sqrt(kx[:,None]**2 + ky[None,:]**2)
        k_mean = (E.ravel() * k_h.ravel()).sum() / E.sum()
        L_dom[k] = 2*np.pi / k_mean


    # Compute simple importance score: mean spread across runs
    nd_profiles_all_runs = np.array(var_nd)
    spread = np.std(nd_profiles_all_runs, axis=0)
    importance_score = np.mean(spread)

    # Nondimensionalize
    z_nd = z / L_dom       # pick first horizontal slice
    var_nd = np.mean(var, axis=(0,1)) / var_rms
    
    return var_nd, z_nd, importance_score

def fft_plotting(fig_folder, name, num_cases, case_names, num_var, var_names, var_nd, z_nd, color_opt):
    """Plot the nondimensionalized variable profile and save frame for video creation.
    Args:       var_names (str): Name of the variables being plotted across cases
                var_nd (array): Nondimensionalized variable profile (mean across horizontal) for current case
                z_nd (array): Nondimensionalized vertical coordinate
    Returns:    outdir (str): Directory where the frame is saved, used for video creation
    '"""
    outdir = os.path.join(fig_folder, name + "fft frames")
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(2, 4, dpi=150, figsize=(30, 12))

    ax = ax.ravel()

    for i in range(num_var):
        for n in range(num_cases):
            ax[i].plot(var_nd[:, n], z_nd, alpha=0.7, color = color_opt[n])

        ax[i].set_xlabel(f'{var_names[i]} / RMS', fontsize=18)
        ax[i].set_ylabel('z / L_dom', fontsize=18)
        ax[i].grid(True)
        ax[i].set_title(f'{var_names[i]}', fontsize=20)
    ax[num_var].remove()
    case_handles = [
        Line2D([0], [0], color=color_opt[i], linestyle='solid', label=case_names[i])
        for i in range(num_cases)]
    fig.legend(handles=case_handles,
            ncol=num_cases,
            bbox_to_anchor=(0.8, 0.2))
    plt.tight_layout()
    # --- Save Frame ---
    frame_path = os.path.join(outdir, f"fft_{it:04d}.png")
    plt.savefig(frame_path)
    plt.close(fig)
    print(f"Time step {it + 1} captured: {frame_path}")

    return outdir

# Set up folder and simulation parameters
universal_folder = '/Users/annapauls/Library/CloudStorage/OneDrive-UCB-O365/CU-Boulder/TESLa/Carbon Sequestration/Simulations/Oceananigans/NBP/salinity and temperature/'
folder_names =['beta = default S0 = 0.1 dTdz = 0.005', 'beta = default S0 = 0.1', 'beta = default S0 = 0.1 dTdz = 0.05', 'beta = default S0 = 0.1 dTdz = 0.1'] 
case_names =[r'dT/dz = 0.005', r'dT/dz = 0.01', r'dT/dz = 0.05', r'dT/dz = 0.10'] 
fig_folder = os.path.join(universal_folder, 'fft analysis')
num_cases = len(case_names)

# flags for what to plot
ND = True
transient_mld = True

# flags for how to read data
with_halos = False
stokes = False * np.ones(num_cases) 
salinity = True


# variable names
var_names = ['Buoyancy', 'Temperature', 'Tracer', 'Velocity', 'Perturbed Buoyancy Flux', 'Perturbed Temperature Flux', 'Perturbed Tracer Flux', 'Radius of Plume']
num_var = len(var_names)

# physical parameters
rj = 10 # m, radius of salinity flux circle at the surface
g = 9.80665  # gravity in m/s^2
dTdz = np.array([0.005, 0.01, 0.05, 0.1]) # background temperature gradient in K/m
mld = 30 * np.ones(num_cases) # 
T0 = 25.0
S0 = 0 
wp = 0.001
Sj = 0.1 * np.ones(num_cases) # np.array([0.05, 0.1, 0.15, 0.2])# 
F_s = np.dot(Sj, wp)

S_value = np.array([0.034487168519906714, 0.03602588163919859, 0.03995705848735615, 0.042189206877616705]) # for dTdz variations at max bw index
S_contour = S_value*0.15 
w_avg_centerline = np.array([-0.043499393099289844, -0.03394752674800345, -0.018453789243636633, -0.01406895477434289]) # for strat centerline w values thorughout time


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

    alpha, beta = collect_temp_and_sal(fid, salinity)
    t_save.append(t_save_temp)

nt = np.arange(0, len(t_save_temp))

#N2 = g * alpha * dTdz 
mld_idx = []
for i in range(num_cases):
    mld_idx.append(np.argmin(np.abs(z[:, i]+mld[i])))

for it in nt:
    dbdz = np.zeros((nx[2], num_cases))
    r_profile = np.zeros((nx[2], num_cases))
    b_center = np.zeros((nx[2], num_cases))
    T_fluc_center = np.zeros((nx[2], num_cases))
    S_fluc_center = np.zeros((nx[2], num_cases))
    T_nd_profiles_all_runs = []
    b_nd_profiles_all_runs = []
    S_nd_profiles_all_runs = []
    r_nd_profiles_all_runs = []
    bw_nd_profiles_all_runs = []
    for i, folder in enumerate(folders):
        # Load data from files
        if not salinity:
            u, v, w, T, Pdynamic, Pstatic = collect_fields_distributed(Nranks, folder, dtn, t_save[i][it], hx, nx, True, salinity, with_halos)
        else:
            u, v, w, T, S, Pdynamic, Pstatic = collect_fields_distributed(Nranks, folder, dtn, t_save[i][it], hx, nx, True, salinity, with_halos)
        # convert temperature and salinity to buoyancy 
        b = g*alpha*(T - T0) - g*beta*(S - S0)

        wc = 0.5 * (w[..., :-1] + w[..., 1:])
        if transient_mld and it != 0:
            dbdz[:, i] = np.gradient(np.mean(b, axis=(0,1)), z)
            dbdz_tol = dbdz[:, i] <= (5.0*10**(-7))#dbdz[:, i] <= 0.01*N2[i]#
            if np.any(dbdz_tol):
                mld_idx[i] = np.min(np.where(dbdz_tol))
            else:
                mld_idx[i] = nx[2] - 1
            mld[i] = -z[mld_idx[i], i]
        b_avg = np.mean(b[:, i], axis=(0,1))
        wc_avg = np.mean(wc[:, i], axis=(0,1))
        bw_fluc, bw_fluc_avg = ab_fluc_mean(b, wc, b_avg, wc_avg)
        bw_idx = np.where(bw_fluc_avg==np.max(bw_fluc_avg))[0][0]
        rp_profile, plume_index, S_contour_temp = plume_tracer_radius(x, y, nx, centerline_index, S, S_contour[i]) 
        r_profile[:, i] = rp_profile
        # computing importances scores of variables of interest
        b_nd, z_ndb, importance_score = compute_importance_scores(b, nx, dx, z, fig_folder)
        b_nd_profiles_all_runs.append(b_nd)
        with open(os.path.join(fig_folder, 'b_importance_scores.txt'), 'a') as f:
            f.write(f"b at {it}: {importance_score:.3f}\n")

        T_nd, z_ndT, importance_score = compute_importance_scores(T, nx, dx, z, fig_folder)
        T_nd_profiles_all_runs.append(T_nd)
        with open(os.path.join(fig_folder, 'T_importance_scores.txt'), 'a') as f:
            f.write(f"T at {it}: {importance_score:.3f}\n")
        
        S_nd, z_ndS, importance_score = compute_importance_scores(S, nx, dx, z, fig_folder)
        S_nd_profiles_all_runs.append(S_nd)
        with open(os.path.join(fig_folder, 'S_importance_scores.txt'), 'a') as f:
            f.write(f"S at {it}: {importance_score:.3f}\n")

        r_nd, z_ndr, importance_score = compute_importance_scores(rp_profile, nx, dx, z, fig_folder)
        r_nd_profiles_all_runs.append(r_nd)
        with open(os.path.join(fig_folder, 'r_importance_scores.txt'), 'a') as f:
            f.write(f"r at {it}: {importance_score:.3f}\n")

        bw_nd, z_ndbw, importance_score = compute_importance_scores(bw_fluc, nx, dx, z, fig_folder)
        bw_nd_profiles_all_runs.append(bw_nd)
        with open(os.path.join(fig_folder, 'bw_importance_scores.txt'), 'a') as f:
            f.write(f"bw at {it}: {importance_score:.3f}\n")
    
        var_nd = [b_nd, T_nd, S_nd, w_nd, bw_nd, Tw_nd, Sw_nd, r_nd]
        # plotting
        outdir = fft_plotting(fig_folder, name, num_cases, case_names, num_var, var_names, var_nd, z_nd, color_opt)

create_video(outdir, fig_folder, name, "fft")