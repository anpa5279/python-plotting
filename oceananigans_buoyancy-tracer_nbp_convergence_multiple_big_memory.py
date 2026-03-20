import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from scipy.interpolate import make_interp_spline
import imageio.v2 as imageio
import matplotlib.ticker as mticker
import re
def fluct2_mean(a, b, a_avg, b_avg):
    """
    Computes the mean fluctuations in x and y directions.

    Parameters:
        a: ndarray of shape (nx, ny, nz)
        b: ndarray of shape (nx, ny, nz)
        a_avg: 1D array of shape (nz,) or (1, 1, nz)
        b_avg: 1D array of shape (nz,) or (1, 1, nz)
    Returns:
        ab_fluct_avg: 1D array of shape (nz,)
    """
    # Compute squared fluctuation a'*b' = (a-a_avg)(B-b_avg)
    ab_fluct = (a - a_avg)*(b - b_avg)
    # Mean over x and y, result shape: (nz,)
    ab_fluct_avg = np.mean(ab_fluct, axis=(0, 1))

    return ab_fluct_avg

def stratification_profile(z, dadz, ml):
    """Returns a linear stratification profile."""
    a = np.zeros(len(z))
    for k in range(len(z)):
        if z[k] <= -ml:
            a[k] = dadz * (z[k] + ml)  # linear increase from 0 at -ml to 0.05 at surface
        else:
            a[k] = 0.0  # constant below mixed layer
    return a

def parse_case(case, N):
    Nx, Ny, Nz = map(int, re.findall(r'\d+', case))

    is_cube = (Nx == Ny == Nz)
    is_horiz_sweep = (Nz == N and Nx != N)
    is_vert_sweep  = (Nx == N and Nz != N)

    # Priority:
    # 0 → cube
    # 1 → horizontal sweep (Nx varies)
    # 2 → vertical sweep (Nz varies)
    if is_cube:
        group = 0
    elif is_horiz_sweep:
        group = 1
    elif is_vert_sweep:
        group = 2
    else:
        group = 3  # fallback, just in case

    return (group, Nx, Nz)

# Set up folder and simulation parameters
folder = '/Users/annapauls/Library/CloudStorage/OneDrive-UCB-O365(2)/CU-Boulder/TESLa/Carbon Sequestration/Simulations/Oceananigans/NBP/b tracer for NBP/with closure Re 3000/'
fig_folder = folder + 'figures/'
os.makedirs(fig_folder, exist_ok=True)
name = 'NBP-'
const_N = 128
matrix_N = 256
case_titles = []
# flags for what to plot
buoyancy_profiles = True
buoyancy_planeslices = False
point_plot = False
with_halos = False

# parameters
ml = 30.0  # mixed layer depth in meters
g = 9.80665  # gravity in m/s^2
alpha = 2.0e-4 
dTdz = 0.01 

# plot setup
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

levels =500
formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-3, 3))

line_styles = ['-', '--', '-.', ':'] 
colors = ['black', 'blue', 'green', 'red', 'purple', 'pink', 'gray', 'orange', 'cyan', 'olive']
# plotting ranges
vel_range = [-0.0002, 0.0002]#[-1*vel_max, vel_max]#
rms_range = [0, 0.003]#[0, np.max([u_rms.max(), v_rms.max(), w_rms.max()])]#
restress_range = [-3*10**(-6), 3*10**(-6)]#[-1*restress_max, restress_max]#
richardson_range = [0, 5*10**(-3)]# Ri_avg.max()]#
P_d_range = [-0.005, 0.005]#
P_s_range = [-0.05, 0.05]#[-Pstatic.max(), Pstatic.max()]
B_range = [-1.5*10**(-3), 10**(-6)]#[B.min(), B.max()]
b_avg_range = [-1.5*10**(-3), 1.0*10**(-4)]
lamb_avg_range = [-2*10**(-6), 2*10**(-6)]#[-1*lamb_avg_max, lamb_avg_max]
b_rms_range = [0, 5*10**(-5)]
b_flux_rms_range = [-2*10**(-4), 2*10**(-4)]

# List JLD2 files
cases = sorted([f for f in os.listdir(folder) if str(const_N) in f])
cases_sorted = sorted(cases, key=lambda c: parse_case(c, const_N))
ncases = len(cases_sorted)
print(f"Number of cases: {ncases}")
dtn = sorted([f for f in os.listdir(os.path.join(folder, cases_sorted[0])) if f.endswith('.jld2')])
fid = os.path.join(folder, cases_sorted[0], dtn[-1])
with h5py.File(fid, 'r') as f:
    timeseries_group = [g for g in f.keys() if 'timeseries' in g][0]
    t_group = f[timeseries_group + '/t']
    t_save = sorted([float(k) for k in t_group.keys()])
    t_save = np.array(t_save)
    t = np.array([t_group[str(int(k))][()] for k in t_save])
    nt = 61 #len(t)
    lx = [f['grid/Lx'][()], f['grid/Ly'][()], f['grid/Lz'][()]]
    if point_plot:
        days = 0.25
        t_desired = days*24*60*60
        nt_desired = np.where(t==t_desired)[0][0]
print(f"Number of time steps: {nt} \n")
if point_plot or buoyancy_profiles:
    res_test = np.zeros(ncases).astype(int)
    case_titles = ["" for x in range(ncases)]

t_save = np.zeros((nt, ncases))
nx = np.zeros((3, ncases)).astype(int)
x = np.zeros((matrix_N, ncases))
y = np.zeros((matrix_N, ncases))
z = np.zeros((matrix_N, ncases))
xf = np.zeros((matrix_N + 1, ncases))
yf = np.zeros((matrix_N + 1, ncases))
zf = np.zeros((matrix_N + 1, ncases))
for caseindex, case in enumerate(cases_sorted):
    dtn = sorted([f for f in os.listdir(os.path.join(folder, case)) if f.endswith('.jld2')])
    nf = len(dtn)
    # Read t steps
    fid = os.path.join(folder, case, dtn[-1])
    with h5py.File(fid, 'r') as f:
        timeseries_group = [g for g in f.keys() if 'timeseries' in g][0]
        t_group = f[timeseries_group + '/t']
        t_save[0:nt, caseindex] = np.array(sorted([float(k) for k in t_group.keys()]))[0:nt]
        nx[:, caseindex] = [int(f['grid/Nx'][()]), int(f['grid/Ny'][()]), int(f['grid/Nz'][()])]
        hx = [f['grid/Hx'][()], f['grid/Hy'][()], f['grid/Hz'][()]]
        x[0:nx[0, caseindex], caseindex] = f['grid/xᶜᵃᵃ'][hx[0]:-hx[0]]
        y[0:nx[1, caseindex], caseindex] = f['grid/yᵃᶜᵃ'][hx[1]:-hx[1]]
        z[0:nx[2, caseindex], caseindex] = f['grid/z/cᵃᵃᶜ'][hx[2]:-hx[2]]
        xf[0:nx[0, caseindex], caseindex] = f['grid/xᶠᵃᵃ'][hx[0]:-hx[0]]
        yf[0:nx[1, caseindex], caseindex] = f['grid/yᵃᶠᵃ'][hx[1]:-hx[1]]
        zf[0:nx[2, caseindex] + 1, caseindex] = f['grid/z/cᵃᵃᶠ'][hx[2]:-hx[2]]
        dz = f['grid/z/Δᵃᵃᶜ'][()]
        dy = f['grid/Δyᵃᶜᵃ'][()]
        dx = f['grid/Δxᶜᵃᵃ'][()]
        if point_plot or buoyancy_profiles:
            if nx[0, caseindex] == const_N and nx[1, caseindex] == const_N and nx[2, caseindex] == const_N:
                res_test[caseindex] = 2 # both convergence plots
                case_titles[caseindex] = f"N = {nx[0, caseindex]}"
            elif nx[2, caseindex] == const_N:
                res_test[caseindex] = 1 # horizontal resolution is varied
                case_titles[caseindex] = f"N = {nx[0, caseindex]}"
            elif nx[0, caseindex] == const_N and nx[1, caseindex] == const_N:
                res_test[caseindex] = 0 #vertical resolution is varied
                case_titles[caseindex] = f"N = {nx[2, caseindex]}"
L_ozmidov = np.zeros((nt, ncases))
L_ozmidov_background = np.zeros((nt, ncases))
for it in range(nt):
    print(f"Processing time step {it + 1}/{nt}...")
    # Initialize arrays
    b_max_sign_change_to_negative_loc = np.zeros((ncases))
    b_max_sign_change_to_positive_loc = np.zeros((ncases))
    b_rms_sign = np.zeros((ncases))
    w_rms_sign = np.zeros((ncases))
    b_flux_rms_sign = np.zeros((ncases))
    idx_neg = np.zeros((ncases)).astype(int)
    u2_fluc = np.zeros((matrix_N, ncases))
    v2_fluc = np.zeros((matrix_N, ncases))
    w2_fluc = np.zeros((matrix_N + 1, ncases))
    b2_fluc = np.zeros((matrix_N, ncases))
    wc2_fluc = np.zeros((matrix_N, ncases))
    uv_fluc = np.zeros((matrix_N, ncases))
    uw_fluc = np.zeros((matrix_N, ncases))
    vw_fluc = np.zeros((matrix_N, ncases))
    bw_fluc = np.zeros((matrix_N, ncases))
    u_avg = np.zeros((matrix_N, ncases))
    v_avg = np.zeros((matrix_N, ncases))
    w_avg = np.zeros((matrix_N + 1, ncases))
    wc_avg = np.zeros((matrix_N, ncases))
    b_avg = np.zeros((matrix_N, ncases))
    b_flux_avg = np.zeros((matrix_N, ncases))
    b_nostrat_flux_avg = np.zeros((matrix_N, ncases))
    bw_fluc = np.zeros((matrix_N, ncases))
    u_rms = np.zeros((matrix_N, ncases))
    v_rms = np.zeros((matrix_N, ncases))
    w_rms = np.zeros((matrix_N + 1, ncases))
    b_rms = np.zeros((matrix_N, ncases))
    b = np.zeros((ncases, matrix_N, matrix_N, matrix_N))
    for caseindex, case in enumerate(cases_sorted):
        dtn = sorted([f for f in os.listdir(os.path.join(folder, case)) if f.endswith('.jld2')])
        nf = len(dtn)
        xs = slice(0, nx[0, caseindex])
        ys = slice(0, nx[1, caseindex])
        zs = slice(0, nx[2, caseindex])
        zfs = slice(0, nx[2, caseindex] + 1)

        # initialize arrays for this case
        u = np.zeros((nx[0, caseindex], nx[1, caseindex], nx[2, caseindex]))
        v = np.zeros((nx[0, caseindex], nx[1, caseindex], nx[2, caseindex]))
        w = np.zeros((nx[0, caseindex], nx[1, caseindex], nx[2, caseindex] + 1))
        b_fluc = np.zeros((nx[0, caseindex], nx[1, caseindex], nx[2, caseindex]))
        b_flux = np.zeros((nx[0, caseindex], nx[1, caseindex], nx[2, caseindex]))
        Ri = np.zeros((nx[0, caseindex], nx[1, caseindex], nx[2, caseindex]))
        for r in range(nf):
            fname = os.path.join(folder, case, dtn[r])
            # .transpose(2, 1, 0)) to get (z, y, x) ordering to be (x, y, z)
            with h5py.File(fname, 'r') as f:
                if with_halos:
                    u = (f['timeseries/u'][f'{int(t_save[it, caseindex])}'][hx[0]:-hx[0], hx[1]:-hx[1], hx[2]:-hx[2]].transpose(2, 1, 0))
                    v = (f['timeseries/v'][f'{int(t_save[it, caseindex])}'][hx[0]:-hx[0], hx[1]:-hx[1], hx[2]:-hx[2]].transpose(2, 1, 0))
                    w = (f['timeseries/w'][f'{int(t_save[it, caseindex])}'][hx[0]:-hx[0], hx[1]:-hx[1], hx[2]:-hx[2]].transpose(2, 1, 0))
                    b[0:nx[0, caseindex], 0:nx[1, caseindex], 0:nx[2, caseindex]] = (f['timeseries/b'][f'{int(t_save[it, caseindex])}'][hx[0]:-hx[0], hx[1]:-hx[1], hx[2]:-hx[2]].transpose(2, 1, 0))
                else:
                    u = (f['timeseries/u'][f'{int(t_save[it, caseindex])}'][:, :, :].transpose(2, 1, 0))
                    v = (f['timeseries/v'][f'{int(t_save[it, caseindex])}'][:, :, :].transpose(2, 1, 0))
                    w = (f['timeseries/w'][f'{int(t_save[it, caseindex])}'][:, :, :].transpose(2, 1, 0))
                    b[caseindex, 0:nx[0, caseindex], 0:nx[1, caseindex], 0:nx[2, caseindex]] = (f['timeseries/b'][f'{int(t_save[it, caseindex])}'][:, :, :].transpose(2, 1, 0))

        #interpolate w to cell centers
        w_face = make_interp_spline(zf[0:nx[2, caseindex] + 1, caseindex], w, axis=-1, k=1)
        wc = w_face(z[0:nx[2, caseindex], caseindex])

        # prepping variables for plume statistics
        b_background = stratification_profile(z[0:nx[2, caseindex], caseindex], alpha*g*dTdz, ml)
        b_fluc = b[caseindex, 0:nx[0, caseindex], 0:nx[1, caseindex], 0:nx[2, caseindex]] - b_background
        b_nostrat_avg = np.mean(b_fluc, axis=(-3, -2))

        b_flux = b[caseindex, 0:nx[0, caseindex], 0:nx[1, caseindex], 0:nx[2, caseindex]] * wc
        b_nostrat_flux = b_fluc * wc

        u = u #-u_s
        #calculate means
        u_avg[0:nx[2, caseindex], caseindex] = np.mean(u, axis=(-3, -2))
        v_avg[0:nx[2, caseindex], caseindex] = np.mean(v, axis=(-3, -2))
        w_avg[0:nx[2, caseindex] + 1, caseindex] = np.mean(w, axis=(-3, -2))
        wc_avg[0:nx[2, caseindex], caseindex] = np.mean(wc, axis=(-3, -2))
        b_avg[0:nx[2, caseindex], caseindex] = np.mean(b[caseindex, 0:nx[0, caseindex], 0:nx[1, caseindex], 0:nx[2, caseindex]], axis=(-3, -2))
        b_flux_avg[0:nx[2, caseindex], caseindex] = np.mean(b_flux, axis=(-3, -2))
        b_nostrat_flux_avg[0:nx[2, caseindex], caseindex] = np.mean(b_nostrat_flux, axis=(-3, -2))

        #calcualte reynolds stresses
        u2_fluc[0:nx[2, caseindex], caseindex] = fluct2_mean(u, u, u_avg[0:nx[2, caseindex], caseindex], u_avg[0:nx[2, caseindex], caseindex])
        v2_fluc[0:nx[2, caseindex], caseindex] = fluct2_mean(v, v, v_avg[0:nx[2, caseindex], caseindex], v_avg[0:nx[2, caseindex], caseindex])
        w2_fluc[0:nx[2, caseindex]+1, caseindex] = fluct2_mean(w, w, w_avg[0:nx[2, caseindex]+1, caseindex], w_avg[0:nx[2, caseindex]+1, caseindex])
        b2_fluc[0:nx[2, caseindex], caseindex] = fluct2_mean(b[caseindex, 0:nx[0, caseindex], 0:nx[1, caseindex], 0:nx[2, caseindex]], b[caseindex, 0:nx[0, caseindex], 0:nx[1, caseindex], 0:nx[2, caseindex]], b_avg[0:nx[2, caseindex], caseindex], b_avg[0:nx[2, caseindex], caseindex])
        wc2_fluc[0:nx[2, caseindex], caseindex] = fluct2_mean(wc, wc, wc_avg[0:nx[2, caseindex], caseindex], wc_avg[0:nx[2, caseindex], caseindex])
        uv_fluc[0:nx[2, caseindex], caseindex] = fluct2_mean(u, v, u_avg[0:nx[2, caseindex], caseindex], v_avg[0:nx[2, caseindex], caseindex])
        uw_fluc[0:nx[2, caseindex], caseindex] = fluct2_mean(u, wc, u_avg[0:nx[2, caseindex], caseindex], wc_avg[0:nx[2, caseindex], caseindex])
        vw_fluc[0:nx[2, caseindex], caseindex] = fluct2_mean(v, wc, v_avg[0:nx[2, caseindex], caseindex], wc_avg[0:nx[2, caseindex], caseindex])
        bw_fluc[0:nx[2, caseindex], caseindex] = fluct2_mean(b[caseindex, 0:nx[0, caseindex], 0:nx[1, caseindex], 0:nx[2, caseindex]], wc, b_avg[0:nx[2, caseindex], caseindex], wc_avg[0:nx[2, caseindex], caseindex])
        # Ozmidov scale calculations
        ke = 0.5 * (u2_fluc + v2_fluc + wc2_fluc)
        espilon = - ke[0:nx[2, caseindex], caseindex]#shear_tot - ke[0:nx[2, caseindex], caseindex] + bw_fluc[0:nx[2, caseindex], caseindex]
        espilon_avg = np.mean(espilon)
        L_ozmidov[it, caseindex] = (espilon_avg/np.mean(b_avg[0:nx[2, caseindex], caseindex]))**0.5
        L_ozmidov_background[it, caseindex] = (espilon_avg/(-alpha*g*dTdz))**0.5
        # rms fluctuations
        u_rms[0:nx[2, caseindex], caseindex] = u2_fluc[0:nx[2, caseindex], caseindex]**0.5
        v_rms[0:nx[2, caseindex], caseindex] = v2_fluc[0:nx[2, caseindex], caseindex]**0.5
        w_rms[0:nx[2, caseindex]+1, caseindex] = w2_fluc[0:nx[2, caseindex]+1, caseindex]**0.5
        b_rms[0:nx[2, caseindex], caseindex] = b2_fluc[0:nx[2, caseindex], caseindex]**0.5

        if point_plot:
            b_sign = np.sign(b_nostrat_avg)
            sign_change_z = b_sign[1:] * b_sign[:-1] < 0
            diff_z = b_nostrat_avg[1:] - b_nostrat_avg[:-1]#wrong
            diff_z_masked = np.where(sign_change_z, diff_z, np.nan)
            if np.all(np.isnan(diff_z_masked[:, caseindex])):
                print(f"No sign change found for case {caseindex} at time step {it}.")
                break
            idx_neg[caseindex] = np.nanargmin(diff_z_masked[:, caseindex])
            idx_pos = np.nanargmax(diff_z_masked[:, caseindex])
            b_max_sign_change_to_negative_loc[caseindex] = z[idx_neg[caseindex], caseindex] # NBP is being stopped from continuing downwards - biggest change from negative to positive buoyancy from the surface per case per time
            b_rms_sign[caseindex] = b_rms[idx_neg[caseindex], caseindex]
            w_rms_sign[caseindex] = w_rms[idx_neg[caseindex], caseindex]
            b_flux_rms_sign[caseindex] = b_flux_avg[idx_neg[caseindex], caseindex]
            b_max_sign_change_to_positive_loc[caseindex] = z[idx_pos, caseindex]# biggest change from positive to negative buoyancy from the surface per case per time

    ############ PLOTTING ############
    if point_plot:
        brms_sign_range = [0, b_rms_sign.max()]
        brms_range = [0, b_rms.max()]
        w_rms_range = [0, w_rms.max()]
        bflux_rms_range = [b_flux_avg.min(), b_flux_avg.max()]
        z_sign_range = [b_max_sign_change_to_negative_loc.min(), b_max_sign_change_to_negative_loc.max()]
        lengthscale_range = [0, 0.3]

        outdir = fig_folder + 'convergence testing/'
        os.makedirs(outdir, exist_ok=True)
        hor = np.array(np.isin(res_test, [1, 2]), dtype=bool)
        ver = np.array(np.isin(res_test, [0, 2]), dtype=bool)
        td = t[it] / 3600 / 24
        fig = plt.figure(figsize=(20, 8))
        fig.tight_layout()
        fig.suptitle(f'{td:.2f} days', fontsize=12) 
        # Titles for each row
        fig.text(0.5, 0.94, "Vertical resolution convergence", 
                ha="center", va="center", fontsize=14)

        fig.text(0.5, 0.48, "Horizontal resolution convergence", 
                ha="center", va="center", fontsize=14)

        # z location of buoyancy sign change as a function of resolution
        ax1 = fig.add_subplot(2, 5,  1)
        ax1.plot(nx[2, ver], b_max_sign_change_to_negative_loc[ver], marker='o', linestyle='none')
        ax1.set_ylabel("[m]")
        ax1.set_title("Neutrally buoyant depth")
        ax1.set_ylim(z_sign_range)

        ax4 = fig.add_subplot(2, 5,  6)
        ax4.plot(nx[1, hor], b_max_sign_change_to_negative_loc[hor], marker='o', linestyle='none')
        ax4.set_ylabel("[m]")
        ax4.set_title("Neutrally buoyant depth")
        ax4.set_ylim(z_sign_range)

        # RMS buoyancy as a function of resolution 
        ax2 = fig.add_subplot(2, 5,  2)
        ax2.plot(nx[2, ver], b_rms_sign[ver], marker='o', linestyle='none', label = "at neutrally buoyant depth")
        ax2.plot(nx[2, ver], b_rms_sign[ver-1], marker='o', linestyle='none', color = 'black', label = "above neutrally buoyant depth")
        ax2.legend(loc='upper right', handlelength=0.5)
        ax2.set_ylabel("[m/s$^{2}$]")
        ax2.set_title("Buoyancy RMS")
        ax2.set_ylim(brms_sign_range)

        ax5 = fig.add_subplot(2, 5,  7)
        ax5.plot(nx[1, hor], b_rms_sign[hor], marker='o', linestyle='none', label = "at neutrally buoyant depth")
        ax5.plot(nx[1, hor], b_rms_sign[hor-1], marker='o', linestyle='none', color = 'black', label = "above neutrally buoyant depth")
        ax5.legend(loc='upper right', handlelength=0.5)
        ax5.set_ylabel("[m/s$^{2}$]")
        ax5.set_title("Buoyancy RMS")
        ax5.set_ylim(brms_sign_range)

        # RMS w as a function of resolution
        ax2 = fig.add_subplot(2, 5,  3)
        ax2.plot(nx[2, ver], w_rms[idx_neg[ver], ver], marker='o', linestyle='none', label = "at neutrally buoyant depth")
        ax2.plot(nx[2, ver], w_rms[idx_neg[ver]-1, ver], marker='o', linestyle='none', color = 'black', label = "above neutrally buoyant depth")
        ax2.legend(loc='upper right', handlelength=0.5)
        ax2.set_ylabel("[m/s]")
        ax2.set_title("w RMS")
        ax2.set_ylim(w_rms_range)
        ax2.ticklabel_format(axis='y', style='sci', scilimits=(-1,1), useMathText=True)

        ax5 = fig.add_subplot(2, 5,  8)
        ax5.plot(nx[1, hor], w_rms[idx_neg[hor], hor], marker='o', linestyle='none', label = "at neutrally buoyant depth")
        ax5.plot(nx[1, hor], w_rms[idx_neg[hor]-1, hor], marker='o', linestyle='none', color = 'black', label = "above neutrally buoyant depth")
        ax5.legend(loc='upper right', handlelength=0.5)
        ax5.set_ylabel("[m/s]")
        ax5.set_title("w RMS")
        ax5.set_ylim(w_rms_range)
        ax5.ticklabel_format(axis='y', style='sci', scilimits=(-1,1), useMathText=True)

        # RMS buoyancy flux as a function of resolution
        ax4 = fig.add_subplot(2, 5,  4)
        ax4.plot(nx[2, ver], bw_fluc[idx_neg[ver], ver], marker='o', linestyle='none', label = "at neutrally buoyant depth")
        ax4.plot(nx[2, ver], bw_fluc[idx_neg[ver]-1, ver], marker='o', linestyle='none', color = 'black', label = "above neutrally buoyant depth")
        ax4.legend(loc='upper right', handlelength=0.5)
        ax4.set_ylabel("[m$^{2}$/s$^{3}$]")
        ax4.set_title("Buoyancy Flux Flucts")
        ax4.set_ylim(bflux_rms_range)

        ax8 = fig.add_subplot(2, 5,  9)
        ax8.plot(nx[1, hor], bw_fluc[idx_neg[hor], hor], marker='o', linestyle='none', label = "at neutrally buoyant depth")
        ax8.plot(nx[1, hor], bw_fluc[idx_neg[hor]-1, hor], marker='o', linestyle='none', color = 'black', label = "above neutrally buoyant depth")
        ax8.legend(loc='upper right', handlelength=0.5)
        ax8.set_ylabel("[m$^{2}$/s$^{3}$]")
        ax8.set_title("Buoyancy Flux Flucts")
        ax8.set_ylim(bflux_rms_range)

        # RMS buoyancy flux as a function of resolution
        ax5 = fig.add_subplot(2, 5, 5)
        ax5.plot(nx[2, ver], L_ozmidov[it, ver], marker='o', linestyle='none', color = 'black', label = r"b$_{\text{average}, 3}$ L$_{O}$")
        ax5.plot(nx[2, ver], L_ozmidov_background[it, ver], marker='o', linestyle='none', color = 'blue', label = r"b$_{\text{stratified}, 3}$ L$_{O}$")
        ax5.legend(loc='upper right', handlelength=0.5)
        ax5.set_ylabel("[m]")
        ax5.set_title("Ozmidov Length Scale")
        ax5.set_ylim(lengthscale_range)

        ax10 = fig.add_subplot(2, 5,  10)
        ax10.plot(nx[1, hor], L_ozmidov[it, hor], marker='o', linestyle='none', color = 'black', label = r"b$_{\text{average}, 3}$ L$_{O}$")
        ax10.plot(nx[1, hor], L_ozmidov_background[it, hor], marker='o', linestyle='none', color = 'blue', label = r"b$_{\text{stratified}, 3}$ L$_{O}$")
        ax10.legend(loc='upper right', handlelength=0.5)
        ax10.set_title("Ozmidov Length Scale")
        ax10.set_ylabel("[m]")
        ax10.set_ylim(lengthscale_range)

        fig.supxlabel("Number of Grid Cells", fontsize=12)
        # --- Save Frame ---
        frame_path = os.path.join(outdir, f"convergence_test_{it:04d}.png")
        plt.tight_layout()
        plt.savefig(frame_path)
        print(f"Time step {it + 1}/{nt} captured")
        plt.close(fig)
    if buoyancy_profiles:
        brms_range = [0, 1.5*10**(-5)]
        w_rms_range = [0, 1*10**(-2)]
        bflux_rms_range = [-1*10**(-8), 1*10**(-8)]
        lengthscale_range = [0, 0.4]
        outdir = fig_folder + 'buoyancy profiles/'
        os.makedirs(outdir, exist_ok=True)
        hor = np.array(np.isin(res_test, [1, 2]), dtype=bool)
        ver = np.array(np.isin(res_test, [0, 2]), dtype=bool)
        td = t[it] / 3600 / 24
        
        fig, ax = plt.subplots(3, 5, figsize=(20, 8), height_ratios = [1, 0.2, 1])

        fig.text(0.5, 1.08, f'{td:.2f} days', ha="center", fontsize=12) 
        # Titles for each row
        fig.text(0.5, 1.05, "Vertical resolution convergence", ha="center", fontsize=14)
        fig.text(0.5, 0.52, "Horizontal resolution convergence", ha="center", fontsize=14)
        
        ax1 = ax[0, 0]
        ax2 = ax[0, 1]
        ax3 = ax[0, 2]
        ax4 = ax[0, 3]
        ax5 = ax[0, 4]
        ax6 = ax[2, 0]
        ax7 = ax[2, 1]
        ax8 = ax[2, 2]
        ax9 = ax[2, 3]
        ax10 = ax[2, 4]
        for a in ax[1, :]:
            a.remove()
        for caseindex, case in enumerate(cases_sorted):
            if ver[caseindex] and hor[caseindex]:
                name_case = case.replace('flux b tracer ', "")
                # buoyancy profile
                ax[0, 0].plot(b_avg[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex], label = name_case)
                # RMS buoyancy 
                ax[0, 1].plot(b_rms[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex])
                # RMS w 
                #ax[0, 2].plot(u_rms[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex])#, linestyle = ':', label = r"$\langle$u$_{rms}$$\rangle_{xy}$")
                #ax[0, 2].plot(v_rms[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex])#, linestyle = '-.', label = r"$\langle$v$_{rms}$$\rangle_{xy}$")
                ax[0, 2].plot(w_rms[0:nx[2, caseindex]+1, caseindex], zf[0:nx[2, caseindex]+1, caseindex], color = colors[caseindex])#, linestyle = '--', label = r"$\langle$w$_{rms}$$\rangle_{xy}$")
                # RMS buoyancy flux 
                ax[0, 3].plot(bw_fluc[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex])
                # buoyancy profile
                ax[2, 0].plot(b_avg[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex], label = name_case)
                # RMS buoyancy 
                ax[2, 1].plot(b_rms[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex])
                # RMS w 
                #ax[2, 2].plot(u_rms[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex])#, linestyle = ':', label = r"$\langle$u$_{rms}$$\rangle_{xy}$")
                #ax[2, 2].plot(v_rms[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex])#, linestyle = '-.', label = r"$\langle$v$_{rms}$$\rangle_{xy}$")
                ax[2, 2].plot(w_rms[0:nx[2, caseindex]+1, caseindex], zf[0:nx[2, caseindex]+1, caseindex], color = colors[caseindex])#, linestyle = '--', label = r"$\langle$w$_{rms}$$\rangle_{xy}$")
                # RMS buoyancy flux 
                ax[2, 3].plot(bw_fluc[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex])
            elif ver[caseindex] and not hor[caseindex]:
                name_case = case.replace('flux b tracer ', "")
                # buoyancy profile
                ax[0, 0].plot(b_avg[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex], label = name_case)
                # RMS buoyancy 
                ax[0, 1].plot(b_rms[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex])
                # RMS w 
                #ax[0, 2].plot(u_rms[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex])#, linestyle = ':')
                #ax[0, 2].plot(v_rms[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex])#, linestyle = '-.')
                ax[0, 2].plot(w_rms[0:nx[2, caseindex]+1, caseindex], zf[0:nx[2, caseindex]+1, caseindex], color = colors[caseindex])#, linestyle = '--')
                # RMS buoyancy flux 
                ax[0, 3].plot(bw_fluc[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex])
            elif hor[caseindex] and not ver[caseindex]:
                name_case = case.replace('flux b tracer ', "")
                # buoyancy profile
                ax[2, 0].plot(b_avg[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex], label = name_case)
                # RMS buoyancy 
                ax[2, 1].plot(b_rms[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex])
                # RMS w 
                #ax[2, 2].plot(u_rms[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex])#, linestyle = '--')
                #ax[2, 2].plot(v_rms[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex])#, linestyle = '-.')
                ax[2, 2].plot(w_rms[0:nx[2, caseindex]+1, caseindex], zf[0:nx[2, caseindex]+1, caseindex], color = colors[caseindex])#, linestyle = ':')
                # RMS buoyancy flux 
                ax[2, 3].plot(bw_fluc[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex])

        # Ozmidov length scale
        ax[2, 4].plot(nx[0, hor], L_ozmidov_background[it, hor], marker = '+', label = r"b$_{\text{stratified}, 3}$ L$_{O}$", linestyle = 'none')
        ax[2, 4].plot(nx[0, hor], L_ozmidov[it, hor], marker = 'o', label = r"b$_{\text{average}, 3}$ L$_{O}$", linestyle = 'none')
        ax[0, 4].plot(nx[2, ver], L_ozmidov_background[it, ver], marker = '+', label = r"b$_{\text{stratified}, 3}$ L$_{O}$", linestyle = 'none')
        ax[0, 4].plot(nx[2, ver], L_ozmidov[it, ver], marker = 'o', label = r"b$_{\text{average}, 3}$ L$_{O}$", linestyle = 'none')

        ax[0, 0].set_xlabel("[m/s$^{2}$]")
        ax[0, 0].set_ylim([-lx[2], 0])
        ax[0, 0].set_title("Buoyancy")
        ax[0, 0].set_ylim([-lx[2], 0])
        ax[0, 0].set_ylabel("Depth [m]")
        ax[0, 0].set_xlim(b_avg_range)
        ax[0, 0].ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True)
        
        ax[0, 1].set_xlabel("[m/s$^{2}$]")
        ax[0, 1].set_title("Buoyancy RMS")
        ax[0, 1].set_xlim(brms_range)
        ax[0, 1].set_ylim([-lx[2], 0])
        #ax[0, 1].set_ylabel("Depth [m]")
    
        #ax[0, 2].legend(loc='upper right')#, handlelength=0.75)
        ax[0, 2].set_xlabel("[m/s]")
        ax[0, 2].set_title("w RMS")
        ax[0, 2].set_xlim(w_rms_range)
        ax[0, 2].ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True)
        ax[0, 2].set_ylim([-lx[2], 0])
        #ax[0, 2].set_ylabel("Depth [m]")

        ax[0, 3].set_xlabel("[m$^{2}$/s$^{3}$]")
        ax[0, 3].set_title("Buoyancy Flux Flucts")
        ax[0, 3].set_xlim(bflux_rms_range)
        ax[0, 3].set_ylim([-lx[2], 0])
        #ax[0, 3].set_ylabel("Depth [m]")

        ax[0, 4].legend(loc='upper right', handlelength=0.75)
        ax[0, 4].set_ylabel("Length Scale [m]")
        ax[0, 4].set_title("Ozmidov Length Scale")
        ax[0, 4].set_ylim(lengthscale_range)
        ax[0, 4].set_xlabel("Time [days]")
        ax[0, 4].set_xlim([0, matrix_N +10])
        
        ax[2, 0].set_xlabel("[m/s$^{2}$]")
        ax[2, 0].set_xlim(b_avg_range)
        ax[2, 0].set_title("Buoyancy")
        ax[2, 0].set_ylabel("Depth [m]")
        ax[2, 0].set_ylim([-lx[2], 0])
        ax[2, 0].set_xlim(b_avg_range)
        ax[2, 0].ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True)

        ax[2, 1].set_xlabel("[m/s$^{2}$]")
        ax[2, 1].set_title("Buoyancy RMS")
        ax[2, 1].set_xlim(brms_range)
        ax[2, 1].set_ylim([-lx[2], 0])

        #ax[2, 2].legend(loc='upper right')#, handlelength=0.75)
        ax[2, 2].set_xlabel("[m/s]")
        ax[2, 2].set_title("w RMS")
        ax[2, 2].set_xlim(w_rms_range)
        ax[2, 2].ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True)
        ax[2, 2].set_ylim([-lx[2], 0]) 

        ax[2, 3].set_xlabel("[m$^{2}$/s$^{3}$]")
        ax[2, 3].set_title("Buoyancy Flux Flucts")
        ax[2, 3].set_xlim(bflux_rms_range)
        ax[2, 3].set_ylim([-lx[2], 0])

        ax[2, 4].set_title("Ozmidov Length Scale")
        ax[2, 4].set_ylabel("Length Scale [m]")
        ax[2, 4].set_ylim(lengthscale_range)
        ax[2, 4].set_xlabel("Time [days]")
        ax[2, 4].set_xlim([0, matrix_N +10])
        ax[2, 4].legend(loc='upper right', handlelength=0.75)

        # universal legend
        handles0, labels0 = ax1.get_legend_handles_labels()
        handles1, labels1 = ax6.get_legend_handles_labels()

        fig.legend(handles0, labels0, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.99))
        fig.legend(handles1, labels1, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.46))


        # --- Save Frame ---
        frame_path = os.path.join(outdir, f"convergence_test_{it:04d}.png")
        plt.savefig(frame_path, bbox_inches="tight")
        print(f"Time step {it + 1}/{nt} captured")
        plt.close(fig)
    if buoyancy_planeslices:
        outdir = fig_folder + 'buoyancy planeslices/'
        os.makedirs(outdir, exist_ok=True)
        fig, ax = plt.subplots(3, 3, figsize=(18, 10), constrained_layout=True)
        td = t[it] / 3600 / 24
        fig.suptitle(f'{td:.2f} days') 
        norm = mcolors.Normalize(vmin=B_range[0], vmax=B_range[-1])
        mappable = cm.ScalarMappable(norm=norm) 
        hor = 0
        ver = 0
        for caseindex, case in enumerate(cases_sorted):
            X, Y, Z = np.meshgrid(x[0:nx[0, caseindex], caseindex] , y[0:nx[1, caseindex], caseindex] , z[0:nx[2, caseindex], caseindex])
            name_case = case.replace('flux b tracer ', "")
            ax[hor, ver].contourf(X[int(nx[0, caseindex]/2), :, :], Z[int(nx[0, caseindex]/2), :, :], b[caseindex, int(nx[0, caseindex]/2), 0:nx[1, caseindex], 0:nx[2, caseindex]], levels, norm=norm)
            ax[hor, ver].set_title(name_case)
            ax[hor, ver].set_xlabel("y [m]")
            ax[hor, ver].set_ylabel("z [m]")
            ax[hor, ver].set_aspect('equal')
            if ver > 1:
                hor += 1
                ver = 0
            else:
                ver += 1

        fig.colorbar(mappable, ax=ax, label=r"m/s$^2$", location='bottom', shrink=0.5, orientation='horizontal', format = formatter)
        
        # --- Save Frame ---
        frame_path = os.path.join(outdir, f"planeslices_{it:04d}.png")
        plt.savefig(frame_path, bbox_inches="tight")
        print(f"Time step {it + 1}/{nt} captured")
        plt.close(fig)


if point_plot:
    outdir = fig_folder + 'NBP and turb stats/'
    print("Creating video...")
    filenames = sorted([f for f in os.listdir(outdir) if f.endswith(".png")])
    with imageio.get_writer(fig_folder + 'point_plot.mp4', fps=10) as writer:
        for filename in filenames:
            image = imageio.imread(f"{outdir}/{filename}")
            writer.append_data(image) 
if buoyancy_profiles:
    outdir = fig_folder + 'buoyancy profiles/'
    print("Creating video...")
    filenames = sorted([f for f in os.listdir(outdir) if f.endswith(".png")])
    with imageio.get_writer(fig_folder + 'buoyancy_profiles.mp4', fps=10) as writer:
        for filename in filenames:
            image = imageio.imread(f"{outdir}/{filename}")
            writer.append_data(image) 
if buoyancy_planeslices:
    outdir = fig_folder + 'buoyancy planeslices/'
    print("Creating video...")
    filenames = sorted([f for f in os.listdir(outdir) if f.endswith(".png")])
    with imageio.get_writer(fig_folder + 'buoyancy_planeslices.mp4', fps=10) as writer:
        for filename in filenames:
            image = imageio.imread(f"{outdir}/{filename}")
            writer.append_data(image) 