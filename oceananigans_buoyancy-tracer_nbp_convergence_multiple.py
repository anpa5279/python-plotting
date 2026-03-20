import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from scipy.interpolate import make_interp_spline
import imageio.v2 as imageio
import matplotlib.ticker as mticker

def fluct2_mean(a, b, a_avg, b_avg):
    """
    Computes the mean fluctuations in x and y directions.

    Parameters:
        a: ndarray of shape (nt, nx, ny, nz)
        b: ndarray of shape (nt, nx, ny, nz)
        a_avg: 1D array of shape (nt, nz,) or (nt, 1, 1, nz)
        b_avg: 1D array of shape (nt, nz,) or (nt, 1, 1, nz)
    Returns:
        ab_fluct_avg: 1D array of shape (nt, nz,)
    """
    # Compute squared fluctuation a'*b' = (a-a_avg)(B-b_avg)
    ab_fluct = (a - a_avg)*(b - b_avg)
    # Mean over x and y, result shape: (nt, nz,)
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

# Set up folder and simulation parameters
folder = '/Users/annapauls/Library/CloudStorage/OneDrive-UCB-O365(2)/CU-Boulder/TESLa/Carbon Sequestration/Simulations/Oceananigans/NBP/b tracer for NBP/with closure Re 3000/'
fig_folder = folder + 'figures/'
os.makedirs(fig_folder, exist_ok=True)
name = 'NBP-'
const_N = 128
matrix_N = 256
case_titles = []
# flags for what to plot
turb_stats_video = False
convergence_test = True
with_halos = False

# parameters
ml = 30.0  # mixed layer depth in meters
g = 9.80665  # gravity in m/s^2
alpha = 2.0e-4 
dTdz = 0.01 
# List JLD2 files
cases = sorted([f for f in os.listdir(folder) if str(const_N) in f])
ncases = len(cases)
print(f"Number of cases: {ncases}")
dtn = sorted([f for f in os.listdir(os.path.join(folder, cases[0])) if f.endswith('.jld2')])
fid = os.path.join(folder, cases[0], dtn[-1])
with h5py.File(fid, 'r') as f:
    timeseries_group = [g for g in f.keys() if 'timeseries' in g][0]
    t_group = f[timeseries_group + '/t']
    t_save = sorted([float(k) for k in t_group.keys()])
    t_save = np.array(t_save)
    t = np.array([t_group[str(int(k))][()] for k in t_save])
    nt = len(t)
    lx = [f['grid/Lx'][()], f['grid/Ly'][()], f['grid/Lz'][()]]
    if convergence_test:
        days = 0.25
        t_desired = days*24*60*60
        nt_desired = np.where(t==t_desired)[0][0]
print(f"Number of time steps: {nt} \n")
# Initialize arrays
u2_fluc = np.zeros((nt, matrix_N, ncases))
v2_fluc = np.zeros((nt, matrix_N, ncases))
w2_fluc = np.zeros((nt, matrix_N + 1, ncases))
b2_fluc = np.zeros((nt, matrix_N, ncases))
wc2_fluc = np.zeros((nt, matrix_N, ncases))
uv_fluc = np.zeros((nt, matrix_N, ncases))
uw_fluc = np.zeros((nt, matrix_N, ncases))
vw_fluc = np.zeros((nt, matrix_N, ncases))
bw_fluc = np.zeros((nt, matrix_N, ncases))
u_avg = np.zeros((nt, matrix_N, ncases))
v_avg = np.zeros((nt, matrix_N, ncases))
w_avg = np.zeros((nt, matrix_N + 1, ncases))
wc_avg = np.zeros((nt, matrix_N, ncases))
b_avg = np.zeros((nt, matrix_N, ncases))
b_nostrat_avg = np.zeros((nt, matrix_N, ncases))
b_flux_avg = np.zeros((nt, matrix_N, ncases))
b_nostrat_flux_avg = np.zeros((nt, matrix_N, ncases))
bw_fluc = np.zeros((nt, matrix_N, ncases))
L_ozmidov = np.zeros((nt, matrix_N, ncases))
nx = np.zeros((3, ncases)).astype(int)
x = np.zeros((matrix_N, ncases))
y = np.zeros((matrix_N, ncases))
z = np.zeros((matrix_N, ncases))
xf = np.zeros((matrix_N + 1, ncases))
yf = np.zeros((matrix_N + 1, ncases))
zf = np.zeros((matrix_N + 1, ncases))
if convergence_test:
    res_test = np.zeros(ncases).astype(int)
    case_titles = ["" for x in range(ncases)]

for caseindex, case in enumerate(cases):
    dtn = sorted([f for f in os.listdir(os.path.join(folder, case)) if f.endswith('.jld2')])
    nf = len(dtn)
    # Read t steps
    fid = os.path.join(folder, case, dtn[-1])
    with h5py.File(fid, 'r') as f:
        timeseries_group = [g for g in f.keys() if 'timeseries' in g][0]
        t_group = f[timeseries_group + '/t']
        t_save = sorted([float(k) for k in t_group.keys()])
        t_save = np.array(t_save)
        t = np.array([t_group[str(int(k))][()] for k in t_save])
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
    if convergence_test:
        if nx[0, caseindex] == const_N and nx[1, caseindex] == const_N and nx[2, caseindex] == const_N:
            res_test[caseindex] = 2 # both convergence plots
            case_titles[caseindex] = f"N = {nx[0, caseindex]}"
        elif nx[2, caseindex] == const_N:
            res_test[caseindex] = 1 # horizontal resolution is varied
            case_titles[caseindex] = f"N = {nx[0, caseindex]}"
        elif nx[0, caseindex] == const_N and nx[1, caseindex] == const_N:
            res_test[caseindex] = 0 #vertical resolution is varied
            case_titles[caseindex] = f"N = {nx[2, caseindex]}"
    xs = slice(0, nx[0, caseindex])
    ys = slice(0, nx[1, caseindex])
    zs = slice(0, nx[2, caseindex])
    zfs = slice(0, nx[2, caseindex] + 1)

    # initialize arrays for this case
    u = np.zeros((nt, nx[0, caseindex], nx[1, caseindex], nx[2, caseindex]))
    v = np.zeros((nt, nx[0, caseindex], nx[1, caseindex], nx[2, caseindex]))
    w = np.zeros((nt, nx[0, caseindex], nx[1, caseindex], nx[2, caseindex] + 1))
    b = np.zeros((nt, nx[0, caseindex], nx[1, caseindex], nx[2, caseindex]))
    b_nostrat = np.zeros((nt, nx[0, caseindex], nx[1, caseindex], nx[2, caseindex]))
    b_flux = np.zeros((nt, nx[0, caseindex], nx[1, caseindex], nx[2, caseindex]))
    Ri = np.zeros((nt, nx[0, caseindex], nx[1, caseindex], nx[2, caseindex]))
    for it in range(nt):
        for r in range(nf):
            fname = os.path.join(folder, case, dtn[r])
            # .transpose(2, 1, 0)) to get (z, y, x) ordering to be (x, y, z)
            with h5py.File(fname, 'r') as f:
                if with_halos:
                    u[it, :, :, :] = (f['timeseries/u'][f'{int(t_save[it])}'][hx[0]:-hx[0], hx[1]:-hx[1], hx[2]:-hx[2]].transpose(2, 1, 0))
                    v[it, :, :, :] = (f['timeseries/v'][f'{int(t_save[it])}'][hx[0]:-hx[0], hx[1]:-hx[1], hx[2]:-hx[2]].transpose(2, 1, 0))
                    w[it, :, :, :] = (f['timeseries/w'][f'{int(t_save[it])}'][hx[0]:-hx[0], hx[1]:-hx[1], hx[2]:-hx[2]].transpose(2, 1, 0))
                    b[it, :, :, :] = (f['timeseries/b'][f'{int(t_save[it])}'][hx[0]:-hx[0], hx[1]:-hx[1], hx[2]:-hx[2]].transpose(2, 1, 0))
                else:
                    u[it, :, :, :] = (f['timeseries/u'][f'{int(t_save[it])}'][:, :, :].transpose(2, 1, 0))
                    v[it, :, :, :] = (f['timeseries/v'][f'{int(t_save[it])}'][:, :, :].transpose(2, 1, 0))
                    w[it, :, :, :] = (f['timeseries/w'][f'{int(t_save[it])}'][:, :, :].transpose(2, 1, 0))
                    b[it, :, :, :] = (f['timeseries/b'][f'{int(t_save[it])}'][:, :, :].transpose(2, 1, 0))
    print(f"Data loaded for Nx={nx[0, caseindex]}, Ny={nx[1, caseindex]}, Nz={nx[2, caseindex]}")

    #interpolate w to cell centers
    w_face = make_interp_spline(zf[0:nx[2, caseindex] + 1, caseindex], w, axis=3, k=1)
    wc = w_face(z[0:nx[2, caseindex], caseindex])
    print(f"\tFace Interpolations completed")

    # prepping variables for plume statistics
    b_background = stratification_profile(z[0:nx[2, caseindex], caseindex], alpha*g*dTdz, ml)
    for it in np.arange(nt):
        b_nostrat[it, :, :, :] = b[it, :, :, :] - b_background
    b_nostrat_avg[:, 0:nx[2, caseindex], caseindex] = np.mean(b_nostrat, axis=(1, 2))

    b_flux = b * wc
    b_nostrat_flux = b_nostrat * wc

    u = u #-u_s
    #calculate means
    u_avg[:, 0:nx[2, caseindex], caseindex] = np.mean(u, axis=(1, 2))
    v_avg[:, 0:nx[2, caseindex], caseindex] = np.mean(v, axis=(1, 2))
    w_avg[:, 0:nx[2, caseindex] + 1, caseindex] = np.mean(w, axis=(1, 2))
    wc_avg[:, 0:nx[2, caseindex], caseindex] = np.mean(wc, axis=(1, 2))
    b_avg[:, 0:nx[2, caseindex], caseindex] = np.mean(b, axis=(1, 2))
    b_flux_avg[:, 0:nx[2, caseindex], caseindex] = np.mean(b_flux, axis=(1, 2))
    b_nostrat_flux_avg[:, 0:nx[2, caseindex], caseindex] = np.mean(b_nostrat_flux, axis=(1, 2))

    #calcualte reynolds stresses
    for it in np.arange(nt):
        u2_fluc[it, 0:nx[2, caseindex], caseindex] = fluct2_mean(u[it, :, :, :], u[it, :, :, :], u_avg[it, 0:nx[2, caseindex], caseindex], u_avg[it, 0:nx[2, caseindex], caseindex])
        v2_fluc[it, 0:nx[2, caseindex], caseindex] = fluct2_mean(v[it, :, :, :], v[it, :, :, :], v_avg[it, 0:nx[2, caseindex], caseindex], v_avg[it, 0:nx[2, caseindex], caseindex])
        w2_fluc[it, 0:nx[2, caseindex]+1, caseindex] = fluct2_mean(w[it, :, :, :], w[it, :, :, :], w_avg[it, 0:nx[2, caseindex]+1, caseindex], w_avg[it, 0:nx[2, caseindex]+1, caseindex])
        b2_fluc[it, 0:nx[2, caseindex], caseindex] = fluct2_mean(b[it, :, :, :], b[it, :, :, :], b_avg[it, 0:nx[2, caseindex], caseindex], b_avg[it, 0:nx[2, caseindex], caseindex])
        wc2_fluc[it, 0:nx[2, caseindex], caseindex] = fluct2_mean(wc[it, :, :, :], wc[it, :, :, :], wc_avg[it, 0:nx[2, caseindex], caseindex], wc_avg[it, 0:nx[2, caseindex], caseindex])
        uv_fluc[it, 0:nx[2, caseindex], caseindex] = fluct2_mean(u[it, :, :, :], v[it, :, :, :], u_avg[it, 0:nx[2, caseindex], caseindex], v_avg[it, 0:nx[2, caseindex], caseindex])
        uw_fluc[it, 0:nx[2, caseindex], caseindex] = fluct2_mean(u[it, :, :, :], wc[it, :, :, :], u_avg[it, 0:nx[2, caseindex], caseindex], wc_avg[it, 0:nx[2, caseindex], caseindex])
        vw_fluc[it, 0:nx[2, caseindex], caseindex] = fluct2_mean(v[it, :, :, :], wc[it, :, :, :], v_avg[it, 0:nx[2, caseindex], caseindex], wc_avg[it, 0:nx[2, caseindex], caseindex])
        bw_fluc[it, 0:nx[2, caseindex], caseindex] = fluct2_mean(b[it, :, :, :], wc[it, :, :, :], b_avg[it, 0:nx[2, caseindex], caseindex], wc_avg[it, 0:nx[2, caseindex], caseindex])
    print(f"\tReynolds stresses calculated")
    
    # Ozmidov scale calculations
    shear_u = np.zeros((nt, nx[0, caseindex], nx[1, caseindex], nx[2, caseindex]))
    shear_v = np.zeros((nt, nx[0, caseindex], nx[1, caseindex], nx[2, caseindex]))
    for it in np.arange(nt):
        shear_u[it, :, :, :] = np.gradient(u[it, :, :, :], dz, axis=-1)
        shear_v[it, :, :, :] = np.gradient(u[it, :, :, :], dz, axis=-1)
    ke = 0.5 * (u2_fluc + v2_fluc + wc2_fluc)
    shear_tot = -(np.mean(shear_u, axis=(1,2))*uw_fluc[: , 0:nx[2, caseindex], caseindex] + 
                np.mean(np.gradient(u[it, :, :, :], dy, axis=-1), axis=(1,2))*uv_fluc[: , 0:nx[2, caseindex], caseindex] +
                np.mean(shear_v, axis=(1,2))*vw_fluc[: , 0:nx[2, caseindex], caseindex] + 
                np.mean(np.gradient(u[it, :, :, :], dx, axis=-1), axis=(1,2))*uv_fluc[: , 0:nx[2, caseindex], caseindex] +
                np.mean(np.gradient(w[it, :, :, :], dy, axis=-1), axis=(1,2))*vw_fluc[: , 0:nx[2, caseindex], caseindex] + 
                np.mean(np.gradient(w[it, :, :, :], dx, axis=-1), axis=(1,2))*uw_fluc[: , 0:nx[2, caseindex], caseindex])

    espilon = shear_tot - ke[: , 0:nx[2, caseindex], caseindex] + bw_fluc[: , 0:nx[2, caseindex], caseindex]
    espilon_avg = np.mean(espilon, axis=(1,2))
    for it in np.arange(nt):
        L_ozmidov[it, 0:nx[2, caseindex], caseindex] = (espilon_avg/b_background)**0.5
    print(f"Ozmidov scale calculated\n")
# rms fluctuations
u_rms = u2_fluc**0.5
v_rms = v2_fluc**0.5
w_rms = w2_fluc**0.5
b_rms = b2_fluc**0.5


print("All cases loaded and processed")
if convergence_test:
    b_sign = np.sign(b_nostrat_avg)
    sign_change_z = b_sign[..., 1:, :] * b_sign[..., :-1, :] < 0
    diff_z = b_nostrat_avg[..., 1:, :] - b_nostrat_avg[..., :-1, :]
    diff_z_masked = np.where(sign_change_z, diff_z, np.nan)
    b_max_sign_change_to_negative_loc = np.zeros((nt, ncases))
    b_max_sign_change_to_positive_loc = np.zeros((nt, ncases))
    b_rms_sign = np.zeros((nt, ncases))
    w_rms_sign = np.zeros((nt, ncases))
    b_flux_rms_sign = np.zeros((nt, ncases))
    idx_neg = np.zeros((nt, ncases)).astype(int)
    for caseindex in range(ncases):
        for it in range(nt):
            idx_neg[it, caseindex] = np.nanargmin(diff_z_masked[it, :, caseindex])
            idx_pos = np.nanargmax(diff_z_masked[it, :, caseindex])
            b_max_sign_change_to_negative_loc[it, caseindex] = z[idx_neg[it, caseindex], caseindex] # NBP is being stopped from continuing downwards - biggest change from negative to positive buoyancy from the surface per case per time
            b_rms_sign[it, caseindex] = b_rms[it, idx_neg[it, caseindex], caseindex]
            w_rms_sign[it, caseindex] = w_rms[it, idx_neg[it, caseindex], caseindex]
            b_flux_rms_sign[it, caseindex] = b_flux_avg[it, idx_neg[it, caseindex], caseindex]
            b_max_sign_change_to_positive_loc[it, caseindex] = z[idx_pos, caseindex]# biggest change from positive to negative buoyancy from the surface per case per time
    print("Convergence test calculations complete")


############ PLOTTING ############
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

# plot ranges
fac = 1*10**(-1)
vel_max = np.max([np.abs(u_avg).max(), np.abs(v_avg).max(), np.abs(w_avg).max()])
vel_range = [-1*vel_max, vel_max]#[-0.01, 0.01]#
rms_range = [0, np.max([u_rms.max(), v_rms.max(), w_rms.max()])]#[0, 0.02]#
restress_max = np.max([np.abs(uv_fluc).max(), np.abs(uw_fluc).max(), np.abs(vw_fluc).max()])
restress_range = [-1*restress_max, restress_max]#[-3*10**(-5), 3*10**(-5)]#
richardson_range = [0, 5*10**(-5)]# Ri_avg.max()]#
B_avg_range = [b_avg.min(), b_avg.max()]
if convergence_test:
    brms_sign_range = [0, b_rms_sign.max()]
    brms_range = [0, b_rms.max()]
    w_rms_range = [0, w_rms.max()]
    bflux_rms_range = [b_flux_avg.min(), b_flux_avg.max()]
    z_sign_range = [b_max_sign_change_to_negative_loc.min(), b_max_sign_change_to_negative_loc.max()]
    lengthscale_range = [0, L_ozmidov.max()]

levels =500
formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-3, 3))

line_styles = ['-', '--', '-.', ':'] 
if convergence_test:
    outdir = fig_folder + 'convergence testing/'
    os.makedirs(outdir, exist_ok=True)
    filenames = []
    hor = np.array(np.isin(res_test, [1, 2]), dtype=bool)
    ver = np.array(np.isin(res_test, [0, 2]), dtype=bool)
    print(nx[1, hor])
    print(nx[2, ver])
    for it in range(0, nt):
        td = t[it] / 3600 / 24
        fig = plt.figure(figsize=(12, 5))
        fig.tight_layout()
        fig.suptitle(f'{td:.2f} days', fontsize=12) 
        # Titles for each row
        fig.text(0.5, 0.94, "Vertical resolution convergence", 
                ha="center", va="center", fontsize=12)

        fig.text(0.5, 0.5, "Horizontal resolution convergence", 
                ha="center", va="center", fontsize=12)

        # z location of buoyancy sign change as a function of resolution
        ax1 = fig.add_subplot(2, 5,  1)
        ax1.plot(nx[2, ver], b_max_sign_change_to_negative_loc[it, ver], marker='o', linestyle='none')
        ax1.set_ylabel("Lengthscale [m]")
        ax1.set_ylim(z_sign_range)

        ax4 = fig.add_subplot(2, 5,  6)
        ax4.plot(nx[1, hor], b_max_sign_change_to_negative_loc[it, hor], marker='o', linestyle='none')
        ax4.set_ylabel("Lengthscale [m]")
        ax4.set_ylim(z_sign_range)

        # RMS buoyancy as a function of resolution 
        ax2 = fig.add_subplot(2, 5,  2)
        ax2.plot(nx[2, ver], b_rms_sign[it, ver], marker='o', linestyle='none')
        ax2.set_ylabel("Buoyancy Root Mean Square Error [m/s$^{2}$]")
        ax2.set_ylim(brms_sign_range)

        ax5 = fig.add_subplot(2, 5,  7)
        ax5.plot(nx[1, hor], b_rms_sign[it, hor], marker='o', linestyle='none')
        ax5.set_ylabel("Buoyancy Root Mean Square Error [m/s$^{2}$]")
        ax5.set_ylim(brms_sign_range)

        # RMS w as a function of resolution
        ax2 = fig.add_subplot(2, 5,  3)
        ax2.plot(nx[2, ver], w_rms[it, idx_neg[it, ver], ver], marker='o', linestyle='none')
        ax2.set_ylabel("Root Mean Square Error [m/s]")
        ax2.set_ylim(w_rms_range)

        ax5 = fig.add_subplot(2, 5,  8)
        ax5.plot(nx[1, hor], w_rms[it, idx_neg[it, hor], hor], marker='o', linestyle='none')
        ax5.set_ylabel("[m/s]")
        ax5.set_ylim(w_rms_range)

        # RMS buoyancy flux as a function of resolution
        ax4 = fig.add_subplot(2, 5,  4)
        ax4.plot(nx[2, ver], bw_fluc[it, idx_neg[it, ver], ver], marker='o', linestyle='none')
        ax4.set_ylabel("Buoyancy Flux Fluctuations [m$^{2}$/s$^{3}$]")
        ax4.set_ylim(bflux_rms_range)

        ax8 = fig.add_subplot(2, 5,  9)
        ax8.plot(nx[1, hor], bw_fluc[it, idx_neg[it, hor], hor], marker='o', linestyle='none')
        ax8.set_ylabel("Buoyancy Flux Fluctuations [m$^{2}$/s$^{3}$]")
        ax8.set_ylim(bflux_rms_range)

        # RMS buoyancy flux as a function of resolution
        ax5 = fig.add_subplot(2, 5, 5)
        ax5.plot(nx[2, ver], L_ozmidov[it, idx_neg[it, ver], ver], marker='o', linestyle='none')
        ax5.set_ylabel("Buoyancy Flux Fluctuations [m$^{2}$/s$^{3}$]")
        ax5.set_ylim(bflux_rms_range)

        ax10 = fig.add_subplot(2, 5,  10)
        ax10.plot(nx[1, hor], L_ozmidov[it, idx_neg[it, hor], hor], marker='o', linestyle='none')
        ax10.set_ylabel("Buoyancy Flux Fluctuations [m$^{2}$/s$^{3}$]")
        ax10.set_ylim(bflux_rms_range)

        fig.supxlabel("Number of Grid Cells", fontsize=12)
        # --- Save Frame ---
        frame_path = os.path.join(outdir, f"convergence_test_{it:04d}.png")
        plt.tight_layout()
        plt.savefig(frame_path)
        filenames.append(frame_path)
        print(f"Time step {it + 1}/{nt} captured: {frame_path}")
        plt.close(fig)
    print("Creating video..")
    filenames = sorted([f for f in os.listdir(outdir) if f.endswith(".png")])
    with imageio.get_writer(fig_folder + 'convergence_test.mp4', fps=10) as writer:
        for filename in filenames:
            image = imageio.imread(f"{outdir}/{filename}")
            writer.append_data(image)
