import os
import numpy as np
import matplotlib.pyplot as plt
import scipy


from reader import OceananigansData
from interpolation import point
from plotting_general import create_video, save_frame
from self_similarity_functions import compute_delta, fit_power_law_all_times, fit_power_law, plot_loglog_fit

# ==========================================================
# MODEL INFORMATION
# ==========================================================
folder = '/Users/annapauls/Documents/Github repositories/3d_langmuir_gpu/localoutputs/scheme-tests/longer/WENO9/dx0.5'

opt = 'fft convolve w_c*10**-5'
outdir = os.path.join(folder, 'figures', 'full MLD constant alpha_2 ' + opt)
reader = OceananigansData(folder, salinity = True, with_halos=True)

nt = reader.nt
lx = reader.lx
dx = reader.dx
time = reader.t
dx_scale = max(dx[:-1]) # not including dz
r = np.arange(dx[0]/2, lx[0]/2, dx_scale)
# ==========================================================
# PARAMETERS
# ==========================================================
mld = 60
g = 9.80665
w0 = 0.001

# ==========================================================
# ANALYSIS
# ==========================================================
above_mld = np.where(reader.z>=-mld+1.0)[0]
# ignoring first few time steps and information below the MLD
if reader.nt < 10:
    it_range = np.arange(6, reader.nt)
else:
    it_range = np.arange(6, 10)


w_rz = -reader.load_scaling_analysis('outer length scale/w filtered') 
surf_remove = int(len(w_rz[0, 0, :]) - 12.0/dx[-1]) 
above_mld_remove = int(5.0/dx[-1])
#above_mld = above_mld[above_mld_remove:surf_remove]
z_constrained = reader.z[above_mld]
w_rz = w_rz[:, :, :]#above_mld_remove:surf_remove]
w_c = w_rz[:, 0, :]

w_rz_avg = np.mean(w_rz, axis=0)
w_c_avg = np.mean(w_c, axis=0)

# calculating delta
delta_avg = np.empty_like(w_c_avg)
delta = np.empty_like(w_c)
for k in range(len(z_constrained)):
    delta_avg[k] = np.max(point(w_rz_avg[:, k], r, f0 = 0.5*w_c_avg[k]))
    for it in range(len(it_range)):
        delta[it, k] = np.max(point(w_rz[it, :, k], r, f0 = 0.5*w_c[it, k]))

Cw, alpha1, _, _ = fit_power_law_all_times(z_constrained, w_c)
Cd, alpha2, _, _ = fit_power_law_all_times(z_constrained, delta)


Cw_avg, alpha1_avg, _, _ = fit_power_law(z_constrained, w_c_avg)
Cd_avg, alpha2_avg, _, _ = fit_power_law(z_constrained, delta_avg)
# defining plume width based on delta
width = delta*2
width_avg = delta_avg*2

# calculating eta, F(eta) at given depths
eta = r[None, :, None]/width[:, None, :]
F = w_rz/w_c[:, None, :]
ur_rz = reader.load_binning_var('ur')
ur_rz = ur_rz[:, :, above_mld]#_remove:surf_remove]
ur_rz = ur_rz[it_range, :, :]
F_transverse = ur_rz/w_c[:, None, :]
for it in range(len(it_range)):
    for k in range(len(z_constrained)):
        eta[it, :, k][r > width[it, k]] = np.nan
        F[it, :, k][r > width[it, k]] = np.nan
        F_transverse[it, :, k][r > width[it, k]] = np.nan

eta_avg = r[:, None]/delta_avg[None, :]
ur_rz_avg = np.mean(ur_rz, axis = 0)
F_avg = w_rz_avg/w_c_avg[None, :]
# ==========================================================
# PLOTTING
# ==========================================================

Cw_dir = os.path.join(outdir, 'w_c_fit')
os.makedirs(Cw_dir, exist_ok=True)

Cd_dir = os.path.join(outdir, 'delta_fit')
os.makedirs(Cd_dir, exist_ok=True)

range_scales = (0.8, 1.2)

vars = {}
vars['w_c'] = {'var': w_c, 'C': Cw, 'alpha': alpha1, 'ylabel': r'w$_c$', 'title': r'log-log fit of w$_c$ vs z', 'legend': '', 'outdir': Cw_dir, 'range': (np.min(w_c)*range_scales[0], np.max(w_c)*range_scales[1])}
vars['delta'] = {'var': delta, 'C': Cd, 'alpha': alpha2, 'ylabel': r'$\delta$', 'title': r'log-log fit of $\delta$ vs z', 'legend': '', 'outdir': Cd_dir, 'range': (np.min(delta)*range_scales[0], np.max(delta)*range_scales[1])}

for it in range(len(it_range)): 
    vars['w_c']['title'] =r'log-log fit of w$_c$ vs z, t = ' + str(time[it_range][it]/3600) + ' hours'
    vars['w_c']['legend'] = fr"w$_c$ = {Cw[it]:.3f} |z|$^{{{alpha1[it]:.3f}}}$"
    vars['delta']['title'] = r'log-log fit of $\delta$ vs z, t = ' + str(time[it_range][it]/3600) + ' hours'
    vars['delta']['legend'] = fr"$\delta$ = {Cd[it]:.3f} |z|$^{{{alpha2_avg:.3f}}}$" #alpha2[it]
    for var in vars.keys():
        plot_loglog_fit(vars[var]['outdir'], z_constrained, vars[var]['var'][it, :], vars[var]['C'][it], vars[var]['alpha'][it], vars[var]['ylabel'], vars[var]['legend'], vars[var]['title'], vars[var]['range'], it = it_range[it])

create_video(Cw_dir, outdir, opt, 'w_c_fit')
create_video(Cd_dir, outdir, opt, 'delta_fit')

plot_loglog_fit(outdir, z_constrained, w_c_avg, Cw_avg, alpha1_avg, r'w$_c$', fr"w$_c$ = {Cw_avg:.3f} |z|$^{{{alpha1_avg:.3f}}}$", r'log-log fit of w$_c$ vs z, time average', (np.min(w_c_avg)*range_scales[0], np.max(w_c_avg)*range_scales[1]), it = 'w_c_avg')
plot_loglog_fit(outdir, z_constrained, delta_avg, Cd_avg, alpha2_avg, r'$\delta$', fr"$\delta$ = {Cd_avg:.3f} |z|$^{{{alpha2_avg:.3f}}}$", r'log-log fit of $\delta$ vs z, time average', (np.min(delta_avg)*range_scales[0], np.max(delta_avg)*range_scales[1]), it = 'delta_avg')

size_in = (12, 6)

fig_folder = os.path.join(outdir, 'scaling analysis/')
os.makedirs(fig_folder, exist_ok=True)

cmap = plt.get_cmap('viridis', len(above_mld))

def _F(eta, alpha):
    return np.exp(-alpha * eta**2)

for it in range(len(it_range)):
    fig, ax = plt.subplots(1, 1, figsize=size_in)
    loc = np.argsort(eta[it, :, :].ravel())
    F_it_k = F[it, :, :].ravel()
    eta_it_k = eta[it, :, :].ravel()
    eta_it_k = eta_it_k[loc]
    F_it_k = F_it_k[loc]
    n_keep = np.where(~np.isnan(F_it_k))
    F_it_k = F_it_k[n_keep]
    eta_it_k = eta_it_k[n_keep]

    alpha = scipy.optimize.curve_fit(_F, eta_it_k, F_it_k)[0][0]

    ax.plot(eta_it_k, _F(eta_it_k, alpha), color='black',
            label=rf"exp(-{alpha:.2f}$\eta$)")

    for n in range(len(above_mld)):
        loc = np.argsort(eta[it, :, n].ravel())
        F_it_k = F[it, :, n].ravel()
        eta_it_k = eta[it, :, n].ravel()
        eta_it_k = eta_it_k[loc]
        F_it_k = F_it_k[loc]


        n_keep = np.where(~np.isnan(F_it_k))
        F_it_k = F_it_k[n_keep]
        eta_it_k = eta_it_k[n_keep]
        color = cmap(n)
        if n == 0 or n == len(above_mld)-1:
            ax.scatter(eta_it_k, F_it_k, marker='o', s=10, color=color, label = f'z = {z_constrained[n]}')
        else:
            ax.scatter(eta_it_k, F_it_k, marker='o', s=10, color=color)

    ax.set_xlabel(r"$\eta$")
    ax.set_xlim(0, 1.1)
    ax.set_ylabel(r"$F(\eta)$")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"t = {time[it_range[it]]/3600:.2f} hours")
    ax.legend(loc='upper right', fontsize='small')
    save_frame(fig, fig_folder, it, size_in)

create_video(fig_folder, outdir, opt, 'F(eta)')