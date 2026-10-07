import os
import numpy as np
import matplotlib.pyplot as plt
import scipy


from reader import OceananigansData
from interpolation import point, horizontal_line
from plotting_general import create_video, save_frame, comparison_plot_opt
from self_similarity_functions import fit_power_law_all_times, fit_power_law, plot_loglog_fit

# ==========================================================
# FLAGS
# ==========================================================
check_steady_state = True
all_depth_opts = False # if True, use the entire domain that is available. If False, it will use specified depths
coeff_calc = False
centerline_plot = False
plot_depth_opt = False
plot_nd = True

if not coeff_calc and plot_nd:
    coeff_calc = True # necessary to calculate delta and w_c for the scaling analysis
# ==========================================================
# MODEL INFORMATION
# ==========================================================
folder = '/Users/annapauls/Documents/Github repositories/3d_langmuir_gpu/localoutputs/scheme-tests/longer/WENO9/dx0.5'

opt = 'fft convolve w_rz'
if all_depth_opts:
    data_opt = 'full MLD'
else:
    data_opt = '1.8hrs to 4hrs specified depths'
outdir = os.path.join(folder, 'figures', 'scaling analysis - ' + data_opt + ' ' + opt)
reader = OceananigansData(folder, salinity = True, with_halos=True, Sval=0.1)

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
Sval = reader.Sval
B0 = -g * reader.beta * Sval * w0 # [m^2/s^3]

# ==========================================================
# ANALYSIS
# ==========================================================
# ignoring first few time steps and information below the MLD
it_start= int(1.8*3600/(reader.t[1] - reader.t[0]))
it_range = np.arange(it_start, reader.nt)
above_mld = np.where(reader.z>=-mld+1.0)[0]
z_constrained = reader.z[above_mld]

w_rz = -reader.load_binning_var('/filtered vertical velocity/'+opt) 
w_rz_full = -reader.load_binning_var('w') 
if (reader.nt-it_start) != len(w_rz[:, 0, 0]):
    w_rz = w_rz[(len(w_rz[:, 0, 0])-(reader.nt-it_start)):, :, :]

if all_depth_opts:
    w_c = w_rz[:, 0, :]
    z_opt = z_constrained
    w_c_full = w_rz_full[:, 0, :]
else: # now only contains the values at the specified depths
    z_opt = np.array([-50, -40, -30, -20, -10])
    r_opt = [dx_scale/2, 5, 10, 15, 20, 30]
    w_rz_loc = np.empty((len(it_range), len(r), len(z_opt)))
    w_c_full = np.empty((reader.nt, len(z_opt)))
    for k, z_loc in enumerate(z_opt):
        for it in range(len(it_range)):
            w_rz_loc[it, :, k] = horizontal_line(w_rz[it, :, :], z = z_constrained, z0 = z_loc)
    w_c = w_rz_loc[:, 0, :]
    w_rz = w_rz_loc 

if check_steady_state:
    w_rz_check = np.empty((reader.nt, len(r_opt), len(z_opt)))
    for it in range(reader.nt):
        for k, z_loc in enumerate(z_opt):
            for ij, r_loc in enumerate(r_opt):
                w_rz_check[it, ij, k] = horizontal_line(w_rz_full[it, :, :], hor = r, hor0 = r_loc, z = reader.z, z0 = z_loc)
    r_opt[0] = 0.0

w_rz_avg = np.mean(w_rz, axis=0)
w_c_avg = np.mean(w_c, axis=0)

# calculating delta
if coeff_calc:
    delta_avg = np.empty_like(w_c_avg)
    delta = np.empty_like(w_c)
    for k, z_loc in enumerate(z_opt):
        delta_avg[k] = np.max(point(w_rz_avg[:, k], r, f0 = 0.5*w_c_avg[k]))
        for it in range(len(it_range)):
            delta[it, k] = np.max(point(w_rz[it, :, k], r, f0 = 0.5*w_c[it, k]))

    Cw, alpha1, _, _ = fit_power_law_all_times(z_opt, w_c)
    Cd, alpha2, _, _ = fit_power_law_all_times(z_opt, delta)


    Cw_avg, alpha1_avg, _, _ = fit_power_law(z_opt, w_c_avg)
    Cd_avg, alpha2_avg, _, _ = fit_power_law(z_opt, delta_avg)
    # defining plume width based on delta
    width = delta*2
    width_avg = delta_avg*2

    # calculating eta, F(eta) at given depths
    eta = r[None, :, None]/width[:, None, :]
    F = w_rz/w_c[:, None, :]
    ur_rz = reader.load_binning_var('ur')
    if all_depth_opts:
        ur_rz = ur_rz[:, :, above_mld]#_remove:surf_remove]
        ur_rz = ur_rz[it_range, :, :]
    else:
        ur_rz_loc = np.empty((len(it_range), len(r), len(z_opt)))
        for k, z_loc in enumerate(z_opt):
            for it in range(len(it_range)):
                ur_rz_loc[it, :, k] = horizontal_line(ur_rz[it, :, :], z = reader.z[above_mld], z0 = z_loc)
        ur_rz = ur_rz_loc
    F_transverse = ur_rz/w_c[:, None, :]
    for it in range(len(it_range)):
        for k, z_loc in enumerate(z_opt):
            eta[it, :, k][r > width[it, k]] = np.nan
            F[it, :, k][r > width[it, k]] = np.nan
            F_transverse[it, :, k][r > width[it, k]] = np.nan

    eta_avg = np.nanmean(eta, axis = 0)
    ur_rz_avg = np.mean(ur_rz, axis = 0)
    F_avg = np.nanmean(F, axis = 0)
# ==========================================================
# PLOTTING
# ==========================================================
if coeff_calc:
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
            plot_loglog_fit(vars[var]['outdir'], z_opt, vars[var]['var'][it, :], vars[var]['C'][it], vars[var]['alpha'][it], vars[var]['ylabel'], vars[var]['legend'], vars[var]['title'], vars[var]['range'], it = it_range[it])

    create_video(Cw_dir, outdir, opt, 'w_c_fit')
    create_video(Cd_dir, outdir, opt, 'delta_fit')

    plot_loglog_fit(outdir, z_opt, w_c_avg, Cw_avg, alpha1_avg, r'w$_c$', fr"w$_c$ = {Cw_avg:.3f} |z|$^{{{alpha1_avg:.3f}}}$", r'log-log fit of w$_c$ vs z, time average', (np.min(w_c_avg)*range_scales[0], np.max(w_c_avg)*range_scales[1]), it = 'w_c_avg')
    plot_loglog_fit(outdir, z_opt, delta_avg, Cd_avg, alpha2_avg, r'$\delta$', fr"$\delta$ = {Cd_avg:.3f} |z|$^{{{alpha2_avg:.3f}}}$", r'log-log fit of $\delta$ vs z, time average', (np.min(delta_avg)*range_scales[0], np.max(delta_avg)*range_scales[1]), it = 'delta_avg')

size_in = (12, 6)
case_opt = comparison_plot_opt(len(z_opt)+1)
colors = colors[1:] # first color is black, which is used for the fit
if check_steady_state:
    fig_folder = os.path.join(outdir, 'steady state check')
    os.makedirs(fig_folder, exist_ok=True)

    nsteps = 11 # number of time steps to average over
    n_start = (nsteps-1)//2
    n_end = reader.nt - 1 - (nsteps-1)//2
    t_avg = reader.t[n_start:n_end+1]/3600 # in hours
    w_avg_t = np.empty((len(t_avg), len(r_opt), len(z_opt)))
    for it in range(len(t_avg)):
        for k, z_loc in enumerate(z_opt):
            for ij, r_loc in enumerate(r_opt):
                w_avg_t[it, ij, k] = np.mean(w_rz_check[it:it+nsteps, ij, k], axis=0)


    wmin = np.min(w_avg_t)
    wmax = np.max(w_avg_t)

    fig, axes = plt.subplots(len(r_opt), 1, figsize=(8, 10), sharex = True)
    axes = axes.ravel()

    for ij, r_loc in enumerate(r_opt):
        for k, z_loc in enumerate(z_opt):
            if ij ==0:
                axes[ij].plot(t_avg, w_avg_t[:, ij, k], marker = case_opt['marker'][k], color = case_opt['color'][k], label = f'z = {z_loc} m')
                axes[ij].legend(loc='upper right', fontsize = 20, ncols = len(z_opt))
            elif ij == len(r_opt)-1:
                axes[ij].plot(t_avg, w_avg_t[:, ij, k], marker = case_opt['marker'][k], color = case_opt['color'][k])
                axes[ij].set_xlabel(r"time [hrs]")
            else:
                axes[ij].plot(t_avg, w_avg_t[:, ij, k], marker = case_opt['marker'][k], color = case_opt['color'][k])
        axes[ij].set_ylabel(r"w [m/s]")

        #axes[ij].set_xlim(min(reader.t)/3600, max(reader.t)/3600)
        #axes[ij].set_ylim(wmin*0.9, wmax*1.1)
        axes[ij].set_title(f"r = {r_loc} m")
        axes[ij].ticklabel_format(axis='y', style='sci', scilimits=(-2,2), useMathText=True) 

    frame_path = os.path.join(outdir, f'steady state check nsteps={nsteps}.png')
    plt.savefig(frame_path, dpi = 200)
    plt.close(fig)

if not all_depth_opts:
    fig_folder = os.path.join(outdir, 'w vs r at specified depths/')
    os.makedirs(fig_folder, exist_ok=True)

    wmin = np.min(w_rz)
    wmax = np.max(w_rz)

    for it in range(len(it_range)):
        fig, ax = plt.subplots(1, 1, figsize=size_in)

        for k, z_loc in enumerate(z_opt):
            ax.plot(r, w_rz[it, :, k], marker = case_opt['marker'][k], color = case_opt['color'][k], label = f'z = {z_loc} m')

        ax.set_xlabel(r"r [m]")
        ax.set_xlim(0, max(r)*0.625)
        ax.set_ylabel(r"w [m/s]")
        ax.set_ylim(wmin, wmax)
        ax.set_title(f"t = {time[it_range[it]]/3600:.2f} hours")
        ax.legend(loc='upper right', fontsize = 20)
        save_frame(fig, fig_folder, it, size_in)

    create_video(fig_folder, outdir, opt, 'w vs r at specified depths')

    # temporal average plot
    fig, ax = plt.subplots(1, 1, figsize=size_in)

    for k, z_loc in enumerate(z_opt):
        ax.plot(r, w_rz_avg[:, k], marker = case_opt['marker'][k], color = case_opt['color'][k], label = f'z = {z_loc} m')

    ax.set_xlabel(r"r [m]")
    ax.set_xlim(0, max(r)*0.625)
    ax.set_ylabel(r"w [m/s]")
    ax.set_ylim(np.min(w_rz_avg), np.max(w_rz_avg))
    ax.set_title(f"Temporal average, t = {time[it_range[0]]/3600:.2f} to {time[it_range[-1]]/3600:.2f} hours")
    ax.legend(loc='upper right', fontsize = 20)

    frame_path = os.path.join(outdir, 'temporal average w vs r at specified depths.png')
    plt.savefig(frame_path, dpi = 200)
    plt.close(fig)

if centerline_plot:
    def _wc(z, Cw):
        return Cw*(-B0)**(1/3)*(z)**(-1/3)
    fig_folder = os.path.join(outdir, 'wc vs z')
    os.makedirs(fig_folder, exist_ok=True)

    wmin = np.min(w_c)
    wmax = np.max(w_c)

    for it in range(len(it_range)):
        fig, ax = plt.subplots(1, 1, figsize=size_in)
        Cw = scipy.optimize.curve_fit(_wc, -z_opt, w_c[it, :])[0][0]
        ax.plot(_wc(-z_constrained, Cw), z_constrained, color = 'black', label=rf"{Cw:.3f}|B$_0|^{{1/3}}$|z|$^{{-1/3}}$")

        for k, z_loc in enumerate(z_opt):
            ax.scatter(w_c[it, k], z_loc, marker = case_opt['marker'][k], s=10, color = case_opt['color'][k], label = f'z = {z_loc} m')
        ax.set_xlabel(r"w [m/s]")
        ax.set_xlim(0, 0.1)
        ax.set_ylabel(r"z [m]")
        ax.set_ylim(min(z_constrained), max(z_constrained))
        ax.set_title(f"t = {time[it_range[it]]/3600:.2f} hours")
        ax.legend(loc='upper right', fontsize = 20)
        save_frame(fig, fig_folder, it, size_in)

    create_video(fig_folder, outdir, opt, 'wc vs z')

    # temporal average plot
    fig, ax = plt.subplots(1, 1, figsize=size_in)
    Cw = scipy.optimize.curve_fit(_wc, -z_opt, w_c_avg)[0][0]

    ax.plot(_wc(-z_constrained, Cw), z_constrained, color = 'black', label=rf"{Cw:.3f}|B$_0|^{{1/3}}$|z|$^{{-1/3}}$")
    for k, z_loc in enumerate(z_opt):
        ax.scatter(w_c_avg[k], z_loc, marker = case_opt['marker'][k], s=10, color = case_opt['color'][k], label = f'z = {z_loc} m')

    ax.set_xlabel(r"w [m/s]")
    ax.set_xlim(0, 0.1)
    ax.set_ylabel(r"z [m]")
    ax.set_ylim(min(z_constrained), max(z_constrained))
    ax.set_title(f"Temporal average, t = {time[it_range[0]]/3600:.2f} to {time[it_range[-1]]/3600:.2f} hours")
    ax.legend(loc='upper right', fontsize = 20)

    frame_path = os.path.join(outdir, 'temporal average wc vs z.png')
    plt.savefig(frame_path, dpi = 200)
    plt.close(fig)

if plot_nd:
    def _F(eta, alpha):
        return np.exp(-alpha * eta**2)
    fig_folder = os.path.join(outdir, 'scaling analysis/')
    os.makedirs(fig_folder, exist_ok=True)

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

        for k, z_loc in enumerate(z_opt):
            loc = np.argsort(eta[it, :, k].ravel())
            F_it_k = F[it, :, k].ravel()
            eta_it_k = eta[it, :, k].ravel()
            eta_it_k = eta_it_k[loc]
            F_it_k = F_it_k[loc]

            n_keep = np.where(~np.isnan(F_it_k))
            F_it_k = F_it_k[n_keep]
            eta_it_k = eta_it_k[n_keep]
            if k == 0 or k == len(z_opt)-1 or not all_depth_opts:
                ax.plot(eta_it_k, F_it_k, marker = case_opt['marker'][k], color = case_opt['color'][k], label = f'z = {z_loc} m')
            else:
                ax.plot(eta_it_k, F_it_k, marker = case_opt['marker'][k], color = case_opt['color'][k])

        ax.plot(np.arange(0, 1, 0.01), _F(np.arange(0, 1, 0.01), alpha), color = 'black', linewidth=4.0,
                label=rf"exp(-{alpha:.2f}$\eta^2$)")

        ax.set_xlabel(r"$\eta$")
        ax.set_xlim(0, 1.1)
        ax.set_ylabel(r"$F(\eta)$")
        ax.set_ylim(0, 1.1)
        ax.set_title(f"t = {time[it_range[it]]/3600:.2f} hours")
        ax.legend(loc='upper right', fontsize = 20)
        save_frame(fig, fig_folder, it, size_in)

    create_video(fig_folder, outdir, opt, 'F(eta)')

    # temporal average of F(eta)
    fig, ax = plt.subplots(1, 1, figsize=size_in)
    loc = np.argsort(eta_avg.ravel())
    eta_avg_a = eta_avg.ravel()
    F_avg_a = F_avg.ravel()
    eta_avg_a = eta_avg_a[loc]
    F_avg_a = F_avg_a[loc]
    n_keep = np.where(~np.isnan(F_avg_a))
    F_avg_a = F_avg_a[n_keep]
    eta_avg_a = eta_avg_a[n_keep]

    alpha = scipy.optimize.curve_fit(_F, eta_avg_a, F_avg_a)[0][0]

    for k, z_loc in enumerate(z_opt):
        loc = np.argsort(eta_avg[:, k].ravel())
        F_avg_k = F_avg[:, k].ravel()
        eta_avg_k = eta_avg[:, k].ravel()
        eta_avg_k = eta_avg_k[loc]
        F_avg_k = F_avg_k[loc]

        n_keep = np.where(~np.isnan(F_avg_k))
        F_avg_k = F_avg_k[n_keep]
        eta_avg_k = eta_avg_k[n_keep]
        if k == 0 or k == len(z_opt)-1 or not all_depth_opts:
            ax.plot(eta_avg_k, F_avg_k, marker = case_opt['marker'][k], color = case_opt['color'][k], label = f'z = {z_loc} m')
        else:
            ax.plot(eta_avg_k, F_avg_k, marker = case_opt['marker'][k], color = case_opt['color'][k])

    ax.plot(np.arange(0, 1, 0.01), _F(np.arange(0, 1, 0.01), alpha), color = 'black', linewidth=4.0,
            label=rf"exp(-{alpha:.2f}$\eta^2$)")

    ax.set_xlabel(r"$\eta$")
    ax.set_xlim(0, 1.1)
    ax.set_ylabel(r"$F(\eta)$")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Temporal average, t = {time[it_range[0]]/3600:.2f} to {time[it_range[-1]]/3600:.2f} hours")
    ax.legend(loc='upper right', fontsize = 20)

    frame_path = os.path.join(outdir, 'temporal average F(eta).png')
    plt.savefig(frame_path, dpi = 200)
    plt.close(fig)