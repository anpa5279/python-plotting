import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import imageio.v2 as imageio
import matplotlib.ticker as mticker
### -------------------------PROFILES------------------------- ###
## stratification profile
def stratification_profile(z, a0, dadz, mld):
    """Returns a linear stratification profile."""
    a = a0 * np.ones(len(z))
    a[z<mld] = a0 + dadz * (z[z<mld] - mld)
    return a

### -------------------------PLOTTING PREP FUNCTIONS------------------------- ###
## defining ranges for plotting
def plot_ranges(lz = 96, rho0 = 1026, T0 = 25, dTdz = 0.01, Sj = 0.0):
    ranges = {}
    list_pqr = ['u', 'v', 'w', 'b', 'T', 'S', 'Pdynamic', 'Pstatic', 'rho', 
                'b_flux', 
                'vel_rms', 'b_rms', 
                'b_avg', 'T_avg', 'vel_avg', 'lamb_avg',
                'vel_restress', 'vel_flux', 'Ri', 
                'u_fluc', 'v_fluc', 'w_fluc', 'b_fluc', 'vel_fluc', 'bw_fluc', 'Tw_fluc', 'rho_fluc', 'T_fluc', 'S_fluc',
                'lengthscale', 'gradb', 'alphas', 
                'Q', 'F', 'M', 'B']
    for i in range(0,len(list_pqr),1):
        ranges[list_pqr[i]] = list()
    ranges['u'] = [-0.002, 0.002]
    ranges['v'] = [-0.002, 0.002]
    ranges['w'] = [-0.002, 0.002]
    ranges['u_fluc'] = [-0.002, 0.002]
    ranges['v_fluc'] = [-0.002, 0.002]
    ranges['w_fluc'] = [-0.002, 0.002]
    ranges['b'] = [-1.5*10**(-3), 10**(-5)]
    ranges['T'] = [T0-(dTdz*lz)+0.2, T0 + 0.02]
    ranges['S'] = [0.0, Sj/2]
    ranges['vel'] = [-0.00035, 0.00035]
    ranges['vel_rms'] = [0, 0.004]
    ranges['vel_flux'] = [-1*10**(-2), 1*10**(-2)]
    ranges['restress'] = [-3*10**(-6), 3*10**(-6)]
    ranges['Ri'] = [0, 5*10**(-3)]
    ranges['Pdynamic'] = [-0.005, 0.005]
    ranges['Pstatic'] = [-0.05, 0.05]
    ranges['b_avg'] = [-1.5*10**(-3), 1.0*10**(-5)]
    ranges['T_avg'] = [T0-(dTdz*lz)+0.2, T0 + 0.05]
    ranges['lamb_avg'] = [-4*10**(-6), 4*10**(-6)]
    ranges['b_rms'] = [0, 2*10**(-5)]
    ranges['bw_fluc'] = [-1*10**(-8), 1*10**(-8)]
    ranges['b_flux'] = [-1*10**(-8), 1*10**(-8)]
    ranges['b_fluc'] = [-1*10**(-4), 1*10**(-4)]
    ranges['lengthscale'] = [0, 0.4]
    ranges['rho'] = [rho0-0.02, rho0+0.15]
    ranges['rho_fluc'] = [-0.02, 0.02]
    ranges['brms_sign'] = [0, 5*10**(-5)]
    ranges['bflux_rms'] = [-1*10**(-5), 1*10**(-5)]
    ranges['z_sign'] = [-1*lz, 0]
    ranges['alphas'] = [0, 0.2]
    ranges['gradb'] = [-3*10**(-5), 3*10**(-5)]
    ranges['T_fluc'] = [-5*10**(-1), 5*10**(-1)]
    ranges['S_fluc'] = [-1*10**(-1), 1*10**(-1)]
    for key in ranges:
        ranges[key] = np.array(ranges[key])
    return ranges

### -------------------------PLOTTING FUNCTIONS------------------------- ###
## turb statistics
def turb_stats(time, it, ranges, fig_folder, lx, nx, z, zf, mld, u_avg, v_avg, w_avg, u_rms, v_rms, w_rms, uv_fluc, uw_fluc, vw_fluc, bu_fluc_avg, bv_fluc_avg, bw_fluc_avg, b_rms, rho, plume_info = []):
    outdir = os.path.join(fig_folder, 'turb stats/')
    os.makedirs(outdir, exist_ok=True)
    td = time[it] / 3600 / 24 # convert time to days

    fig = plt.figure()
    fig.set_figheight(8)
    fig.set_figwidth(11)
    fig.suptitle(f'{td:.2f} days', fontsize=12) 

    ax1 = plt.subplot2grid(shape=(2, 3), loc=(0, 0)) # velocity profiles
    ax2 = plt.subplot2grid(shape=(2, 3), loc=(0, 1)) # velocity rms w/ vertical velocity of plume
    ax3 = plt.subplot2grid(shape=(2, 3), loc=(0, 2)) # Reynolds stress
    ax4 = plt.subplot2grid(shape=(2, 3), loc=(1, 0)) # buoyancy profiles of interest
    ax5 = plt.subplot2grid(shape=(2, 3), loc=(1, 1)) # buoyancy flux fluctuations
    ax6 = plt.subplot2grid(shape=(2, 3), loc=(1, 2)) # buoyancy rms 

    fig.subplots_adjust(hspace=0.05)

    if len(plume_info)> 0:
        plume_depth_intrusion = plume_info[0]
        plume_depth_neutral = plume_info[1]
        rho_tracer = plume_info[2]
        ax1.plot([-1*10**6, 1*10**6], plume_depth_intrusion*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')
        ax1.plot([-1*10**6, 1*10**6], plume_depth_neutral*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')
        ax2.plot([-1*10**6, 1*10**6], plume_depth_intrusion*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')
        ax2.plot([-1*10**6, 1*10**6], plume_depth_neutral*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')
        ax3.plot([-1*10**6, 1*10**6], plume_depth_intrusion*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')
        ax3.plot([-1*10**6, 1*10**6], plume_depth_neutral*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')
        ax4.plot([-1*10**6, 1*10**6], plume_depth_intrusion*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')
        ax4.plot([-1*10**6, 1*10**6], plume_depth_neutral*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')
        ax5.plot([-1*10**6, 1*10**6], plume_depth_intrusion*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')
        ax5.plot([-1*10**6, 1*10**6], plume_depth_neutral*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')
        ax6.plot([-1*10**6, 1*10**6], plume_depth_intrusion*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')
        ax6.plot([-1*10**6, 1*10**6], plume_depth_neutral*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')

    # velocity profiles
    ax1.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')#, label = "MLD")
    ax1.plot(u_avg, z, label=r"$\langle$u$\rangle_{xy}$", color='green')
    ax1.plot(v_avg, z, label=r"$\langle$v$\rangle_{xy}$", color='red')
    ax1.plot(w_avg, zf, label=r"$\langle$w$\rangle_{xy}$", color='blue')
    ax1.set_xlabel("[m/s]")
    ax1.set_ylabel("Depth [m]")
    ax1.set_title('Velocity Profiles')
    ax1.set_ylim(-lx[2], 0)
    ax1.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True)
    ax1.set_xlim(ranges['vel'])
    ax1.legend(loc='lower right', handlelength=0.9)

    # velocity rms
    ax2.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')
    ax2.plot(u_rms, z, label=r"$\langle$u$_{rms}$$\rangle_{xy}$", color='green')
    ax2.plot(v_rms, z, label=r"$\langle$v$_{rms}$$\rangle_{xy}$", color='red')
    ax2.plot(w_rms, zf, label=r"$\langle$w$_{rms}$$\rangle_{xy}$", color='blue')
    ax2.set_xlabel("[m/s]")
    ax2.set_title("Root Mean Square Velocities")
    ax2.set_ylim(-lx[2], 0)
    ax2.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True)
    ax2.set_xlim(ranges['vel_rms'])
    ax2.legend(loc='lower right', handlelength=0.9)

    # reynolds stresses 
    ax3.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')
    ax3.plot(uv_fluc, z, label=r"$\langle$uv$\rangle_{xy}$", color='green')
    ax3.plot(uw_fluc, z, label=r"$\langle$uw$\rangle_{xy}$", color='red')
    ax3.plot(vw_fluc, z, label=r"$\langle$vw$\rangle_{xy}$", color='blue')
    ax3.set_xlabel(r"[m$^2$/s$^2$]")
    ax3.set_title("Reynolds Stresses")
    ax3.set_ylim(-lx[2], 0)
    ax3.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True)
    ax3.set_xlim(ranges['restress'])
    ax3.legend(loc='lower right', handlelength=0.9)

    # density profiles
    ax4.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')
    ax4.plot(rho, z, label = r"$\rho_{\text{avg}}$")
    if len(plume_info)> 0:
        ax4.plot(rho_tracer, z, color = 'blue', label = r"$\rho_{\text{centerline}}$")
        ax4.set_title("Density Profiles")
        ax4.legend(loc='upper right', handlelength=0.9)
    else:
        ax4.set_title("Density Profile")
    ax4.set_ylabel("Depth [m]")
    ax4.set_xlabel(r"[kg/m$^3$]")
    ax4.set_ylim(-lx[2], 0)
    ax4.set_xlim(ranges['rho'])
    ax4.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True)

    # perturbed buoyancy flux 
    ax5.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')
    ax5.set_xlim(ranges['bw_fluc'])
    ax5.plot(bu_fluc_avg, z, color='green', label = r"b'u'")
    ax5.plot(bv_fluc_avg, z, color='blue', label = r"b'v'")
    ax5.plot(bw_fluc_avg, z, color='red', label = r"b'w'")
    ax5.legend(loc='lower right', handlelength=0.9)
    ax5.set_xlabel(r"[m$^{2}$/s$^{3}$]")
    ax5.set_title("Buoyancy Flux Fluctuations")
    ax5.set_ylim(-lx[2], 0)
    ax5.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True) 

    # buoyancy brms 
    ax6.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')
    ax6.plot(b_rms, z, color = 'black')
    ax6.set_xlabel(r"[m/s$^2$]")
    ax6.set_title("Root Mean Square Buoyancy")
    ax6.set_ylim(-lx[2], 0)
    ax6.set_xlim(ranges['b_rms'])
    ax6.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True)

    # --- Save Frame ---
    frame_path = os.path.join(outdir, f"turb_stats_{it:04d}.png")
    plt.tight_layout()
    plt.savefig(frame_path)
    plt.close(fig)
    print(f"Time step {it + 1} captured: {frame_path}")
    plt.close(fig)

    return outdir # return the directory where frames are saved for video creation
## 3D fields 
def plot_3d_fields(time, it, ranges, fig_folder, lx, X, Y, Z, Xf, Yf, Zf, u, v, w, T = np.array([]), S = np.array([]), b = np.array([])):

    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))

    outdir = os.path.join(fig_folder, 'oc 3D video frames/')
    os.makedirs(outdir, exist_ok=True)
    td = time[it] / 3600 / 24

    levels = 500
    if S.size != 0:
        fig, ax = plt.subplots(1, 5, figsize=(15, 4), subplot_kw=dict(projection='3d'))
        ax1 = ax[0]
        ax2 = ax[1]
        ax3 = ax[2]
        ax4 = ax[3]
        ax5 = ax[4]
    else:
        fig, ax = plt.subplots(1, 4, figsize=(12, 4), subplot_kw=dict(projection='3d'))
        ax1 = ax[0]
        ax2 = ax[1]
        ax3 = ax[2]
        ax4 = ax[3]

    fig.tight_layout()
    fig.suptitle(f'{td:.2f} days', fontsize=12)
    # --- 3D Plots ---
    # u velocity
    norm = mcolors.Normalize(vmin=ranges['u'][0], vmax=ranges['u'][-1])
    mappable = cm.ScalarMappable(norm=norm, cmap='RdBu_r')
    ax1.set(xlim=[np.min(X[:, :, idx_loc]), np.max(X[:, :, idx_loc])], ylim=[np.min(Y[:, :, idx_loc]), np.max(Y[:, :, idx_loc])], zlim=[-lx[2], 0])
    ax1.contourf(X[:, :, -1], Y[:, :, -1], u[:, :, -1], levels, zdir='z', offset=0, cmap='RdBu_r', norm=norm)
    ax1.contourf(X[-1, :, :], u[-1, :, :], Z[-1, :, :], levels, zdir='y', offset=0, cmap='RdBu_r', norm=norm)
    ax1.contourf(u[:, 0, :], Y[:, 0, :], Z[:, 0, :], levels, zdir='x', offset=lx[0], cmap='RdBu_r', norm=norm)
    ax1.set_title('u')
    ax1.set(xlabel="y", ylabel="x", zlabel="z")
    ax1.set_aspect('equal')
    cbar = fig.colorbar(mappable, ax=ax1, label='[m/s]', location='bottom', orientation='horizontal', shrink=0.75)
    cbar.formatter.set_powerlimits((-3, 2))
    cbar.update_ticks()

    # v velocity
    norm = mcolors.Normalize(vmin=ranges['v'][0], vmax=ranges['v'][-1])
    mappable = cm.ScalarMappable(norm=norm, cmap='RdBu_r')
    ax2.set(xlim=[np.min(X[:, :, idx_loc]), np.max(X[:, :, idx_loc])], ylim=[np.min(Y[:, :, idx_loc]), np.max(Y[:, :, idx_loc])], zlim=[-lx[2], 0])
    ax2.contourf(X[:, :, -1], Y[:, :, -1], v[:, :, -1], levels, zdir='z', offset=0, cmap='RdBu_r', norm=norm)
    ax2.contourf(X[-1, :, :], v[-1, :, :], Z[-1, :, :], levels, zdir='y', offset=0, cmap='RdBu_r', norm=norm)
    ax2.contourf(v[:, 0, :], Y[:, 0, :], Z[:, 0, :], levels, zdir='x', offset=lx[0], cmap='RdBu_r', norm=norm)
    ax2.set_title('v')
    ax2.set(xlabel="y", ylabel="x", zlabel="z")
    ax2.set_aspect('equal')
    cbar = fig.colorbar(mappable, ax=ax2, label='[m/s]', location='bottom', orientation='horizontal', shrink=0.75)
    cbar.formatter.set_powerlimits((-3, 2))
    cbar.update_ticks()

    # w velocity
    norm = mcolors.Normalize(vmin=ranges['w'][0], vmax=ranges['w'][-1])
    mappable = cm.ScalarMappable(norm=norm, cmap='RdBu_r')
    ax3.set(xlim=[np.min(X[:, :, idx_loc]), np.max(X[:, :, idx_loc])], ylim=[np.min(Y[:, :, idx_loc]), np.max(Y[:, :, idx_loc])], zlim=[-lx[2], 0])
    ax3.contourf(Xf[:, :, -1], Yf[:, :, -2], w[:, :, -2], levels, zdir='z', offset=0, cmap='RdBu_r', norm=norm)
    ax3.contourf(Xf[-1, :, :], w[-1, :, :], Zf[-1, :, :], levels, zdir='y', offset=0, cmap='RdBu_r', norm=norm)
    ax3.contourf(w[:, 0, :], Yf[:, 0, :], Zf[:, 0, :], levels, zdir='x', offset=lx[0], cmap='RdBu_r', norm=norm)
    ax3.set_title('w')
    ax3.set(xlabel="y", ylabel="x", zlabel="z")
    ax3.set_aspect('equal')
    cbar = fig.colorbar(mappable, ax=ax3, label='[m/s]', location='bottom', orientation='horizontal', shrink=0.75)
    cbar.formatter.set_powerlimits((-3, 2))
    cbar.update_ticks()

    if T.size != 0:
        # temperature field
        norm = mcolors.Normalize(vmin=ranges['T'][0], vmax=ranges['T'][-1])
        mappable = cm.ScalarMappable(norm=norm)
        ax4.set(xlim=[np.min(X[:, :, idx_loc]), np.max(X[:, :, idx_loc])], ylim=[np.min(Y[:, :, idx_loc]), np.max(Y[:, :, idx_loc])], zlim=[-lx[2], 0])
        ax4.contourf(X[:, :, -1], Y[:, :, -1], T[:, :, -1], 50, zdir='z', offset=0, norm=norm)
        ax4.contourf(X[-1, :, :], T[-1, :, :], Z[-1, :, :], levels, zdir='y', offset=0, norm=norm)
        ax4.contourf(T[:, 0, :], Y[:, 0, :], Z[:, 0, :], levels, zdir='x', offset=lx[0], norm=norm)
        ax4.set_title(r"Temperature")
        ax4.set(xlabel="y", ylabel="x", zlabel="z")
        ax4.set_aspect('equal')
        cbar = fig.colorbar(mappable, ax=ax4, label=r"$^\circ$C", location='bottom', orientation='horizontal', shrink=0.75)

    else:
        # buoyancy field
        norm = mcolors.Normalize(vmin=ranges['b'][0], vmax=ranges['b'][-1])
        mappable = cm.ScalarMappable(norm=norm)
        ax4.set(xlim=[np.min(X[:, :, idx_loc]), np.max(X[:, :, idx_loc])], ylim=[np.min(Y[:, :, idx_loc]), np.max(Y[:, :, idx_loc])], zlim=[-lx[2], 0])
        ax4.contourf(X[:, :, -1], Y[:, :, -1], b[:, :, -1], levels, zdir='z', offset=0, norm=norm)
        ax4.contourf(X[-1, :, :], b[-1, :, :], Z[-1, :, :], levels, zdir='y', offset=0, norm=norm)
        ax4.contourf(b[:, 0, :], Y[:, 0, :], Z[:, 0, :], levels, zdir='x', offset=lx[0], norm=norm)
        ax4.set_title(r"Buoyancy")
        ax4.set(xlabel="y", ylabel="x", zlabel="z")
        ax4.set_aspect('equal')
        cbar = fig.colorbar(mappable, ax=ax4, label=r"m/s$^{2}$", location='bottom', orientation='horizontal', shrink=0.75)
        cbar.formatter.set_powerlimits((-3, 2))
        cbar.update_ticks()
    
    if S.size != 0:
        norm = mcolors.Normalize(vmin=ranges['S'][0], vmax=ranges['S'][-1])
        mappable = cm.ScalarMappable(norm=norm)
        ax5.set(xlim=[np.min(X[:, :, idx_loc]), np.max(X[:, :, idx_loc])], ylim=[np.min(Y[:, :, idx_loc]), np.max(Y[:, :, idx_loc])], zlim=[-lx[2], 0])
        ax5.contourf(X[:, :, -1], Y[:, :, -1], S[:, :, -1], levels, zdir='z', offset=0, norm=norm)
        ax5.contourf(X[-1, :, :], S[-1, :, :], Z[-1, :, :], levels, zdir='y', offset=0, norm=norm)
        ax5.contourf(S[:, 0, :], Y[:, 0, :], Z[:, 0, :], levels, zdir='x', offset=lx[0], norm=norm)
        ax5.set_title(r"Tracer")
        ax5.set(xlabel="y", ylabel="x", zlabel="z")
        ax5.set_aspect('equal')
        cbar = fig.colorbar(mappable, ax=ax5, location='bottom', orientation='horizontal', shrink=0.75)
        cbar.formatter.set_powerlimits((-3, 2))
        cbar.update_ticks()

    # --- Save Frame ---
    frame_path = os.path.join(outdir, f"oc_frame_{it:04d}.png")
    plt.tight_layout()
    plt.savefig(frame_path)
    plt.close(fig)
    print(f"Time step {it + 1} captured: {frame_path}")
    return outdir # return the directory where frames are saved for video creation
## vertical plane slices 
def vert_plane_slices(time, it, ranges, fig_folder, lx, X, Y, z, u, v, w, rho, rho_perturbed, T = np.array([]), S = np.array([]), depths = np.array([]), yz=True):
    if yz: #yz plane
        ar = lx[1]/lx[2]
        plane = 'YZ plane'
        hor = Y
        hor_range = (np.min(Y), np.max(Y))
    else: #xz plane
        ar = lx[0]/lx[2]
        plane = 'XZ plane'
        hor = X
        hor_range = (np.min(X), np.max(X))
    levels = 500
    outdir = os.path.join(fig_folder, 'vertical plane slices/', plane)
    os.makedirs(outdir, exist_ok=True)
    td = time[it] / 3600 / 24
    ncols = 4
    nrows = 2
    hor_len = 12.0
    vert_len = hor_len * nrows / (ncols * ar) + 0.75 * nrows + 1.1
    fig, ax = plt.subplots(nrows, ncols, figsize=(hor_len, vert_len), sharey = True, sharex = True, constrained_layout=True)
    fig.suptitle(plane + f', {td:.2f} days', y = 0.99, fontsize=12)
    ax = ax.ravel()
    ax[3].remove() 
    """
    ax0 = ax[1, 0] # temperature
    ax1 = ax[1, 1] # tracer
    ax2 = ax[1, 2] # density
    ax3 = ax[1, 3] # perturbed density 
    ax4 = ax[0, 0] # u velocity
    ax5 = ax[0, 1] # v velocity
    ax6 = ax[0, 2] # w velocity
    ax[0, 3].remove() # remove
    """

    norm = mcolors.Normalize(vmin=ranges['T'][0], vmax=ranges['T'][-1])
    mappable = cm.ScalarMappable(norm=norm)
    ax[4].contourf(hor, z, T, levels, norm=norm)
    ax[4].set_xlabel("[m]")
    ax[4].set_ylabel("Depth [m]")
    ax[4].set_title("Temperature")
    ax[4].set_ylim(-lx[2], 0)
    ax[4].set_xlim(hor_range)
    ax[4].set_aspect('equal')
    cbar  = fig.colorbar(mappable, ax=ax[4], label=r"$^\circ$C", anchor = (0.5, -0.3), orientation='horizontal', shrink=0.75)
    
    norm = mcolors.Normalize(vmin=ranges['S'][0], vmax=ranges['S'][-1])
    mappable = cm.ScalarMappable(norm=norm)
    ax[5].contourf(hor, z, S, levels, norm=norm)
    ax[5].set_xlabel("[m]")
    ax[5].set_title("Tracer")
    ax[5].set_ylim(-lx[2], 0)
    ax[5].set_xlim(hor_range)
    ax[5].set_aspect('equal')
    cbar  = fig.colorbar(mappable, ax=ax[5], label=r"g/kg", anchor = (0.5, -0.3), orientation='horizontal', shrink=0.75)

    norm = mcolors.Normalize(vmin=(ranges['rho'][0]), vmax=(ranges['rho'][-1]))
    mappable = cm.ScalarMappable(norm=norm)
    ax[6].contourf(hor, z, rho, levels, norm=norm)
    ax[6].set_xlabel("[m]")
    ax[6].set_title("Density")
    ax[6].set_ylim(-lx[2], 0)
    ax[6].set_xlim(hor_range)
    ax[6].set_aspect('equal')
    cbar  = fig.colorbar(mappable, ax=ax[6], label=r"kg/m$^3$", anchor = (0.5, -0.3), orientation='horizontal', shrink=0.75)
    cbar.formatter.set_scientific(False)
    cbar.formatter.set_useOffset(False)
    cbar.set_ticks([(ranges['rho'][0]), (ranges['rho'][-1])])
    cbar.update_ticks()

    norm = mcolors.Normalize(vmin=ranges['rho_fluc'][0], vmax=ranges['rho_fluc'][-1])
    mappable = cm.ScalarMappable(norm=norm, cmap='RdBu_r')
    ax[7].contourf(hor, z, rho_perturbed, levels, norm=norm, cmap='RdBu_r')
    ax[7].set_xlabel("[m]")
    ax[7].set_title("Perturbed Density")
    ax[7].set_ylim(-lx[2], 0)
    ax[7].set_xlim(hor_range)
    ax[7].set_aspect('equal')
    cbar  = fig.colorbar(mappable, ax=ax[7], label=r"kg/m$^3$", anchor = (0.5, -0.3), orientation='horizontal', shrink=0.75)

    norm = mcolors.Normalize(vmin=ranges['u'][0], vmax=ranges['u'][-1])
    mappable = cm.ScalarMappable(norm=norm, cmap = 'RdBu_r')
    ax[0].contourf(hor, z, u, levels, norm=norm, cmap='RdBu_r')
    ax[0].set_ylabel("Depth [m]")
    ax[0].set_title("u")
    ax[0].set_ylim(-lx[2], 0)
    ax[0].set_xlim(hor_range)
    ax[0].set_aspect('equal')
    cbar  = fig.colorbar(mappable, ax=ax[0], label=r"m/s", anchor = (0.5, -0.05), orientation='horizontal', shrink=0.75)
    cbar.formatter.set_powerlimits((-3, 2))
    cbar.update_ticks()

    norm = mcolors.Normalize(vmin=ranges['v'][0], vmax=ranges['v'][-1])
    mappable = cm.ScalarMappable(norm=norm, cmap = 'RdBu_r')
    ax[1].contourf(hor, z, v, levels, norm=norm, cmap='RdBu_r')
    ax[1].set_title("v")
    ax[1].set_ylim(-lx[2], 0)
    ax[1].set_xlim(hor_range)
    ax[1].set_aspect('equal')
    cbar  = fig.colorbar(mappable, ax=ax[1], label=r"m/s", anchor = (0.5, -0.05), orientation='horizontal', shrink=0.75)
    cbar.formatter.set_powerlimits((-3, 2))
    cbar.update_ticks()

    norm = mcolors.Normalize(vmin=ranges['w'][0], vmax=ranges['w'][-1])
    mappable = cm.ScalarMappable(norm=norm, cmap = 'RdBu_r')
    ax[2].contourf(hor, z, w, levels, norm=norm, cmap='RdBu_r')
    ax[2].set_title("w")
    ax[2].set_ylim(-lx[2], 0)
    ax[2].set_xlim(hor_range)
    ax[2].set_aspect('equal')
    cbar  = fig.colorbar(mappable, ax=ax[2], label=r"m/s", anchor = (0.5, -0.05), orientation='horizontal', shrink=0.75)
    cbar.formatter.set_powerlimits((-3, 2))
    cbar.update_ticks()
    
    if depths.size > 0:
        for depth in depths:
            ax[0].plot(hor, depth*np.ones_like(hor), linestyle='--', linewidth = 0.2, color = 'black')
            ax[1].plot(hor, depth*np.ones_like(hor), linestyle='--', linewidth = 0.2, color = 'black')
            ax[2].plot(hor, depth*np.ones_like(hor), linestyle='--', linewidth = 0.2, color = 'black')
            ax[4].plot(hor, depth*np.ones_like(hor), linestyle='--', linewidth = 0.2, color = 'black')
            ax[5].plot(hor, depth*np.ones_like(hor), linestyle='--', linewidth = 0.2, color = 'black')
            ax[6].plot(hor, depth*np.ones_like(hor), linestyle='--', linewidth = 0.2, color = 'black')
            ax[7].plot(hor, depth*np.ones_like(hor), linestyle='--', linewidth = 0.2, color = 'black')
    # --- Save Frame ---
    frame_path = os.path.join(outdir, f"oc_plane_slices_{it:04d}.png")
    plt.savefig(frame_path)
    plt.close(fig)
    print(f"Time step {it + 1} captured: {frame_path}")
    return outdir # return the directory where frames are saved for video creation
## surface plane slices 
def xy_plane_slices(time, it, ranges, fig_folder, X, Y, u, v, w, Pdynamic, rho, rho_perturbed, plane, T = np.array([]), S = np.array([])):
    outdir = os.path.join(fig_folder, 'horizontal plane slices/', 'XY plane slice at ' + plane)
    os.makedirs(outdir, exist_ok=True)
    td = time[it] / 3600 / 24

    fig, ax = plt.subplots(2, 4, figsize=(12, 8.5))
    fig.tight_layout()
    fig.suptitle(f'{td:.2f} days', fontsize=12)

    levels = 500

    ax1 = ax[0, 0]
    ax2 = ax[0, 1]
    ax3 = ax[0, 2]
    ax4 = ax[0, 3]
    ax5 = ax[1, 0]
    ax6 = ax[1, 1]
    ax7 = ax[1, 2]
    ax8 = ax[1, 3]

    norm = mcolors.Normalize(vmin=(ranges['rho_fluc'][0]), vmax=(ranges['rho_fluc'][-1]))
    mappable = cm.ScalarMappable(norm=norm, cmap='RdBu_r')
    ax1.contourf(X, Y, rho_perturbed, levels, norm=norm, cmap='RdBu_r')
    #ax1.set_xlabel("[m]")
    ax1.set_ylabel("[m]")
    ax1.set_title("Perturbed Density")
    ax1.set_ylim(np.min(Y), np.max(Y))
    ax1.set_xlim(np.min(X), np.max(X))
    ax1.set_aspect('equal')
    cbar = fig.colorbar(mappable, ax=ax1, label=r"kg/m$^3$", orientation='horizontal', shrink=0.75)
    cbar.formatter.set_powerlimits((-2, 2))
    cbar.update_ticks()
        
    norm = mcolors.Normalize(vmin=ranges['rho'][0], vmax=ranges['rho'][-1])
    mappable = cm.ScalarMappable(norm=norm)
    ax2.contourf(X, Y, rho, levels, norm=norm)
    #ax2.set_xlabel("[m]")
    ax2.set_title("Density")
    ax2.set_ylim(np.min(Y), np.max(Y))
    ax2.set_xlim(np.min(X), np.max(X))
    ax2.set_aspect('equal')
    cbar  = fig.colorbar(mappable, ax=ax2, label=r"kg/m$^3$", orientation='horizontal', shrink=0.75)
    cbar.formatter.set_scientific(False)
    cbar.formatter.set_useOffset(False)
    cbar.set_ticks([(ranges['rho'][0]), (ranges['rho'][-1])])
    cbar.update_ticks()
    
    norm = mcolors.Normalize(vmin=ranges['S'][0], vmax=ranges['S'][-1])
    mappable = cm.ScalarMappable(norm=norm)
    ax3.contourf(X, Y, S, levels, norm=norm)
    #ax3.set_xlabel("[m]")
    ax3.set_title("Tracer")
    ax3.set_ylim(np.min(Y), np.max(Y))
    ax3.set_xlim(np.min(X), np.max(X))
    ax3.set_aspect('equal')
    cbar  = fig.colorbar(mappable, ax=ax3, orientation='horizontal', shrink=0.75)
    
    T_slice = T
    if (np.max(T_slice) - np.min(T_slice))<=1e-6:
        levelsT = 10
    else:
        levelsT = levels
    norm = mcolors.Normalize(vmin=ranges['T'][0], vmax=ranges['T'][-1])
    mappable = cm.ScalarMappable(norm=norm)
    ax4.contourf(X, Y, T_slice, levelsT, norm=norm)
    #ax4.set_xlabel("[m]")
    ax4.set_title("Temperature")
    ax4.set_ylim(np.min(Y), np.max(Y))
    ax4.set_xlim(np.min(X), np.max(X))
    ax4.set_aspect('equal')
    cbar  = fig.colorbar(mappable, ax=ax4, label=r"$^\circ$C", orientation='horizontal', shrink=0.75)
    
    norm = mcolors.Normalize(vmin=ranges['Pdynamic'][0], vmax=ranges['Pdynamic'][-1])
    mappable = cm.ScalarMappable(norm=norm, cmap='RdBu_r')
    ax5.contourf(X, Y, Pdynamic, levels, norm=norm, cmap='RdBu_r')
    ax5.set_xlabel("[m]")
    ax5.set_ylabel("[m]")
    ax5.set_title("Hydrodynamic Pressure")
    ax5.set_ylim(np.min(Y), np.max(Y))
    ax5.set_xlim(np.min(X), np.max(X))
    ax5.set_aspect('equal')
    cbar  = fig.colorbar(mappable, ax=ax5, label=r"m/s$^{2}$", orientation='horizontal', shrink=0.75)
    cbar.formatter.set_powerlimits((-3, 2))
    cbar.update_ticks()

    norm = mcolors.Normalize(vmin=ranges['u'][0], vmax=ranges['u'][-1])
    mappable = cm.ScalarMappable(norm=norm, cmap = 'RdBu_r')
    ax6.contourf(X, Y, u, levels, norm=norm, cmap='RdBu_r')
    ax6.set_xlabel("[m]")
    ax6.set_title("u")
    ax6.set_ylim(np.min(Y), np.max(Y))
    ax6.set_xlim(np.min(X), np.max(X))
    ax6.set_aspect('equal')
    cbar  = fig.colorbar(mappable, ax=ax6, label=r"m/s", orientation='horizontal', shrink=0.75)
    cbar.formatter.set_powerlimits((-3, 2))
    cbar.update_ticks()

    norm = mcolors.Normalize(vmin=ranges['v'][0], vmax=ranges['v'][-1])
    mappable = cm.ScalarMappable(norm=norm, cmap = 'RdBu_r')
    ax7.contourf(X, Y, v, levels, norm=norm, cmap='RdBu_r')
    ax7.set_xlabel("[m]")
    ax7.set_title("v")
    ax7.set_ylim(np.min(Y), np.max(Y))
    ax7.set_xlim(np.min(X), np.max(X))
    ax7.set_aspect('equal')
    cbar  = fig.colorbar(mappable, ax=ax7, label=r"m/s", orientation='horizontal', shrink=0.75)
    cbar.formatter.set_powerlimits((-3, 2))
    cbar.update_ticks()

    norm = mcolors.Normalize(vmin=ranges['w'][0], vmax=ranges['w'][-1])
    mappable = cm.ScalarMappable(norm=norm, cmap = 'RdBu_r')
    ax8.contourf(X, Y, w, levels, norm=norm, cmap='RdBu_r')
    ax8.set_xlabel("[m]")
    #ax8.set_ylabel("[m]")
    ax8.set_title("w")
    ax8.set_ylim(np.min(Y), np.max(Y))
    ax8.set_xlim(np.min(X), np.max(X))
    ax8.set_aspect('equal')
    cbar  = fig.colorbar(mappable, ax=ax8, label=r"m/s", orientation='horizontal', shrink=0.75)
    cbar.formatter.set_powerlimits((-3, 2))
    cbar.update_ticks()

    # --- Save Frame ---
    frame_path = os.path.join(outdir, f"oc_plane_slices_{it:04d}.png")
    plt.tight_layout()
    plt.savefig(frame_path)
    plt.close(fig)
    print(f"Time step {it + 1} captured: {frame_path}")
    return outdir # return the directory where frames are saved for video creation


### -------------------------SAVING FRAMES AND MAKING VIDEOS------------------------- ###
def create_video(outdir, fig_folder, name, plot_type):
    print("Creating video...")
    filenames = sorted([f for f in os.listdir(outdir) if f.endswith(".png")])
    vid_name = os.path.join(fig_folder, name + plot_type + '.mp4')
    with imageio.get_writer(vid_name, fps=10) as writer:
        for filename in filenames:
            image = imageio.imread(f"{outdir}/{filename}")
            writer.append_data(image)