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
                'vel_restress', 'vel_flux', 'richardson', 
                'u_fluc', 'v_fluc', 'w_fluc', 'b_fluc', 'vel_fluc', 'bw_fluc', 'Tw_fluc', 'rho_fluc', 'T_fluc', 'S_fluc',
                'lengthscale', 'gradb', 'alphas']
    for i in range(0,len(list_pqr),1):
        ranges[list_pqr[i]] = list()
    ranges['u'] = [-0.002, 0.002]
    ranges['v'] = [-0.002, 0.002]
    ranges['w'] = [-0.002, 0.002]
    ranges['u_fluc'] = [-0.002, 0.002]
    ranges['v_fluc'] = [-0.002, 0.002]
    ranges['w_fluc'] = [-0.002, 0.002]
    ranges['b'] = [-1.5*10**(-3), 10**(-5)]
    ranges['T'] = [T0-(dTdz*lz)+0.2, T0 + 0.05]
    ranges['S'] = [0.0, Sj/2]
    ranges['vel'] = [-0.00035, 0.00035]
    ranges['vel_rms'] = [0, 0.004]
    ranges['vel_flux'] = [-1*10**(-2), 1*10**(-2)]
    ranges['restress'] = [-3*10**(-6), 3*10**(-6)]
    ranges['richardson'] = [0, 5*10**(-3)]
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
    ax1.set(xlim=[-lx[0]/2, lx[0]/2], ylim=[-lx[1]/2, lx[1]/2], zlim=[-lx[2], 0])
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
    ax2.set(xlim=[-lx[0]/2, lx[0]/2], ylim=[-lx[1]/2, lx[1]/2], zlim=[-lx[2], 0])
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
    ax3.set(xlim=[-lx[0]/2, lx[0]/2], ylim=[-lx[1]/2, lx[1]/2], zlim=[-lx[2], 0])
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
        ax4.set(xlim=[-lx[0]/2, lx[0]/2], ylim=[-lx[1]/2, lx[1]/2], zlim=[-lx[2], 0])
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
        ax4.set(xlim=[-lx[0]/2, lx[0]/2], ylim=[-lx[1]/2, lx[1]/2], zlim=[-lx[2], 0])
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
        ax5.set(xlim=[-lx[0]/2, lx[0]/2], ylim=[-lx[1]/2, lx[1]/2], zlim=[-lx[2], 0])
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
def vert_plane_slices(time, it, ranges, fig_folder, lx, nx, X, Xf, Y, Yf, Z, Zf, u, v, w, u_fluc, v_fluc, w_fluc, b_fluc, Pstatic, Pdynamic, rho, rho_perturbed, b, T = np.array([]), S = np.array([]), yz=True):
    smaller = 0.1
    if yz: #yz plane
        plane = 'YZ plane'
        hor = Y[:, int(nx[1]/2), :]
        horf = Yf[:, int(nx[1]/2), :]
        z = Z[:, int(nx[1]/2), :]
        x_data = int(nx[0]/2)
        y_data = np.arange(0, nx[1])
        hor_range = (-lx[1]/2, lx[1]/2)
        urange = [ranges['u'][0]*smaller, ranges['u'][-1]*smaller]
        vrange = ranges['v']
        uflucrange = [ranges['u_fluc'][0]*smaller, ranges['u_fluc'][-1]*smaller]
        vflucrange = ranges['v_fluc']
    else: #xz plane
        plane = 'XZ plane'
        hor = X[int(nx[0]/2), :, :]
        horf = Xf[int(nx[0]/2), :, :]
        z = Z[int(nx[0]/2), :, :]
        x_data = np.arange(0, nx[0])
        y_data = int(nx[1]/2)
        hor_range = (-lx[0]/2, lx[0]/2)
        urange = ranges['u']
        vrange = [ranges['v'][0]*smaller, ranges['v'][-1]*smaller]
        uflucrange = ranges['u_fluc']
        vflucrange = [ranges['v_fluc'][0]*smaller, ranges['v_fluc'][-1]*smaller]

    outdir = os.path.join(fig_folder, 'vertical plane slices/', plane)
    os.makedirs(outdir, exist_ok=True)
    td = time[it] / 3600 / 24

    fig, ax = plt.subplots(3, 4, figsize=(12, 7.5))
    fig.tight_layout()
    fig.suptitle(plane + f', {td:.2f} days', y = 0.9, fontsize=12)

    levels = 500
    ax5 = ax[0, 0] # typically ax1
    ax1 = ax[0, 1] # typically ax2
    ax2 = ax[0, 2] # typically ax3
    ax4 = ax[0, 3]
    ax9 = ax[1, 0] # typically ax5 
    ax6 = ax[1, 1]
    ax7 = ax[1, 2]
    ax8 = ax[1, 3]
    ax3 = ax[2, 0] # typically ax9
    ax10 = ax[2, 1]
    ax11 = ax[2, 2]
    ax12 = ax[2, 3]

    norm = mcolors.Normalize(vmin=(ranges['rho'][0]), vmax=(ranges['rho'][-1]))
    mappable = cm.ScalarMappable(norm=norm)
    ax1.contourf(hor, z, rho[x_data, y_data, :], levels, norm=norm)
    ax1.set_xlabel("[m]")
    #ax1.set_ylabel("Depth [m]")
    ax1.set_title("Density")
    ax1.set_ylim(-lx[2], 0)
    ax1.set_xlim(hor_range)
    ax1.set_aspect('equal')
    cbar  = fig.colorbar(mappable, ax=ax1, label=r"kg/m$^3$", anchor = (0.5, -0.1), orientation='horizontal', shrink=0.75)
    cbar.formatter.set_scientific(False)
    cbar.formatter.set_useOffset(False)
    cbar.set_ticks([(ranges['rho'][0]), (ranges['rho'][-1])])
    cbar.update_ticks()

    norm = mcolors.Normalize(vmin=ranges['rho_fluc'][0], vmax=ranges['rho_fluc'][-1])#SymLogNorm(linthresh=1e-3, vmin=ranges['rho_fluc'][0], vmax=ranges['rho_fluc'][-1], base=10)
    mappable = cm.ScalarMappable(norm=norm, cmap='RdBu_r')
    ax2.contourf(hor, z, rho_perturbed[x_data, y_data, :], levels, norm=norm, cmap='RdBu_r')
    ax2.set_xlabel("[m]")
    #ax2.set_ylabel("Depth [m]")
    ax2.set_title("Perturbed Density")
    ax2.set_ylim(-lx[2], 0)
    ax2.set_xlim(hor_range)
    ax2.set_aspect('equal')
    cbar  = fig.colorbar(mappable, ax=ax2, label=r"kg/m$^3$", anchor = (0.5, -0.1), orientation='horizontal', shrink=0.75)

    norm = mcolors.Normalize(vmin=ranges['b'][0], vmax=ranges['b'][-1])
    mappable = cm.ScalarMappable(norm=norm, cmap='RdBu_r')
    ax3.contourf(hor, z, b[x_data, y_data, :], levels, norm=norm, cmap='RdBu_r')
    ax3.set_xlabel("[m]")
    ax3.set_title("Buoyancy")
    ax3.set_ylim(-lx[2], 0)
    ax3.set_xlim(hor_range)
    ax3.set_aspect('equal')
    cbar  = fig.colorbar(mappable, ax=ax3, label=r"m/s$^2$", anchor = (0.5, -0.1), orientation='horizontal', shrink=0.75)
    cbar.formatter.set_powerlimits((-3, 2))
    cbar.update_ticks()

    if T.size != 0 and S.size == 0:
        norm = mcolors.Normalize(vmin=ranges['T'][0], vmax=ranges['T'][-1])
        mappable = cm.ScalarMappable(norm=norm, cmap='RdBu_r')
        ax4.contourf(hor, z, T[x_data, y_data, :], levels, norm=norm, cmap='RdBu_r')
        ax4.set_xlabel("[m]")
        ax4.set_title("Temperature")
        ax4.set_ylim(-lx[2], 0)
        ax4.set_xlim(hor_range)
        ax4.set_aspect('equal')
        cbar  = fig.colorbar(mappable, ax=ax4, label=r"$^\circ$C", anchor = (0.5, -0.1), orientation='horizontal', shrink=0.75)
        cbar.formatter.set_powerlimits((-3, 2))
        cbar.update_ticks()
    else:
        norm = mcolors.Normalize(vmin=ranges['b_fluc'][0], vmax=ranges['b_fluc'][-1])#SymLogNorm(linthresh=1e-5, vmin=ranges['b_fluc'][0], vmax=ranges['b_fluc'][-1], base=10)
        mappable = cm.ScalarMappable(norm=norm, cmap='RdBu_r')
        ax4.contourf(hor, z, b_fluc[x_data, y_data, :], levels, norm=norm, cmap='RdBu_r')
        ax4.set_xlabel("[m]")
        ax4.set_title("Buoyancy Perturbations")
        ax4.set_ylim(-lx[2], 0)
        ax4.set_xlim(hor_range)
        ax4.set_aspect('equal')
        cbar  = fig.colorbar(mappable, ax=ax4, label=r"m/s$^2$", anchor = (0.5, -0.1), orientation='horizontal', shrink=0.75)
        cbar.formatter.set_powerlimits((-3, 2))
        cbar.update_ticks()
    if T.size != 0 and S.size != 0:
        norm = mcolors.Normalize(vmin=ranges['T'][0], vmax=ranges['T'][-1])
        mappable = cm.ScalarMappable(norm=norm)
        ax5.contourf(hor, z, T[x_data, y_data, :], levels, norm=norm)
        ax5.set_xlabel("[m]")
        ax5.set_ylabel("Depth [m]")
        ax5.set_title("Temperature")
        ax5.set_ylim(-lx[2], 0)
        ax5.set_xlim(hor_range)
        ax5.set_aspect('equal')
        cbar  = fig.colorbar(mappable, ax=ax5, label=r"$^\circ$C", anchor = (0.5, -0.1), orientation='horizontal', shrink=0.75)
    else:
        norm = mcolors.Normalize(vmin=ranges['Pstatic'][0], vmax=ranges['Pstatic'][-1])
        mappable = cm.ScalarMappable(norm=norm, cmap='RdBu_r')
        ax5.contourf(hor, z, Pstatic[x_data, y_data, :], levels, norm=norm, cmap='RdBu_r')
        ax5.set_xlabel("[m]")
        ax5.set_ylabel("Depth [m]")
        ax5.set_title("Hydrostatic Pressure")
        ax5.set_ylim(-lx[2], 0)
        ax5.set_xlim(hor_range)
        ax5.set_aspect('equal')
        cbar  = fig.colorbar(mappable, ax=ax5, label=r"m/s$^{2}$", anchor = (0.5, -0.1), orientation='horizontal', shrink=0.75)

    norm = mcolors.Normalize(vmin=urange[0], vmax=urange[-1])
    mappable = cm.ScalarMappable(norm=norm, cmap = 'RdBu_r')
    ax6.contourf(hor, z, u[x_data, y_data, :], levels, norm=norm, cmap='RdBu_r')
    ax6.set_xlabel("[m]")
    #ax6.set_ylabel("Depth [m]")
    ax6.set_title("u")
    ax6.set_ylim(-lx[2], 0)
    ax6.set_xlim(hor_range)
    ax6.set_aspect('equal')
    cbar  = fig.colorbar(mappable, ax=ax6, label=r"m/s", anchor = (0.5, -0.1), orientation='horizontal', shrink=0.75)
    cbar.formatter.set_powerlimits((-3, 2))
    cbar.update_ticks()

    norm = mcolors.Normalize(vmin=vrange[0], vmax=vrange[-1])
    mappable = cm.ScalarMappable(norm=norm, cmap = 'RdBu_r')
    ax7.contourf(hor, z, v[x_data, y_data, :], levels, norm=norm, cmap='RdBu_r')
    ax7.set_xlabel("[m]")
    #ax7.set_ylabel("Depth [m]")
    ax7.set_title("v")
    ax7.set_ylim(-lx[2], 0)
    ax7.set_xlim(hor_range)
    ax7.set_aspect('equal')
    cbar  = fig.colorbar(mappable, ax=ax7, label=r"m/s", anchor = (0.5, -0.1), orientation='horizontal', shrink=0.75)
    cbar.formatter.set_powerlimits((-3, 2))
    cbar.update_ticks()

    norm = mcolors.Normalize(vmin=ranges['w'][0], vmax=ranges['w'][-1])
    mappable = cm.ScalarMappable(norm=norm, cmap = 'RdBu_r')
    ax8.contourf(horf, Zf[x_data, y_data, :], w[x_data, y_data, :], levels, norm=norm, cmap='RdBu_r')
    ax8.set_xlabel("[m]")
    ax8.set_title("w")
    ax8.set_ylim(-lx[2], 0)
    ax8.set_xlim(hor_range)
    ax8.set_aspect('equal')
    cbar  = fig.colorbar(mappable, ax=ax8, label=r"m/s", anchor = (0.5, -0.1), orientation='horizontal', shrink=0.75)
    cbar.formatter.set_powerlimits((-3, 2))
    cbar.update_ticks()
    if T.size != 0 and S.size != 0:
        norm = mcolors.Normalize(vmin=ranges['S'][0], vmax=ranges['S'][-1]) # SymLogNorm(linthresh=1e-3, vmin=ranges['S'][0], vmax=ranges['S'][-1], base=10)
        mappable = cm.ScalarMappable(norm=norm)
        ax9.contourf(hor, z, S[x_data, y_data, :], levels, norm=norm)
        ax9.set_xlabel("[m]")
        ax9.set_ylabel("Depth [m]")
        ax9.set_title("Tracer")
        ax9.set_ylim(-lx[2], 0)
        ax9.set_xlim(hor_range)
        ax9.set_aspect('equal')
        cbar  = fig.colorbar(mappable, ax=ax9, anchor = (0.5, -0.1), orientation='horizontal', shrink=0.75)
        #cbar.locator = mticker.MaxNLocator(nbins=3)
        #cbar.update_ticks()
    else:
        norm = mcolors.Normalize(vmin=ranges['Pdynamic'][0], vmax=ranges['Pdynamic'][-1])
        mappable = cm.ScalarMappable(norm=norm, cmap='RdBu_r')
        ax9.contourf(hor, z, Pdynamic[x_data, y_data, :], levels, norm=norm, cmap='RdBu_r')
        ax9.set_xlabel("[m]")
        ax9.set_ylabel("Depth [m]")
        ax9.set_title("Hydrodynamic Pressure")
        ax9.set_ylim(-lx[2], 0)
        ax9.set_xlim(hor_range)
        ax9.set_aspect('equal')
        cbar  = fig.colorbar(mappable, ax=ax9, label=r"m/s$^{2}$", anchor = (0.5, -0.1), orientation='horizontal', shrink=0.75)

    norm = mcolors.Normalize(vmin=uflucrange[0], vmax=uflucrange[-1])
    mappable = cm.ScalarMappable(norm=norm, cmap = 'RdBu_r')
    ax10.contourf(hor, z, u_fluc[x_data, y_data, :], levels, norm=norm, cmap='RdBu_r')
    ax10.set_xlabel("[m]")
    #ax10.set_ylabel("Depth [m]")
    ax10.set_title("u'")
    ax10.set_ylim(-lx[2], 0)
    ax10.set_xlim(hor_range)
    ax10.set_aspect('equal')
    cbar  = fig.colorbar(mappable, ax=ax10, label=r"m/s", anchor = (0.5, -0.1), orientation='horizontal', shrink=0.75)
    cbar.formatter.set_powerlimits((-3, 2))
    cbar.update_ticks()

    norm = mcolors.Normalize(vmin=vflucrange[0], vmax=vflucrange[-1])
    mappable = cm.ScalarMappable(norm=norm, cmap = 'RdBu_r')
    ax11.contourf(hor, z, v_fluc[x_data, y_data, :], levels, norm=norm, cmap='RdBu_r')
    ax11.set_xlabel("[m]")
    #ax11.set_ylabel("Depth [m]")
    ax11.set_title("v'")
    ax11.set_ylim(-lx[2], 0)
    ax11.set_xlim(hor_range)
    ax11.set_aspect('equal')
    cbar  = fig.colorbar(mappable, ax=ax11, label=r"m/s", anchor = (0.5, -0.1), orientation='horizontal', shrink=0.75)
    cbar.formatter.set_powerlimits((-3, 2))
    cbar.update_ticks()

    norm = mcolors.Normalize(vmin=ranges['w_fluc'][0], vmax=ranges['w_fluc'][-1])
    mappable = cm.ScalarMappable(norm=norm, cmap = 'RdBu_r')
    ax12.contourf(horf, Zf[x_data, y_data, :], w_fluc[x_data, y_data, :], levels, norm=norm, cmap='RdBu_r')
    ax12.set_xlabel("[m]")
    #ax12.set_ylabel("Depth [m]")
    ax12.set_title("w'")
    ax12.set_ylim(-lx[2], 0)
    ax12.set_xlim(hor_range)
    ax12.set_aspect('equal')
    cbar  = fig.colorbar(mappable, ax=ax12, label=r"m/s", anchor = (0.5, -0.1), orientation='horizontal', shrink=0.75)
    cbar.formatter.set_powerlimits((-3, 2))
    cbar.update_ticks()

    # --- Save Frame ---
    frame_path = os.path.join(outdir, f"oc_plane_slices_{it:04d}.png")
    plt.tight_layout()
    plt.savefig(frame_path)
    plt.close(fig)
    print(f"Time step {it + 1} captured: {frame_path}")
    return outdir # return the directory where frames are saved for video creation
## surface plane slices 
def xy_plane_slices(time, it, ranges, fig_folder, lx, X, Y, u, v, w, b, b_fluc, Pdynamic, rho, rho_perturbed, idx_loc, plane, T = np.array([]), S = np.array([])):
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
    ax1.contourf(X[:, :, idx_loc], Y[:, :, idx_loc], rho_perturbed[:, :, idx_loc], levels, norm=norm, cmap='RdBu_r')
    #ax1.set_xlabel("[m]")
    ax1.set_ylabel("[m]")
    ax1.set_title("Perturbed Density")
    ax1.set_ylim(-lx[1]/2, lx[1]/2)
    ax1.set_xlim(-lx[0]/2, lx[0]/2)
    ax1.set_aspect('equal')
    cbar = fig.colorbar(mappable, ax=ax1, label=r"kg/m$^3$", orientation='horizontal', shrink=0.75)
    #cbar.formatter.set_scientific(False)
    #cbar.formatter.set_useOffset(False)
    #cbar.set_ticks([(ranges['rho'][0]), (ranges['rho'][-1])])
    cbar.formatter.set_powerlimits((-2, 2))
    cbar.update_ticks()
        
    norm = mcolors.Normalize(vmin=ranges['b_fluc'][0], vmax=ranges['b_fluc'][-1])
    mappable = cm.ScalarMappable(norm=norm, cmap='RdBu_r')
    ax2.contourf(X[:, :, idx_loc], Y[:, :, idx_loc], b_fluc[:, :, idx_loc], levels, norm=norm, cmap='RdBu_r')
    #ax2.set_xlabel("[m]")
    ax2.set_title("Perturbed Buoyancy")
    ax2.set_ylim(-lx[1]/2, lx[1]/2)
    ax2.set_xlim(-lx[0]/2, lx[0]/2)
    ax2.set_aspect('equal')
    cbar  = fig.colorbar(mappable, ax=ax2, label=r"m/s$^2$", orientation='horizontal', shrink=0.75)
    cbar.formatter.set_powerlimits((-3, 2))
    cbar.update_ticks()

    if np.size(S)==0:
        norm = mcolors.Normalize(vmin=ranges['rho_fluc'][0], vmax=ranges['rho_fluc'][-1])
        mappable = cm.ScalarMappable(norm=norm, cmap='RdBu_r')
        ax3.contourf(X[:, :, idx_loc], Y[:, :, idx_loc], rho_perturbed[:, :, idx_loc], levels, norm=norm, cmap='RdBu_r')
        #ax3.set_xlabel("[m]")
        ax3.set_title("Perturbed Density")
        ax3.set_ylim(-lx[1]/2, lx[1]/2)
        ax3.set_xlim(-lx[0]/2, lx[0]/2)
        ax3.set_aspect('equal')
        cbar  = fig.colorbar(mappable, ax=ax3, label=r"kg/m$^3$", orientation='horizontal', shrink=0.75)
    else:
        norm = mcolors.Normalize(vmin=ranges['S'][0], vmax=ranges['S'][-1])
        mappable = cm.ScalarMappable(norm=norm)
        ax3.contourf(X[:, :, idx_loc], Y[:, :, idx_loc], S[:, :, idx_loc], levels, norm=norm)
        #ax3.set_xlabel("[m]")
        ax3.set_title("Tracer")
        ax3.set_ylim(-lx[1]/2, lx[1]/2)
        ax3.set_xlim(-lx[0]/2, lx[0]/2)
        ax3.set_aspect('equal')
        cbar  = fig.colorbar(mappable, ax=ax3, orientation='horizontal', shrink=0.75)

    if np.size(T)==0:
        norm = mcolors.Normalize(vmin=ranges['b_fluc'][0], vmax=ranges['b_fluc'][-1])
        mappable = cm.ScalarMappable(norm=norm, cmap='RdBu_r')
        ax4.contourf(X[:, :, idx_loc], Y[:, :, idx_loc], b_fluc[:, :, idx_loc], levels, norm=norm, cmap='RdBu_r')
        #ax4.set_xlabel("[m]")
        ax4.set_title("Buoyancy Perturbations")
        ax4.set_ylim(-lx[1]/2, lx[1]/2)
        ax4.set_xlim(-lx[0]/2, lx[0]/2)
        ax4.set_aspect('equal')
        cbar  = fig.colorbar(mappable, ax=ax4, label=r"m/s$^2$", orientation='horizontal', shrink=0.75)
        cbar.formatter.set_powerlimits((-3, 2))
        cbar.update_ticks()
    else:
        T_slice = T[:, :, idx_loc]
        if (np.max(T_slice) - np.min(T_slice))<=1e-6:
            levelsT = 10
        else:
            levelsT = levels
        norm = mcolors.Normalize(vmin=ranges['T'][0], vmax=ranges['T'][-1])
        mappable = cm.ScalarMappable(norm=norm)
        ax4.contourf(X[:, :, idx_loc], Y[:, :, idx_loc], T_slice, levelsT, norm=norm)
        #ax4.set_xlabel("[m]")
        ax4.set_title("Temperature")
        ax4.set_ylim(-lx[1]/2, lx[1]/2)
        ax4.set_xlim(-lx[0]/2, lx[0]/2)
        ax4.set_aspect('equal')
        cbar  = fig.colorbar(mappable, ax=ax4, label=r"$^\circ$C", orientation='horizontal', shrink=0.75)
    
    norm = mcolors.Normalize(vmin=ranges['Pdynamic'][0], vmax=ranges['Pdynamic'][-1])
    mappable = cm.ScalarMappable(norm=norm, cmap='RdBu_r')
    ax5.contourf(X[:, :, idx_loc], Y[:, :, idx_loc], Pdynamic[:, :, idx_loc], levels, norm=norm, cmap='RdBu_r')
    ax5.set_xlabel("[m]")
    ax5.set_ylabel("[m]")
    ax5.set_title("Hydrodynamic Pressure")
    ax5.set_ylim(-lx[1]/2, lx[1]/2)
    ax5.set_xlim(-lx[0]/2, lx[0]/2)
    ax5.set_aspect('equal')
    cbar  = fig.colorbar(mappable, ax=ax5, label=r"m/s$^{2}$", orientation='horizontal', shrink=0.75)
    cbar.formatter.set_powerlimits((-3, 2))
    cbar.update_ticks()

    norm = mcolors.Normalize(vmin=ranges['u'][0], vmax=ranges['u'][-1])
    mappable = cm.ScalarMappable(norm=norm, cmap = 'RdBu_r')
    ax6.contourf(X[:, :, idx_loc], Y[:, :, idx_loc], u[:, :, idx_loc], levels, norm=norm, cmap='RdBu_r')
    ax6.set_xlabel("[m]")
    ax6.set_title("u")
    ax6.set_ylim(-lx[1]/2, lx[1]/2)
    ax6.set_xlim(-lx[0]/2, lx[0]/2)
    ax6.set_aspect('equal')
    cbar  = fig.colorbar(mappable, ax=ax6, label=r"m/s", orientation='horizontal', shrink=0.75)
    cbar.formatter.set_powerlimits((-3, 2))
    cbar.update_ticks()

    norm = mcolors.Normalize(vmin=ranges['v'][0], vmax=ranges['v'][-1])
    mappable = cm.ScalarMappable(norm=norm, cmap = 'RdBu_r')
    ax7.contourf(X[:, :, idx_loc], Y[:, :, idx_loc], v[:, :, idx_loc], levels, norm=norm, cmap='RdBu_r')
    ax7.set_xlabel("[m]")
    ax7.set_title("v")
    ax7.set_ylim(-lx[1]/2, lx[1]/2)
    ax7.set_xlim(-lx[0]/2, lx[0]/2)
    ax7.set_aspect('equal')
    cbar  = fig.colorbar(mappable, ax=ax7, label=r"m/s", orientation='horizontal', shrink=0.75)
    cbar.formatter.set_powerlimits((-3, 2))
    cbar.update_ticks()

    norm = mcolors.Normalize(vmin=ranges['w'][0], vmax=ranges['w'][-1])
    mappable = cm.ScalarMappable(norm=norm, cmap = 'RdBu_r')
    ax8.contourf(X[:, :, idx_loc], Y[:, :, idx_loc], w[:, :, idx_loc], levels, norm=norm, cmap='RdBu_r')
    ax8.set_xlabel("[m]")
    #ax8.set_ylabel("[m]")
    ax8.set_title("w")
    ax8.set_ylim(-lx[1]/2, lx[1]/2)
    ax8.set_xlim(-lx[0]/2, lx[0]/2)
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