import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import imageio.v2 as imageio
import matplotlib.ticker as mticker
### ----------------------------------PROFILES------------------------------- ###
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
def turb_stats(time, it, ranges, fig_folder, lx, nx, z, mld, u_avg, v_avg, w_avg, u_rms, v_rms, w_rms, uv_fluc, uw_fluc, vw_fluc, bu_fluc_avg, bv_fluc_avg, bw_fluc_avg, b_rms, rho, plume_info = []):
    outdir = os.path.join(fig_folder, 'turb stats/')
    os.makedirs(outdir, exist_ok=True)
    td = time / 3600 / 24 # convert time to days

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
    ax1.plot(w_avg, z, label=r"$\langle$w$\rangle_{xy}$", color='blue')
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
    ax2.plot(w_rms, z, label=r"$\langle$w$_{rms}$$\rangle_{xy}$", color='blue')
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
## vertical plane slices 
def vert_plane_slices(time, it, ranges, fig_folder, lx, x, y, z, u, v, w, rho, rho_perturbed, T = np.array([]), S = np.array([]), depths = np.array([]), yz=True):
    if yz: #yz plane
        ar = lx[1]/lx[2]
        plane = 'YZ plane'
        hor = y
    else: #xz plane
        ar = lx[0]/lx[2]
        plane = 'XZ plane'
        hor = x
    outdir = os.path.join(fig_folder, 'vertical plane slices/', plane)
    os.makedirs(outdir, exist_ok=True)
    td = time / 3600 / 24
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

    im = ax[4].imshow(T.T, vmin=ranges['T'][0], vmax=ranges['T'][-1], extent =[hor.min(), hor.max(), z.min(), z.max()], interpolation ='none', origin ='lower')
    ax[4].set_xlabel("[m]")
    ax[4].set_ylabel("Depth [m]")
    ax[4].set_title("Temperature")
    ax[4].set_aspect('equal')
    cbar = fig.colorbar(im, ax = ax[4], label=r"$^\circ$C", anchor = (0.5, -0.3), orientation='horizontal', shrink=0.75)
    
    im = ax[5].imshow(S.T, vmin=ranges['S'][0], vmax=ranges['S'][-1], extent =[hor.min(), hor.max(), z.min(), z.max()], interpolation ='none', origin ='lower')
    ax[5].set_xlabel("[m]")
    ax[5].set_title("Tracer")
    ax[5].set_aspect('equal')
    cbar = fig.colorbar(im, ax = ax[5], label=r"g/kg", anchor = (0.5, -0.3), orientation='horizontal', shrink=0.75)

    im = ax[6].imshow(rho.T, vmin=ranges['rho'][0], vmax=ranges['rho'][-1], extent =[hor.min(), hor.max(), z.min(), z.max()], interpolation ='none', origin ='lower')
    ax[6].set_xlabel("[m]")
    ax[6].set_title("Density")
    ax[6].set_aspect('equal')
    cbar = fig.colorbar(im, ax = ax[6], label=r"kg/m$^3$", anchor = (0.5, -0.3), orientation='horizontal', shrink=0.75)
    cbar.formatter.set_useOffset(False)
    cbar.set_ticks([ranges['rho'][0], ranges['rho'][-1]])
    cbar.update_ticks()
    
    im = ax[7].imshow(rho_perturbed.T, vmin=ranges['rho_fluc'][0], vmax=ranges['rho_fluc'][-1], extent =[hor.min(), hor.max(), z.min(), z.max()], interpolation ='none', origin ='lower', cmap='RdBu_r')
    ax[7].set_xlabel("[m]")
    ax[7].set_title("Perturbed Density")
    ax[7].set_aspect('equal')
    cbar = fig.colorbar(im, ax = ax[7], label=r"kg/m$^3$", anchor = (0.5, -0.3), orientation='horizontal', shrink=0.75)

    im = ax[0].imshow(u.T, vmin=ranges['u'][0], vmax=ranges['u'][-1], extent =[hor.min(), hor.max(), z.min(), z.max()], interpolation ='none', origin ='lower', cmap='RdBu_r')
    ax[0].set_ylabel("Depth [m]")
    ax[0].set_title("u")
    ax[0].set_aspect('equal')
    cbar = fig.colorbar(im, ax = ax[0], label=r"m/s", anchor = (0.5, -0.05), orientation='horizontal', shrink=0.75)
    cbar.formatter.set_powerlimits((-3, 2))
    cbar.update_ticks()

    im = ax[1].imshow(v.T, vmin=ranges['v'][0], vmax=ranges['v'][-1], extent =[hor.min(), hor.max(), z.min(), z.max()], interpolation ='none', origin ='lower', cmap='RdBu_r')
    ax[1].set_title("v")
    ax[1].set_aspect('equal')
    cbar = fig.colorbar(im, ax = ax[1], label=r"m/s", anchor = (0.5, -0.05), orientation='horizontal', shrink=0.75)
    cbar.formatter.set_powerlimits((-3, 2))
    cbar.update_ticks()

    im = ax[2].imshow(w.T, vmin=ranges['w'][0], vmax=ranges['w'][-1], extent =[hor.min(), hor.max(), z.min(), z.max()], interpolation ='none', origin ='lower', cmap='RdBu_r')
    ax[2].set_title("w")
    ax[2].set_aspect('equal')
    cbar = fig.colorbar(im, ax = ax[2], label=r"m/s", anchor = (0.5, -0.05), orientation='horizontal', shrink=0.75)
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
def xy_plane_slices(time, it, ranges, fig_folder, x, y, u, v, w, Pdynamic, rho, rho_perturbed, plane, T = np.array([]), S = np.array([])):
    outdir = os.path.join(fig_folder, 'horizontal plane slices/', 'XY plane slice at ' + plane)
    os.makedirs(outdir, exist_ok=True)
    td = time / 3600 / 24

    fig, ax = plt.subplots(2, 4, figsize=(12, 8.5))
    fig.tight_layout()
    fig.suptitle(f'{td:.2f} days', fontsize=12)

    ax1 = ax[0, 0]
    ax2 = ax[0, 1]
    ax3 = ax[0, 2]
    ax4 = ax[0, 3]
    ax5 = ax[1, 0]
    ax6 = ax[1, 1]
    ax7 = ax[1, 2]
    ax8 = ax[1, 3]

    im = ax1.imshow(rho_perturbed.T, vmin=ranges['rho_fluc'][0], vmax=ranges['rho_fluc'][-1], extent =[x.min(), x.max(), y.min(), y.max()], interpolation ='none', origin ='lower', cmap='RdBu_r')
    #ax1.set_xlabel("[m]")
    ax1.set_ylabel("[m]")
    ax1.set_title("Perturbed Density")
    ax1.set_aspect('equal')
    cbar = fig.colorbar(im, ax = ax1, label=r"kg/m$^3$", orientation='horizontal', shrink=0.75)
    cbar.formatter.set_powerlimits((-2, 2))
    cbar.update_ticks()

    im = ax2.imshow(rho.T, vmin=ranges['rho'][0], vmax=ranges['rho'][-1], extent =[x.min(), x.max(), y.min(), y.max()], interpolation ='none', origin ='lower')
    #ax2.set_xlabel("[m]")
    ax2.set_title("Density")
    ax2.set_aspect('equal')
    cbar = fig.colorbar(im, ax = ax2, label=r"kg/m$^3$", orientation='horizontal', shrink=0.75)
    cbar.formatter.set_useOffset(False)
    cbar.set_ticks([ranges['rho'][0], ranges['rho'][-1]])
    cbar.update_ticks()

    im = ax3.imshow(S.T, vmin=ranges['S'][0], vmax=ranges['S'][-1], extent =[x.min(), x.max(), y.min(), y.max()], interpolation ='none', origin ='lower')
    #ax3.set_xlabel("[m]")
    ax3.set_title("Tracer")
    ax3.set_aspect('equal')
    cbar = fig.colorbar(im, ax = ax3, label = r"g/kg", orientation='horizontal', shrink=0.75)
    
    im = ax4.imshow(T.T, vmin=ranges['T'][0], vmax=ranges['T'][-1], extent =[x.min(), x.max(), y.min(), y.max()], interpolation ='none', origin ='lower')
    #ax4.set_xlabel("[m]")
    ax4.set_title("Temperature")
    ax4.set_aspect('equal')
    cbar = fig.colorbar(im, ax = ax4, label=r"$^\circ$C", orientation='horizontal', shrink=0.75)
    
    im = ax5.imshow(Pdynamic.T, vmin=ranges['Pdynamic'][0], vmax=ranges['Pdynamic'][-1], extent =[x.min(), x.max(), y.min(), y.max()], interpolation ='none', origin ='lower', cmap='RdBu_r')
    ax5.set_xlabel("[m]")
    ax5.set_ylabel("[m]")
    ax5.set_title("Hydrodynamic Pressure")
    ax5.set_aspect('equal')
    cbar = fig.colorbar(im, ax = ax5, label=r"m/s$^{2}$", orientation='horizontal', shrink=0.75)
    cbar.formatter.set_powerlimits((-3, 2))
    cbar.update_ticks()

    im = ax6.imshow(u.T, vmin=ranges['u'][0], vmax=ranges['u'][-1], extent =[x.min(), x.max(), y.min(), y.max()], interpolation ='none', origin ='lower', cmap='RdBu_r')
    ax6.set_xlabel("[m]")
    ax6.set_title("u")
    ax6.set_aspect('equal')
    cbar = fig.colorbar(im, ax = ax6, label=r"m/s", orientation='horizontal', shrink=0.75)
    cbar.formatter.set_powerlimits((-3, 2))
    cbar.update_ticks()

    im = ax7.imshow(v.T, vmin=ranges['v'][0], vmax=ranges['v'][-1], extent =[x.min(), x.max(), y.min(), y.max()], interpolation ='none', origin ='lower', cmap='RdBu_r')
    ax7.set_xlabel("[m]")
    ax7.set_title("v")
    ax7.set_aspect('equal')
    cbar = fig.colorbar(im, ax = ax7, label=r"m/s", orientation='horizontal', shrink=0.75)
    cbar.formatter.set_powerlimits((-3, 2))
    cbar.update_ticks()

    im = ax8.imshow(w.T, vmin=ranges['w'][0], vmax=ranges['w'][-1], extent =[x.min(), x.max(), y.min(), y.max()], interpolation ='none', origin ='lower', cmap='RdBu_r')
    ax8.set_xlabel("[m]")
    #ax8.set_ylabel("[m]")
    ax8.set_title("w")
    ax8.set_aspect('equal')
    cbar = fig.colorbar(im, ax = ax8, label=r"m/s", orientation='horizontal', shrink=0.75)
    cbar.formatter.set_powerlimits((-3, 2))
    cbar.update_ticks()

    # --- Save Frame ---
    frame_path = os.path.join(outdir, f"oc_plane_slices_{it:04d}.png")
    plt.tight_layout()
    plt.savefig(frame_path)
    plt.close(fig)
    print(f"Time step {it + 1} captured: {frame_path}")
    return outdir # return the directory where frames are saved for video creation
### -------------------------PLOTTING COMPARISON FUNCTIONS------------------------- ###
### temporal analysis ###
def plume_temporal_analysis(time, ranges, color_opt, fig_folder, case_names, name, lx, start_neutral, mld, h_neutral, h_max, r_mld, r_neutral, r_hmax, w_mld, w_neutral, w_hmax, b_mld, b_neutral, b_hmax, T_mld, T_neutral, T_hmax, tracer_mld, tracer_neutral, tracer_hmax, tracerw_fluc_avg, Tw_fluc_avg, ND = False):
    num_cases = len(case_names)
    if num_cases==1:
        fig, ax = plt.subplots(2, 4, figsize=(12, 5))
        outdir = os.path.join(fig_folder, 'plume analysis/')
        os.makedirs(outdir, exist_ok=True)
    else:
        outdir = os.path.join(fig_folder, 'comparison plume analysis/')
        os.makedirs(outdir, exist_ok=True)
        gridspec_kw={'height_ratios': [1, 1, 0.1]} # add space for universal legend
        fig, ax = plt.subplots(3, 4, figsize=(12, 6.5), gridspec_kw=gridspec_kw)
        for a in ax[2, :]:
            a.remove()
        case_handles = [
            Line2D([0], [0], color=color_opt[i], linestyle='solid', label=case_names[i])
            for i in range(num_cases)
        ]
        fig.legend(handles=case_handles,
                loc='lower center',
                ncol=num_cases,
                bbox_to_anchor=(0.52, 0.015))
    #fig.tight_layout()
    ax1 = ax[0, 0] # depth of plume through time 
    ax2 = ax[0, 1] # max and average radius of plume through time 
    ax3 = ax[0, 2] # vertical velocity at depth through time
    ax4 = ax[0, 3] # perturbed buoyancy at depth through time
    ax5 = ax[1, 0] # perturbed Temperature at depth through time
    ax6 = ax[1, 1] # perturbed tracer at depth through time
    ax7 = ax[1, 2] # average tracer at MLD through time
    ax8 = ax[1, 3] # w_avg at MLD through time 
    if ND:
        ax1.set_ylabel(r"z/h$_{\text{MLD}}$")
        ax1.set_ylim(ymin = -lx[-1], ymax = 0)
        ax2.set_ylabel(r"$\langle$r$\rangle_{\text{xy}}$/l$_{j}$") #(r"$\langle$r$\rangle_{\text{xy}}$/h$_{\text{MLD}}$") #
        ax2.set_ylim(ymin = ranges['radius'][0], ymax = ranges['radius'][-1])
        ax3.set_ylabel(r"w/(h$_{\mathrm{MLD}_0} \sqrt{N^{2}})$")
        ax3.set_ylim(ymin = ranges['w'][0], ymax = ranges['w'][-1])
        ax4.set_ylabel(r"b'/(h$_{\mathrm{MLD}_0} N^{2}$)")
        ax4.set_ylim(ymin = ranges['b_fluc'][0], ymax = ranges['b_fluc'][-1])
        ax5.set_ylabel(r"$\langle$T$'\rangle_{\text{xy}}$/T$_{0}$")
        ax5.set_ylim(ymin = ranges['T_fluc'][0], ymax = ranges['T_fluc'][-1])
        ax6.set_ylabel(r"$\langle$C'$\rangle_{\text{xy}}$/S$_{\text{max}}$") #(\text{h}_{mld} \sqrt{N^{2}}$)/(F$_{\text{C}}$)") #(r"$\langle$C$'\sqrt{g\text{r}_{j}}$/(F$_{\text{C}}$)") #
        ax6.set_ylim(ymin = ranges['S_fluc'][0], ymax = ranges['S_fluc'][-1])
        ax7.set_ylabel(r"$\langle$C'w$\rangle_{\text{xy}}$/F$_{\text{C}}$") #(\text{h}_{mld}\sqrt{N^{2}}$)/(F$_{\text{C}}$)")# (r"$\langle$C$\rangle_{\text{xy}}$\sqrt{g\text{r}_{j}}$/(F$_{\text{C}}$)")#
        ax7.set_ylim(ymin = ranges['Sw_fluc'][0], ymax = ranges['Sw_fluc'][-1])
        ax8.set_ylabel(r"$\langle$T$'$w$\rangle_{\text{xy}}$/(h$_{\mathrm{MLD}_0} \sqrt{N^{2}}$)")
        ax8.set_ylim(ymin = ranges['Tw_fluc'][0], ymax = ranges['Tw_fluc'][-1])
    else:
        ax1.set_ylabel("[m]")
        ax1.set_ylim(ymin = -lx[-1], ymax = 0)
        ax2.set_ylabel("[m]")
        ax2.set_ylim(ymin = ranges['radius'][0], ymax = ranges['radius'][-1])
        ax3.set_ylabel("[m/s]")
        ax3.set_ylim(ymin = ranges['w'][0], ymax = ranges['w'][-1])
        ax4.set_ylabel(r"[m/s$^2$]")
        ax4.set_ylim(ymin = ranges['b_fluc'][0], ymax = ranges['b_fluc'][-1])
        ax5.set_ylabel(r"$\langle$T$'\rangle_{\text{xy}}$ [$^{\circ}$C]")
        ax5.set_ylim(ymin = ranges['T_fluc'][0], ymax = ranges['T_fluc'][-1])
        ax6.set_ylabel(r"$\langle$C$'\rangle_{\text{xy}}$ [g/kg]")
        ax6.set_ylim(ymin = ranges['S_fluc'][0], ymax = ranges['S_fluc'][-1])
        ax7.set_ylabel(r"$\langle$C$'\text{w}\rangle_{\text{xy}}$ [g/kg]")
        ax7.set_ylim(ymin = ranges['Sw_fluc'][0], ymax = ranges['Sw_fluc'][-1])
        ax8.set_ylabel(r"$\langle$T$'$w$\rangle_{\text{xy}}$ [$^{\circ}$C $\cdot$ m/s]")
        ax8.set_ylim(ymin = ranges['Tw_fluc'][0], ymax = ranges['Tw_fluc'][-1])
    # Depth of plume through time 
    for i in range(num_cases):
        if i == 0:
            ax1.plot(time/ 3600 / 24, -mld[:, i], label = r"h$_{\text{MLD}}$", linewidth = 0.75, linestyle = 'dashed', color = color_opt[i])
            ax1.plot(time[start_neutral[i]::]/ 3600 / 24, h_neutral[start_neutral[i]::, i], label = r"h$_{\text{neutral}}$", linewidth = 0.75, linestyle = 'dotted', color = color_opt[i])
            ax1.plot(time/ 3600 / 24, h_max[:, i], label = r"h$_{\text{intrusion}}$", linewidth = 0.75, linestyle = 'solid', color = color_opt[i])
        else: 
            ax1.plot(time/ 3600 / 24, -mld[:, i], linewidth = 0.75, linestyle = 'dashed', color = color_opt[i])
            ax1.plot(time/ 3600 / 24, h_max[:, i], linewidth = 0.75, linestyle = 'solid', color = color_opt[i])
            ax1.plot(time[start_neutral[i]::]/ 3600 / 24, h_neutral[start_neutral[i]::, i], linewidth = 0.75, linestyle = 'dotted', color = color_opt[i])
    ax1.set_title("Plume Depths", size = 10)
    #ax1.set_xlabel("Time [days]") 
    ax1.legend(loc='lower right', labelspacing = 0.25, handlelength=0.75)
    ax1.set_xlim([time.min() / 3600 / 24, time.max() / 3600 / 24])
    # radius of plume 
    for i in range(num_cases):
        if i == 0:
            ax2.plot(time/ 3600 / 24, r_mld[:, i], label = r"r$_{\text{MLD}}$", linewidth = 0.75, linestyle = 'dashed', color = color_opt[i])
            ax2.plot(time[start_neutral[i]::]/ 3600 / 24, r_neutral[start_neutral[i]::, i], label = r"r$_{\text{neutral}}$", linewidth = 0.75, linestyle = 'dotted', color = color_opt[i])
            ax2.plot(time/ 3600 / 24, r_hmax[:, i], label = r"r$_{\text{intrusion}}$", linewidth = 0.75, linestyle = 'solid', color = color_opt[i])
        else: 
            ax2.plot(time/ 3600 / 24, r_mld[:, i], linewidth = 0.75, linestyle = 'dashed', color = color_opt[i])
            ax2.plot(time[start_neutral[i]::]/ 3600 / 24, r_neutral[start_neutral[i]::, i], linewidth = 0.75, linestyle = 'dotted', color = color_opt[i])
            ax2.plot(time/ 3600 / 24, r_hmax[:, i], linewidth = 0.75, linestyle = 'solid', color = color_opt[i])
    ax2.set_xlim([time.min() / 3600 / 24, time.max() / 3600 / 24])
    ax2.set_title("Plume Radii", size = 10)
    #ax2.set_xlabel("Time [days]") 
    ax2.legend(loc='upper left', labelspacing = 0.25, handlelength=0.75)
    # vertical velocity
    for i in range(num_cases):
        if i == 0:
            ax3.plot(time/ 3600 / 24, w_mld[:, i], label = r"w$_{\text{MLD}}$", linewidth = 0.75, linestyle = 'dashed', color = color_opt[i])
            ax3.plot(time[start_neutral[i]::]/ 3600 / 24, w_neutral[start_neutral[i]::, i], label = r"w$_{\text{neutral}}$", linewidth = 0.75, linestyle = 'dotted', color = color_opt[i])
            ax3.plot(time/ 3600 / 24, w_hmax[:, i],label = r"w$_{\text{intrusion}}$", linewidth = 0.75, linestyle = 'solid', color = color_opt[i])
        else:
            ax3.plot(time/ 3600 / 24, w_mld[:, i], linewidth = 0.75, linestyle = 'dashed', color = color_opt[i])
            ax3.plot(time[start_neutral[i]::]/ 3600 / 24, w_neutral[start_neutral[i]::, i], linewidth = 0.75, linestyle = 'dotted', color = color_opt[i])
            ax3.plot(time/ 3600 / 24, w_hmax[:, i], linewidth = 0.75, linestyle = 'solid', color = color_opt[i])
    #ax3.set_xlabel("Time [days]") 
    ax3.set_title("Vertical Velocity", size = 10)
    ax3.legend(loc='upper right', labelspacing = 0.25, handlelength=0.75)
    ax3.set_xlim([time.min() / 3600 / 24, time.max() / 3600 / 24]) 
    # buoyancy perturbations 
    for i in range(num_cases):
        if i == 0:
            ax4.plot(time/ 3600 / 24, b_mld[:, i], label = r"b$'_{\text{MLD}}$", linewidth = 0.75, linestyle = 'dashed', color = color_opt[i])
            ax4.plot(time[start_neutral[i]::]/ 3600 / 24, b_neutral[start_neutral[i]::, i], label = r"b$'_{\text{neutral}}$", linewidth = 0.75, linestyle = 'dotted', color = color_opt[i])
            ax4.plot(time/ 3600 / 24, b_hmax[:, i], label = r"b$'_{\text{intrusion}}$", linewidth = 0.75, linestyle = 'solid', color = color_opt[i])
        else:
            ax4.plot(time/ 3600 / 24, b_mld[:, i], linewidth = 0.75, linestyle = 'dashed', color = color_opt[i])
            ax4.plot(time/ 3600 / 24, b_hmax[:, i], linewidth = 0.75, linestyle = 'solid', color = color_opt[i])
            ax4.plot(time[start_neutral[i]::]/ 3600 / 24, b_neutral[start_neutral[i]::, i], linewidth = 0.75, linestyle = 'dotted', color = color_opt[i])
    #ax4.set_xlabel("Time [days]") 
    ax4.set_title("Perturbed Buoyancy", size = 10)
    ax4.legend(loc='upper right', labelspacing = 0.25, handlelength=0.75)
    ax4.set_xlim([time.min() / 3600 / 24, time.max() / 3600 / 24])
    ax4.ticklabel_format(axis='y', style='sci', scilimits=(-3,2), useMathText=True)
    # temperature perturbations 
    for i in range(num_cases):
        if i == 0:
            ax5.plot(time/ 3600 / 24, T_mld[:, i], label = r"T$'_{\text{MLD}}$", linewidth = 0.75, linestyle = 'dashed', color = color_opt[i])
            ax5.plot(time[start_neutral[i]::]/ 3600 / 24, T_neutral[start_neutral[i]::, i], label = r"T$'_{\text{neutral}}$", linewidth = 0.75, linestyle = 'dotted', color = color_opt[i])
            ax5.plot(time/ 3600 / 24, T_hmax[:, i], label = r"T$'_{\text{intrusion}}$", linewidth = 0.75, linestyle = 'solid', color = color_opt[i])
        else:
            ax5.plot(time/ 3600 / 24, T_mld[:, i], linewidth = 0.75, linestyle = 'dashed', color = color_opt[i])
            ax5.plot(time/ 3600 / 24, T_hmax[:, i], linewidth = 0.75, linestyle = 'solid', color = color_opt[i])
            ax5.plot(time[start_neutral[i]::]/ 3600 / 24, T_neutral[start_neutral[i]::, i], linewidth = 0.75, linestyle = 'dotted', color = color_opt[i])
    ax5.set_xlabel("Time [days]") 
    ax5.set_title("Perturbed Temperature", size = 10)
    ax5.legend(loc='lower right', labelspacing = 0.25, handlelength=0.75)
    ax5.set_xlim([time.min() / 3600 / 24, time.max() / 3600 / 24])
    ax5.ticklabel_format(axis='y', style='sci', scilimits=(-3,2), useMathText=True)
    # tracer perturbations 
    for i in range(num_cases):
        if i == 0:
            ax6.plot(time/ 3600 / 24, tracer_mld[:, i], label = r"C$'_{\text{MLD}}$", linewidth = 0.75, linestyle = 'dashed', color = color_opt[i])
            ax6.plot(time[start_neutral[i]::]/ 3600 / 24, tracer_neutral[start_neutral[i]::, i], label = r"C$'_{\text{neutral}}$", linewidth = 0.75, linestyle = 'dotted', color = color_opt[i])
            ax6.plot(time/ 3600 / 24, tracer_hmax[:, i], label = r"C$'_{\text{intrusion}}$", linewidth = 0.75, linestyle = 'solid', color = color_opt[i])
        else:
            ax6.plot(time/ 3600 / 24, tracer_mld[:, i], linewidth = 0.75, linestyle = 'dashed', color = color_opt[i])
            ax6.plot(time/ 3600 / 24, tracer_hmax[:, i], linewidth = 0.75, linestyle = 'solid', color = color_opt[i])
            ax6.plot(time[start_neutral[i]::]/ 3600 / 24, tracer_neutral[start_neutral[i]::, i], linewidth = 0.75, linestyle = 'dotted', color = color_opt[i])
    ax6.set_xlabel("Time [days]") 
    ax6.set_title("Perturbed Tracer", size = 10)
    ax6.legend(loc='lower right', labelspacing = 0.25, handlelength=0.75)
    ax6.set_xlim([time.min() / 3600 / 24, time.max() / 3600 / 24])
    ax6.ticklabel_format(axis='y', style='sci', scilimits=(-3,2), useMathText=True)
    # average salinity at MLD
    for i in range(num_cases):
        ax7.plot(time/ 3600 / 24, tracerw_fluc_avg[:, i], linewidth = 0.75, color = color_opt[i])
    ax7.set_xlabel("Time [days]")
    ax7.set_title(r"$\langle$C$'$w$\rangle_{\text{xy}}$ at MLD", size = 10)
    ax7.ticklabel_format(axis='y', scilimits=(-1,1), useMathText=True)
    ax7.set_xlim([time.min() / 3600 / 24, time.max() / 3600 / 24])
    # root mean square w at MLD 
    for i in range(num_cases):
        ax8.plot(time/ 3600 / 24, Tw_fluc_avg[:, i], linewidth = 0.75, color = color_opt[i])
    ax8.set_xlabel("Time [days]")
    ax8.set_title(r"$\langle$T$'$w$\rangle_{\text{xy}}$ at MLD", size = 10)
    ax8.ticklabel_format(axis='y', style='sci', scilimits=(-3,2), useMathText=True)
    ax8.set_xlim([time.min() / 3600 / 24, time.max() / 3600 / 24])
    # --- Save Frame ---
    frame_path = os.path.join(outdir, f"{name}_temporal_comparison.png")
    plt.savefig(frame_path)
    plt.close(fig)
    print("Temporal Plot Saved: ", frame_path)
### spatial vertical analysis ###
def plume_vertical_spatial_plot(time, it, ranges, color_opt, fig_folder, case_names, name, lx, z, tracer_avg, u_rms, v_rms, w_rms, b_avg, b_center, r_profile, bu_fluc_avg, bv_fluc_avg, bw_fluc_avg, T_avg, T_fluc, tracer_fluc, ND = False, z_nd = r"(z - h$_{\mathrm{MLD}_0}$)/l$_{j}$"):
    num_cases = len(case_names)
    if num_cases==0:
        fig, ax = plt.subplots(2, 4, figsize=(12, 8))
        outdir = os.path.join(fig_folder, 'vertical centerline-' + name)
        os.makedirs(outdir, exist_ok=True)
    else:
        outdir = os.path.join(fig_folder, 'vertical centerline-' + name)
        os.makedirs(outdir, exist_ok=True)
        gridspec_kw={'height_ratios': [1, 1, 0.02]} # add space for universal legend
        fig, ax = plt.subplots(3, 4, figsize=(12, 10), gridspec_kw=gridspec_kw)
        for a in ax[2, :]:
            a.remove()
        case_handles = [
            Line2D([0], [0], color=color_opt[i], linestyle='solid', label=case_names[i])
            for i in range(num_cases)
        ]

        fig.legend(handles=case_handles,
                loc='lower center',
                ncol=num_cases,
                bbox_to_anchor=(0.52, 0.015))

    td = time[it] / 3600 / 24
    fig.suptitle(f'{td:.2f} days', fontsize=12)

    ax1 = ax[0, 0]
    ax2 = ax[0, 1]
    ax3 = ax[0, 2]
    ax4 = ax[0, 3]
    ax5 = ax[1, 0]
    ax6 = ax[1, 1]
    ax7 = ax[1, 2]
    ax8 = ax[1, 3]

    if ND:
        ax1.set_ylabel(z_nd) 
        ax1.set_xlabel(r"(u$_{i}$/$\sqrt{\text{g l}_{j}}$)Ri$_{g}^{1/3}$")
        ax2.set_xlabel(r"($\langle$C$\rangle_{\text{xy}} \beta$)Ri$_{g}^{-1/3}$Fr$_{flux}^{-1}$")
        ax3.set_xlabel(r"(b/g)(Ri$_{g}^{1/3}$)Fr$_{flux}^{-1}$")
        ax4.set_xlabel(r"(C'$_{\text{centerline}} \beta$)Ri$_{g}^{-1/3}$Fr$_{flux}^{-1}$") 
        ax5.set_ylabel(z_nd)
        ax5.set_xlabel(r"(r/l$_{j}$)Fr$_{flux}^{-1}$")
        ax6.set_xlabel(r"($\langle$b'u'$_{i}\rangle_{xy}$/$\sqrt{\text{g}^3 \text{r}_{j}}$)Fr$_{flux}^{-1}$")
        ax7.set_xlabel(r"(b$_{rms}$/g)(Ri$_{g}^{1/3}$)Fr$_{flux}^{-1}$")
        ax8.set_xlabel(r"$(\text{T'}_{\text{centerline}}\alpha$)Ri$_{g}^{-1/3}$Fr$_{flux}^{-1}$")
    else:
        ax1.set_ylabel("Depth [m]")
        ax1.set_xlabel("[m/s]")
        ax2.set_xlabel(r"$\langle$C$\rangle_{\text{xy}}$ [g/kg]")
        ax3.set_xlabel(r"[m/s$^2$]")
        ax4.set_xlabel(r"C$'_{\text{centerline}}$ [g/kg]")
        ax5.set_ylabel("Depth [m]")
        ax5.set_xlabel("[m]")
        ax6.set_xlabel(r"[m$^2$/s$^3$]")
        ax7.set_xlabel(r"$\langle$T$\rangle_{\text{xy}}$ [$^{\circ}$ C]")
        ax8.set_xlabel(r"T$'_{\text{centerline}}$ [$^{\circ}$ C]")

    # velocity rms
    for i in range(num_cases):
        if i == 0:
            ax1.plot(u_rms[i], z[i], label=r"$\langle$u$_{\text{rms}}\rangle_{\text{xy}}$", color = color_opt[i], linestyle='dotted', linewidth = 0.75)
            ax1.plot(v_rms[i], z[i], label=r"$\langle$v$_{\text{rms}}\rangle_{\text{xy}}$", color = color_opt[i], linestyle='dashed', linewidth = 0.75)
            ax1.plot(w_rms[i], z[i], label=r"$\langle$w$_{\text{rms}}\rangle_{\text{xy}}$", color = color_opt[i], linestyle='solid', linewidth = 0.75)
        else:
            ax1.plot(u_rms[i], z[i], color = color_opt[i], linestyle='dotted', linewidth = 0.75)
            ax1.plot(v_rms[i], z[i], color = color_opt[i], linestyle='dashed', linewidth = 0.75)
            ax1.plot(w_rms[i], z[i], color = color_opt[i], linestyle='solid', linewidth = 0.75)
    ax1.set_title("Root Mean Square Velocities")
    #ax1.set_ylim(ymin = np.min(z), ymax = np.max(z))
    ax1.set_xlim(ranges['vel_rms'])
    ax1.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
    ax1.legend(loc='lower right')

    # tracer profile 
    for i in range(num_cases):
        ax2.plot(tracer_avg[i], z[i], color = color_opt[i], linestyle='solid', linewidth = 0.75)
    ax2.set_title('Tracer')
    #ax2.set_ylim(ymin = np.min(z), ymax = np.max(z))
    ax2.set_xlim(ranges['S'])
    ax2.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)

    # buoyancy profiles
    for i in range(num_cases):
        if i == 0:
            ax3.plot(b_avg[i], z[i], color = color_opt[i], label = r"$\langle$b$\rangle_{\text{xy}}$", linestyle='solid', linewidth = 0.75)
            ax3.plot(b_center[i], z[i], color = color_opt[i], label = r"b$_{\text{centerline}}$", linestyle='dashed', linewidth = 0.75)
        else:
            ax3.plot(b_avg[i], z[i], color = color_opt[i], linestyle='solid', linewidth = 0.75)
            ax3.plot(b_center[i], z[i], color = color_opt[i], linestyle='dashed', linewidth = 0.75)
    ax3.set_title("Buoyancy Profile")
    #ax3.set_ylim(ymin = np.min(z), ymax = np.max(z))
    ax3.set_xlim(ranges['b_avg'])
    ax3.legend(loc='upper left')
    ax3.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)

    # temperature fluctuations 
    for i in range(num_cases):
        ax4.plot(tracer_fluc[i], z[i], color = color_opt[i], linestyle='solid', linewidth = 0.75)
    ax4.set_title("Perturbed Tracer")
    #ax4.set_ylim(ymin = np.min(z), ymax = np.max(z))
    ax4.set_xlim(ranges['S_fluc'])
    ax4.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)

    # plume radius
    for i in range(num_cases):
        ax5.plot(r_profile[i], z[i], color = color_opt[i], linestyle='solid', linewidth = 0.75)
    ax5.set_title("Plume Radius with Depth")
    #ax5.set_ylim(ymin = np.min(z), ymax = np.max(z))
    ax5.set_xlim(0, lx[0][0]/2)

    # perturbed buoyancy flux 
    for i in range(num_cases):
        if i == 0:
            ax6.plot(bu_fluc_avg[i], z[i], color = color_opt[i], label = r"$\langle$b'u'$\rangle_{\text{xy}}$", linestyle='dotted', linewidth = 0.75)
            ax6.plot(bv_fluc_avg[i], z[i], color = color_opt[i], label = r"$\langle$b'v'$\rangle_{\text{xy}}$", linestyle='dashed', linewidth = 0.75)
            ax6.plot(bw_fluc_avg[i], z[i], color = color_opt[i], label = r"$\langle$b'w'$\rangle_{\text{xy}}$", linestyle='solid', linewidth = 0.75)
        else:
            ax6.plot(bu_fluc_avg[i], z[i], color = color_opt[i], linestyle='dotted', linewidth = 0.75)
            ax6.plot(bv_fluc_avg[i], z[i], color = color_opt[i], linestyle='dashed', linewidth = 0.75)
            ax6.plot(bw_fluc_avg[i], z[i], color = color_opt[i], linestyle='solid', linewidth = 0.75)
    ax6.legend(loc='lower right')
    ax6.set_title("Buoyancy Flux Fluctuations")
    #ax6.set_ylim(ymin = np.min(z), ymax = np.max(z))
    ax6.set_xlim(ranges['bw_fluc'])
    ax6.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True) 

    # buoyancy brms 
    for i in range(num_cases):
        ax7.plot(T_avg[i], z[i], color = color_opt[i], linestyle='solid', linewidth = 0.75)
    ax7.set_title("Temperature")
    #ax7.set_ylim(ymin = np.min(z), ymax = np.max(z))
    ax7.set_xlim(ranges['T'])
    ax7.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)

    # temperature fluctuations 
    for i in range(num_cases):
        ax8.plot(T_fluc[i], z[i], color = color_opt[i], linestyle='solid', linewidth = 0.75)
    ax8.set_title("Perturbed Temperature")
    #ax8.set_ylim(ymin = np.min(z), ymax = np.max(z))
    ax8.set_xlim(ranges['T_fluc'])
    ax8.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)

    # --- Save Frame ---
    frame_path = os.path.join(outdir, f"comparison_vert_buoyancy_{it:04d}.png")
    plt.savefig(frame_path)
    plt.close(fig)
    print(f"Time step {it + 1} captured: {frame_path}")

    return outdir # return the directory where frames are saved for video creation
### spatial horizontal analysis ###
def plume_horizontal_spatial_plot(time, it, ranges, color_opt, fig_folder, case_names, name, lx, y, u, v, w, b_center, bu_fluc, bv_fluc, bw_fluc, T, tracer, ND = False):
    num_cases = len(case_names)
    if num_cases==0:
        fig, ax = plt.subplots(2, 3, figsize=(12, 7))
        outdir = os.path.join(fig_folder, 'horizontal centerline-' + name)
        os.makedirs(outdir, exist_ok=True)
    else:
        outdir = os.path.join(fig_folder, 'horizontal centerline-' + name)
        os.makedirs(outdir, exist_ok=True)
        gridspec_kw={'height_ratios': [1, 1, 0.02]} # add space for universal legend
        fig, ax = plt.subplots(3, 3, figsize=(12, 9), gridspec_kw=gridspec_kw)
        for a in ax[2, :]:
            a.remove()
        case_handles = [
            Line2D([0], [0], color=color_opt[i], linestyle='solid', label=case_names[i])
            for i in range(num_cases)
        ]

        fig.legend(handles=case_handles,
                loc='lower center',
                ncol=num_cases,
                bbox_to_anchor=(0.52, 0.015))

    td = time[it] / 3600 / 24
    fig.suptitle(f'{td:.2f} days', fontsize=12)

    ax1 = ax[0, 0] # u, v, w through horizontal centerline
    ax2 = ax[0, 1] # horizontal buoyancy flux through horizontal centerline
    ax3 = ax[0, 2] # tracer through horizontal centerline
    ax4 = ax[1, 0] # perturbed buoyancy through horizontal centerline
    ax5 = ax[1, 1] # vertical buoyancy flux through horizontal centerline
    ax6 = ax[1, 2] # temperature through horizontal centerline

    if ND:
        ax1.set_xlabel(r"y/l$_{j}$") 
        ax1.set_ylabel(r"u$_{i}$/(F$_{\text{C}} \beta$ dT/dz l$_{j}$ T$_{0}$)") 
        ax2.set_xlabel(r"y/l$_{j}$")
        ax2.set_ylabel(r"(b' u'$_{i}$)/(F$_{\text{C}} \beta$ g dT/dz l$_{j}$/T$_{0}$)") 
        ax3.set_xlabel(r"y/l$_{j}$")
        ax3.set_ylabel(r"C$_{\text{centerline}} \sqrt{\text{g l}_{j}}$/(F$_{\text{C}}$)") 
        ax4.set_xlabel(r"y/l$_{j}$")
        ax4.set_ylabel(r"b'/(F$_{\text{C}} \beta \sqrt{\text{g dT/dz l}_{j}/\text{T}_{0}}$)")
        ax5.set_xlabel(r"y/l$_{j}$")
        ax5.set_ylabel(r"b'w'/(F$_{\text{C}} \beta$ g dT/dz l$_{j}$/T$_{0}$)")
        ax6.set_xlabel(r"y/l$_{j}$")
        ax6.set_ylabel(r"T$_{\text{centerline}\sqrt{\text{g l}_{j}}}$/(F$_{\text{C}}\beta$dT/dz l$_{j}$)")
    else:
        ax1.set_xlabel("x [m]")
        ax1.set_ylabel("[m/s]")
        ax2.set_xlabel("x [m]")
        ax2.set_ylabel(r"b'u'$_{i}$ [m$^2$/s$^3$]")
        ax3.set_xlabel("x [m]")
        ax3.set_ylabel(r"C$_{\text{centerline}}$ [g/kg]")
        ax4.set_xlabel("x [m]")
        ax4.set_ylabel(r"[m/s$^2$]")
        ax5.set_xlabel("x [m]")
        ax5.set_ylabel(r"b'w' [m$^2$/s$^3$]")
        ax6.set_xlabel("x [m]")
        ax6.set_ylabel(r"T$_{\text{centerline}}$ [$^{\circ}$ C]")

    # velocity
    for i in range(num_cases):
        if i == 0:
            ax1.plot(y, u[i], label=r"u$_{\text{centerline}}$", color = color_opt[i], linestyle='dotted', linewidth = 0.75)
            ax1.plot(y, v[i], label=r"v$_{\text{centerline}}$", color = color_opt[i], linestyle='dashed', linewidth = 0.75)
            ax1.plot(y, w[i], label=r"w$_{\text{centerline}}$", color = color_opt[i], linestyle='solid', linewidth = 0.75)
        else:
            ax1.plot(y, u[i], color = color_opt[i], linestyle='dotted', linewidth = 0.75)
            ax1.plot(y, v[i], color = color_opt[i], linestyle='dashed', linewidth = 0.75)
            ax1.plot(y, w[i], color = color_opt[i], linestyle='solid', linewidth = 0.75)
    ax1.set_title("Velocity")
    ax1.set_xlim(-lx[0][1]/2, lx[0][1]/2)
    ax1.set_ylim(ranges['w'])
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(-3,2), useMathText=True)
    ax1.legend(loc='lower right')

    # horizontal buoyancy flux 
    for i in range(num_cases):
        if i == 0:
            ax2.plot(y, bu_fluc[i], color = color_opt[i], label = r"b$'$u$'_{\text{centerline}}$", linestyle='dotted', linewidth = 0.75)
            ax2.plot(y, bv_fluc[i], color = color_opt[i], label = r"b$'$v$'_{\text{centerline}}$", linestyle='dashed', linewidth = 0.75)
        else:
            ax2.plot(y, bu_fluc[i], color = color_opt[i], linestyle='dotted', linewidth = 0.75)
            ax2.plot(y, bv_fluc[i], color = color_opt[i], linestyle='dashed', linewidth = 0.75)
    ax2.set_title('Horizontal Buoyancy Flux Fluctuations')
    ax2.set_xlim(-lx[0][1]/2, lx[0][1]/2)
    ax2.set_ylim(ymin = ranges['b_flux'][0], ymax = ranges['b_flux'][1])
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(-3,2), useMathText=True)

    # tracer  
    for i in range(num_cases):
        ax3.plot(y, tracer[i], color = color_opt[i], linestyle='solid', linewidth = 0.75)
    ax3.set_title("Tracer")
    ax3.set_xlim(-lx[0][1]/2, lx[0][1]/2)
    ax3.set_ylim(ranges['S'])
    ax3.ticklabel_format(axis='y', style='sci', scilimits=(-3,2), useMathText=True)

    # Perturbed buoyancy 
    for i in range(num_cases):
        ax4.plot(y, b_center[i], color = color_opt[i], linestyle='solid', linewidth = 0.75)
    ax4.set_title("Perturbed Buoyancy")
    ax4.set_xlim(-lx[0][1]/2, lx[0][1]/2)
    ax4.set_ylim(ranges['b_fluc'])
    ax4.ticklabel_format(axis='y', style='sci', scilimits=(-3,2), useMathText=True)
    # perturbed buoyancy flux 
    for i in range(num_cases):
        ax5.plot(y, bw_fluc[i], color = color_opt[i], linestyle='solid', linewidth = 0.75)
    ax5.set_title("Vertical Buoyancy Flux Fluctuations")
    ax5.set_xlim(-lx[0][1]/2, lx[0][1]/2)
    ax5.set_ylim(ranges['bw_fluc'])
    ax5.ticklabel_format(axis='y', style='sci', scilimits=(-3,2), useMathText=True) 

    # temperature  
    for i in range(num_cases):
        ax6.plot(y, T[i], color = color_opt[i], linestyle='solid', linewidth = 0.75)
    ax6.set_title("Temperature")
    ax6.set_xlim(-lx[0][1]/2, lx[0][1]/2)
    ax6.set_ylim(ranges['T'])

    # --- Save Frame ---
    frame_path = os.path.join(outdir, f"hor_centerline_comparisons_{it:04d}.png")
    plt.savefig(frame_path)
    plt.close(fig)
    print(f"Time step {it + 1} captured: {frame_path}")

    return outdir # return the directory where frames are saved for video creation

### -------------------------PLOTTING DENSE PLUME FUNCTIONS------------------------- ###
## buoyancy analysis 
def buoyancy_analysis(time, it, ranges, fig_folder, lx, nx, z, zf, X, Z, mld, b_avg, w_avg, b_center, w_center, b_rms, bu_fluc_avg, bv_fluc_avg, bw_fluc_avg, b_fluc, rho_perturbed, Ri_avg, Ri_strat, Ri_plume, plume_depth_intrusion, plume_depth_neutral, w_neutral, w_intrusion, w_mld, rho_perturbed_neutral, rho_perturbed_intrusion, rho_perturbed_mld, bwfluc_neutral, bwfluc_intrusion, bwfluc_mld):

    outdir = os.path.join(fig_folder, 'NBP buoyancy analysis/')
    os.makedirs(outdir, exist_ok=True)
    td = time[it] / 3600 / 24

    fig = plt.figure()
    fig.set_figheight(7)
    fig.set_figwidth(22)
    fig.suptitle(f'{td:.2f} days', fontsize=12) 

    ax1 = plt.subplot2grid(shape=(4, 10), loc=(0, 0), rowspan=2, colspan=2)
    ax2 = plt.subplot2grid(shape=(4, 10), loc=(0, 2), rowspan=2, colspan=2)
    ax3 = plt.subplot2grid(shape=(4, 10), loc=(0, 4), rowspan=2, colspan=2)
    ax4 = plt.subplot2grid(shape=(4, 10), loc=(0, 6), rowspan=2, colspan=2)
    ax5 = plt.subplot2grid(shape=(4, 10), loc=(0, 8), rowspan=2, colspan=2)
    ax6 = plt.subplot2grid(shape=(4, 10), loc=(2, 0), rowspan=2, colspan=2)
    ax7 = plt.subplot2grid(shape=(4, 10), loc=(2, 2), rowspan=2, colspan=2)
    ax8 = plt.subplot2grid(shape=(4, 10), loc=(2, 4), rowspan=2, colspan=2)
    ax9 = plt.subplot2grid(shape=(4, 10), loc=(2, 6), rowspan=2, colspan=2)
    ax10 = plt.subplot2grid(shape=(4, 10), loc=(2, 8), rowspan=1, colspan=2)
    ax11 = plt.subplot2grid(shape=(4, 10), loc=(3, 8), rowspan=1, colspan=2, sharex=ax10)

    fig.subplots_adjust(hspace=0.05)

    levels = 500

    # Depth of plume through time
    ax1.plot(time/ 3600 / 24, -mld*np.ones(len(time)), linestyle='--', linewidth = 0.5, color = 'black', label = "MLD")
    ax1.plot(time[:it+1]/ 3600 / 24, plume_depth_intrusion, color = 'blue', label = r"h$_{\text{intrusion}}$")
    ax1.plot(time[:it+1]/ 3600 / 24, plume_depth_neutral, color = 'red', label = r"h$_{\text{neutral}}$")
    ax1.set_xlabel("Time [days]") 
    ax1.set_ylabel("[m]")
    ax1.set_title("Plume Depths")
    ax1.set_ylim(ymin = -lx[-1], ymax = 0)
    ax1.legend(loc='upper right', handlelength=0.9)
    ax1.set_xlim([0, time.max() / 3600 / 24])  

    # vertical velocity
    ax2.plot(time[:it+1]/ 3600 / 24, w_mld, color = 'black', label = r"w$_{\text{MLD}}$")
    ax2.plot(time[:it+1]/ 3600 / 24, w_intrusion, color = 'blue',label = r"w$_{\text{intrusion}}$")
    ax2.plot(time[:it+1]/ 3600 / 24, w_neutral, color = 'red', label = r"w$_{\text{neutral}}$")
    ax2.set_xlabel("Time [days]") 
    ax2.set_ylabel("[m/s]")
    ax2.set_title("w at Depth")
    ax2.set_ylim(ymin = ranges['w'][0]*10, ymax = ranges['w'][-1]*10)
    ax2.legend(loc='upper right', handlelength=0.9)
    ax2.set_xlim([0, time.max() / 3600 / 24])  

    # density perturbations 
    ax3.plot(time[:it+1]/ 3600 / 24, rho_perturbed_mld, color = 'black', label = r"$\rho$'$_{\text{MLD}}$")
    ax3.plot(time[:it+1]/ 3600 / 24, rho_perturbed_intrusion, color = 'blue', label = r"$\rho$'$_{\text{intrusion}}$")
    ax3.plot(time[:it+1]/ 3600 / 24, rho_perturbed_neutral, color = 'red', label = r"$\rho$'$_{\text{neutral}}$")
    ax3.set_xlabel("Time [days]") 
    ax3.set_ylabel(r"[kg/m$^3$]")
    ax3.set_title("Perturbed Density at Depth")
    ax3.legend(loc='upper right', handlelength=0.9)
    ax3.set_ylim(ymin = ranges['rho_fluc'][0]*2, ymax = ranges['rho_fluc'][-1]*2)
    ax3.set_xlim([0, time.max() / 3600 / 24])

    # buoyancy perturbations 
    ax4.plot(time[:it+1]/ 3600 / 24, bwfluc_mld, color = 'black', label = r"b'w$_{\text{MLD}}$")
    ax4.plot(time[:it+1]/ 3600 / 24, bwfluc_intrusion, color = 'blue', label = r"b'w$_{\text{intrusion}}$")
    ax4.plot(time[:it+1]/ 3600 / 24, bwfluc_neutral, color = 'red', label = r"b'w$_{\text{neutral}}$")
    ax4.set_xlabel("Time [days]") 
    ax4.set_ylabel(r"[m$^{2}$/s$^{3}$]")
    ax4.set_title("Perturbed Buoyancy Flux at Depth")
    ax4.legend(loc='lower right', handlelength=0.9)
    ax4.set_ylim(ymin = ranges['bw_fluc'][0]*10, ymax = ranges['bw_fluc'][-1]*10)
    ax4.set_xlim([0, time.max() / 3600 / 24])

    #Richardson profile
    ax5.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')
    ax5.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')
    ax5.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')
    ax5.set_xlim(ranges['Ri'])
    if len(np.shape(Ri_strat))>1: # if we have the Hassanzadeh Richardson number, plot that as well
        Ri_strat_h = Ri_strat[:, 1]
        Ri_avg_h = Ri_avg[:, 1]
        Ri_plume_h = Ri_plume[:, 1]
        Ri_strat = Ri_strat[:, 0]
        Ri_avg = Ri_avg[:, 0]
        Ri_plume = Ri_plume[:, 0]
        ax5.plot(Ri_strat_h, z, linestyle='--', label = r"Ri$_{\text{stratified, Hassanzadeh}}$")
        ax5.plot(Ri_avg_h, z, color = 'black', linestyle='--', label = r"Ri$_{\text{average, Hassanzadeh}}$")
        ax5.plot(Ri_plume_h, z, color = 'red', linestyle='--', label = r"Ri$_{\text{centerline, Hassanzadeh}}$")
    ax5.plot(Ri_strat, z, label = r"Ri$_{\text{stratified}}$")
    ax5.plot(Ri_avg, z, color = 'black', label = r"Ri$_{\text{average}}$")
    ax5.plot(Ri_plume, z, color = 'red', label = r"Ri$_{\text{centerline}}$")
    ax5.set_xlabel("Richardson Number") 
    ax5.set_ylabel("Depth [m]")
    ax5.set_title("Richardson Number")
    ax5.legend(loc='lower right', handlelength=0.9)
    ax5.set_ylim(-lx[2], 0)

    # buoyancy 
    ax6.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')
    ax6.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')
    ax6.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')
    ax6.set_xlim(ranges['b'])
    ax6.plot(b_avg, z, color='black', label = r"$\langle$b$\rangle_{\text{xy}}")
    ax6.plot(b_center, z, color='red', label = r"b$_{\text{centerline}}$")
    ax6.set_title("Buoyancy Profile")
    ax6.set_xlabel("[m/s$^{2}$]")
    ax6.set_ylabel("Depth [m]")
    ax6.set_ylim(-lx[2], 0)
    ax6.legend(loc='upper left', handlelength=0.9)
    ax6.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True)

    # w
    ax7.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')
    ax7.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')
    ax7.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')
    ax7.set_xlim(xmin = ranges['w'][0]*10, xmax = ranges['w'][-1]*1)
    ax7.plot(w_avg, zf, color='black', label = r"w$_{\text{average}}$")
    ax7.plot(w_center, zf, color='red', label = r"w$_{\text{centerline}}$")
    ax7.set_title("Vertical Velocity Profile")
    ax7.set_xlabel("[m/s]")
    ax7.set_ylim(-lx[2], 0)
    ax7.legend(loc='lower right', handlelength=0.9)
    ax7.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True)

    # RMS buoyancy 
    ax8.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')
    ax8.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')
    ax8.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')
    ax8.set_xlim(ranges['b_rms'])
    ax8.plot(b_rms, z, color='black')
    ax8.set_title("Buoyancy Root Mean Square Error")
    ax8.set_xlabel(r"[m/s$^{2}$]")
    ax8.set_ylabel("Depth [m]")
    ax8.set_ylim(-lx[2], 0)
    ax8.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True)

    # RMS buoyancy flux 
    ax9.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')
    ax9.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')
    ax9.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')
    ax9.set_xlim(ranges['bw_fluc'])
    ax9.plot(bu_fluc_avg, z, color='black', label = r"b'u'")
    ax9.plot(bv_fluc_avg, z, color='blue', label = r"b'v'")
    ax9.plot(bw_fluc_avg, z, color='red', label = r"b'w'")
    ax9.legend(loc='lower right', handlelength=0.9)
    ax9.set_xlabel(r"[m$^{2}$/s$^{3}$]")
    #ax9.set_ylabel("Depth [m]")
    ax9.set_title("Buoyancy Flux Fluctuations")
    ax9.set_ylim(-lx[2], 0)
    ax9.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True) 

    # Perturbed density
    norm = mcolors.Normalize(vmin=ranges['rho_fluc'][0], vmax=ranges['rho_fluc'][-1])
    ax10.contourf(X[int(nx[1]/2), :, :], Z[int(nx[1]/2), :, :], rho_perturbed[int(nx[1]/2), :, :], levels, norm=norm, cmap='RdBu_r')
    #ax10.set_xlabel("[m]")
    #ax10.set_ylabel("Depth [m]")
    ax10.set_ylim(-lx[2], 0)
    ax10.set_xlim(0, lx[1])
    ax10.set_aspect('equal')
    mappable = cm.ScalarMappable(norm=norm, cmap='RdBu_r')
    cbar = fig.colorbar(mappable, ax=ax10, label=r"[kg/m$^3$]", shrink=0.9, pad=0.1)#, fraction=0.046, pad=0.1) #anchor = (0.5, -0.4), orientation='horizontal', 
    cbar.formatter.set_scientific(True)

    # Buoyancy fluctuations
    norm = mcolors.Normalize(vmin=ranges['b_fluc'][0], vmax=ranges['b_fluc'][-1])
    ax11.contourf(X[int(nx[1]/2), :, :], Z[int(nx[1]/2), :, :], b_fluc[int(nx[1]/2), :, :], levels, norm=norm, cmap='RdBu_r')
    ax11.set_xlabel("[m]")
    #ax11.set_ylabel("Depth [m]")
    ax11.set_ylim(-lx[2], 0)
    ax11.set_xlim(0, lx[1])
    ax11.set_aspect('equal')
    mappable = cm.ScalarMappable(norm=norm, cmap='RdBu_r')
    cbar = fig.colorbar(mappable, ax=ax11, label=r"[m/s$^{2}$]", shrink=0.9, pad=0.1)
    cbar.formatter.set_powerlimits((-3, 2))
    cbar.update_ticks()

    # --- Save Frame ---
    frame_path = os.path.join(outdir, f"oc_frame_{it:04d}.png")
    plt.tight_layout()
    plt.savefig(frame_path)
    plt.close(fig)
    print(f"Time step {it + 1} captured: {frame_path}")
    plt.close(fig)
    return outdir # return the directory where frames are saved for video creation
## dense tracer buoyancy analysis 
def plot_tracer_plume(time, it, ranges, fig_folder, lx, nx, z, zf, Y, Z, mld, u_avg, v_avg, w_avg, uv_fluc, uw_fluc, vw_fluc, u_rms, v_rms, w_rms, dbdx, dbdy, dbdz, b_avg, b_center, b_rms, bu_fluc_avg, bv_fluc_avg, bw_fluc_avg, b_fluc, rho_perturbed, tracer_avg, rp, plume_depths, ws, rhos, bw_flucs, l_scale):
    plume_depth_intrusion = plume_depths[0]
    plume_depth_neutral = plume_depths[1]

    w_intrusion = ws[0]
    w_neutral = ws[1]
    w_mld = ws[2]

    rho_perturbed_intrusion = rhos[0]
    rho_perturbed_neutral = rhos[1]
    rho_perturbed_mld = rhos[2]

    bwfluc_intrusion = bw_flucs[0]
    bwfluc_neutral = bw_flucs[1]
    bwfluc_mld = bw_flucs[2]

    outdir = os.path.join(fig_folder, 'NBP analysis/')
    os.makedirs(outdir, exist_ok=True)
    td = time[it] / 3600 / 24

    fig = plt.figure()
    fig.set_figheight(12)
    fig.set_figwidth(22)
    fig.suptitle(f'{td:.2f} days', fontsize=12) 

    ax1a = plt.subplot2grid(shape=(6, 10), loc=(0, 0), rowspan=2, colspan=2) # velocity profiles
    ax2a = plt.subplot2grid(shape=(6, 10), loc=(0, 2), rowspan=2, colspan=2) # velocity rms w/ vertical velocity of plume
    ax3a = plt.subplot2grid(shape=(6, 10), loc=(0, 4), rowspan=2, colspan=2) # Reynolds stress
    ax4a = plt.subplot2grid(shape=(6, 10), loc=(0, 6), rowspan=2, colspan=2) # buoyancy gradients
    ax5a = plt.subplot2grid(shape=(6, 10), loc=(0, 8), rowspan=2, colspan=2) # buoyancy flux u, v, w
    ax1 = plt.subplot2grid(shape=(6, 10), loc=(2, 0), rowspan=2, colspan=2) # plume depths
    ax2 = plt.subplot2grid(shape=(6, 10), loc=(2, 2), rowspan=2, colspan=2) # w at depth
    ax3 = plt.subplot2grid(shape=(6, 10), loc=(2, 4), rowspan=2, colspan=2) # perturbed density at depth
    ax4 = plt.subplot2grid(shape=(6, 10), loc=(2, 6), rowspan=2, colspan=2) # perturbed buoyancy flux at depth
    ax5 = plt.subplot2grid(shape=(6, 10), loc=(2, 8), rowspan=2, colspan=2) # entrainment coefficient
    ax6 = plt.subplot2grid(shape=(6, 10), loc=(4, 0), rowspan=2, colspan=2) # max radius of plume 
    ax7 = plt.subplot2grid(shape=(6, 10), loc=(4, 2), rowspan=2, colspan=2) # buoyancy profiles
    ax8 = plt.subplot2grid(shape=(6, 10), loc=(4, 4), rowspan=2, colspan=2) # buoyancy rms
    ax9 = plt.subplot2grid(shape=(6, 10), loc=(4, 6), rowspan=2, colspan=2) # tracer plane slice with plume contour
    ax10 = plt.subplot2grid(shape=(6, 10), loc=(4, 8), rowspan=1, colspan=2) # perturbed density place slice
    ax11 = plt.subplot2grid(shape=(6, 10), loc=(5, 8), rowspan=1, colspan=2, sharex=ax10) # perturbed buoyancy plane slice

    fig.subplots_adjust(hspace=0.05)

    levels = 500

    # velocity profiles
    ax1a.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')#, label = "MLD")
    ax1a.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')#, label = "Intrusion Depth")
    ax1a.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')#, label = "Neutral Buoyancy")
    ax1a.plot(u_avg, z, label=r"$\langle$u$\rangle_{xy}$", color='green')
    ax1a.plot(v_avg, z, label=r"$\langle$v$\rangle_{xy}$", color='red')
    ax1a.plot(w_avg, zf, label=r"$\langle$w$\rangle_{xy}$", color='blue')
    ax1a.set_xlabel("[m/s]")
    ax1a.set_ylabel("Depth [m]")
    ax1a.set_title('Velocity Profiles')
    ax1a.set_ylim(-lx[2], 0)
    ax1a.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True)
    ax1a.set_xlim(ranges['vel'])
    ax1a.legend(loc='lower right', handlelength=0.9)

    # velocity rms
    ax2a.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')
    ax2a.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')
    ax2a.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')
    ax2a.plot(u_rms, z, label=r"$\langle$u$_{rms}$$\rangle_{xy}$", color='green')
    ax2a.plot(v_rms, z, label=r"$\langle$v$_{rms}$$\rangle_{xy}$", color='red')
    ax2a.plot(w_rms, zf, label=r"$\langle$w$_{rms}$$\rangle_{xy}$", color='blue')
    ax2a.set_xlabel("[m/s]")
    #ax2a.set_ylabel("Depth [m]")
    ax2a.set_title("Root Mean Square Velocities")
    ax2a.set_ylim(-lx[2], 0)
    ax2a.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True)
    ax2a.set_xlim(ranges['vel_rms'])
    ax2a.legend(loc='lower right', handlelength=0.9)

    # reynolds stresses 
    ax3a.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')
    ax3a.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')
    ax3a.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')
    ax3a.plot(uv_fluc, z, label=r"$\langle$uv$\rangle_{xy}$", color='green')
    ax3a.plot(uw_fluc, z, label=r"$\langle$uw$\rangle_{xy}$", color='red')
    ax3a.plot(vw_fluc, z, label=r"$\langle$vw$\rangle_{xy}$", color='blue')
    ax3a.set_xlabel(r"[m$^2$/s$^2$]")
    #ax3a.set_ylabel("Depth [m]")
    ax3a.set_title("Reynolds Stresses")
    ax3a.set_ylim(-lx[2], 0)
    ax3a.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True)
    ax3a.set_xlim(ranges['restress'])
    ax3a.legend(loc='lower right', handlelength=0.9)

    # buoyancy gradients 
    ax4a.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')
    ax4a.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')
    ax4a.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')
    ax4a.plot(dbdx, z, color='green', label = r"db/dx")
    ax4a.plot(dbdy, z, label=r"db/dy", color='red')
    ax4a.plot(dbdz, z, label=r"db/dz", color='blue')
    ax4a.set_xlabel(r"[1/s$^2$]")
    ax4a.set_title("Buoyancy Gradient")
    ax4a.set_ylim(-lx[2], 0)
    ax4a.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True)
    ax4a.set_xlim(ranges['gradb'])
    ax4a.legend(loc='lower left', handlelength=0.9)

    # perturbed buoyancy flux 
    ax5a.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')
    ax5a.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')
    ax5a.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')
    ax5a.set_xlim(ranges['bw_fluc'])
    ax5a.plot(bu_fluc_avg, z, color='green', label = r"b'u'")
    ax5a.plot(bv_fluc_avg, z, color='blue', label = r"b'v'")
    ax5a.plot(bw_fluc_avg, z, color='red', label = r"b'w'")
    ax5a.legend(loc='lower right', handlelength=0.9)
    ax5a.set_xlabel(r"[m$^{2}$/s$^{3}$]")
    ax5a.set_title("Buoyancy Flux Fluctuations")
    ax5a.set_ylim(-lx[2], 0)
    ax5a.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True) 

    # Depth of plume through time
    ax1.plot(time/ 3600 / 24, -mld*np.ones(len(time)), linestyle='--', linewidth = 0.5, color = 'black', label = "MLD")
    ax1.plot(time[:it+1]/ 3600 / 24, plume_depth_intrusion, color = 'cornflowerblue', label = r"h$_{\text{intrusion}}$")
    ax1.plot(time[:it+1]/ 3600 / 24, plume_depth_neutral, color = 'mediumblue', label = r"h$_{\text{neutral}}$")
    ax1.set_xlabel("Time [days]") 
    ax1.set_ylabel("[m]")
    ax1.set_title("Plume Depths")
    ax1.set_ylim(ymin = -lx[-1], ymax = 0)
    ax1.legend(loc='upper right', handlelength=0.9)
    ax1.set_xlim([0, time.max() / 3600 / 24])  

    # vertical velocity
    ax2.plot(time[:it+1]/ 3600 / 24, w_mld, color = 'black', label = r"w$_{\text{MLD}}$")
    ax2.plot(time[:it+1]/ 3600 / 24, w_intrusion, color = 'cornflowerblue',label = r"w$_{\text{intrusion}}$")
    ax2.plot(time[:it+1]/ 3600 / 24, w_neutral, color = 'mediumblue', label = r"w$_{\text{neutral}}$")
    ax2.set_xlabel("Time [days]") 
    ax2.set_ylabel("[m/s]")
    ax2.set_title("w at Depth")
    ax2.set_ylim(ymin = ranges['w'][0]*10.1, ymax = ranges['w'][-1]*10.1)
    ax2.legend(loc='upper right', handlelength=0.9)
    ax2.set_xlim([0, time.max() / 3600 / 24])  

    # density perturbations 
    ax3.plot(time[:it+1]/ 3600 / 24, rho_perturbed_mld, color = 'black', label = r"$\rho$'$_{\text{MLD}}$")
    ax3.plot(time[:it+1]/ 3600 / 24, rho_perturbed_intrusion, color = 'cornflowerblue', label = r"$\rho$'$_{\text{intrusion}}$")
    ax3.plot(time[:it+1]/ 3600 / 24, rho_perturbed_neutral, color = 'mediumblue', label = r"$\rho$'$_{\text{neutral}}$")
    ax3.set_xlabel("Time [days]") 
    ax3.set_ylabel(r"[kg/m$^3$]")
    ax3.set_title("Perturbed Density at Depth")
    ax3.legend(loc='upper right', handlelength=0.9)
    ax3.set_ylim(ymin = ranges['rho_fluc'][0]*4, ymax = ranges['rho_fluc'][-1]*4)
    ax3.set_xlim([0, time.max() / 3600 / 24])

    # buoyancy perturbations 
    ax4.plot(time[:it+1]/ 3600 / 24, bwfluc_mld, color = 'black', label = r"b'w$_{\text{MLD}}$")
    ax4.plot(time[:it+1]/ 3600 / 24, bwfluc_intrusion, color = 'cornflowerblue', label = r"b'w$_{\text{intrusion}}$")
    ax4.plot(time[:it+1]/ 3600 / 24, bwfluc_neutral, color = 'mediumblue', label = r"b'w$_{\text{neutral}}$")
    ax4.set_xlabel("Time [days]") 
    ax4.set_ylabel(r"[m$^{2}$/s$^{3}$]")
    ax4.set_title("Perturbed Buoyancy Flux at Depth")
    ax4.legend(loc='lower right', handlelength=0.9)
    ax4.set_ylim(ymin = ranges['bw_fluc'][0]*10**(3), ymax = ranges['bw_fluc'][-1]*10**(3))
    ax4.set_xlim([0, time.max() / 3600 / 24])

    # length scale ratio 
    ax5.plot(time[:it+1]/ 3600 / 24, l_scale)
    ax5.set_xlabel("Time [days]") 
    ax5.set_ylabel(r"r$_p$/h$_intrusion$")
    ax5.set_title("Lengthscale ratio")
    ax5.set_ylim(ymin = 0.0, ymax = 1.0)
    ax5.set_xlim([0, time.max() / 3600 / 24])

    # radius of plume 
    ax6.plot(time[:it+1]/ 3600 / 24, rp, color = 'black')
    ax6.set_xlim([0, time.max() / 3600 / 24])
    ax6.set_title("Horizontal Plume Spread")
    ax6.set_xlabel("Time [days]") 
    ax6.set_ylabel("Radius [m]")
    ax6.set_ylim(0, lx[0]/4)
    ax6.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True)

    # buoyancy
    ax7.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')
    ax7.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')
    ax7.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')
    ax7.set_xlim(ranges['b'])
    ax7.plot(b_avg, z, color='green', label = r"$\langle$b$\rangle_{\text{xy}}")
    ax7.plot(b_center, z, color='red', label = r"b$_{\text{centerline}}$")
    ax7.set_title("Buoyancy Profile")
    ax7.set_xlabel("[m/s$^{2}$]")
    ax7.set_ylabel("Depth [m]")
    ax7.set_ylim(-lx[2], 0)
    ax7.legend(loc='upper left', handlelength=0.9)
    ax7.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True)

    # RMS buoyancy 
    ax8.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')
    ax8.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')
    ax8.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')
    ax8.set_xlim(ranges['b_rms'])
    ax8.plot(b_rms, z, color='black')
    ax8.set_title("Buoyancy Root Mean Square Error")
    ax8.set_xlabel(r"[m/s$^{2}$]")
    ax8.set_ylim(-lx[2], 0)
    ax8.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True)

    # tracer average with depth  
    ax9.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')
    ax9.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')
    ax9.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')
    ax9.plot(tracer_avg, z, color='black')
    ax9.set_xlabel(r"S/S$_{source}$")
    ax9.set_ylabel("Depth [m]")
    ax9.set_title("Tracer")
    ax9.set_ylim(-lx[2], 0)
    ax9.set_xlim(xmin = ranges['S'][0], xmax = ranges['S'][-1]*10**(-2))
    ax9.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True)

    # Perturbed density plane slice
    norm = mcolors.Normalize(vmin=ranges['rho_fluc'][0], vmax=ranges['rho_fluc'][-1])
    ax10.contourf(Y[:, int(nx[1]/2), :], Z[:, int(nx[1]/2), :], rho_perturbed[:, int(nx[1]/2), :], levels, norm=norm, cmap='RdBu_r')
    ax10.set_ylim(-lx[2], 0)
    ax10.set_xlim(0, lx[1])
    ax10.set_ylabel("Depth [m]")
    ax10.set_aspect('equal')
    ax10.set_title("Perturbed Density")
    mappable = cm.ScalarMappable(norm=norm, cmap='RdBu_r')
    cbar = fig.colorbar(mappable, ax=ax10, label=r"[kg/m$^3$]", shrink=0.9, pad=0.1)#, fraction=0.046, pad=0.1) #anchor = (0.5, -0.4), orientation='horizontal', 
    cbar.formatter.set_scientific(True)

    # Buoyancy fluctuations plane slice
    norm = mcolors.Normalize(vmin=ranges['b_fluc'][0], vmax=ranges['b_fluc'][-1])
    ax11.contourf(Y[:, int(nx[1]/2), :], Z[:, int(nx[1]/2), :], b_fluc[:, int(nx[1]/2), :], levels, norm=norm, cmap='RdBu_r')
    ax11.set_xlabel("[m]")
    ax11.set_ylim(-lx[2], 0)
    ax11.set_xlim(0, lx[1])
    ax11.set_aspect('equal')
    ax11.set_title("Perturbed Buoyancy")
    mappable = cm.ScalarMappable(norm=norm, cmap='RdBu_r')
    cbar = fig.colorbar(mappable, ax=ax11, label=r"[m/s$^{2}$]", shrink=0.9, pad=0.1)
    cbar.formatter.set_powerlimits((-3, 2))
    cbar.update_ticks()

    # --- Save Frame ---
    frame_path = os.path.join(outdir, f"oc_frame_{it:04d}.png")
    plt.tight_layout()
    plt.savefig(frame_path)
    plt.close(fig)
    print(f"Time step {it + 1} captured: {frame_path}")
    plt.close(fig)
    return outdir # return the directory where frames are saved for video creation
## dense tracer buoyancy analysis via momentum
def plot_momentum_plume(time, it, ranges, fig_folder, lx, z, zf, mld, b_avg, tracer_avg, u_rms, v_rms, w_rms, b_rms, bu_fluc_avg, bv_fluc_avg, bw_fluc_avg, tracer_fluc, T_fluc, Q, M, F, B, wm, dm, bm, Ri, r_profile, b_center, plume_depths, ND = False):
    plume_depth_intrusion = plume_depths[0]
    plume_depth_neutral = plume_depths[1]

    outdir = os.path.join(fig_folder, 'plume momentum analysis/')
    os.makedirs(outdir, exist_ok=True)
    td = time[it] / 3600 / 24
    gridspec_kw={'height_ratios': [1, 1, 1, 1, 0.02]} # add space for universal legend
    fig, ax = plt.subplots(5, 4, figsize=(15, 18), gridspec_kw=gridspec_kw)
    for a in ax[4, :]:
        a.remove()
    fig.suptitle(f'{td:.2f} days', fontsize=12) 

    ax1 = ax[0, 0]
    ax2 = ax[0, 1]
    ax3 = ax[0, 2]
    ax4 = ax[0, 3]
    ax5 = ax[1, 0]
    ax6 = ax[1, 1]
    ax7 = ax[1, 2]
    ax8 = ax[1, 3]
    ax9 = ax[2, 0] # Q - volume flux
    ax10 = ax[2, 1] # M - momentum flux
    ax11 = ax[2, 2] # F - buoyancy flux
    ax12 = ax[2, 3] # B - buoyancy integral
    ax13 = ax[3, 0] # wm - characteristic velocity
    ax14 = ax[3, 1] # dm - characteristic length
    ax15 = ax[3, 2] # bm - characteristic buoyancy
    ax16 = ax[3, 3] # Ri - Richardson number

    if ND:
        ax1.set_ylabel(r"z/h$_{\text{MLD_{0}}}$")
        ax1.set_xlabel(r"u$_{i, rms}$/(h$_{\text{MLD_{0}}} \sqrt{N^{2}}$)")
        #ax2.set_ylabel(r"z/h$_{\text{MLD_{0}}}$")
        ax2.set_xlabel(r"S$(\text{h}_{mld} \sqrt{N^{2}}$)/(J$^{\text{S}}$)") #(r"S$\sqrt{g\text{r}_j}$/(J$^{\text{S}}$)")
        #ax3.set_ylabel(r"z/h$_{\text{MLD_{0}}}$")
        ax3.set_xlabel(r"b/(h$_{\text{MLD_{0}}} N^{2}$)")
        #ax4.set_ylabel(r"z/h$_{\text{MLD_{0}}}$")
        ax4.set_xlabel(r"S$'(\text{h}_{mld} \sqrt{N^{2}}$)/(J$^{\text{S}}$)") #(r"S$'\sqrt{g\text{r}_j}$/(J$^{\text{S}}$)")
        ax5.set_ylabel(r"z/h$_{\text{MLD_{0}}}$")
        ax5.set_xlabel(r"r/h$_{\text{MLD_{0}}}$") #(r"r/r$_{j}$")
        #ax6.set_ylabel(r"z/h$_{\text{MLD_{0}}}$")
        ax6.set_xlabel(r"$\langle$b'u'$_{i}\rangle_{\text{xy}}$/(h$_{\text{MLD_{0}}}^2 (N^{2})^{3/2}$)")
        #ax7.set_ylabel(r"z/h$_{\text{MLD_{0}}}$")
        ax7.set_xlabel(r"b$_{rms}$/(h$_{\text{MLD_{0}}} N^{2}$)")

        #ax8.set_ylabel(r"z/h$_{\text{MLD_{0}}}$")
        ax8.set_xlabel(r"$\langle$T$'\rangle_{\text{xy}}$/T$_{0}$")
    else:
        ax1.set_ylabel("Depth [m]")
        ax1.set_xlabel("[m/s]")
        #ax2.set_ylabel("[m]")
        ax2.set_xlabel(r"$\langle$C$\rangle_{\text{xy}}$[g/kg]")
        #ax3.set_ylabel("[m]")
        ax3.set_xlabel(r"[m/s$^2$]")
        #ax4.set_ylabel("[m]")
        ax4.set_xlabel(r"$\langle$C$'\rangle_{\text{xy}}$[g/kg]")
        ax5.set_ylabel("Depth [m]")
        ax5.set_xlabel("[m]")
        #ax6.set_ylabel("[m]")
        ax6.set_xlabel(r"[m$^2$/s$^3$]")
        #ax7.set_ylabel("[m]")
        ax7.set_xlabel(r"[m/s$^2$]")
        #ax8.set_ylabel("[m]")
        ax8.set_xlabel(r"$\langle$T$'\rangle_{\text{xy}}$[$^{\circ}$ C]")
        ax9.set_xlabel("[m]")
        ax9.set_xlabel(r"[m$^3$/s]")
        ax10.set_xlabel(r"[m$^4$/s$^2$]")
        ax11.set_xlabel(r"[m$^4$/s$^3$]")
        ax12.set_xlabel(r"[m$^3$/s$^2$]")
        ax13.set_xlabel(r"[m/s]")
        ax14.set_xlabel(r"[m]")
        ax15.set_xlabel(r"[-]")
        ax16.set_xlabel(r"[-]")

    # velocity rms
    mld_handle, = ax1.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')#, label = "MLD")
    intrusion_handle, = ax1.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')#, label = "Intrusion Depth")
    neutral_handle, = ax1.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')#, label = "Neutral Buoyancy")
    ax1.plot(u_rms, z, label=r"$\langle$u$_{\text{rms}}\rangle_{\text{xy}}$", color = 'blue')
    ax1.plot(v_rms, z, label=r"$\langle$v$_{\text{rms}}\rangle_{\text{xy}}$", color = 'green')
    ax1.plot(w_rms, zf, label=r"$\langle$w$_{\text{rms}}\rangle_{\text{xy}}$", color = 'red')
    ax1.set_title("Root Mean Square Velocities")
    ax1.set_ylim(-lx[2], 0)
    ax1.set_xlim(ranges['vel_rms'])
    ax1.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True)
    ax1.legend(loc='lower right')

    # tracer profile 
    ax2.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')#, label = "MLD")
    ax2.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')#, label = "Intrusion Depth")
    ax2.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')#, label = "Neutral Buoyancy")
    ax2.plot(tracer_avg, z, color = 'black')
    ax2.set_title('Tracer')
    ax2.set_ylim(-lx[2], 0)
    ax2.set_xlim(ranges['S'])
    ax2.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True)

    # buoyancy profiles
    ax3.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')#, label = "MLD")
    ax3.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')#, label = "Intrusion Depth")
    ax3.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')#, label = "Neutral Buoyancy")
    ax3.plot(b_avg, z, color = 'black', label = r"$\langle$b$\rangle_{\text{xy}}$")
    ax3.plot(b_center, z, color = 'black', label = r"b$_{\text{centerline}}$", linestyle = 'dashed')
    ax3.set_title("Buoyancy Profile")
    ax3.set_ylim(-lx[2], 0)
    ax3.set_xlim(ranges['b_avg'])
    if ND:
        ax3.legend(loc='lower right')
    else:
        ax3.legend(loc='upper left')
    ax3.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True)

    # temperature fluctuations 
    ax4.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')#, label = "MLD")
    ax4.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')#, label = "Intrusion Depth")
    ax4.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')#, label = "Neutral Buoyancy")
    ax4.plot(tracer_fluc, z, color = 'black')
    ax4.set_title("Perturbed Tracer")
    ax4.set_ylim(-lx[2], 0)
    ax4.set_xlim(ranges['S_fluc'])
    ax4.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)

    # tracer plume radius
    ax5.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')#, label = "MLD")
    ax5.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')#, label = "Intrusion Depth")
    ax5.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')#, label = "Neutral Buoyancy")
    ax5.plot(r_profile, z, color = 'black')
    ax5.set_title("Tracer Plume Radius with Depth")
    ax5.set_ylim(-lx[2], 0)
    ax5.set_xlim(0, lx[0]/4)

    # perturbed buoyancy flux 
    ax6.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')#, label = "MLD")
    ax6.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')#, label = "Intrusion Depth")
    ax6.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')#, label = "Neutral Buoyancy")
    ax6.plot(bu_fluc_avg, z, color = 'blue', label = r"$\langle$b'u'$\rangle_{\text{xy}}$")
    ax6.plot(bv_fluc_avg, z, color = 'green', label = r"$\langle$b'v'$\rangle_{\text{xy}}$")
    ax6.plot(bw_fluc_avg, z, color = 'red', label = r"$\langle$b'w'$\rangle_{\text{xy}}$")
    ax6.legend(loc='lower right')
    ax6.set_title("Buoyancy Flux Fluctuations")
    ax6.set_ylim(-lx[2], 0)
    ax6.set_xlim(ranges['bw_fluc'])
    ax6.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True) 

    # buoyancy brms 
    ax7.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')#, label = "MLD")
    ax7.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')#, label = "Intrusion Depth")
    ax7.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')#, label = "Neutral Buoyancy")
    ax7.plot(b_rms, z, color = 'black')
    ax7.set_title("Root Mean Square Buoyancy")
    ax7.set_ylim(-lx[2], 0)
    ax7.set_xlim(ranges['b_rms'])
    ax7.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True)

    # temperature fluctuations 
    ax8.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')#, label = "MLD")
    ax8.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')#, label = "Intrusion Depth")
    ax8.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')#, label = "Neutral Buoyancy")
    ax8.plot(T_fluc, z, color = 'black')
    ax8.set_title("Perturbed Temperature")
    ax8.set_ylim(-lx[2], 0)
    ax8.set_xlim(ranges['T_fluc'])

    # Q
    ax9.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')#, label = "MLD")
    ax9.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')#, label = "Intrusion Depth")
    ax9.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')#, label = "Neutral Buoyancy")
    ax9.plot(Q, z, color = 'black')
    ax9.set_title("Volume Flux")
    ax9.set_ylim(-lx[2], 0)
    ax9.set_xlim(ranges['Q'])

    # M
    ax10.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')#, label = "MLD")
    ax10.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')#, label = "Intrusion Depth")
    ax10.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')#, label = "Neutral Buoyancy")
    ax10.plot(M, z, color = 'black')
    ax10.set_title("Momentum Flux")
    ax10.set_ylim(-lx[2], 0)
    ax10.set_xlim(ranges['M'])

    # F perturbed 
    ax11.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')#, label = "MLD")
    ax11.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')#, label = "Intrusion Depth")
    ax11.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')#, label = "Neutral Buoyancy")
    ax11.plot(F, z, color = 'black')
    ax11.set_title("Buoyancy Perturbed Flux")
    ax11.set_ylim(-lx[2], 0)
    ax11.set_xlim(xmin = ranges['F'][0], xmax = ranges['F'][-1])

    # B
    ax12.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')#, label = "MLD")
    ax12.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')#, label = "Intrusion Depth")
    ax12.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')#, label = "Neutral Buoyancy")
    ax12.plot(B, z, color = 'black')
    ax12.set_title("Buoyancy Integral")
    ax12.set_ylim(-lx[2], 0)
    ax12.set_xlim(xmin = ranges['B'][0], xmax = ranges['B'][-1])

    # wm
    ax13.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')#, label = "MLD")
    ax13.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')#, label = "Intrusion Depth")
    ax13.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')#, label = "Neutral Buoyancy")
    ax13.plot(wm, z, color = 'red')
    ax13.set_title("Characteristic Velocity")
    ax13.set_ylim(-lx[2], 0)
    ax13.set_xlim(xmin = ranges['w'][0]*10.0, xmax = ranges['w'][-1]*10.0)

    # dm
    ax14.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')#, label = "MLD")
    ax14.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')#, label = "Intrusion Depth")
    ax14.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')#, label = "Neutral Buoyancy")
    ax14.plot(dm, z, color = 'black')
    ax14.set_title("Characteristic Length")
    ax14.set_ylim(-lx[2], 0)
    ax14.set_xlim(xmin = 0, xmax = lx[0]/4)

    # Ri
    ax15.plot([-1*10**6, 1*10**6], -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')#, label = "MLD")
    ax15.plot([-1*10**6, 1*10**6], plume_depth_intrusion[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')#, label = "Intrusion Depth")
    ax15.plot([-1*10**6, 1*10**6], plume_depth_neutral[it]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')#, label = "Neutral Buoyancy")
    ax15.plot(Ri, z, color = 'black')
    ax15.set_title("Richardson Number")
    ax15.set_ylim(-lx[2], 0)
    ax15.set_xlim(ranges['Ri'])

    fig.legend(handles=[mld_handle, intrusion_handle, neutral_handle], labels=["MLD", "Intrusion Depth", "Neutral Buoyancy Depth"],
            loc='lower center',
            ncol=3,
            bbox_to_anchor=(0.52, 0.015))

    # --- Save Frame ---
    frame_path = os.path.join(outdir, f"plume_momentum_analysis_{it:04d}.png")
    plt.savefig(frame_path)
    plt.close(fig)
    print(f"Time step {it + 1} captured: {frame_path}")

    return outdir # return the directory where frames are saved for video creation

### -------------------------PLOTTING ND FUNCTIONS------------------------- ###
## ND Richardson 
def plot_rig_exponents(color_opt, title, file_name, fig_folder, w_rms, b_center, bw, rp, T, S, z_nd, Ri_g, case_names, exponents = [-0.5, -1/3, -0.25, 0.0, 0.25, 1/3, 0.5], z_str = rf"(z-h$_{{ML}}$)h$_{{ML}}^{{1/3}}$/L$_N^{{4/3}}$"):
    num_cases = len(case_names)
    scale = np.ones(7) 
    scale[-1] = 0.02
    gridspec_kw={'height_ratios': scale}
    fig, axes = plt.subplots(7, len(exponents), figsize=(len(exponents)*3, 25), sharey=True, gridspec_kw = gridspec_kw)
    plt.subplots_adjust(top=0.9)
    for a in axes[-1, :]:
        a.remove()
    case_handles = [Line2D([0], [0], color=color_opt[i], linestyle='solid', label=case_names[i])for i in range(num_cases)]
    fig.legend(handles=case_handles,
            loc='lower center',
            ncol=num_cases,
            bbox_to_anchor=(0.52, 0.005), fontsize = 16)
    fig.suptitle(title, fontsize = 20, y = 0.99)
    """
    axes[0, :] = ND rms velocity vs z_nd varied exponent of Ri_g
    axes[1, :] = ND centerline buoyancy vs z_nd varied exponent of Ri_g
    axes[2, :] = ND average buoyancy flux vs z_nd varied exponent of Ri_g
    axes[3, :] = ND radius of plume vs z_nd varied exponent of Ri_g
    axes[4, :] = ND perturbed temperature vs z_nd varied exponent of Ri_g
    axes[5, :] = ND average salinity vs z_nd varied exponent of Ri_g
    """
    for ax in axes[:, 0]:
        ax.set_ylabel(z_str, fontsize = 16)
    for ax, exp in zip(axes[0, :], exponents):
        for i in range(num_cases):
            correction = Ri_g[i]**exp
            ax.plot(w_rms[:, i] * (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'Ri$_g^{{{exp:.2f}}}$', fontsize = 16)
        ax.set_xlabel(rf"$w_{{rms}}/\sqrt{{\text{{g r}}_{{j}}}} \cdot Ri_g^{{{exp:.2f}}}$", fontsize = 16)

    for ax, exp in zip(axes[1, :], exponents):
        for i in range(num_cases):
            correction = Ri_g[i]**exp
            ax.plot(b_center[:, i] *(correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'Ri$_g^{{{exp:.2f}}}$', fontsize = 16)
        ax.set_xlabel(rf"$b_{{\text{{centerline}}}}/g \cdot Ri_g^{{{exp:.2f}}}$", fontsize = 16)

    for ax, exp in zip(axes[2, :], exponents):
        for i in range(num_cases):
            correction = Ri_g[i]**exp
            ax.plot(bw[:, i] * (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'Ri$_g^{{{exp:.2f}}}$', fontsize = 16)
        ax.set_xlabel(rf"$(\langle b'w'\rangle_{{xy}}/\sqrt{{\text{{g}}^3 \text{{r}}_{{j}}}})\cdot Ri_g^{{{exp:.2f}}}$", fontsize = 16)

    for ax, exp in zip(axes[3, :], exponents):
        for i in range(num_cases):
            correction = Ri_g[i]**exp
            ax.plot(rp[:, i] * (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'Ri$_g^{{{exp:.2f}}}$', fontsize = 16)
        ax.set_xlabel(rf"(r/r$_{{j}})\cdot Ri_g^{{{exp:.2f}}}$", fontsize = 16)
    
    for ax, exp in zip(axes[4, :], exponents):
        for i in range(num_cases):
            correction = Ri_g[i]**exp
            ax.plot(T[:, i] * (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'Ri$_g^{{{exp:.2f}}}$', fontsize = 16)
        ax.set_xlabel(rf"$(\text{{T'}}_{{\text{{centerline}}}}\alpha)\cdot Ri_g^{{{exp:.2f}}}$", fontsize = 16)
    
    for ax, exp in zip(axes[5, :], exponents):
        for i in range(num_cases):
            correction = Ri_g[i]**exp
            ax.plot(S[:, i] * (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'Ri$_g^{{{exp:.2f}}}$', fontsize = 16)
        ax.set_xlabel(rf"($\langle$C$\rangle_{{\text{{xy}}}} \beta)\cdot Ri_g^{{{exp:.2f}}}$", fontsize = 16)
    #axes[5, 0].legend()

    plt.tight_layout()

    # --- Save Frame ---
    str_exp = list(map(str, exponents))
    str_exp = '_'.join(f"{x:.3g}" for x in exponents)
    frame_path = os.path.join(fig_folder, f"Ri_{file_name} _pow{str_exp}.png")
    plt.savefig(frame_path)
    plt.close(fig)
## ND Froude
def plot_Fr_exponents(color_opt, title, file_name, fig_folder, w_rms, b_center, bw, rp, T, S, z_nd, Fr, case_names, exponents = [-0.5, -1/3, -0.25, 0.0, 0.25, 1/3, 0.5], z_str = rf"(z-h$_{{ML}}$)h$_{{ML}}^{{1/3}}$/L$_N^{{4/3}}$"):
    num_cases = len(case_names)
    scale = np.ones(7) 
    scale[-1] = 0.02
    gridspec_kw={'height_ratios': scale}
    fig, axes = plt.subplots(7, len(exponents), figsize=(len(exponents)*3, 25), sharey=True, gridspec_kw = gridspec_kw)
    plt.subplots_adjust(top=0.9)
    for a in axes[-1, :]:
        a.remove()
    case_handles = [Line2D([0], [0], color=color_opt[i], linestyle='solid', label=case_names[i])for i in range(num_cases)]
    fig.legend(handles=case_handles,
            loc='lower center',
            ncol=num_cases,
            bbox_to_anchor=(0.52, 0.005), fontsize = 16)
    fig.suptitle(title, fontsize = 20, y = 0.99)
    """
    axes[0, :] = ND rms velocity vs z_nd varied exponent of Fr
    axes[1, :] = ND centerline buoyancy vs z_nd varied exponent of Fr
    axes[2, :] = ND average buoyancy flux vs z_nd varied exponent of Fr
    axes[3, :] = ND radius of plume vs z_nd varied exponent of Fr
    axes[4, :] = ND perturbed temperature vs z_nd varied exponent of Fr
    axes[5, :] = ND average salinity vs z_nd varied exponent of Fr
    """
    for ax in axes[:, 0]:
        ax.set_ylabel(z_str, fontsize = 16)
    for ax, exp in zip(axes[0, :], exponents):
        for i in range(num_cases):
            correction = Fr[i]**exp
            ax.plot(w_rms[:, i] * (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'Fr$^{{{exp:.2f}}}$', fontsize = 16)
        ax.set_xlabel(rf"$w_{{rms}}/\sqrt{{\text{{g r}}_{{j}}}} \cdot Fr^{{{exp:.2f}}}$", fontsize = 16)
    
    for ax, exp in zip(axes[1, :], exponents):
        for i in range(num_cases):
            correction = Fr[i]**exp
            ax.plot(b_center[:, i] * (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'Fr$^{{{exp:.2f}}}$', fontsize = 16)
        ax.set_xlabel(rf"$b_{{\text{{centerline}}}}/g \cdot Fr^{{{exp:.2f}}}$", fontsize = 16)

    for ax, exp in zip(axes[2, :], exponents):
        for i in range(num_cases):
            correction = Fr[i]**exp
            ax.plot(bw[:, i] * (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'Fr$^{{{exp:.2f}}}$', fontsize = 16)
        ax.set_xlabel(rf"$(\langle b'w'\rangle_{{xy}}/\sqrt{{\text{{g}}^3 \text{{r}}_{{j}}}})\cdot Fr^{{{exp:.2f}}}$", fontsize = 16)
    #axes[2, 0].legend()

    for ax, exp in zip(axes[3, :], exponents):
        for i in range(num_cases):
            correction = Fr[i]**exp
            ax.plot(rp[:, i] * (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'Fr$^{{{exp:.2f}}}$', fontsize = 16)
        ax.set_xlabel(rf"(r/r$_{{j}})\cdot Fr^{{{exp:.2f}}}$", fontsize = 16)
    #axes[3, 0].legend()

    for ax, exp in zip(axes[4, :], exponents):
        for i in range(num_cases):
            correction = Fr[i]**exp
            ax.plot(T[:, i] * (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'Fr$^{{{exp:.2f}}}$', fontsize = 16)
        ax.set_xlabel(rf"$(\text{{T'}}_{{\text{{centerline}}}}\alpha)\cdot Fr^{{{exp:.2f}}}$", fontsize = 16)
    #axes[4, 0].legend()

    for ax, exp in zip(axes[5, :], exponents):
        for i in range(num_cases):
            correction = Fr[i]**exp
            ax.plot(S[:, i] * (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'Fr$^{{{exp:.2f}}}$', fontsize = 16)
        ax.set_xlabel(rf"($\langle$C$\rangle_{{\text{{xy}}}} \beta)\cdot Fr^{{{exp:.2f}}}$", fontsize = 16)

    plt.tight_layout()

    # --- Save Frame ---
    str_exp = '_'.join(f"{x:.3g}" for x in exponents)
    frame_path = os.path.join(fig_folder, f"Fr_{file_name} _pow{str_exp}.png")
    plt.savefig(frame_path)
    plt.close(fig)
## ND MLD
def plot_mld_exponents(color_opt, title, file_name, fig_folder, w_rms, b_center, bw, rp, T, S, z_nd, mld, case_names, exponents = [-0.5, -1/3, -0.25, 0.0, 0.25, 1/3, 0.5], z_str = rf"(z-h$_{{ML}}$)h$_{{ML}}^{{1/3}}$/L$_N^{{4/3}}$"):
    num_cases = len(case_names)
    scale = np.ones(7) 
    scale[-1] = 0.02
    gridspec_kw={'height_ratios': scale}
    fig, axes = plt.subplots(7, len(exponents), figsize=(len(exponents)*3, 25), sharey=True, gridspec_kw = gridspec_kw)
    plt.subplots_adjust(top=0.9)
    for a in axes[-1, :]:
        a.remove()
    case_handles = [Line2D([0], [0], color=color_opt[i], linestyle='solid', label=case_names[i])for i in range(num_cases)]
    fig.legend(handles=case_handles,
            loc='lower center',
            ncol=num_cases,
            bbox_to_anchor=(0.52, 0.005), fontsize = 16)
    fig.suptitle(title, fontsize = 20, y = 0.99)
    """
    axes[0, :] = ND rms velocity vs z_nd varied exponent of MLD
    axes[1, :] = ND centerline buoyancy vs z_nd varied exponent of MLD
    axes[2, :] = ND average buoyancy flux vs z_nd varied exponent of MLD
    axes[3, :] = ND radius of plume vs z_nd varied exponent of MLD
    axes[4, :] = ND perturbed temperature vs z_nd varied exponent of MLD
    axes[5, :] = ND average salinity vs z_nd varied exponent of MLD
    """

    for ax in axes[:, 0]:
        ax.set_ylabel(z_str, fontsize = 16)

    for ax, exp in zip(axes[0, :], exponents):
        for i in range(num_cases):
            correction = mld[i]**exp
            ax.plot(w_rms[:, i] * (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'$\hat{{h}}_{{ML}}^{{{exp:.2f}}}$', fontsize = 16)
        ax.set_xlabel(rf"$w_{{rms}}/\sqrt{{\text{{g}} \text{{r}}_{{j}}}} \cdot \hat{{h}}_{{ML}}^{{{exp:.2f}}}$", fontsize = 16)

    for ax, exp in zip(axes[1, :], exponents):
        for i in range(num_cases):
            correction = mld[i]**exp
            ax.plot(b_center[:, i] *(correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'$\hat{{h}}_{{ML}}^{{{exp:.2f}}}$', fontsize = 16)
        ax.set_xlabel(rf"$b_{{\text{{centerline}}}}/g \cdot \hat{{h}}_{{ML}}^{{{exp:.2f}}}$", fontsize = 16)

    for ax, exp in zip(axes[2, :], exponents):
        for i in range(num_cases):
            correction = mld[i]**exp
            ax.plot(bw[:, i] * (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'$\hat{{h}}_{{ML}}^{{{exp:.2f}}}$', fontsize = 16)
        ax.set_xlabel(rf"$(\langle b'w'\rangle_{{\text{{xy}}}}/\sqrt{{\text{{g}}^3 \text{{r}}_{{j}}}})\cdot \hat{{h}}_{{ML}}^{{{exp:.2f}}}$", fontsize = 16)

    for ax, exp in zip(axes[3, :], exponents):
        for i in range(num_cases):
            correction = mld[i]**exp
            ax.plot(rp[:, i] * (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'$\hat{{h}}_{{ML}}^{{{exp:.2f}}}$', fontsize = 16)
        ax.set_xlabel(rf"(r/r$_{{j}})\cdot \hat{{h}}_{{ML}}^{{{exp:.2f}}}$", fontsize = 16)

    for ax, exp in zip(axes[4, :], exponents):
        for i in range(num_cases):
            correction = mld[i]**exp
            ax.plot(T[:, i] * (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'$\hat{{h}}_{{ML}}^{{{exp:.2f}}}$', fontsize = 16)
        ax.set_xlabel(rf"$(\text{{T'}}_{{\text{{centerline}}}}\alpha)\cdot \hat{{h}}_{{ML}}^{{{exp:.2f}}}$", fontsize = 16)

    for ax, exp in zip(axes[5, :], exponents):
        for i in range(num_cases):
            correction = mld[i]**exp
            ax.plot(S[:, i] * (correction), 
                    z_nd[:, i], color=color_opt[i])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
        ax.set_title(rf'$\hat{{h}}_{{ML}}^{{{exp:.2f}}}$', fontsize = 16)
        ax.set_xlabel(rf"($\langle$C$\rangle_{{\text{{xy}}}} \beta)\cdot \hat{{h}}_{{ML}}^{{{exp:.2f}}}$", fontsize = 16)

    plt.tight_layout()

    # --- Save Frame ---
    str_exp = '_'.join(f"{x:.3g}" for x in exponents)
    frame_path = os.path.join(fig_folder, f"MLD_{file_name} _pow{str_exp}.png")
    plt.savefig(frame_path)
    plt.close(fig)
## All ND
def plot_combo_exponents(color_opt, title, file_name, fig_folder, w_rms, b_center, bw, rp, T, S, z_nd, vars_exps, Ri_g, Fr, mld, case_names, z_str = rf"(z-h$_{{ML}}$)h$_{{ML}}^{{1/3}}$/L$_N^{{4/3}}$"):
    NDs = [rf"Ri$_g^", rf"Fr$^", rf"$\hat{{h}}_{{ML}}^"] 
    NDs_filtered = [[("" if str(Fraction(x).limit_denominator()) == '0' 
                else NDs[j] + "{"+str(Fraction(x).limit_denominator())+"}$")
                for j, x in enumerate(row)] for row in vars_exps]
    vars_str = [''.join(row) for row in NDs_filtered]
    num_cases = len(case_names)
    if num_cases > 4:
        gridspec_kw={'height_ratios': [1, 0.3]}
        n_col = num_cases//2
    else:
        gridspec_kw={'height_ratios': [1, 0.1]}
        n_col = num_cases

    fig, axes = plt.subplots(2, len(NDs_filtered), figsize=(25, 7), sharey=True, gridspec_kw = gridspec_kw)
    plt.subplots_adjust(top=0.9)
    for a in axes[-1, :]:
        a.remove()
    case_handles = [Line2D([0], [0], color=color_opt[i], linestyle='solid', label=case_names[i])for i in range(num_cases)]
    fig.legend(handles=case_handles,
            loc='lower center',
            ncol=n_col,
            bbox_to_anchor=(0.52, 0.005), fontsize = 16)
    fig.suptitle(title, fontsize = 20, y = 0.99)
    """
    axes[0] = ND rms velocity vs z_nd varied exponent of all
    axes[1] = ND centerline buoyancy vs z_nd varied exponent of all
    axes[2] = ND average buoyancy flux vs z_nd varied exponent of all
    axes[3] = ND radius of plume vs z_nd varied exponent of all
    axes[4] = ND perturbed temperature vs z_nd varied exponent of all
    axes[5] = ND average salinity vs z_nd varied exponent of all
    """

    for ax in axes[:, 0]:
        ax.set_ylabel(z_str, fontsize = 16)

    for i in range(num_cases):
        axes[0, 0].plot(w_rms[:, i] * mld[i]**vars_exps[0, 2] * Ri_g[i]**vars_exps[0, 0] * Fr[i]**vars_exps[0, 1], 
                z_nd[:, i], color=color_opt[i])
    axes[0, 0].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
    axes[0, 0].set_xlabel(rf"$w_{{rms}}/\sqrt{{\text{{g r}}_{{j}}}}\cdot$ {vars_str[0]}", fontsize = 16)

    for i in range(num_cases):
        axes[0, 1].plot(b_center[:, i] * mld[i]**vars_exps[1, 2] * Ri_g[i]**vars_exps[1, 0] * Fr[i]**vars_exps[1, 1], 
                z_nd[:, i], color=color_opt[i])
    axes[0, 1].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
    axes[0, 1].set_xlabel(rf"$b_{{\text{{centerline}}}}/g \cdot$ {vars_str[1]}", fontsize = 16)

    for i in range(num_cases):
        axes[0, 2].plot(bw[:, i] * mld[i]**vars_exps[2, 2] * Ri_g[i]**vars_exps[2, 0] * Fr[i]**vars_exps[2, 1], 
                z_nd[:, i], color=color_opt[i])
    axes[0, 2].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
    axes[0, 2].set_xlabel(rf"$(\langle b'w'\rangle_{{xy}}/\sqrt{{\text{{g}}^3 \text{{r}}_{{j}}}})\cdot$ {vars_str[2]}", fontsize = 16)

    for i in range(num_cases):
        axes[0, 3].plot(rp[:, i] * mld[i]**vars_exps[3, 2] * Ri_g[i]**vars_exps[3, 0] * Fr[i]**vars_exps[3, 1], 
                z_nd[:, i], color=color_opt[i])
    axes[0, 3].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
    axes[0, 3].set_xlabel(rf"(r/r$_{{j}})\cdot$ {vars_str[3]}", fontsize = 16)

    for i in range(num_cases):
        axes[0, 4].plot(T[:, i] * mld[i]**vars_exps[4, 2] * Ri_g[i]**vars_exps[4, 0] * Fr[i]**vars_exps[4, 1],
                z_nd[:, i], color=color_opt[i])
    axes[0, 4].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
    axes[0, 4].set_xlabel(rf"$(\text{{T'}}_{{\text{{centerline}}}}\alpha)\cdot$ {vars_str[4]}", fontsize = 16)

    for i in range(num_cases):
        axes[0, 5].plot(S[:, i] * mld[i]**vars_exps[5, 2] * Ri_g[i]**vars_exps[5, 0] * Fr[i]**vars_exps[5, 1], 
                z_nd[:, i], color=color_opt[i])
    axes[0, 5].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
    axes[0, 5].set_xlabel(rf"($\langle$C$\rangle_{{\text{{xy}}}} \beta)\cdot$ {vars_str[5]}", fontsize = 16)

    plt.tight_layout()

    # --- Save Frame ---
    frame_path = os.path.join(fig_folder, f"{file_name} _combined.png")
    i = 0
    while True:
        i += 1
        frame_path = os.path.join(fig_folder, f"{file_name} _combined_{i}.png")
        if os.path.exists(frame_path):
            continue
        plt.savefig(frame_path)
        break
    plt.close(fig)

### ---------------------- CONVERGENCE TESTS ----------------------------- ###
def convergence_tests(time, it, ranges, fig_folder, lx, nx, x, y, z, cases_sorted, matrix_N, ver, hor, 
                      b, b_avg, b_rms_sign, w_rms, b_rms, bw_fluc, b_flux_avg, b_max_sign_change_to_negative_loc, L_ozmidov, L_ozmidov_background, idx_neg, plot_points = False):

    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))

    levels = 500

    colors = ['black', 'blue', 'green', 'red', 'purple', 'pink', 'gray', 'orange', 'cyan', 'olive']
    td = time[it] / 3600 / 24

    ## buoyancy analysis profiles
    outdir0 = [fig_folder + 'convergence tests buoyancy profiles/']
    os.makedirs(outdir0, exist_ok=True)

    fig, ax = plt.subplots(3, 5, figsize=(20, 8), height_ratios = [1, 0.2, 1])

    fig.text(0.5, 1.08, f'{td:.2f} days', ha="center", fontsize=12) 
    # Titles for each row
    fig.text(0.5, 1.05, "Vertical resolution convergence", ha="center", fontsize=14)
    fig.text(0.5, 0.52, "Horizontal resolution convergence", ha="center", fontsize=14)
    
    ax1 = ax[0, 0]
    ax6 = ax[2, 0]
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
            ax[0, 2].plot(w_rms[0:nx[2, caseindex]+1, caseindex], z[0:nx[2, caseindex]+1, caseindex], color = colors[caseindex])#, linestyle = '--', label = r"$\langle$w$_{rms}rangle_{\text{xy}}$")
            # RMS buoyancy flux 
            ax[0, 3].plot(bw_fluc[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex])
            # buoyancy profile
            ax[2, 0].plot(b_avg[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex], label = name_case)
            # RMS buoyancy 
            ax[2, 1].plot(b_rms[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex])
            # RMS w 
            ax[2, 2].plot(w_rms[0:nx[2, caseindex]+1, caseindex], z[0:nx[2, caseindex]+1, caseindex], color = colors[caseindex])#, linestyle = '--', label = r"$\langle$w$_{rms}\rangle_{\text{xy}}$")
            # RMS buoyancy flux 
            ax[2, 3].plot(bw_fluc[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex])
        elif ver[caseindex] and not hor[caseindex]:
            name_case = case.replace('flux b tracer ', "")
            # buoyancy profile
            ax[0, 0].plot(b_avg[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex], label = name_case)
            # RMS buoyancy 
            ax[0, 1].plot(b_rms[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex])
            # RMS w 
            ax[0, 2].plot(w_rms[0:nx[2, caseindex]+1, caseindex], z[0:nx[2, caseindex]+1, caseindex], color = colors[caseindex])#, linestyle = '--')
            # RMS buoyancy flux 
            ax[0, 3].plot(bw_fluc[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex])
        elif hor[caseindex] and not ver[caseindex]:
            name_case = case.replace('flux b tracer ', "")
            # buoyancy profile
            ax[2, 0].plot(b_avg[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex], label = name_case)
            # RMS buoyancy 
            ax[2, 1].plot(b_rms[0:nx[2, caseindex], caseindex], z[0:nx[2, caseindex], caseindex], color = colors[caseindex])
            # RMS w 
            ax[2, 2].plot(w_rms[0:nx[2, caseindex]+1, caseindex], z[0:nx[2, caseindex]+1, caseindex], color = colors[caseindex])#, linestyle = ':')
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
    ax[0, 0].set_ylabel("y [m]")
    ax[0, 0].set_xlim(ranges['b_avg'])
    ax[0, 0].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
    
    ax[0, 1].set_xlabel("[m/s$^{2}$]")
    ax[0, 1].set_title("Buoyancy RMS")
    ax[0, 1].set_xlim(ranges['b_rms'])
    ax[0, 1].set_ylim([-lx[2], 0])

    ax[0, 2].set_xlabel("[m/s]")
    ax[0, 2].set_title("w RMS")
    ax[0, 2].set_xlim(ranges['vel_rms'])
    ax[0, 2].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
    ax[0, 2].set_ylim([-lx[2], 0])

    ax[0, 3].set_xlabel("[m$^{2}$/s$^{3}$]")
    ax[0, 3].set_title("Buoyancy Flux Flucts")
    ax[0, 3].set_xlim(ranges['bflux_rms'])
    ax[0, 3].set_ylim([-lx[2], 0])

    ax[0, 4].legend(loc='upper right', handlelength=0.75)
    ax[0, 4].set_ylabel("Length Scale [m]")
    ax[0, 4].set_title("Ozmidov Length Scale")
    ax[0, 4].set_ylim(ranges['b_avg'])
    ax[0, 4].set_xlabel("Time [days]")
    ax[0, 4].set_xlim([0, matrix_N +10])
    
    ax[2, 0].set_xlabel("[m/s$^{2}$]")
    ax[2, 0].set_xlim(ranges['b_avg'])
    ax[2, 0].set_title("Buoyancy")
    ax[2, 0].set_ylabel("Depth [m]")
    ax[2, 0].set_ylim([-lx[2], 0])
    ax[2, 0].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)

    ax[2, 1].set_xlabel("[m/s$^{2}$]")
    ax[2, 1].set_title("Buoyancy RMS")
    ax[2, 1].set_xlim(ranges['b_rms'])
    ax[2, 1].set_ylim([-lx[2], 0])

    ax[2, 2].set_xlabel("[m/s]")
    ax[2, 2].set_title("w RMS")
    ax[2, 2].set_xlim(ranges['vel_rms'])
    ax[2, 2].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True)
    ax[2, 2].set_ylim([-lx[2], 0]) 

    ax[2, 3].set_xlabel("[m$^{2}$/s$^{3}$]")
    ax[2, 3].set_title("Buoyancy Flux Flucts")
    ax[2, 3].set_xlim(ranges['bflux_rms'])
    ax[2, 3].set_ylim([-lx[2], 0])

    ax[2, 4].set_title("Ozmidov Length Scale")
    ax[2, 4].set_ylabel("Length Scale [m]")
    ax[2, 4].set_ylim(ranges['lengthscale'])
    ax[2, 4].set_xlabel("Time [days]")
    ax[2, 4].set_xlim([0, matrix_N +10])
    ax[2, 4].legend(loc='upper right', handlelength=0.75)

    # universal legend
    handles0, labels0 = ax1.get_legend_handles_labels()
    handles1, labels1 = ax6.get_legend_handles_labels()

    fig.legend(handles0, labels0, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.99))
    fig.legend(handles1, labels1, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.46))

    # --- Save Frame ---
    frame_path = os.path.join(outdir0, f"convergence_test_{it:04d}.png")
    plt.savefig(frame_path, bbox_inches="tight")
    print(f"Time step {it + 1} captured")
    plt.close(fig)

    ## buoyancy plane slices ##
    outdir1 = fig_folder + 'buoyancy planeslices/'
    os.makedirs(outdir1, exist_ok=True)
    fig, ax = plt.subplots(3, 3, figsize=(18, 10), constrained_layout=True)
    fig.suptitle(f'{td:.2f} days') 
    norm = mcolors.Normalize(vmin=ranges['b_avg'][0], vmax=ranges['b_avg'][-1])
    mappable = cm.ScalarMappable(norm=norm) 
    hor_plot = 0
    ver_plot = 0
    for caseindex, case in enumerate(cases_sorted):
        X, Y, Z = np.meshgrid(x[0:nx[0, caseindex], caseindex] , y[0:nx[1, caseindex], caseindex] , z[0:nx[2, caseindex], caseindex])
        name_case = case.replace('flux b tracer ', "")
        ax[hor_plot, ver_plot].contourf(X[int(nx[0, caseindex]/2), :, :], Z[int(nx[0, caseindex]/2), :, :], b[caseindex, int(nx[0, caseindex]/2), 0:nx[1, caseindex], 0:nx[2, caseindex]], levels, norm=norm)
        ax[hor_plot, ver_plot].set_title(name_case)
        ax[hor_plot, ver_plot].set_xlabel("y [m]")
        ax[hor_plot, ver_plot].set_ylabel("z [m]")
        ax[hor_plot, ver_plot].set_aspect('equal')
        if ver_plot > 1:
            hor_plot += 1
            ver_plot = 0
        else:
            ver_plot += 1

    cbar = fig.colorbar(mappable, ax=ax, label=r"m/s$^2$", location='bottom', shrink=0.5, orientation='horizontal')
    
    # --- Save Frame ---
    frame_path = os.path.join(outdir1, f"planeslices_{it:04d}.png")
    plt.savefig(frame_path, bbox_inches="tight")
    print(f"Time step {it + 1} captured")
    plt.close(fig)

    if plot_points:
        ranges['brms_sign'] = [0, b_rms_sign.max()]
        ranges['bflux_rms'] = [b_flux_avg.min(), b_flux_avg.max()]
        ranges['z_sign'] = [b_max_sign_change_to_negative_loc.min(), b_max_sign_change_to_negative_loc.max()]

        outdir2 = fig_folder + 'convergence testing/'
        os.makedirs(outdir2, exist_ok=True)
        td = time[it] / 3600 / 24
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
        ax1.set_ylim(ranges['z_sign'])

        ax4 = fig.add_subplot(2, 5,  6)
        ax4.plot(nx[1, hor], b_max_sign_change_to_negative_loc[hor], marker='o', linestyle='none')
        ax4.set_ylabel("[m]")
        ax4.set_title("Neutrally buoyant depth")
        ax4.set_ylim(ranges['z_sign'])

        # RMS buoyancy as a function of resolution 
        ax2 = fig.add_subplot(2, 5,  2)
        ax2.plot(nx[2, ver], b_rms_sign[ver], marker='o', linestyle='none', label = "at neutrally buoyant depth")
        ax2.plot(nx[2, ver], b_rms_sign[ver-1], marker='o', linestyle='none', color = color_opt[i], label = "above neutrally buoyant depth")
        ax2.legend(loc='upper right', handlelength=0.75)
        ax2.set_ylabel("[m/s$^{2}$]")
        ax2.set_title("Buoyancy RMS")
        ax2.set_ylim(ranges['brms_sign'])

        ax5 = fig.add_subplot(2, 5,  7)
        ax5.plot(nx[1, hor], b_rms_sign[hor], marker='o', linestyle='none', label = "at neutrally buoyant depth")
        ax5.plot(nx[1, hor], b_rms_sign[hor-1], marker='o', linestyle='none', color = color_opt[i], label = "above neutrally buoyant depth")
        ax5.legend(loc='upper right', handlelength=0.75)
        ax5.set_ylabel("[m/s$^{2}$]")
        ax5.set_title("Buoyancy RMS")
        ax5.set_ylim(ranges['brms_sign'])

        # RMS w as a function of resolution
        ax2 = fig.add_subplot(2, 5,  3)
        ax2.plot(nx[2, ver], w_rms[idx_neg[ver], ver], marker='o', linestyle='none', label = "at neutrally buoyant depth")
        ax2.plot(nx[2, ver], w_rms[idx_neg[ver]-1, ver], marker='o', linestyle='none', color = color_opt[i], label = "above neutrally buoyant depth")
        ax2.legend(loc='upper right', handlelength=0.75)
        ax2.set_ylabel("[m/s]")
        ax2.set_title("w RMS")
        ax2.set_ylim(ranges['vel_rms'])
        ax2.ticklabel_format(axis='y', style='sci', scilimits=(-3,2), useMathText=True)

        ax5 = fig.add_subplot(2, 5,  8)
        ax5.plot(nx[1, hor], w_rms[idx_neg[hor], hor], marker='o', linestyle='none', label = "at neutrally buoyant depth")
        ax5.plot(nx[1, hor], w_rms[idx_neg[hor]-1, hor], marker='o', linestyle='none', color = color_opt[i], label = "above neutrally buoyant depth")
        ax5.legend(loc='upper right', handlelength=0.75)
        ax5.set_ylabel("[m/s]")
        ax5.set_title("w RMS")
        ax5.set_ylim(ranges['vel_rms'])
        ax5.ticklabel_format(axis='y', style='sci', scilimits=(-3,2), useMathText=True)

        # RMS buoyancy flux as a function of resolution
        ax4 = fig.add_subplot(2, 5,  4)
        ax4.plot(nx[2, ver], bw_fluc[idx_neg[ver], ver], marker='o', linestyle='none', label = "at neutrally buoyant depth")
        ax4.plot(nx[2, ver], bw_fluc[idx_neg[ver]-1, ver], marker='o', linestyle='none', color = color_opt[i], label = "above neutrally buoyant depth")
        ax4.legend(loc='upper right', handlelength=0.75)
        ax4.set_ylabel("[m$^{2}$/s$^{3}$]")
        ax4.set_title("Buoyancy Flux Flucts")
        ax4.set_ylim(ranges['bflux_rms'])

        ax8 = fig.add_subplot(2, 5,  9)
        ax8.plot(nx[1, hor], bw_fluc[idx_neg[hor], hor], marker='o', linestyle='none', label = "at neutrally buoyant depth")
        ax8.plot(nx[1, hor], bw_fluc[idx_neg[hor]-1, hor], marker='o', linestyle='none', color = color_opt[i], label = "above neutrally buoyant depth")
        ax8.legend(loc='upper right', handlelength=0.75)
        ax8.set_ylabel("[m$^{2}$/s$^{3}$]")
        ax8.set_title("Buoyancy Flux Flucts")
        ax8.set_ylim(ranges['bflux_rms'])

        # RMS buoyancy flux as a function of resolution
        ax5 = fig.add_subplot(2, 5, 5)
        ax5.plot(nx[2, ver], L_ozmidov[it, ver], marker='o', linestyle='none', color = color_opt[i], label = r"b$_{\text{average}, 3}$ L$_{O}$")
        ax5.plot(nx[2, ver], L_ozmidov_background[it, ver], marker='o', linestyle='none', color = 'blue', label = r"b$_{\text{stratified}, 3}$ L$_{O}$")
        ax5.legend(loc='upper right', handlelength=0.75)
        ax5.set_ylabel("[m]")
        ax5.set_title("Ozmidov Length Scale")
        ax5.set_ylim(ranges['lengthscale'])
        ax10 = fig.add_subplot(2, 5,  10)
        ax10.plot(nx[1, hor], L_ozmidov[it, hor], marker='o', linestyle='none', color = color_opt[i], label = r"b$_{\text{average}, 3}$ L$_{O}$")
        ax10.plot(nx[1, hor], L_ozmidov_background[it, hor], marker='o', linestyle='none', color = 'blue', label = r"b$_{\text{stratified}, 3}$ L$_{O}$")
        ax10.legend(loc='upper right', handlelength=0.75)
        ax10.set_title("Ozmidov Length Scale")
        ax10.set_ylabel("[m]")
        ax10.set_ylim(ranges['lengthscale'])

        fig.supxlabel("Number of Grid Cells", fontsize=12)
        # --- Save Frame ---
        frame_path = os.path.join(outdir, f"convergence_test_{it:04d}.png")
        plt.tight_layout()
        plt.savefig(frame_path)
        print(f"Time step {it + 1} captured")
        plt.close(fig)
        return outdir0, outdir1, outdir2
    return outdir0, outdir1 # return the directory where frames are saved for video

### -------------------------SAVING FRAMES AND MAKING VIDEOS------------------------- ###
def create_video(outdir, fig_folder, name, plot_type):
    print("Creating video...")
    filenames = sorted([f for f in os.listdir(outdir) if f.endswith(".png")])
    vid_name = os.path.join(fig_folder, name + plot_type + '.mp4')
    with imageio.get_writer(vid_name, fps=10) as writer:
        for filename in filenames:
            image = imageio.imread(f"{outdir}/{filename}")
            writer.append_data(image)