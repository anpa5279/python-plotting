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
    # Compute squared fluctuation a'*B' = (a-a_avg)(B-b_avg)
    ab_fluct = (a - a_avg)*(b - b_avg)
    # Mean over x and y, result shape: (nt, nz,)
    ab_fluct_avg = np.mean(ab_fluct, axis=(0, 1))

    return ab_fluct_avg

# Set up folder and simulation parameters
folder = '/Users/annapauls/Library/CloudStorage/OneDrive-UCB-O365(2)/CU-Boulder/TESLa/Carbon Sequestration/Simulations/Oceananigans/NBP/b tracer for NBP/with closure only visc Re 3000/'
fig_folder = 'figures and videos/'
const_N = 128
# flags for what to plot
video_3d_flag = False
turb_stats_video = True
vel_plane_slice_video = False
with_halos = False

# List JLD2 files
cases = sorted([f for f in os.listdir(folder) if str(const_N) in f])
ncases = len(cases)
print(f"Number of cases: {ncases}")
for caseindex, case in enumerate(cases):
    print(f"Processing case {caseindex + 1}/{ncases}: {case}")
    name = 'NBP-'
    dtn = sorted([f for f in os.listdir(os.path.join(folder, case)) if f.endswith('.jld2')])
    nf = len(dtn)
    # Read t steps
    fid = os.path.join(folder, case, dtn[-1])
    with h5py.File(fid, 'r') as f:
        timeseries_group = [g for g in f.keys() if 'timeseries' in g][0]
        t_group = f[timeseries_group + '/t']
        t_save = sorted([float(k) for k in t_group.keys()])
        t_save = np.array(t_save)
        time = np.array([t_group[str(int(k))][()] for k in t_save])
        nx = [int(f['grid/Nx'][()]), int(f['grid/Ny'][()]), int(f['grid/Nz'][()])]
        hx = [f['grid/Hx'][()], f['grid/Hy'][()], f['grid/Hz'][()]]
        lx = [f['grid/Lx'][()], f['grid/Ly'][()], f['grid/Lz'][()]]
        x = f['grid/xᶜᵃᵃ'][hx[0]:-hx[0]]
        y = f['grid/yᵃᶜᵃ'][hx[1]:-hx[1]]
        z = f['grid/z/cᵃᵃᶜ'][hx[2]:-hx[2]]
        xf = f['grid/xᶠᵃᵃ'][hx[0]:-hx[0]]
        yf = f['grid/yᵃᶠᵃ'][hx[1]:-hx[1]]
        zf = f['grid/z/cᵃᵃᶠ'][hx[2]:-hx[2]]
        dz = f['grid/z/Δᵃᵃᶜ'][()]
    name+=f'Nx{nx[0]}_Ny{nx[1]}_Nz{nx[2]}'
    print(name)
    
    if video_3d_flag or turb_stats_video or vel_plane_slice_video:
        nt = len(t_save)
    else:
        nt = 1  # only last time step
    X, Y, Z = np.meshgrid(x, y, z)
    Xf, Yf, Zf = np.meshgrid(xf, yf, zf)
    for it in range(0, nt):
        # Initialize arrays
        u = np.zeros((nt, nx[0], nx[1], nx[2]))
        v = np.zeros((nt, nx[0], nx[1], nx[2]))
        w = np.zeros((nt, nx[0], nx[1], nx[2] + 1))
        B = np.zeros((nt, nx[0], nx[1], nx[2]))
        Pdynamic = np.zeros((nt, nx[0], nx[1], nx[2]))
        Pstatic = np.zeros((nt, nx[0], nx[1], nx[2]))
        u2_fluc = np.zeros((nt, nx[2]))
        v2_fluc = np.zeros((nt, nx[2]))
        w2_fluc = np.zeros((nt, nx[2] + 1))
        b2_fluc = np.zeros((nt, nx[2]))
        wc2_fluc = np.zeros((nt, nx[2]))
        uv_fluc = np.zeros((nt, nx[2]))
        uw_fluc = np.zeros((nt, nx[2]))
        vw_fluc = np.zeros((nt, nx[2]))
        # Load data from files
        xrange = range(0, nx[0])
        for r in range(nf):
            fname = os.path.join(folder, case, dtn[r])
            if "fields" in fname:
                with h5py.File(fname, 'r') as f:
                    if with_halos:
                        u[xrange, :, :] = (f['timeseries/u'][f'{int(t_save[it])}'])[hx[0]:-hx[0], hx[1]:-hx[1], hx[2]:-hx[2]].transpose(2, 1, 0)
                        v[xrange, :, :] = (f['timeseries/v'][f'{int(t_save[it])}'])[hx[0]:-hx[0], hx[1]:-hx[1], hx[2]:-hx[2]].transpose(2, 1, 0)
                        w[xrange, :, :] = (f['timeseries/w'][f'{int(t_save[it])}'])[hx[0]:-hx[0], hx[1]:-hx[1], hx[2]:-hx[2]].transpose(2, 1, 0)
                        B[xrange, :, :] = (f['timeseries/b'][f'{int(t_save[it])}'])[hx[0]:-hx[0], hx[1]:-hx[1], hx[2]:-hx[2]].transpose(2, 1, 0) 
                    else:
                        u[xrange, :, :] = (f['timeseries/u'][f'{int(t_save[it])}'][:, :, :].transpose(2, 1, 0))
                        v[xrange, :, :] = (f['timeseries/v'][f'{int(t_save[it])}'][:, :, :].transpose(2, 1, 0)) 
                        w[xrange, :, :] = (f['timeseries/w'][f'{int(t_save[it])}'][:, :, :].transpose(2, 1, 0)) 
                        B[xrange, :, :] = (f['timeseries/b'][f'{int(t_save[it])}'][:, :, :].transpose(2, 1, 0)) 
                        Pdynamic[xrange, :, :] = (f['timeseries/P_dynamic'][f'{int(t_save[it])}'][:, :, :].transpose(2, 1, 0)) 
                        Pstatic[xrange, :, :] = (f['timeseries/P_static'][f'{int(t_save[it])}'][:, :, :].transpose(2, 1, 0)) 
                    
                    new_range = range(xrange.start + (nx[0]), xrange.stop + (nx[0]))
                    xrange = new_range

        #interpolate so all values are from the center, center, center of the grid cell
        w_face = make_interp_spline(zf, w, axis=-1, k=1)
        wc = w_face(z)

        u = u #-u_s
        #calculate means
        u_avg = np.mean(u, axis=(0, 1))
        v_avg = np.mean(v, axis=(0, 1))
        w_avg = np.mean(w, axis=(0, 1))
        wc_avg = np.mean(wc, axis=(0, 1))
        b_avg = np.mean(B, axis=(0, 1))
        #calcualte reynolds stresses
        u2_fluc = fluct2_mean(u, u, u_avg, u_avg)
        v2_fluc = fluct2_mean(v, v, v_avg, v_avg)
        w2_fluc = fluct2_mean(w, w, w_avg, w_avg)
        wc2_fluc = fluct2_mean(wc, wc, wc_avg, wc_avg)
        uv_fluc = fluct2_mean(u, v, u_avg, v_avg)
        uw_fluc = fluct2_mean(u, wc, u_avg, wc_avg)
        vw_fluc = fluct2_mean(v, wc, v_avg, wc_avg)

        b2_fluc = fluct2_mean(B, B, b_avg, b_avg)
        # rms fluctuations
        u_rms = u2_fluc**0.5
        v_rms = v2_fluc**0.5
        w_rms = w2_fluc**0.5
        b_rms = b2_fluc**0.5

        # calculate lamb vector
        omega_x = np.gradient(wc, y, axis=1) - np.gradient(v, z, axis=-1)
        omega_y = np.gradient(u, z, axis=-1) - np.gradient(wc, x, axis=0)
        omega_z = np.gradient(v, x, axis=0) - np.gradient(u, y, axis=1)
        lamb_x = v * omega_z - wc * omega_y
        lamb_y = wc * omega_x - u * omega_z
        lamb_z = u * omega_y - v * omega_x
        lamb_x_avg = np.mean(lamb_x, axis=(0, 1))
        lamb_y_avg = np.mean(lamb_y, axis=(0, 1))
        lamb_z_avg = np.mean(lamb_z, axis=(0, 1))

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
        if video_3d_flag:
            fac = 1*10**(-1)
            u_max = np.abs(u).max()
            v_max = np.abs(v).max()
            w_max = np.abs(w).max()
            u_range = [-1*u_max, u_max]
            v_range = [-1*v_max, v_max]
            w_range = [-1*w_max, w_max]
            print("u max:", u_max)
            print("v max:", v_max)
            print("w max:", w_max)
        vel_range = [-0.0002, 0.0002]#[-1*vel_max, vel_max]#
        rms_range = [0, 0.003]#[0, np.max([u_rms.max(), v_rms.max(), w_rms.max()])]#
        restress_range = [-3*10**(-6), 3*10**(-6)]#[-1*restress_max, restress_max]#
        richardson_range = [0, 5*10**(-3)]# Ri_avg.max()]#
        P_d_range = [-0.005, 0.005]#
        P_s_range = [-0.05, 0.05]#[-Pstatic.max(), Pstatic.max()]
        B_range = [-1.5*10**(-3), 10**(-5)]#[B.min(), B.max()]
        B_avg_range = [-1.5*10**(-3), 1.0*10**(-4)]
        lamb_avg_range = [-2*10**(-6), 2*10**(-6)]#[-1*lamb_avg_max, lamb_avg_max]
        b_rms_range = [0, 5*10**(-5)]

        levels =200
        formatter = mticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, 3))

        # --- Create Video ---
        if turb_stats_video:
            
            outdir = folder + case + '/' + fig_folder
            print("Output directory:" + outdir)
            os.makedirs(outdir, exist_ok=True)
            td = time[it] / 3600 / 24
            fig = plt.figure(figsize=(10, 9))
            fig.tight_layout()
            fig.suptitle(f'{td:.2f} days', fontsize=12)
            ax1 = fig.add_subplot(3, 3, 1)  # average velocities
            ax1.plot(u_avg, z, label=r"$\langle$u$\rangle_{xy}$", color='green')
            ax1.plot(v_avg, z, label=r"$\langle$v$\rangle_{xy}$", color='red')
            ax1.plot(w_avg, zf, label=r"$\langle$w$\rangle_{xy}$", color='blue')
            ax1.set_xlabel("[m/s]")
            ax1.set_ylabel("Depth [m]")
            ax1.set_title('Velocity Profiles')
            ax1.set_ylim(-lx[2], 0)
            ax1.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True)
            ax1.set_xlim(vel_range)
            ax1.legend(loc='lower right', handlelength=0.5)

            ax2 = fig.add_subplot(3, 3, 2)  # rms
            ax2.plot(u_rms, z, label=r"$\langle$u$_{rms}$$\rangle_{xy}$", color='green')
            ax2.plot(v_rms, z, label=r"$\langle$v$_{rms}$$\rangle_{xy}$", color='red')
            ax2.plot(w_rms, zf, label=r"$\langle$w$_{rms}$$\rangle_{xy}$", color='blue')
            ax2.set_xlabel("[m/s]")
            ax2.set_ylabel("Depth [m]")
            ax2.set_title("Root Mean Square Velocities")
            ax2.set_ylim(-lx[2], 0)
            ax2.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True)
            ax2.set_xlim(rms_range)
            ax2.legend(loc='lower right', handlelength=0.5)

            ax3 = fig.add_subplot(3, 3, 4)  # Reynolds stress
            ax3.plot(uv_fluc, z, label=r"$\langle$uv$\rangle_{xy}$", color='green')
            ax3.plot(uw_fluc, z, label=r"$\langle$uw$\rangle_{xy}$", color='red')
            ax3.plot(vw_fluc, z, label=r"$\langle$vw$\rangle_{xy}$", color='blue')
            ax3.set_xlabel(r"[m$^2$/s$^2$]")
            ax3.set_ylabel("Depth [m]")
            ax3.set_title("Reynolds Stresses")
            ax3.set_ylim(-lx[2], 0)
            ax3.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True)
            ax3.set_xlim(restress_range)
            ax3.legend(loc='lower right', handlelength=0.5)

            ax4 = fig.add_subplot(3, 3, 3)  
            ax4.plot(b_rms, z, color='black')
            ax4.set_xlabel("[m/s$^{2}$]")
            ax4.set_ylabel("Depth [m]")
            ax4.set_title("Buoyancy RMS")
            ax4.set_ylim(-lx[2], 0)
            ax4.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True)
            ax4.set_xlim(b_rms_range)

            ax5 = fig.add_subplot(3, 3, 5)
            ax5.plot(B_avg, z, color='black')
            ax5.set_xlabel(r"[m/s$^{2}$]")
            ax5.set_ylabel("Depth [m]")
            ax5.set_title("Buoyancy")
            ax5.set_ylim(-lx[2], 0)
            ax5.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True)
            ax5.set_xlim(B_avg_range)

            ax6 = fig.add_subplot(3, 3, 6)
            ax6.plot(lamb_x_avg, z, label = r"$\langle$u$\times \omega\rangle_{xy}$", color='green')
            ax6.plot(lamb_y_avg, z, label = r"$\langle$v$\times \omega\rangle_{xy}$", color='red')
            ax6.plot(lamb_z_avg, z, label = r"$\langle$w$\times \omega\rangle_{xy}$", color='blue')
            ax6.set_xlabel(r"[m/s$^{2}$]")
            ax6.set_ylabel("Depth [m]")
            ax6.set_title("Lamb Vector")
            ax6.set_ylim(-lx[2], 0)
            ax6.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True)
            ax6.set_xlim(lamb_avg_range)
            ax6.legend(loc='lower right', handlelength=0.5)

            norm = mcolors.Normalize(vmin=B_range[0], vmax=B_range[-1])
            ax7 = fig.add_subplot(3, 3, 7)
            ax7.contourf(X[int(nx[1]/2), :, :], Z[int(nx[1]/2), :, :], B[it, int(nx[1]/2), :, :], levels, norm=norm)
            ax7.set_xlabel("[m]")
            ax7.set_ylabel("Depth [m]")
            ax7.set_title("Buoyancy")
            ax7.set_ylim(-lx[2], 0)
            ax7.set_xlim(0, lx[0])
            mappable = cm.ScalarMappable(norm=norm)
            #formatter.set_powerlimits((-3, 0)) 
            fig.colorbar(mappable, ax=ax7, label=r"m/s$^2$", anchor = (0.5, -0.15), orientation='horizontal', shrink=0.75, format = formatter)

            norm = mcolors.Normalize(vmin=P_d_range[0], vmax=P_d_range[-1])
            mappable = cm.ScalarMappable(norm=norm, cmap='RdBu_r')
            ax8 = fig.add_subplot(3, 3, 8)
            ax8.contourf(X[int(nx[1]/2), :, :], Z[int(nx[1]/2), :, :], Pdynamic[it, int(nx[1]/2), :, :], levels, norm=norm, cmap = 'RdBu_r')
            ax8.set_xlabel("[m]")
            ax8.set_ylabel("Depth [m]")
            ax8.set_title("Hydrodynamic Pressure")
            ax8.set_ylim(-lx[2], 0)
            ax8.set_xlim(0, lx[0])
            #formatter.set_powerlimits((-3, 0)) 
            fig.colorbar(mappable, ax=ax8, label=r"m/s$^{2}$", anchor = (0.5, -0.15), orientation='horizontal', shrink=0.75, format = formatter)

            norm = mcolors.Normalize(vmin=P_s_range[0], vmax=P_s_range[-1])
            mappable = cm.ScalarMappable(norm=norm, cmap='RdBu_r')
            ax9 = fig.add_subplot(3, 3, 9)
            ax9.contourf(X[int(nx[1]/2), :, :], Z[int(nx[1]/2), :, :], Pstatic[it, int(nx[1]/2), :, :], levels, norm=norm, cmap = 'RdBu_r')
            ax9.set_xlabel("[m]")
            ax9.set_ylabel("Depth [m]")
            ax9.set_title("Hydrostatic Pressure")
            ax9.set_ylim(-lx[2], 0)
            ax9.set_xlim(0, lx[0])
            #formatter.set_powerlimits((-2, 0)) 
            fig.colorbar(mappable, ax=ax9, label=r"m/s$^{2}$", anchor = (0.5, -0.15), orientation='horizontal', shrink=0.75, format = formatter)

            # --- Save Frame ---
            frame_path = os.path.join(outdir, f"oc_frame_{it:04d}.png")
            plt.tight_layout()
            plt.savefig(frame_path)
            plt.close(fig)
            print(f"Time step {it + 1}/{nt} captured")
            plt.close(fig)

        if video_3d_flag:
            
            outdir = fig_folder + 'oc 3D video frames'
            os.makedirs(outdir, exist_ok=True)
            td = time[it] / 3600 / 24
            fig = plt.figure(figsize=(12, 4))
            fig.tight_layout()
            fig.suptitle(f'{td:.2f} days', fontsize=12)

            # --- 3D Plots ---
            # u velocity
            norm = mcolors.Normalize(vmin=u_range[0]*fac, vmax=u_range[-1]*fac)
            mappable = cm.ScalarMappable(norm=norm, cmap='RdBu_r')
            ax5 = fig.add_subplot(1, 4, 1, projection='3d')
            ax5.set(xlim=[0, lx[0]], ylim=[0, lx[1]], zlim=[-lx[2], 0])
            ax5.contourf(X[:, :, -1], Y[:, :, -1], u[:, :, -1], levels, zdir='z', offset=0, cmap='RdBu_r', norm=norm)
            ax5.contourf(X[-1, :, :], u[ -1, :, :], Z[-1, :, :], levels, zdir='y', offset=0, cmap='RdBu_r', norm=norm)
            ax5.contourf(u[:, 0, :], Y[:, 0, :], Z[:, 0, :], levels, zdir='x', offset=lx[0], cmap='RdBu_r', norm=norm)
            ax5.set_title('u')
            ax5.set(xlabel="y", ylabel="x", zlabel="z")
            fig.colorbar(mappable, ax=ax5, label='[m/s]', location='bottom', orientation='horizontal', shrink=0.75, format = formatter)
            ax5.set_aspect('equal')
            # w velocity
            norm = mcolors.Normalize(vmin=w_range[0]*fac*0.5, vmax=w_range[-1]*fac*0.5)
            mappable = cm.ScalarMappable(norm=norm, cmap='RdBu_r')
            ax6 = fig.add_subplot(1, 4, 3, projection='3d')
            ax6.set(xlim=[0, lx[0]], ylim=[0, lx[1]], zlim=[-lx[2], 0])
            ax6.contourf(Xf[:, :, -1], Yf[:, :, -2], w[:, :, -2], levels, zdir='z', offset=0, cmap='RdBu_r', norm=norm)
            ax6.contourf(Xf[-1, :, :], w[ -1, :, :], Zf[-1, :, :], levels, zdir='y', offset=0, cmap='RdBu_r', norm=norm)
            ax6.contourf(w[:, 0, :], Yf[:, 0, :], Zf[:, 0, :], levels, zdir='x', offset=lx[0], cmap='RdBu_r', norm=norm)
            ax6.set_title('w')
            ax6.set(xlabel="y", ylabel="x", zlabel="z")
            #formatter.set_powerlimits((-3, 0)) 
            fig.colorbar(mappable, ax=ax6, label='[m/s]', location='bottom', orientation='horizontal', shrink=0.75, format = formatter)
            ax6.set_aspect('equal')
            # v velocity
            norm = mcolors.Normalize(vmin=v_range[0]*fac, vmax=v_range[-1]*fac)
            mappable = cm.ScalarMappable(norm=norm, cmap='RdBu_r')
            ax7 = fig.add_subplot(1, 4, 2, projection='3d')
            ax7.set(xlim=[0, lx[0]], ylim=[0, lx[1]], zlim=[-lx[2], 0])
            ax7.contourf(X[:, :, -1], Y[:, :, -1], v[:, :, -1], levels, zdir='z', offset=0, cmap='RdBu_r', norm=norm)
            ax7.contourf(X[-1, :, :], v[ -1, :, :], Z[-1, :, :], levels, zdir='y', offset=0, cmap='RdBu_r', norm=norm)
            ax7.contourf(v[:, 0, :], Y[:, 0, :], Z[:, 0, :], levels, zdir='x', offset=lx[0], cmap='RdBu_r', norm=norm)
            ax7.set_title('v')
            ax7.set(xlabel="y", ylabel="x", zlabel="z")
            #formatter.set_powerlimits((-3, 0)) 
            fig.colorbar(mappable, ax=ax7, label='[m/s]', location='bottom', orientation='horizontal', shrink=0.75, format = formatter)
            ax7.set_aspect('equal')
            # buoyancy field
            norm = mcolors.Normalize(vmin=B_range[0], vmax=B_range[-1])
            mappable = cm.ScalarMappable(norm=norm)
            ax8 = fig.add_subplot(1, 4, 4, projection='3d')
            ax8.set(xlim=[0, lx[0]], ylim=[0, lx[1]], zlim=[-lx[2], 0])
            ax8.contourf(X[:, :, -1], Y[:, :, -1], B[:, :, -1], levels, zdir='z', offset=0, norm=norm)
            ax8.contourf(X[-1, :, :], B[ -1, :, :], Z[-1, :, :], levels, zdir='y', offset=0, norm=norm)
            ax8.contourf(B[:, 0, :], Y[:, 0, :], Z[:, 0, :], levels, zdir='x', offset=lx[0], norm=norm)
            ax8.set_title(r"Buoyancy")
            ax8.set(xlabel="y", ylabel="x", zlabel="z")
            #formatter.set_powerlimits((-3, 0)) 
            fig.colorbar(mappable, ax=ax8, label=r"[m/s$^{2}$]", location='bottom', orientation='horizontal', shrink=0.75, format = formatter)
            ax8.set_aspect('equal')

            # --- Save Frame ---
            frame_path = os.path.join(outdir, f"oc_frame_{it:04d}.png")
            plt.tight_layout()
            plt.savefig(frame_path)
            plt.close(fig)
            print(f"Time step {it + 1}/{nt} captured: {frame_path}")

        if vel_plane_slice_video:
            
            outdir = fig_folder + 'NBP plane slices/'
            os.makedirs(outdir, exist_ok=True)
            td = time[it] / 3600 / 24
            fig = plt.figure(figsize=(12, 6))
            fig.tight_layout()
            fig.suptitle(f'{td:.2f} days', y = 0.9, fontsize=12)

            norm = mcolors.Normalize(vmin=u_range[0]*fac, vmax=u_range[-1]*fac)
            ax7 = fig.add_subplot(2, 3, 1)
            ax7.contourf(X[int(nx[1]/2), :, :], Z[int(nx[1]/2), :, :], u[it, int(nx[1]/2), :, :], levels, norm=norm, cmap = 'RdBu_r')
            ax7.set_xlabel("[m]")
            ax7.set_ylabel("Depth [m]")
            ax7.set_title("u")
            ax7.set_ylim(-lx[2], 0)
            ax7.set_xlim(0, lx[0])
            ax7.set_aspect('equal')
            mappable = cm.ScalarMappable(norm=norm, cmap = 'RdBu_r')
            fig.colorbar(mappable, ax=ax7, label=r"m/s", anchor = (0.5, -0.1), orientation='horizontal', shrink=0.75, format = formatter)

            norm = mcolors.Normalize(vmin=v_range[0]*fac, vmax=v_range[-1]*fac)
            ax7 = fig.add_subplot(2, 3, 2)
            ax7.contourf(X[int(nx[1]/2), :, :], Z[int(nx[1]/2), :, :], v[it, int(nx[1]/2), :, :], levels, norm=norm, cmap = 'RdBu_r')
            ax7.set_xlabel("[m]")
            ax7.set_ylabel("Depth [m]")
            ax7.set_title("v")
            ax7.set_ylim(-lx[2], 0)
            ax7.set_xlim(0, lx[0])
            ax7.set_aspect('equal')
            mappable = cm.ScalarMappable(norm=norm, cmap = 'RdBu_r')
            fig.colorbar(mappable, ax=ax7, label=r"m/s", anchor = (0.5, -0.1), orientation='horizontal', shrink=0.75, format = formatter)

            norm = mcolors.Normalize(vmin=w_range[0]*fac, vmax=w_range[-1]*fac)
            ax7 = fig.add_subplot(2, 3, 3)
            ax7.contourf(Xf[int(nx[1]/2), :, :], Zf[int(nx[1]/2), :, :], w[it, int(nx[1]/2), :, :], levels, norm=norm, cmap = 'RdBu_r')
            ax7.set_xlabel("[m]")
            ax7.set_ylabel("Depth [m]")
            ax7.set_title("w")
            ax7.set_ylim(-lx[2], 0)
            ax7.set_xlim(0, lx[0])
            ax7.set_aspect('equal')
            mappable = cm.ScalarMappable(norm=norm, cmap = 'RdBu_r')
            fig.colorbar(mappable, ax=ax7, label=r"m/s", anchor = (0.5, -0.1), orientation='horizontal', shrink=0.75, format = formatter)


            norm = mcolors.Normalize(vmin=B_range[0], vmax=B_range[-1])
            ax7 = fig.add_subplot(2, 3, 4)
            ax7.contourf(X[int(nx[1]/2), :, :], Z[int(nx[1]/2), :, :], B[it, int(nx[1]/2), :, :], levels, norm=norm)
            ax7.set_xlabel("[m]")
            ax7.set_ylabel("Depth [m]")
            ax7.set_title("Buoyancy")
            ax7.set_ylim(-lx[2], 0)
            ax7.set_xlim(0, lx[0])
            ax7.set_aspect('equal')
            mappable = cm.ScalarMappable(norm=norm)
            #formatter.set_powerlimits((-3, 0)) 
            fig.colorbar(mappable, ax=ax7, label=r"m/s$^2$", anchor = (0.5, -0.1), orientation='horizontal', shrink=0.75, format = formatter)

            norm = mcolors.Normalize(vmin=P_d_range[0], vmax=P_d_range[-1])
            mappable = cm.ScalarMappable(norm=norm, cmap='RdBu_r')
            ax8 = fig.add_subplot(2, 3, 5)
            ax8.contourf(X[int(nx[1]/2), :, :], Z[int(nx[1]/2), :, :], Pdynamic[it, int(nx[1]/2), :, :], levels, norm=norm, cmap = 'RdBu_r')
            ax8.set_xlabel("[m]")
            ax8.set_ylabel("Depth [m]")
            ax8.set_title("Hydrodynamic Pressure")
            ax8.set_ylim(-lx[2], 0)
            ax8.set_xlim(0, lx[0])
            ax8.set_aspect('equal')
            #formatter.set_powerlimits((-3, 0)) 
            fig.colorbar(mappable, ax=ax8, label=r"m/s$^{2}$", anchor = (0.5, -0.1), orientation='horizontal', shrink=0.75, format = formatter)

            norm = mcolors.Normalize(vmin=P_s_range[0], vmax=P_s_range[-1])
            mappable = cm.ScalarMappable(norm=norm, cmap='RdBu_r')
            ax9 = fig.add_subplot(2, 3, 6)
            ax9.contourf(X[int(nx[1]/2), :, :], Z[int(nx[1]/2), :, :], Pstatic[it, int(nx[1]/2), :, :], levels, norm=norm, cmap = 'RdBu_r')
            ax9.set_xlabel("[m]")
            ax9.set_ylabel("Depth [m]")
            ax9.set_title("Hydrostatic Pressure")
            ax9.set_ylim(-lx[2], 0)
            ax9.set_xlim(0, lx[0])
            ax9.set_aspect('equal')
            #formatter.set_powerlimits((-2, 0)) 
            fig.colorbar(mappable, ax=ax9, label=r"m/s$^{2}$", anchor = (0.5, -0.1), orientation='horizontal', shrink=0.75, format = formatter)

            # --- Save Frame ---
            frame_path = os.path.join(outdir, f"oc_plane_slices_{it:04d}.png")
            plt.tight_layout()
            plt.savefig(frame_path)
            plt.close(fig)
            print(f"Time step {it + 1}/{nt} captured: {frame_path}")
            plt.close(fig)
    
    print(f"All frames created for case {caseindex + 1}/{ncases}: {case}")
    # creating videos
    if turb_stats_video:
        outdir = fig_folder + 'NBP and turb stats/'
        print("Creating video...")
        filenames = sorted([f for f in os.listdir(outdir) if f.endswith(".png")])
        with imageio.get_writer(fig_folder + name + '_w_turb_stats.mp4', fps=10) as writer:
            for filename in filenames:
                image = imageio.imread(f"{outdir}/{filename}")
                writer.append_data(image)   
    if video_3d_flag:
        outdir = fig_folder + 'oc 3D video frames'
        print("Creating video...")
        filenames = sorted([f for f in os.listdir(outdir) if f.endswith(".png")])
        with imageio.get_writer(fig_folder + name + '_3D.mp4', fps=10) as writer:
            for filename in filenames:
                image = imageio.imread(f"{outdir}/{filename}")
                writer.append_data(image)
    if vel_plane_slice_video:
        print("Creating video...")
        outdir = fig_folder + 'NBP plane slices/'
        filenames = sorted([f for f in os.listdir(outdir) if f.endswith(".png")])
        with imageio.get_writer(fig_folder + name + '_vel_plane_slices.mp4', fps=10) as writer:
            for filename in filenames:
                image = imageio.imread(f"{outdir}/{filename}")
                writer.append_data(image)