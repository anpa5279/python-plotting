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
folder = 'output data/'
fig_folder = 'figures and videos/'
name = 'NBP'
# flags for what to plot
video_3d_flag = True
turb_stats_video = True
plot_dense_scalar = False
with_halos = False

# Parameters
T0 = 25.0
g = -9.80665 # m/s^2
alpha = 2e-4
rho_w = 1026 #reference density
molar_calcite = 100.09/1000.0 # kg/mol
rho_calcite = 2710.0

# List JLD2 files
dtn = sorted([f for f in os.listdir(folder) if f.endswith('.jld2')])
nf = len(dtn)
Nranks = nf #// 2

# Read time steps
fid = os.path.join(folder, dtn[-1])
with h5py.File(fid, 'r') as f:
    timeseries_group = [g for g in f.keys() if 'timeseries' in g][0]
    t_group = f[timeseries_group + '/t']
    t_save = sorted([float(k) for k in t_group.keys()])
    t_save = np.array(t_save)
    time = np.array([t_group[str(int(k))][()] for k in t_save])
    nx = [f['grid/Nx'][()] * Nranks, f['grid/Ny'][()], f['grid/Nz'][()]]
    hx = [f['grid/Hx'][()], f['grid/Hy'][()], f['grid/Hz'][()]]
    lx = [f['grid/Lx'][()], f['grid/Ly'][()], f['grid/Lz'][()]]
    x = f['grid/xᶜᵃᵃ'][:]
    y = f['grid/yᵃᶜᵃ'][:]
    z = f['grid/z/cᵃᵃᶜ'][:]
    xf = f['grid/xᶠᵃᵃ'][:]
    yf = f['grid/yᵃᶠᵃ'][:]
    zf = f['grid/z/cᵃᵃᶠ'][:]
x = x[hx[0]:-hx[0]]
y = y[hx[1]:-hx[1]]
z = z[hx[2]:-hx[2]]
xf = xf[hx[0]:-hx[0]]
yf = yf[hx[1]:-hx[1]]
zf = zf[hx[2]:-hx[2]]
if video_3d_flag or turb_stats_video:
    nt = len(t_save)
else:
    nt = 1  # only last time step
X, Y, Z = np.meshgrid(x, y, z)
Xw, Yw, Zf = np.meshgrid(x, y, zf)
# Initialize arrays
u = np.zeros((nt, nx[0], nx[1], nx[2]))
v = np.zeros((nt, nx[0], nx[1], nx[2]))
w = np.zeros((nt, nx[0], nx[1], nx[2] + 1))
T = np.zeros((nt, nx[0], nx[1], nx[2]))
CaCO3 = np.zeros((nt, nx[0], nx[1], nx[2]))
Pdynamic = np.zeros((nt, nx[0], nx[1], nx[2]))
Pstatic = np.zeros((nt, nx[0], nx[1], nx[2]))
u2_fluc = np.zeros((nt, nx[2]))
v2_fluc = np.zeros((nt, nx[2]))
w2_fluc = np.zeros((nt, nx[2] + 1))
wc2_fluc = np.zeros((nt, nx[2]))
uv_fluc = np.zeros((nt, nx[2]))
uw_fluc = np.zeros((nt, nx[2]))
vw_fluc = np.zeros((nt, nx[2]))
# Load data from files
for it in range(0, nt):
    xrange = range(0, nx[0] // Nranks)
    for r in range(nf):
        fname = os.path.join(folder, dtn[r])
        if "fields" in fname:
            with h5py.File(fname, 'r') as f:
                #if ("0" in fname) or Nranks==1:
                #    u_f =np.array(f["IC/"]["friction_velocity"])
                #    u_s =np.array(f["IC/"]["stokes_velocity"])
                if with_halos:
                    u[it, xrange, :, :] = np.array(f['timeseries/u'][f'{int(t_save[it])}'])[hx[0]:-hx[0], hx[1]:-hx[1], hx[2]:-hx[2]].transpose(2, 1, 0)
                    v[it, xrange, :, :] = np.array(f['timeseries/v'][f'{int(t_save[it])}'])[hx[0]:-hx[0], hx[1]:-hx[1], hx[2]:-hx[2]].transpose(2, 1, 0)
                    w[it, xrange, :, :] = np.array(f['timeseries/w'][f'{int(t_save[it])}'])[hx[0]:-hx[0], hx[1]:-hx[1], hx[2]:-hx[2]].transpose(2, 1, 0)
                    T[it, xrange, :, :] = np.array(f['timeseries/T'][f'{int(t_save[it])}'])[hx[0]:-hx[0], hx[1]:-hx[1], hx[2]:-hx[2]].transpose(2, 1, 0) 
                    CaCO3[it, xrange, :, :] = np.array(f['timeseries/CaCO3'][f'{int(t_save[it])}'])[hx[0]:-hx[0], hx[1]:-hx[1], hx[2]:-hx[2]].transpose(2, 1, 0) 
                else:
                    u[it, xrange, :, :] = np.array(f['timeseries/u'][f'{int(t_save[it])}'])[:, :, :].transpose(2, 1, 0)
                    v[it, xrange, :, :] = np.array(f['timeseries/v'][f'{int(t_save[it])}'])[:, :, :].transpose(2, 1, 0) 
                    w[it, xrange, :, :] = np.array(f['timeseries/w'][f'{int(t_save[it])}'])[:, :, :].transpose(2, 1, 0) 
                    T[it, xrange, :, :] = np.array(f['timeseries/T'][f'{int(t_save[it])}'])[:, :, :].transpose(2, 1, 0) 
                    CaCO3[it, xrange, :, :] = np.array(f['timeseries/CaCO3'][f'{int(t_save[it])}'])[:, :, :].transpose(2, 1, 0) 
                    Pdynamic[it, xrange, :, :] = np.array(f['timeseries/P_dynamic'][f'{int(t_save[it])}'])[:, :, :].transpose(2, 1, 0) 
                    Pstatic[it, xrange, :, :] = np.array(f['timeseries/P_static'][f'{int(t_save[it])}'])[:, :, :].transpose(2, 1, 0) 
                
                new_range = range(xrange.start + (nx[0] // Nranks), xrange.stop + (nx[0] // Nranks))
                xrange = new_range
print(f"Data loaded.")
# nbp calculations
rho_tot = rho_w - rho_w/rho_calcite* CaCO3 * molar_calcite + CaCO3 * molar_calcite
rho_nd = (rho_tot - rho_w)/rho_w
B = g * (alpha * (T0-T) + rho_nd)

#interpolate so all values are from the center, center, center of the grid cell
w_face = make_interp_spline(zf, w, axis=3, k=1)
wc = w_face(z)
u_f = 0.001
u = u #-u_s
#calculate means
u_avg = np.mean(u, axis=(1, 2))
v_avg = np.mean(v, axis=(1, 2))
w_avg = np.mean(w, axis=(1, 2))
wc_avg = np.mean(wc, axis=(1, 2))
T_avg = np.mean(T, axis=(1, 2))
B_avg = np.mean(B, axis=(1, 2))
rho_nd_avg = np.mean(rho_nd, axis=(1, 2))
print("B")
print(B_avg[0, :])
print("concentration influence")
print(rho_nd_avg[0, :])
print("temperature influence")
print(alpha *(T0-T_avg[0, :]))
#calcualte reynolds stresses
for i in range(0, nt):
    u2_fluc[i, :] = fluct2_mean(u[i, :, :, :], u[i, :, :, :], u_avg[i, :], u_avg[i, :])
    v2_fluc[i, :] = fluct2_mean(v[i, :, :, :], v[i, :, :, :], v_avg[i, :], v_avg[i, :])
    w2_fluc[i, :] = fluct2_mean(w[i, :, :, :], w[i, :, :, :], w_avg[i, :], w_avg[i, :])
    wc2_fluc[i, :] = fluct2_mean(wc[i, :, :, :], wc[i, :, :, :], wc_avg[i, :], wc_avg[i, :])
    uv_fluc[i, :] = fluct2_mean(u[i, :, :, :], v[i, :, :, :], u_avg[i, :], v_avg[i, :])
    uw_fluc[i, :] = fluct2_mean(u[i, :, :, :], wc[i, :, :, :], u_avg[i, :], wc_avg[i, :])
    vw_fluc[i, :] = fluct2_mean(v[i, :, :, :], wc[i, :, :, :], v_avg[i, :], wc_avg[i, :])
# rms fluctuations
u_rms = u2_fluc**0.5
v_rms = v2_fluc**0.5
w_rms = w2_fluc**0.5
#calculate richardson number
dbdz = np.gradient(b_avg, z, axis=-1)
Ri_avg  = dbdz / ((np.gradient(u_avg, z, axis=-1))**2 + (np.gradient(v_avg, z, axis=-1))**2)
# normalize w2 fluctuation by friction velocity squared
w2_fluc_nd = w2_fluc/u_f**2

############ PLOTTING ############
# font for plotting 
plt.rcParams['font.family'] = 'serif' # or 'sans-serif' or 'monospace'
plt.rcParams['font.serif'] = 'cmr10'
plt.rcParams['font.sans-serif'] = 'cmss10'
plt.rcParams['font.monospace'] = 'cmtt10'
plt.rcParams["axes.formatter.use_mathtext"] = True # to fix the minus signs
plt.rcParams['font.size'] = 12

# plot ranges
fac = 2*10**(-2)
u_max = 5*10**(-3)#np.abs(u).max()*fac
v_max = 5*10**(-3)#np.abs(v).max()*fac
w_max = np.abs(w).max()*fac
u_range = [-1*u_max, u_max]
v_range = [-1*v_max, v_max]
w_range = [-1*w_max, w_max]
vel_max = np.max([np.abs(u_avg).max(), np.abs(v_avg).max(), np.abs(w_avg).max()])
vel_range = [-0.04, 0.04]#[-1*vel_max, vel_max]
rms_range = [0, 0.02]#np.max([u_rms.max(), v_rms.max(), w_rms.max()])]
#restress_max = np.max([np.abs(uv_fluc).max(), np.abs(uw_fluc).max(), np.abs(vw_fluc).max()])
restress_range = [-3*10**(-5), 3*10**(-5)]#[-1*restress_max, restress_max]
richardson_range = [0, 5*10**(-5)]# Ri_avg.max()]#
w2_range = [0, 1.5]#w2_fluc_nd.max()]
T_range = [T_avg.min(), T_avg.max()]
print("max negative: ", CaCO3.min())
c_range = [0.0, CaCO3.max()*0.5]
P_max = np.abs(Pdynamic).max()*10**(-1)
P_d_range = [-1 * P_max, P_max]
P_max = np.abs(Pstatic).max()*10**(-1)
P_s_range = [-np.abs(Pstatic).max(), np.abs(Pstatic).max()]
B_range = [B.min(), B.max()]
B_avg_range = [B_avg.min(), B_avg.max()]

levels =200
formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((0, 0)) 

# --- Create Video ---
if turb_stats_video and plot_dense_scalar:
    print("Creating video frames...")
    outdir = fig_folder + 'oc video frames NBP and turb stats/'
    os.makedirs(outdir, exist_ok=True)
    filenames = []
    for i in range(0, nt):
        td = time[i] / 3600 / 24
        fig = plt.figure(figsize=(10, 9))
        fig.tight_layout()
        fig.suptitle(f'{td:.2f} days', fontsize=12)
        ax1 = fig.add_subplot(3, 3, 1)  # average velocities
        ax1.plot(u_avg[i, :], z, label=r"$\langle$u$\rangle_{xy}$", color='green')
        ax1.plot(v_avg[i, :], z, label=r"$\langle$v$\rangle_{xy}$", color='red')
        ax1.plot(w_avg[i, :], zf, label=r"$\langle$w$\rangle_{xy}$", color='blue')
        ax1.set_xlabel("[m/s]")
        ax1.set_ylabel("Depth [m]")
        ax1.set_title('Velocity Profiles')
        ax1.set_ylim(-lx[2], 0)
        ax1.set_xlim(vel_range)
        ax1.legend(loc='lower right', handlelength=0.5)

        ax2 = fig.add_subplot(3, 3, 2)  # rms
        ax2.plot(u_rms[i, :], z, label=r"$\langle$u$_{rms}$$\rangle_{xy}$", color='green')
        ax2.plot(v_rms[i, :], z, label=r"$\langle$v$_{rms}$$\rangle_{xy}$", color='red')
        ax2.plot(w_rms[i, :], zf, label=r"$\langle$w$_{rms}$$\rangle_{xy}$", color='blue')
        ax2.set_xlabel("[m/s]")
        ax2.set_ylabel("Depth [m]")
        ax2.set_title("Root Mean Square Velocities")
        ax2.set_ylim(-lx[2], 0)
        ax2.set_xlim(rms_range)
        ax2.legend(loc='lower right', handlelength=0.5)

        ax3 = fig.add_subplot(3, 3, 3)  # Reynolds stress
        ax3.plot(uv_fluc[i, :], z, label=r"$\langle$uv$\rangle_{xy}$", color='green')
        ax3.plot(uw_fluc[i, :], z, label=r"$\langle$uw$\rangle_{xy}$", color='red')
        ax3.plot(vw_fluc[i, :], z, label=r"$\langle$vw$\rangle_{xy}$", color='blue')
        ax3.set_xlabel(r"[m$^2$/s$^2$]")
        ax3.set_ylabel("Depth [m]")
        ax3.set_title("Reynolds Stresses")
        ax3.set_ylim(-lx[2], 0)
        ax3.set_xlim(restress_range)
        ax3.legend(loc='lower right', handlelength=0.5)

        ax4 = fig.add_subplot(3, 3, 4)  # Richardson number
        ax4.plot(Ri_avg[i, :], z, label="Ri", color='black')
        ax4.set_xlabel("Ri")
        ax4.set_ylabel("Depth [m]")
        ax4.set_title("Richardson Number")
        ax4.set_ylim(-lx[2], 0)
        ax4.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True)
        ax4.set_xlim(richardson_range)

        ax5 = fig.add_subplot(3, 3, 5)
        ax5.plot(B_avg[i, :], z, color='black')
        ax5.set_xlabel(r"[m/s$^{2}$]")
        ax5.set_ylabel("Depth [m]")
        ax5.set_title("Buoyancy")
        ax5.set_ylim(-lx[2], 0)
        ax5.set_xlim(B_avg_range)

        ax6 = fig.add_subplot(3, 3, 6)
        ax6.plot(T_avg[i, :], z, label=r"[$^\circ$C]", color='black')
        ax6.set_xlabel(r"Temperature [$^\circ$C]")
        ax6.set_ylabel("Depth [m]")
        ax6.set_title(r"Temperature")
        ax6.set_ylim(-lx[2], 0)
        ax6.set_xlim(T_range)

        # tracer plotting 
        norm = mcolors.Normalize(vmin=c_range[0], vmax=c_range[-1])
        ax7 = fig.add_subplot(3, 3, 7)
        ax7.contourf(X[int(nx[1]/2), :, :], Z[int(nx[1]/2), :, :], CaCO3[i, :, int(nx[1]/2), :], levels, norm=norm)
        ax7.set_xlabel("[m]")
        ax7.set_ylabel("Depth [m]")
        ax7.set_title("Dense scalar")
        ax7.set_ylim(-lx[2], 0)
        ax7.set_xlim(0, lx[0])
        mappable = cm.ScalarMappable(norm=norm)
        fig.colorbar(mappable, ax=ax7, label=r"mol/m$^{3}$", anchor = (0.5, -0.15), orientation='horizontal', shrink=0.75)

        norm = mcolors.Normalize(vmin=P_d_range[0], vmax=P_d_range[-1])
        mappable = cm.ScalarMappable(norm=norm, cmap='RdBu_r')
        ax8 = fig.add_subplot(3, 3, 8)
        ax8.contourf(X[int(nx[1]/2), :, :], Z[int(nx[1]/2), :, :], Pdynamic[i, :, int(nx[1]/2), :], levels, norm=norm, cmap = 'RdBu_r')
        ax8.set_xlabel("[m]")
        ax8.set_ylabel("Depth [m]")
        ax8.set_title("Hydrodynamic Pressure")
        ax8.set_ylim(-lx[2], 0)
        ax8.set_xlim(0, lx[0])
        fig.colorbar(mappable, ax=ax8, label=r"m/s$^{2}$", anchor = (0.5, -0.15), orientation='horizontal', shrink=0.75)

        norm = mcolors.Normalize(vmin=B_range[0], vmax=B_range[-1])
        ax9 = fig.add_subplot(3, 3, 9)
        ax9.contourf(X[int(nx[1]/2), :, :], Z[int(nx[1]/2), :, :], B[i, :, int(nx[1]/2), :], levels, norm=norm)
        ax9.set_xlabel("[m]")
        ax9.set_ylabel("Depth [m]")
        ax9.set_title("Buoyancy")
        ax9.set_ylim(-lx[2], 0)
        ax9.set_xlim(0, lx[0])
        mappable = cm.ScalarMappable(norm=norm)
        fig.colorbar(mappable, ax=ax9, anchor = (0.5, -0.15), orientation='horizontal', shrink=0.75)

        # --- Save Frame ---
        frame_path = os.path.join(outdir, f"oc_frame_{i:04d}.png")
        plt.tight_layout()
        plt.savefig(frame_path)
        plt.close(fig)
        filenames.append(frame_path)
        print(f"Time step {i + 1}/{nt} captured: {frame_path}")
        plt.close(fig)
    print("Creating video...")
    filenames = sorted([f for f in os.listdir(outdir) if f.endswith(".png")])
    with imageio.get_writer(fig_folder + name + '_w_turb_stats.mp4', fps=10) as writer:
        for filename in filenames:
            image = imageio.imread(f"{outdir}/{filename}")
            writer.append_data(image)
elif turb_stats_video and not plot_dense_scalar:
    print("Creating video frames...")
    outdir = fig_folder + 'oc video frames NBP and turb stats/'
    os.makedirs(outdir, exist_ok=True)
    filenames = []
    for i in range(0, nt):
        td = time[i] / 3600 / 24
        fig = plt.figure(figsize=(10, 9))
        fig.tight_layout()
        fig.suptitle(f'{td:.2f} days', fontsize=12)
        ax1 = fig.add_subplot(3, 3, 1)  # average velocities
        ax1.plot(u_avg[i, :], z, label=r"$\langle$u$\rangle_{xy}$", color='green')
        ax1.plot(v_avg[i, :], z, label=r"$\langle$v$\rangle_{xy}$", color='red')
        ax1.plot(w_avg[i, :], zf, label=r"$\langle$w$\rangle_{xy}$", color='blue')
        ax1.set_xlabel("[m/s]")
        ax1.set_ylabel("Depth [m]")
        ax1.set_title('Velocity Profiles')
        ax1.set_ylim(-lx[2], 0)
        ax1.set_xlim(vel_range)
        ax1.legend(loc='lower right', handlelength=0.5)

        ax2 = fig.add_subplot(3, 3, 2)  # rms
        ax2.plot(u_rms[i, :], z, label=r"$\langle$u$_{rms}$$\rangle_{xy}$", color='green')
        ax2.plot(v_rms[i, :], z, label=r"$\langle$v$_{rms}$$\rangle_{xy}$", color='red')
        ax2.plot(w_rms[i, :], zf, label=r"$\langle$w$_{rms}$$\rangle_{xy}$", color='blue')
        ax2.set_xlabel("[m/s]")
        ax2.set_ylabel("Depth [m]")
        ax2.set_title("Root Mean Square Velocities")
        ax2.set_ylim(-lx[2], 0)
        ax2.set_xlim(rms_range)
        ax2.legend(loc='lower right', handlelength=0.5)

        ax3 = fig.add_subplot(3, 3, 3)  # Reynolds stress
        ax3.plot(uv_fluc[i, :], z, label=r"$\langle$uv$\rangle_{xy}$", color='green')
        ax3.plot(uw_fluc[i, :], z, label=r"$\langle$uw$\rangle_{xy}$", color='red')
        ax3.plot(vw_fluc[i, :], z, label=r"$\langle$vw$\rangle_{xy}$", color='blue')
        ax3.set_xlabel(r"[m$^2$/s$^2$]")
        ax3.set_ylabel("Depth [m]")
        ax3.set_title("Reynolds Stresses")
        ax3.set_ylim(-lx[2], 0)
        ax3.set_xlim(restress_range)
        ax3.legend(loc='lower right', handlelength=0.5)

        ax4 = fig.add_subplot(3, 3, 4)  # Richardson number
        ax4.plot(Ri_avg[i, :], z, label="Ri", color='black')
        ax4.set_xlabel("Ri")
        ax4.set_ylabel("Depth [m]")
        ax4.set_title("Richardson Number")
        ax4.set_ylim(-lx[2], 0)
        ax4.ticklabel_format(axis='x', style='sci', scilimits=(-1,1), useMathText=True)
        ax4.set_xlim(richardson_range)

        ax5 = fig.add_subplot(3, 3, 5)
        ax5.plot(B_avg[i, :], z, color='black')
        ax5.set_xlabel(r"[m/s$^{2}$]")
        ax5.set_ylabel("Depth [m]")
        ax5.set_title("Buoyancy")
        ax5.set_ylim(-lx[2], 0)
        ax5.set_xlim(B_avg_range)

        ax6 = fig.add_subplot(3, 3, 6)
        ax6.plot(T_avg[i, :], z, label=r"[$^\circ$C]", color='black')
        ax6.set_xlabel(r"Temperature [$^\circ$C]")
        ax6.set_ylabel("Depth [m]")
        ax6.set_title(r"Temperature")
        ax6.set_ylim(-lx[2], 0)
        ax6.set_xlim(T_range)

        norm = mcolors.Normalize(vmin=B_range[0], vmax=B_range[-1])
        ax9 = fig.add_subplot(3, 3, 7)
        ax9.contourf(X[int(nx[1]/2), :, :], Z[int(nx[1]/2), :, :], B[i, :, int(nx[1]/2), :], levels, norm=norm)
        ax9.set_xlabel("[m]")
        ax9.set_ylabel("Depth [m]")
        ax9.set_title("Buoyancy")
        ax9.set_ylim(-lx[2], 0)
        ax9.set_xlim(0, lx[0])
        mappable = cm.ScalarMappable(norm=norm)
        fig.colorbar(mappable, ax=ax9, anchor = (0.5, -0.15), orientation='horizontal', shrink=0.75)
        
        norm = mcolors.Normalize(vmin=P_d_range[0], vmax=P_d_range[-1])
        mappable = cm.ScalarMappable(norm=norm, cmap='RdBu_r')
        ax8 = fig.add_subplot(3, 3, 8)
        ax8.contourf(X[int(nx[1]/2), :, :], Z[int(nx[1]/2), :, :], Pdynamic[i, :, int(nx[1]/2), :], levels, norm=norm, cmap = 'RdBu_r')
        ax8.set_xlabel("[m]")
        ax8.set_ylabel("Depth [m]")
        ax8.set_title("Hydrodynamic Pressure")
        ax8.set_ylim(-lx[2], 0)
        ax8.set_xlim(0, lx[0])
        fig.colorbar(mappable, ax=ax8, label=r"m/s$^{2}$", anchor = (0.5, -0.15), orientation='horizontal', shrink=0.75)

        norm = mcolors.Normalize(vmin=P_s_range[0], vmax=P_s_range[-1])
        mappable = cm.ScalarMappable(norm=norm, cmap='RdBu_r')
        ax9 = fig.add_subplot(3, 3, 9)
        ax9.contourf(X[int(nx[1]/2), :, :], Z[int(nx[1]/2), :, :], Pstatic[i, :, int(nx[1]/2), :], levels, norm=norm, cmap = 'RdBu_r')
        ax9.set_xlabel("[m]")
        ax9.set_ylabel("Depth [m]")
        ax9.set_title("Hydrostatic Pressure")
        ax9.set_ylim(-lx[2], 0)
        ax9.set_xlim(0, lx[0])
        fig.colorbar(mappable, ax=ax9, label=r"m/s$^{2}$", anchor = (0.5, -0.15), orientation='horizontal', shrink=0.75)

        # --- Save Frame ---
        frame_path = os.path.join(outdir, f"oc_frame_{i:04d}.png")
        plt.tight_layout()
        plt.savefig(frame_path)
        plt.close(fig)
        filenames.append(frame_path)
        print(f"Time step {i + 1}/{nt} captured: {frame_path}")
        plt.close(fig)
    print("Creating video...")
    filenames = sorted([f for f in os.listdir(outdir) if f.endswith(".png")])
    with imageio.get_writer(fig_folder + name + '_no-scalar_turb_stats.mp4', fps=10) as writer:
        for filename in filenames:
            image = imageio.imread(f"{outdir}/{filename}")
            writer.append_data(image)
else:
    # collecting ranges for plotting and other values
    td = time[-1] / 3600 / 24
    title = f'{td:.2f} days'  #
    i = -1  # last time step

    print("Creating figure")
    # --- Create Figure ---
    fig = plt.figure(figsize=(18.3, 4))
    fig.suptitle(title)
    fig.tight_layout()
    # --- Vertical Profiles ---
    ax1 = fig.add_subplot(1, 5, 1)  # average velocities
    ax1.plot(u_avg[i, :], z, label=r"$\langle$u$\rangle_{xy}$", color='green')
    ax1.plot(v_avg[i, :], z, label=r"$\langle$v$\rangle_{xy}$", color='red')
    ax1.plot(w_avg[i, :], zf, label=r"$\langle$w$\rangle_{xy}$", color='blue')
    ax1.set_xlabel("[m/s]")
    ax1.set_ylabel("Depth [m]")
    ax1.set_title('Velocity Profiles')
    ax1.set_ylim(-lx[2], 0)
    ax1.set_xlim(vel_range)
    ax1.legend(loc='lower right')

    ax2 = fig.add_subplot(1, 5, 2)  # rms
    ax2.plot(u_rms[i, :], z, label=r"$\langle$u$_{rms}$$\rangle_{xy}$", color='green')
    ax2.plot(v_rms[i, :], z, label=r"$\langle$v$_{rms}$$\rangle_{xy}$", color='red')
    ax2.plot(w_rms[i, :], zf, label=r"$\langle$w$_{rms}$$\rangle_{xy}$", color='blue')
    ax2.set_xlabel("[m/s]")
    ax2.set_ylabel("Depth [m]")
    ax2.set_title("Root Mean Square Velocities")
    ax2.set_ylim(-lx[2], 0)
    ax2.set_xlim(rms_range)
    ax2.legend(loc='lower right')

    ax3 = fig.add_subplot(1, 5, 3)  # Reynolds stress
    ax3.plot(uv_fluc[i, :], z, label=r"$\langle$uv$\rangle_{xy}$", color='green')
    ax3.plot(uw_fluc[i, :], z, label=r"$\langle$uw$\rangle_{xy}$", color='red')
    ax3.plot(vw_fluc[i, :], z, label=r"$\langle$vw$\rangle_{xy}$", color='blue')
    ax3.set_xlabel(r"[m$^2$/s$^2$]")
    ax3.set_ylabel("Depth [m]")
    ax3.set_title("Reynolds Stresses")
    ax3.set_ylim(-lx[2], 0)
    ax3.set_xlim(restress_range)
    ax3.legend(loc='lower right')

    ax4 = fig.add_subplot(1, 5, 4)  # Richardson number
    ax4.plot(Ri_avg[i, :], z, label="Ri", color='black')
    ax4.set_xlabel("Ri")
    ax4.set_ylabel("Depth [m]")
    ax4.set_title("Richardson Number")
    ax4.set_ylim(-lx[2], 0)
    ax4.set_xlim(0.0, Ri_avg[i, :].max())

    ax5 = fig.add_subplot(1, 5, 5)
    ax5.plot(w2_fluc_nd[i, :], zf, label="Ri", color='black')
    ax5.set_xlabel(r"$\langle $w$^{\prime 2} \rangle_{xy}/$u$_{f}^{2}$")
    ax5.set_ylabel("Depth [m]")
    ax5.set_title(r"$\langle $w$^{\prime 2} \rangle_{xy}/$u$_{f}^{2}$")
    ax5.set_ylim(-lx[2], 0)
    ax5.set_xlim(w2_range)

    # --- Save Figure ---
    frame_path = os.path.join(fig_folder, name + '-turb-stats-oc.svg')
    plt.tight_layout()
    plt.savefig(frame_path)
    plt.close(fig)

if video_3d_flag:
    print("Creating video frames...")
    outdir = fig_folder + 'oc 3D video frames'
    os.makedirs(outdir, exist_ok=True)
    filenames = []
    for i in range(0, nt):
        td = time[i] / 3600 / 24
        fig = plt.figure(figsize=(12, 8))
        fig.tight_layout()
        fig.suptitle(f'{td:.2f} days', fontsize=12)

        # --- 3D Plots ---
        # u velocity
        norm = mcolors.Normalize(vmin=u_range[0], vmax=u_range[-1])
        mappable = cm.ScalarMappable(norm=norm, cmap='RdBu_r')
        ax5 = fig.add_subplot(2, 3, 1, projection='3d')
        ax5.set(xlim=[0, lx[0]], ylim=[0, lx[1]], zlim=[-lx[2], 0])
        ax5.contourf(X[:, :, -1], Y[:, :, -1], u[i, :, :, -1], levels, zdir='z', offset=0, cmap='RdBu_r', norm=norm)
        ax5.contourf(X[-1, :, :], u[i, -1, :, :], Z[-1, :, :], levels, zdir='y', offset=0, cmap='RdBu_r', norm=norm)
        ax5.contourf(u[i, :, 0, :], Y[:, 0, :], Z[:, 0, :], levels, zdir='x', offset=lx[0], cmap='RdBu_r', norm=norm)
        ax5.set_title('u')
        ax5.set(xlabel="y", ylabel="x", zlabel="z")
        fig.colorbar(mappable, ax=ax5, label='[m/s]', location='bottom', orientation='horizontal', shrink=0.5, format = formatter)
        ax5.set_aspect('equal')
        # w velocity
        norm = mcolors.Normalize(vmin=w_range[0], vmax=w_range[-1])
        mappable = cm.ScalarMappable(norm=norm, cmap='RdBu_r')
        ax6 = fig.add_subplot(2, 3, 2, projection='3d')
        ax6.set(xlim=[0, lx[0]], ylim=[0, lx[1]], zlim=[-lx[2], 0])
        ax6.contourf(Xw[:, :, -1], Yw[:, :, -2], w[i, :, :, -2], levels, zdir='z', offset=0, cmap='RdBu_r', norm=norm)
        ax6.contourf(Xw[-1, :, :], w[i, -1, :, :], Zf[-1, :, :], levels, zdir='y', offset=0, cmap='RdBu_r', norm=norm)
        ax6.contourf(w[i, :, 0, :], Yw[:, 0, :], Zf[:, 0, :], levels, zdir='x', offset=lx[0], cmap='RdBu_r', norm=norm)
        ax6.set_title('w')
        ax6.set(xlabel="y", ylabel="x", zlabel="z")
        fig.colorbar(mappable, ax=ax6, label='[m/s]', location='bottom', orientation='horizontal', shrink=0.5, format = formatter)
        ax6.set_aspect('equal')
        # v velocity
        norm = mcolors.Normalize(vmin=v_range[0], vmax=v_range[-1])
        mappable = cm.ScalarMappable(norm=norm, cmap='RdBu_r')
        ax7 = fig.add_subplot(2, 3, 3, projection='3d')
        ax7.set(xlim=[0, lx[0]], ylim=[0, lx[1]], zlim=[-lx[2], 0])
        ax7.contourf(X[:, :, -1], Y[:, :, -1], v[i, :, :, -1], levels, zdir='z', offset=0, cmap='RdBu_r', norm=norm)
        ax7.contourf(X[-1, :, :], v[i, -1, :, :], Z[-1, :, :], levels, zdir='y', offset=0, cmap='RdBu_r', norm=norm)
        ax7.contourf(v[i, :, 0, :], Y[:, 0, :], Z[:, 0, :], levels, zdir='x', offset=lx[0], cmap='RdBu_r', norm=norm)
        ax7.set_title('v')
        ax7.set(xlabel="y", ylabel="x", zlabel="z")
        fig.colorbar(mappable, ax=ax7, label='[m/s]', location='bottom', orientation='horizontal', shrink=0.5, format = formatter)
        ax7.set_aspect('equal')
        # buoyancy field
        norm = mcolors.Normalize(vmin=B_range[0], vmax=B_range[-1])
        mappable = cm.ScalarMappable(norm=norm)
        ax8 = fig.add_subplot(2, 3, 4, projection='3d')
        ax8.set(xlim=[0, lx[0]], ylim=[0, lx[1]], zlim=[-lx[2], 0])
        ax8.contourf(X[:, :, -1], Y[:, :, -1], B[i, :, :, -1], levels, zdir='z', offset=0, norm=norm)
        ax8.contourf(X[-1, :, :], B[i, -1, :, :], Z[-1, :, :], levels, zdir='y', offset=0, norm=norm)
        ax8.contourf(B[i, :, 0, :], Y[:, 0, :], Z[:, 0, :], levels, zdir='x', offset=lx[0], norm=norm)
        ax8.set_title(r"Buoyancy")
        ax8.set(xlabel="y", ylabel="x", zlabel="z")
        fig.colorbar(mappable, ax=ax8, label=r"[m/s$^{2}$]", location='bottom', orientation='horizontal', shrink=0.5, format = formatter)
        ax8.set_aspect('equal')
        # tracer field
        norm = mcolors.Normalize(vmin=c_range[0], vmax=c_range[-1])
        mappable = cm.ScalarMappable(norm=norm)
        ax5 = fig.add_subplot(2, 3, 5, projection='3d')
        ax5.set(xlim=[0, lx[0]], ylim=[0, lx[1]], zlim=[-lx[2], 0])
        ax5.contourf(X[:, :, -1], Y[:, :, -1], CaCO3[i, :, :, -1], levels, zdir='z', offset=0, norm=norm)
        ax5.contourf(X[-1, :, :], CaCO3[i, -1, :, :], Z[-1, :, :], levels, zdir='y', offset=0, norm=norm)
        ax5.contourf(CaCO3[i, :, 0, :], Y[:, 0, :], Z[:, 0, :], levels, zdir='x', offset=lx[0], norm=norm)
        ax5.set_title(r"Dense Tracer")
        ax5.set(xlabel="y", ylabel="x", zlabel="z")
        fig.colorbar(mappable, ax=ax5, label=r"[mol/m$^{3}$]", location='bottom', orientation='horizontal', shrink=0.5, format = formatter)
        ax5.set_aspect('equal')
        # tracer field
        norm = mcolors.Normalize(vmin=T_range[0], vmax=T_range[-1])
        mappable = cm.ScalarMappable(norm=norm)
        ax6 = fig.add_subplot(2, 3, 6, projection='3d')
        ax6.set(xlim=[0, lx[0]], ylim=[0, lx[1]], zlim=[-lx[2], 0])
        ax6.contourf(X[:, :, -1], Y[:, :, -1], T[i, :, :, -1], levels, zdir='z', offset=0, norm=norm)
        ax6.contourf(X[-1, :, :], T[i, -1, :, :], Z[-1, :, :], levels, zdir='y', offset=0, norm=norm)
        ax6.contourf(T[i, :, 0, :], Y[:, 0, :], Z[:, 0, :], levels, zdir='x', offset=lx[0], norm=norm)
        ax6.set_title(r"Temperature")
        ax6.set(xlabel="y", ylabel="x", zlabel="z")
        fig.colorbar(mappable, ax=ax6, label=r"[$^\circ$C]", location='bottom', orientation='horizontal', shrink=0.5)
        ax6.set_aspect('equal')

        # --- Save Frame ---
        frame_path = os.path.join(outdir, f"oc_frame_{i:04d}.png")
        plt.tight_layout()
        plt.savefig(frame_path)
        plt.close(fig)
        filenames.append(frame_path)
        print(f"Time step {i + 1}/{nt} captured: {frame_path}")

    print("Creating video...")
    filenames = sorted([f for f in os.listdir(outdir) if f.endswith(".png")])
    with imageio.get_writer(fig_folder + name + '_3D.mp4', fps=10) as writer:
        for filename in filenames:
            image = imageio.imread(f"{outdir}/{filename}")
            writer.append_data(image)
