import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from scipy.interpolate import make_interp_spline
import imageio.v2 as imageio

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
    # Compute squared fluctuation a'*B1' = (a-a_avg)(B1-b_avg)
    ab_fluct = (a - a_avg)*(b - b_avg)
    # Mean over x and y, result shape: (nt, nz,)
    ab_fluct_avg = np.mean(ab_fluct, axis=(0, 1))

    return ab_fluct_avg
def stokes_exp(z):
    g_Earth = 9.80665
    wavelength = 60.0 #m
    amplitude = 0.8 #m
    wavenumber = 2 * np.pi / wavelength
    frequency = np.sqrt(g_Earth * wavenumber)
    vert_scale = wavelength / (4 * np.pi)
    us = 0.05501259798225732#amplitude**2* wavenumber* frequency #0.05501259798225732#
    return us*np.exp(z/vert_scale)
# Set up folder1 and simulation parameters
case1 = 'Flux'
case2 = 'Value'
folder1 = 'output data/data1/'
folder2 = 'output data/data2/'
fig_folder = 'figures and videos/'
# flags for what to plot
turb_stats_video = False
with_halos = False
differences = True
# Parameters
T0 = 25.0
g = -9.81  # m/s^2
alpha = 2e-4

# FIRST CASE DATA AND CALCULATIONS
dtn = sorted([f for f in os.listdir(folder1) if f.endswith('.jld2')])
nf = len(dtn)
Nranks1 = nf #// 2

# Read time steps
fid = os.path.join(folder1, dtn[-1])
with h5py.File(fid, 'r') as f:
    timeseries_group = [g for g in f.keys() if 'timeseries' in g][0]
    t_group = f[timeseries_group + '/t']
    t_save = sorted([float(k) for k in t_group.keys()])
    t_save = np.array(t_save)
    time = np.array([t_group[str(int(k))][()] for k in t_save])
    nx = [f['grid/Nx'][()] * Nranks1, f['grid/Ny'][()], f['grid/Nz'][()]]
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
if turb_stats_video or differences:
    nt = len(t_save)
else:
    nt = 1  # only last time step
X, Y, Z = np.meshgrid(x, y, z)
Xw, Yw, Zf = np.meshgrid(x, y, zf)
# Initialize arrays
u1 = np.zeros((nt, nx[0], nx[1], nx[2]))
v1 = np.zeros((nt, nx[0], nx[1], nx[2]))
w1 = np.zeros((nt, nx[0], nx[1], nx[2] + 1))
T1 = np.zeros((nt, nx[0], nx[1], nx[2]))
u2_fluc1 = np.zeros((nt, nx[2]))
v2_fluc1 = np.zeros((nt, nx[2]))
w2_fluc1 = np.zeros((nt, nx[2] + 1))
wc2_fluc1 = np.zeros((nt, nx[2]))
uv_fluc1 = np.zeros((nt, nx[2]))
uw_fluc1 = np.zeros((nt, nx[2]))
vw_fluc1 = np.zeros((nt, nx[2]))
# Load data from files
for it in range(0, nt):
    xrange = range(0, nx[0] // Nranks1)
    for r in range(nf):
        fname = os.path.join(folder1, dtn[r])
        if "fields" in fname:
            with h5py.File(fname, 'r') as f:
                if ("0" in fname) or Nranks1==1:
                    u_f1 =np.array(f["IC/"]["friction_velocity"])
                #    u_s =np.array(f["IC/"]["stokes_velocity"])
                if with_halos:
                    u1[it, xrange, :, :] = np.array(f['timeseries/u'][f'{int(t_save[it])}'])[hx[0]:-hx[0], hx[1]:-hx[1], hx[2]:-hx[2]].transpose(2, 1, 0)
                    v1[it, xrange, :, :] = np.array(f['timeseries/v'][f'{int(t_save[it])}'])[hx[0]:-hx[0], hx[1]:-hx[1], hx[2]:-hx[2]].transpose(2, 1, 0)
                    w1[it, xrange, :, :] = np.array(f['timeseries/w'][f'{int(t_save[it])}'])[hx[0]:-hx[0], hx[1]:-hx[1], hx[2]:-hx[2]].transpose(2, 1, 0)
                    T1[it, xrange, :, :] = np.array(f['timeseries/T'][f'{int(t_save[it])}'])[hx[0]:-hx[0], hx[1]:-hx[1], hx[2]:-hx[2]].transpose(2, 1, 0) 
                else:
                    u1[it, xrange, :, :] = np.array(f['timeseries/u'][f'{int(t_save[it])}'])[:, :, :].transpose(2, 1, 0)
                    v1[it, xrange, :, :] = np.array(f['timeseries/v'][f'{int(t_save[it])}'])[:, :, :].transpose(2, 1, 0) 
                    w1[it, xrange, :, :] = np.array(f['timeseries/w'][f'{int(t_save[it])}'])[:, :, :].transpose(2, 1, 0) 
                    T1[it, xrange, :, :] = np.array(f['timeseries/T'][f'{int(t_save[it])}'])[:, :, :].transpose(2, 1, 0) 
                
                new_range = range(xrange.start + (nx[0] // Nranks1), xrange.stop + (nx[0] // Nranks1))
                xrange = new_range
print(case1 + " data loaded.")
B1 = -g * alpha * (T1 - T0)
#interpolate so all values are from the center, center, center of the grid cell
w_face = make_interp_spline(zf, w1, axis=3, k=1)
wc1 = w_face(z)
#convert from lagrangian to eulerian
#u_s= stokes_exp(z)
#u1 = u1-u_s

#calculate means
u_avg1 = np.mean(u1, axis=(1, 2))
v_avg1 = np.mean(v1, axis=(1, 2))
w_avg1 = np.mean(w1, axis=(1, 2))
wc_avg = np.mean(wc1, axis=(1, 2))
T_avg1 = np.mean(T1, axis=(1, 2))
B_avg1 = np.mean(B1, axis=(1, 2))
#calcualte reynolds stresses
for i in range(0, nt):
    u2_fluc1[i, :] = fluct2_mean(u1[i, :, :, :], u1[i, :, :, :], u_avg1[i, :], u_avg1[i, :])
    v2_fluc1[i, :] = fluct2_mean(v1[i, :, :, :], v1[i, :, :, :], v_avg1[i, :], v_avg1[i, :])
    w2_fluc1[i, :] = fluct2_mean(w1[i, :, :, :], w1[i, :, :, :], w_avg1[i, :], w_avg1[i, :])
    wc2_fluc1[i, :] = fluct2_mean(wc1[i, :, :, :], wc1[i, :, :, :], wc_avg[i, :], wc_avg[i, :])
    uv_fluc1[i, :] = fluct2_mean(u1[i, :, :, :], v1[i, :, :, :], u_avg1[i, :], v_avg1[i, :])
    uw_fluc1[i, :] = fluct2_mean(u1[i, :, :, :], wc1[i, :, :, :], u_avg1[i, :], wc_avg[i, :])
    vw_fluc1[i, :] = fluct2_mean(v1[i, :, :, :], wc1[i, :, :, :], v_avg1[i, :], wc_avg[i, :])
# rms fluctuations
u_rms1 = u2_fluc1**0.5
v_rms1 = v2_fluc1**0.5
w_rms1 = w2_fluc1**0.5
#calculate richardson number
dbdz = np.gradient(B1, z, axis=3)
Ri1 = dbdz / ((np.gradient(u1, z, axis=3))**2 + (np.gradient(v1, z, axis=3))**2)**0.5
Ri_avg1 = np.mean(Ri1, axis=(1,2))
# normalize w2 fluctuation by friction velocity squared
w2_fluc_nd1 = w2_fluc1/u_f1**2

# SECOND CASE DATA AND CALCULATIONS
dtn = sorted([f for f in os.listdir(folder2) if f.endswith('.jld2')])
nf = len(dtn)
Nranks2 = nf #// 2

fid = os.path.join(folder2, dtn[-1])
with h5py.File(fid, 'r') as f:
    timeseries_group = [g for g in f.keys() if 'timeseries' in g][0]
    t_group = f[timeseries_group + '/t']
    t_save = sorted([float(k) for k in t_group.keys()])
    t_save = np.array(t_save)
    time = np.array([t_group[str(int(k))][()] for k in t_save])
# Initialize arrays
u2 = np.zeros((nt, nx[0], nx[1], nx[2]))
v2 = np.zeros((nt, nx[0], nx[1], nx[2]))
w2 = np.zeros((nt, nx[0], nx[1], nx[2] + 1))
T2 = np.zeros((nt, nx[0], nx[1], nx[2]))
u2_fluc2 = np.zeros((nt, nx[2]))
v2_fluc2 = np.zeros((nt, nx[2]))
w2_fluc2 = np.zeros((nt, nx[2] + 1))
wc2_fluc2 = np.zeros((nt, nx[2]))
uv_fluc2 = np.zeros((nt, nx[2]))
uw_fluc2 = np.zeros((nt, nx[2]))
vw_fluc2 = np.zeros((nt, nx[2]))
# Load data from files
for it in range(0, nt):
    xrange = range(0, nx[0] // Nranks2)
    for r in range(nf):
        fname = os.path.join(folder2, dtn[r])
        if "fields" in fname:
            with h5py.File(fname, 'r') as f:
                if ("0" in fname) or Nranks2==1:
                    u_f2 =np.array(f["IC/"]["friction_velocity"])
                #    u_s =np.array(f["IC/"]["stokes_velocity"])
                if with_halos:
                    u2[it, xrange, :, :] = np.array(f['timeseries/u'][f'{int(t_save[it])}'])[hx[0]:-hx[0], hx[1]:-hx[1], hx[2]:-hx[2]].transpose(2, 1, 0)
                    v2[it, xrange, :, :] = np.array(f['timeseries/v'][f'{int(t_save[it])}'])[hx[0]:-hx[0], hx[1]:-hx[1], hx[2]:-hx[2]].transpose(2, 1, 0)
                    w2[it, xrange, :, :] = np.array(f['timeseries/w'][f'{int(t_save[it])}'])[hx[0]:-hx[0], hx[1]:-hx[1], hx[2]:-hx[2]].transpose(2, 1, 0)
                    T2[it, xrange, :, :] = np.array(f['timeseries/T'][f'{int(t_save[it])}'])[hx[0]:-hx[0], hx[1]:-hx[1], hx[2]:-hx[2]].transpose(2, 1, 0) 
                else:
                    u2[it, xrange, :, :] = np.array(f['timeseries/u'][f'{int(t_save[it])}'])[:, :, :].transpose(2, 1, 0)
                    v2[it, xrange, :, :] = np.array(f['timeseries/v'][f'{int(t_save[it])}'])[:, :, :].transpose(2, 1, 0) 
                    w2[it, xrange, :, :] = np.array(f['timeseries/w'][f'{int(t_save[it])}'])[:, :, :].transpose(2, 1, 0) 
                    T2[it, xrange, :, :] = np.array(f['timeseries/T'][f'{int(t_save[it])}'])[:, :, :].transpose(2, 1, 0) 
                
                new_range = range(xrange.start + (nx[0] // Nranks2), xrange.stop + (nx[0] // Nranks2))
                xrange = new_range
print(case2 + " data loaded.")
T2 = T2
B2 = -g * alpha * (T2 - T0)
#interpolate so all values are from the center, center, center of the grid cell
w_face = make_interp_spline(zf, w2, axis=3, k=1)
wc2 = w_face(z)
#convert from lagrangian to eulerian
#u_s= stokes_exp(z)
#u2 = u2-u_s

#calculate means
u_avg2 = np.mean(u2, axis=(1, 2))
v_avg2 = np.mean(v2, axis=(1, 2))
w_avg2 = np.mean(w2, axis=(1, 2))
wc_avg2 = np.mean(wc2, axis=(1, 2))
T_avg2 = np.mean(T2, axis=(1, 2))
B_avg2 = np.mean(B2, axis=(1, 2))
#calcualte reynolds stresses
for i in range(0, nt):
    u2_fluc2[i, :] = fluct2_mean(u2[i, :, :, :], u2[i, :, :, :], u_avg2[i, :], u_avg2[i, :])
    v2_fluc2[i, :] = fluct2_mean(v2[i, :, :, :], v2[i, :, :, :], v_avg2[i, :], v_avg2[i, :])
    w2_fluc2[i, :] = fluct2_mean(w2[i, :, :, :], w2[i, :, :, :], w_avg2[i, :], w_avg2[i, :])
    wc2_fluc2[i, :] = fluct2_mean(wc2[i, :, :, :], wc2[i, :, :, :], wc_avg2[i, :], wc_avg2[i, :])
    uv_fluc2[i, :] = fluct2_mean(u2[i, :, :, :], v2[i, :, :, :], u_avg2[i, :], v_avg2[i, :])
    uw_fluc2[i, :] = fluct2_mean(u2[i, :, :, :], wc2[i, :, :, :], u_avg2[i, :], wc_avg2[i, :])
    vw_fluc2[i, :] = fluct2_mean(v2[i, :, :, :], wc2[i, :, :, :], v_avg2[i, :], wc_avg2[i, :])
# rms fluctuations
u_rms2 = u2_fluc2**0.5
v_rms2 = v2_fluc2**0.5
w_rms2 = w2_fluc2**0.5
#calculate richardson number
dbdz = np.gradient(B2, z, axis=3)
Ri2 = dbdz / ((np.gradient(u2, z, axis=3))**2 + (np.gradient(v2, z, axis=3))**2)**0.5
Ri_avg2 = np.mean(Ri2, axis=(1,2))
# normalize w2 fluctuation by friction velocity squared
w2_fluc_nd2 = w2_fluc2/u_f2**2

############ PLOTTING ############
# font for plotting 
plt.rcParams['font.family'] = 'serif' # or 'sans-serif' or 'monospace'
plt.rcParams['font.serif'] = 'cmr10'
plt.rcParams['font.sans-serif'] = 'cmss10'
plt.rcParams['font.monospace'] = 'cmtt10'
plt.rcParams["axes.formatter.use_mathtext"] = True # to fix the minus signs
plt.rcParams['font.size'] = 16

# plot ranges
u_max = np.max([np.abs(u1).max(), np.abs(u2).max()])
v_max = np.max([np.abs(v1).max(), np.abs(v2).max()])
w_max = np.max([np.abs(w1).max(), np.abs(w2).max()])
u_range = [-1*u_max, u_max]
v_range = [-1*v_max, v_max]
w_range = [-1*w_max, w_max]
vel_max = np.max([np.abs(u_avg1).max(), np.abs(v_avg1).max(), np.abs(w_avg1).max(), np.abs(u_avg2).max(), np.abs(v_avg2).max(), np.abs(w_avg2).max()])
vel_range = [-1*vel_max, vel_max] #[-0.04, 0.04]#
rms_range = [0, np.max([u_rms1.max(), v_rms1.max(), w_rms1.max(), u_rms2.max(), v_rms2.max(), w_rms2.max()])]#0.02]#
restress_max = np.max([np.abs(uv_fluc1).max(), np.abs(uw_fluc1).max(), np.abs(vw_fluc1).max(), np.abs(uv_fluc2).max(), np.abs(uw_fluc2).max(), np.abs(vw_fluc2).max()])
restress_range = [-1*restress_max, restress_max] #[-3*10**(-5), 3*10**(-5)]#
richardson_range = [0, Ri_avg1.max()] #3*10**(-6)]#
w2_range = [0, w2_fluc_nd1.max()]#3]#
T_range = [T1.min(), T1.max()]#[24.2, 25.0]#
# --- Create Video ---
if turb_stats_video:
    print("Creating video frames...")
    outdir = fig_folder + 'oc comparisons/'
    os.makedirs(outdir, exist_ok=True)
    filenames = []
    for i in range(0, nt):
        td = time[i] / 3600 / 24
        fig = plt.figure(figsize=(20, 12))
        fig.tight_layout()
        fig.suptitle(f'{td:.2f} days', fontsize=20)
        ax1 = fig.add_subplot(2, 3, 1)  # average velocities
        ax1.plot(u_avg1[i, :], z, label=r"$\langle$u$\rangle_{xy}$", color='green')
        ax1.plot(u_avg2[i, :], z, color='green', linestyle="--")
        ax1.plot(v_avg1[i, :], z, label=r"$\langle$v$\rangle_{xy}$", color='red')
        ax1.plot(v_avg2[i, :], z, color='red', linestyle="--")
        ax1.plot(w_avg1[i, :], zf, label=r"$\langle$w$\rangle_{xy}$", color='blue')
        ax1.plot(w_avg2[i, :], zf, color='blue', linestyle="--")
        ax1.set_xlabel("[m/s]")
        ax1.set_ylabel("Depth [m]")
        ax1.set_title('Velocity Profiles')
        ax1.set_ylim(-lx[2], 0)
        ax1.set_xlim(vel_range)
        ax1.legend(loc='lower right')

        ax2 = fig.add_subplot(2, 3, 2)  # rms
        ax2.plot(u_rms1[i, :], z, label=r"$\langle$u$_{rms}$$\rangle_{xy}$", color='green')
        ax2.plot(u_rms2[i, :], z, color='green', linestyle="--")
        ax2.plot(v_rms1[i, :], z, label=r"$\langle$v$_{rms}$$\rangle_{xy}$", color='red')
        ax2.plot(v_rms2[i, :], z, color='red', linestyle="--")
        ax2.plot(w_rms1[i, :], zf, label=r"$\langle$w$_{rms}$$\rangle_{xy}$", color='blue')
        ax2.plot(w_rms2[i, :], zf, color='blue', linestyle="--")
        ax2.set_xlabel("[m/s]")
        ax2.set_ylabel("Depth [m]")
        ax2.set_title("Root Mean Square Velocities")
        ax2.set_ylim(-lx[2], 0)
        ax2.set_xlim(rms_range)
        ax2.legend(loc='lower right')

        ax3 = fig.add_subplot(2, 3, 3)  # Reynolds stress
        ax3.plot(uv_fluc1[i, :], z, label=r"$\langle$uv$\rangle_{xy}$", color='green')
        ax3.plot(uv_fluc2[i, :], z, color='green', linestyle="--")
        ax3.plot(uw_fluc1[i, :], z, label=r"$\langle$uw$\rangle_{xy}$", color='red')
        ax3.plot(uw_fluc2[i, :], z, color='red', linestyle="--")
        ax3.plot(vw_fluc1[i, :], z, label=r"$\langle$vw$\rangle_{xy}$", color='blue')
        ax3.plot(vw_fluc2[i, :], z, color='blue', linestyle="--")
        ax3.set_xlabel(r"[m$^2$/s$^2$]")
        ax3.set_ylabel("Depth [m]")
        ax3.set_title("Reynolds Stresses")
        ax3.set_ylim(-lx[2], 0)
        ax3.set_xlim(restress_range)
        ax3.legend(loc='lower right')

        ax4 = fig.add_subplot(2, 3, 4)  # Richardson number
        ax4.plot(Ri_avg1[i, :], z, label=case1, color='black')
        ax4.plot(Ri_avg2[i, :], z, label=case2, color='black', linestyle="--")
        ax4.set_xlabel("Ri")
        ax4.set_ylabel("Depth [m]")
        ax4.set_title("Richardson Number")
        ax4.set_ylim(-lx[2], 0)
        ax4.set_xlim(richardson_range)

        ax5 = fig.add_subplot(2, 3, 5)
        ax5.plot(w2_fluc_nd1[i, :], zf, label=case1, color='black')
        ax5.plot(w2_fluc_nd2[i, :], zf, label=case2, color='black', linestyle="--")
        ax5.set_xlabel(r"$\langle $w$^{\prime 2} \rangle_{xy}/$u$_{f}^{2}$")
        ax5.set_ylabel("Depth [m]")
        ax5.set_title(r"$\langle $w$^{\prime 2} \rangle_{xy}/$u$_{f}^{2}$")
        ax5.set_ylim(-lx[2], 0)
        ax5.set_xlim(w2_range)

        ax6 = fig.add_subplot(2, 3, 6)
        ax6.plot(T_avg1[i, :], z, label=case1, color='black')
        ax6.plot(T_avg2[i, :], z, label=case2, color='black', linestyle="--")
        ax6.set_xlabel(r"Temperature [$^\circ$C]")
        ax6.set_ylabel("Depth [m]")
        ax6.set_title(r"Temperature")
        ax6.set_ylim(-lx[2], 0)
        ax6.set_xlim(T_range)

        # universal legend
        handles, labels = ax4.get_legend_handles_labels()
        lgd = fig.legend(handles, labels, loc = 'lower center', bbox_to_anchor=(0.1, 0.95), ncol=2)

        # --- Save Frame ---
        frame_path = os.path.join(outdir, f"oc_frame_{i:04d}.png")
        plt.tight_layout()
        plt.savefig(frame_path)
        plt.close(fig)
        filenames.append(frame_path)
        print(f"Time step {i + 1}/{nt} captured: {frame_path}")

    print("Creating video...")
    filenames = sorted([f for f in os.listdir(outdir) if f.endswith(".png")])
    with imageio.get_writer(fig_folder + 'oceananigans-differences.mp4', fps=10) as writer:
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
    ax1.plot(u_avg1[i, :], z, label=r"$\langle$u$\rangle_{xy}$", color='green')
    ax1.plot(v_avg1[i, :], z, label=r"$\langle$v$\rangle_{xy}$", color='red')
    ax1.plot(w_avg1[i, :], zf, label=r"$\langle$w$\rangle_{xy}$", color='blue')
    ax1.set_xlabel("[m/s]")
    ax1.set_ylabel("Depth [m]")
    ax1.set_title('Velocity Profiles')
    ax1.set_ylim(-lx[2], 0)
    ax1.set_xlim(vel_range)
    ax1.legend(loc='lower right')

    ax2 = fig.add_subplot(1, 5, 2)  # rms
    ax2.plot(u_rms1[i, :], z, label=r"$\langle$u$_{rms}$$\rangle_{xy}$", color='green')
    ax2.plot(v_rms1[i, :], z, label=r"$\langle$v$_{rms}$$\rangle_{xy}$", color='red')
    ax2.plot(w_rms1[i, :], zf, label=r"$\langle$w$_{rms}$$\rangle_{xy}$", color='blue')
    ax2.set_xlabel("[m/s]")
    ax2.set_ylabel("Depth [m]")
    ax2.set_title("Root Mean Square Velocities")
    ax2.set_ylim(-lx[2], 0)
    ax2.set_xlim(rms_range)
    ax2.legend(loc='lower right')

    ax3 = fig.add_subplot(1, 5, 3)  # Reynolds stress
    ax3.plot(uv_fluc1[i, :], z, label=r"$\langle$uv$\rangle_{xy}$", color='green')
    ax3.plot(uw_fluc1[i, :], z, label=r"$\langle$uw$\rangle_{xy}$", color='red')
    ax3.plot(vw_fluc1[i, :], z, label=r"$\langle$vw$\rangle_{xy}$", color='blue')
    ax3.set_xlabel(r"[m$^2$/s$^2$]")
    ax3.set_ylabel("Depth [m]")
    ax3.set_title("Reynolds Stresses")
    ax3.set_ylim(-lx[2], 0)
    ax3.set_xlim(restress_range)
    ax3.legend(loc='lower right')

    ax4 = fig.add_subplot(1, 5, 4)  # Richardson number
    ax4.plot(Ri_avg1[i, :], z, label=case1, color='black')
    ax4.set_xlabel("Ri1")
    ax4.set_ylabel("Depth [m]")
    ax4.set_title("Richardson Number")
    ax4.set_ylim(-lx[2], 0)
    ax4.set_xlim(0.0, Ri_avg1[i, :].max())

    ax5 = fig.add_subplot(1, 5, 5)
    ax5.plot(w2_fluc_nd1[i, :], zf, label=case1, color='black')
    ax5.set_xlabel(r"$\langle $w$^{\prime 2} \rangle_{xy}/$u$_{f}^{2}$")
    ax5.set_ylabel("Depth [m]")
    ax5.set_title(r"$\langle $w$^{\prime 2} \rangle_{xy}/$u$_{f}^{2}$")
    ax5.set_ylim(-lx[2], 0)
    ax5.set_xlim(w2_range)

    # --- Save Figure ---
    frame_path = os.path.join(fig_folder, 'oceananigans-differences.svg')
    plt.tight_layout()
    plt.savefig(frame_path)
    plt.close(fig)

if differences:
    print("Creating video frames...")

    # calculating differences
    u_avg_d = (u_avg2-u_avg1)#/u_avg1
    v_avg_d = (v_avg2-v_avg1)#/v_avg1
    w_avg_d = (w_avg2-w_avg1)#/w_avg1
    T_avg_d = (T_avg2-T_avg1)#/T_avg1
    Ri_avg_d = (Ri_avg2-Ri_avg1)#/Ri_avg1
    u_rms_d = (u_rms2-u_rms1)#/u_rms1
    v_rms_d = (v_rms2-v_rms1)#/v_rms1
    w_rms_d = (w_rms2-w_rms1)#/w_rms1
    uv_fluc_d = (uv_fluc2 - uv_fluc1)#/uv_fluc1
    uw_fluc_d = (uw_fluc2 - uw_fluc1)#/uw_fluc1
    vw_fluc_d = (vw_fluc2 - vw_fluc1)#/vw_fluc1
    w2_fluc_nd_d = (w2_fluc_nd2 - w2_fluc_nd1)#/ w2_fluc_nd1


    # plot ranges
    vel_max = np.max([np.abs(u_avg_d).max(), np.abs(v_avg_d).max(), np.abs(w_avg_d).max()])
    vel_range = [-1*vel_max, vel_max]
    rms_max = np.max([u_rms_d.max(), v_rms_d.max(), w_rms_d.max()])
    rms_range = [-1*rms_max, rms_max]
    restress_max = np.max([np.abs(uv_fluc_d).max(), np.abs(uw_fluc_d).max(), np.abs(vw_fluc_d).max()])
    restress_range = [-1*restress_max, restress_max]
    ri_max = np.abs(Ri_avg_d).max()
    richardson_range = [-1*ri_max, ri_max] #3*10**(-6)]#
    w2_max = np.abs(w2_fluc_nd_d).max()
    w2_range = [-1*w2_max, w2_max]#3]#
    T_max = np.abs(T_avg_d).max()
    T_range = [-1*T_max, T_max]

    outdir = fig_folder + 'oc comparisons diff/'
    os.makedirs(outdir, exist_ok=True)
    filenames = []
    for i in range(0, nt):
        td = time[i] / 3600 / 24
        fig = plt.figure(figsize=(20, 12))
        fig.tight_layout()
        fig.suptitle(f'{td:.2f} days', fontsize=20)
        ax1 = fig.add_subplot(2, 3, 1)  # average velocities
        ax1.plot(u_avg_d[i, :], z, label=r"$\langle$u$\rangle_{xy}$", color='green')
        ax1.plot(v_avg_d[i, :], z, label=r"$\langle$v$\rangle_{xy}$", color='red')
        ax1.plot(w_avg_d[i, :], zf, label=r"$\langle$w$\rangle_{xy}$", color='blue')
        ax1.set_xlabel("[m/s]")
        ax1.set_ylabel("Depth [m]")
        ax1.set_title('Velocity Profiles')
        ax1.set_ylim(-lx[2], 0)
        ax1.set_xlim(vel_range)
        ax1.legend(loc='lower right')

        ax2 = fig.add_subplot(2, 3, 2)  # rms
        ax2.plot(u_rms_d[i, :], z, label=r"$\langle$u$_{rms}$$\rangle_{xy}$", color='green')
        ax2.plot(v_rms_d[i, :], z, label=r"$\langle$v$_{rms}$$\rangle_{xy}$", color='red')
        ax2.plot(w_rms_d[i, :], zf, label=r"$\langle$w$_{rms}$$\rangle_{xy}$", color='blue')
        ax2.set_xlabel("[m/s]")
        ax2.set_ylabel("Depth [m]")
        ax2.set_title("Root Mean Square Velocities")
        ax2.set_ylim(-lx[2], 0)
        ax2.set_xlim(rms_range)
        ax2.legend(loc='lower right')

        ax3 = fig.add_subplot(2, 3, 3)  # Reynolds stress
        ax3.plot(uv_fluc_d[i, :], z, label=r"$\langle$uv$\rangle_{xy}$", color='green')
        ax3.plot(uw_fluc_d[i, :], z, label=r"$\langle$uw$\rangle_{xy}$", color='red')
        ax3.plot(vw_fluc_d[i, :], z, label=r"$\langle$vw$\rangle_{xy}$", color='blue')
        ax3.set_xlabel(r"[m$^2$/s$^2$]")
        ax3.set_ylabel("Depth [m]")
        ax3.set_title("Reynolds Stresses")
        ax3.set_ylim(-lx[2], 0)
        ax3.set_xlim(restress_range)
        ax3.legend(loc='lower right')

        ax4 = fig.add_subplot(2, 3, 4)  # Richardson number
        ax4.plot(Ri_avg_d[i, :], z, color='black')
        ax4.set_xlabel("Ri")
        ax4.set_ylabel("Depth [m]")
        ax4.set_title("Richardson Number")
        ax4.set_ylim(-lx[2], 0)
        ax4.set_xlim(richardson_range)

        ax5 = fig.add_subplot(2, 3, 5)
        ax5.plot(w2_fluc_nd_d[i, :], zf, color='black')
        ax5.set_xlabel(r"$\langle $w$^{\prime 2} \rangle_{xy}/$u$_{f}^{2}$")
        ax5.set_ylabel("Depth [m]")
        ax5.set_title(r"$\langle $w$^{\prime 2} \rangle_{xy}/$u$_{f}^{2}$")
        ax5.set_ylim(-lx[2], 0)
        ax5.set_xlim(w2_range)

        ax6 = fig.add_subplot(2, 3, 6)
        ax6.plot(T_avg_d[i, :], z, color='black')
        ax6.set_xlabel(r"Temperature [$^\circ$C]")
        ax6.set_ylabel("Depth [m]")
        ax6.set_title(r"Temperature")
        ax6.set_ylim(-lx[2], 0)
        ax6.set_xlim(T_range)

        # --- Save Frame ---
        frame_path = os.path.join(outdir, f"oc_frame_{i:04d}.png")
        plt.tight_layout()
        plt.savefig(frame_path)
        plt.close(fig)
        filenames.append(frame_path)
        print(f"Time step {i + 1}/{nt} captured: {frame_path}")

    print("Creating video...")
    filenames = sorted([f for f in os.listdir(outdir) if f.endswith(".png")])
    with imageio.get_writer(fig_folder + 'oceananigans-calculated-differences.mp4', fps=10) as writer:
        for filename in filenames:
            image = imageio.imread(f"{outdir}/{filename}")
            writer.append_data(image)