import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import h5py

from reader import OceananigansData
from diagnostics import comparison_info
from plotting_general import plot_format, comparison_plot_opt, create_video

# flags for how to read data
with_halos = False
closure = False
salinity = True
stokes = False

contours = np.array([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05])
universal_folder = '/glade/derecho/scratch/apauls/outputs/'
#harddrive: '/Volumes/Anna External/Oceananigans/dense plume with stratification/salinity and temperature /no noise circle inlet/'#

# selecting cases to compare
variations = 'else' # 'MLD', 'flux', 'strat', 'all', 'length', 'WENO', 'vertical resolution', 'one case', 'else'
if variations != 'else' and variations != 'one':
    cases_info = comparison_info(variations, universal_folder = universal_folder)
    case_names = cases_info['case_names']
    num_cases = cases_info['num_cases']
    fig_folder = cases_info['fig_folder']
    folder_names = cases_info['folder_names']
elif variations == 'one':
    folder_names = ['Lz = 160m/S0 = 0.1 dTdz = 0.01 MLD = 70',]
    fig_folder =os.path.join(universal_folder, folder_names[0], 'plotting outputs')
    os.makedirs(fig_folder, exist_ok=True)
    num_cases = len(folder_names)
    case_names = [r'L$_z = 160$m and MLD  = 70m']
    num_cases = 1
elif variations == 'else':
    folder_names = ['Lz = 160m/S0 = 0.1 dTdz = 0.01 MLD = 60', 'Lz = 160m/S0 = 0.1 dTdz = 0.01 MLD = 70', 'Lz = 160m/S0 = 0.2 dTdz = 0.01 MLD = 60']
    num_cases = len(folder_names)
    fig_folder = os.path.join(universal_folder, 'comparison figures', 'Lz = 160m' + ' comparison figures', 'binning')
    os.makedirs(fig_folder, exist_ok=True)
    case_names =[r'F$_{\text{C}} = -1.0\cdot 10^{-4}$, MLD = 60m, dTdz = 0.01', r'F$_{\text{C}} = -1.0\cdot 10^{-4}$, MLD = 70m, dTdz = 0.01', r'F$_{\text{C}} = - 2.0\cdot 10^{-4}$, MLD = 60m, dTdz = 0.01']#[r'L$_z = 96$m', r'L$_z = 160$m']#r'$\Delta z = 0.5$m', r'$\Delta z = 0.375$m'#

color_opt, _ = comparison_plot_opt(len(contours))

readers = []
for folder in folder_names:
    folder = os.path.join(universal_folder, folder)
    readers.append(OceananigansData(folder, salinity = salinity))

# physical parameters
x0 = 0.0
y0 = 0.0
rj = 5 # m, radius of salinity flux circle at the surface
g = 9.80665  # gravity in m/s^2

# collecting model information for all cases
lz = 0
nt = 1000
nx = []
dense_plume = []
time = []

for i, reader in enumerate(readers):

    reader.load_grid()
    lz = np.max((lz, reader.lx[-1]))
    reader.load_time()
    time.append(reader.time)
    nt = min((nt, len(time[i])))

    S_value, w_value = reader.load_contour_temporal_averages('interp_temporal_averages.h5')
    reader.load_equation_of_state()

z = []
r_scale = []
neutral_depths = []
r_contour = [] 
for i, reader in enumerate(readers):
    start = 7
    r_profile = np.zeros([len(contours), reader.nx[2], len(time[i])])
    z_max = np.zeros([len(time[i]) - start, len(contours)])
    neutral = np.zeros([len(time[i]) - start, len(contours)])
    r_c = np.zeros([len(time[i]) - start, len(contours)])
    params = np.zeros([2, len(contours)])
    for j, contour in enumerate(contours):
        # Load data from files
        fname = os.path.join(reader.folder, 'binning', 'binning_rtz.h5')
        with h5py.File(fname, 'r') as f:
            r_profile[j, :, :] = f[f'r given contour/contour = {contour}'][()]
            z.append(f['ccc/dimensions/z'][()])
            S_rz = f['ccc/S_rz'][()]
            r_bin = f['ccc/dimensions/r_bin'][()]

    r_contour.append(r_profile)

############ PLOTTING ############
plot_format(fontsize = 10)
num_cases = len(case_names)
outdir = os.path.join(fig_folder, 'tracer radius with depth in time')
os.makedirs(outdir, exist_ok=True)
ncols = num_cases
gridspec_kw={'height_ratios': [1, 0.15]}
width = 0.8
for it in range(nt):
    fig, axes = plt.subplots(2, ncols, figsize=(3.5*ncols, 5), sharey = True, gridspec_kw=gridspec_kw)
    for a in axes[-1, :]:
        a.remove()
    fig.suptitle(f"Time = {time[0][it]/3600/24:.2f} days", fontsize=12)
    case_handles = [Line2D([0], [0], color=color_opt[i], linestyle='solid', linewidth=width, label=f"Contour = {contours[i]:.2e} ")for i in range(len(contours))]
    fig.legend(handles=case_handles,
            loc='lower center',
            ncol=3,
            bbox_to_anchor=(0.52, 0.005))

    for i, ax in enumerate(axes[0, :]):
        for n, color in enumerate(color_opt):
            ax.plot(r_contour[i][n, :, it], z[i], color=color, linewidth = width)
        ax.set_ylim(-lz, 0)
        ax.set_xlim(0, max(r_bin))
        ax.set_xlabel("Radius [m]")
        ax.set_title(case_names[i], fontsize = 10)
        ax.set_aspect('equal')
        if i == 0:
            ax.set_ylabel('Depth [m]')
    # --- Save Frame ---
    frame_path = os.path.join(outdir, f"tracer_radius with depth_{it:04d}.png")
    plt.savefig(frame_path, bbox_inches='tight')
    plt.close(fig)


create_video(outdir, fig_folder, 'binning', f'radius contours')