import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from reader import OceananigansData
from physics import buoyancy
from interpolation import point
from diagnostics import comparison_info
from plotting_general import plot_format, comparison_plot_opt

# flags for how to read data
with_halos = False
closure = False
salinity = True
stokes = False

contour_bound = 0.05
name_uni = f'contour-{contour_bound:.2f}'
universal_folder = '/glade/derecho/scratch/apauls/outputs/'

# selecting cases to compare
variations = 'all' # 'MLD', 'flux', 'strat', 'all', 'vertical length', 'vertical resolution', 'else'
cases_info = comparison_info(variations, universal_folder)
mld = cases_info['mld']
dTdz = cases_info['dTdz']
F_s = cases_info['F_s']
case_names = cases_info['case_names']
num_cases = cases_info['num_cases']
fig_folder = cases_info['fig_folder']
os.makedirs(fig_folder, exist_ok=True)

readers = []
for name in cases_info["folder_names"]:
    folder = os.path.join(universal_folder, name)
    readers.append(OceananigansData(folder, salinity = salinity))
    readers[-1].load_grid()
    readers[-1].load_time()
    readers[-1].load_equation_of_state()

z = readers[0].z
nx = readers[0].nx
lx = readers[0].lx
lx = [lx[0]/2, lx[-1]]
nt = readers[0].nt

nz = np.max(nx[:][2])

# physical parameters
rho0 = 1026
T0 = 25
S0 = 0 
rp = 5
alpha = readers[0].alpha
beta = readers[0].beta
g = 9.80665

N = (g * alpha * dTdz)**0.5
F0 = -g * beta * F_s * np.pi * rp**2
Ln = (F0/N**3)**(1/4)

length_scale = []
neutral_depth = []
for i, reader in enumerate(readers):
    ur_rz = reader.load_binning_var('horizontal velocity')
utheta_rz = reader.load_binning_var('rotational velocity')
w_rz = reader.load_binning_var('w')
T_rz = reader.load_binning_var('T')
S_rz = reader.load_binning_var('S')
r = reader.loading_bin_contours()

    S_avg = np.mean(S_rz[0, :, 5:], axis=(-1))
    S_mld = point(S_avg, z, z0 = -mld[i])
    b_mld = g * S_mld * beta
    length_scale.append(b_mld/(g*alpha*dTdz[i]))
    """
    b = buoyancy(reader, T_rz, S_rz)
    bT = b['b_T']
    bS = b['b_C']
    b_fluc = 
    """
    b_fluc_avg = np.mean(b_fluc_rz[0, :, 5:], axis=(-1))
    mld_idx = np.argmin(np.abs(z + mld[i]))
    idxs = np.where(np.diff(np.sign(b_fluc_avg))<0)[0] # goes from pos to neg
    idx = idxs[np.argmin(np.abs(mld_idx - idxs))]
    weight = -b_fluc_avg[idx]/(b_fluc_avg[idx+1] - b_fluc_avg[idx])
    z_neutral = (1 - weight) * z[idx] + weight * z[idx+1]
    neutral_depth.append(np.abs(z_neutral+mld[i]))
    print(f'Case: MLD = {mld[i]} m, dT/dz = {dTdz[i]}, F_s = {F_s[i]} g/kg m/s')
    print('\tLn:', Ln[i])
    print('\t\t(96m-MLD)/Ln:', (96-mld[i])/Ln[i], '\t(160m-MLD)/Ln:', (160-mld[i])/Ln[i])
    print('\tb(S_perturbed_centerline)/(g*alpha*dTdz):', length_scale[i])
    print('\tneutral depth:', neutral_depth[i])
    print('')

lz = np.arange(96, 193)

color_opt, line_opt = comparison_plot_opt(num_cases)
plot_format()
scale = [1, 0.25]
gridspec_kw={'height_ratios': scale}
fig, ax = plt.subplots(2, 3, figsize=(23, 6), dpi = 300, gridspec_kw = gridspec_kw)
for a in ax[-1, :]:
        a.remove()
ax = ax.ravel()
plt.subplots_adjust(top=0.9)
case_handles = [Line2D([0], [0], color=color_opt[i], linestyle='solid', label=case_names[i]) for i in range(num_cases)]
fig.legend(handles=case_handles,
        loc='lower center',
        ncol=num_cases/4,
        bbox_to_anchor=(0.52, 0.005), fontsize = 12)


ax[0].set_title('Neutral Depths', fontsize = 12)
ax[0].set_xlabel(r'L$_z$ [m]', fontsize = 12)
ax[0].set_ylabel(r'(L$_z$ - h$_{ml}$)/(z$_{neutral}$)', fontsize = 12)

ax[1].set_title('Ln', fontsize = 12)
ax[1].set_xlabel(r'L$_z$ [m]', fontsize = 12)
ax[1].set_ylabel(r'(L$_z$ - h$_{ml}$)/(L$_{n}$)', fontsize = 12)

ax[2].set_title('$\Delta$b/(db/dz)', fontsize = 12)
ax[2].set_xlabel(r'L$_z$ [m]', fontsize = 12)
ax[2].set_ylabel(r'(L$_z$ - h$_{ml}$)/($\Delta$b/(db/dz))', fontsize = 12)

for i in range(num_cases):
     ax[0].plot(lz, (lz - mld[i])/(neutral_depth[i]), color = color_opt[i])
     ax[1].plot(lz, (lz - mld[i])/(Ln[i]), color = color_opt[i])
     ax[2].plot(lz, (lz - mld[i])/(length_scale[i]), color = color_opt[i])

neutral_option = (137 - mld[0])/(neutral_depth[0])
Ln_option = (137 - mld[0])/(Ln[0])
length_scale_option = (137 - mld[0])/(length_scale[0])
ax[0].plot(lz, neutral_option*np.ones_like(lz), color = 'black', linestyle = 'dashed', linewidth = 0.85)
ax[1].plot(lz, Ln_option*np.ones_like(lz), color = 'black', linestyle = 'dashed', linewidth = 0.85)
ax[2].plot(lz, length_scale_option*np.ones_like(lz), color = 'black', linestyle = 'dashed', linewidth = 0.85)

plt.savefig(os.path.join(fig_folder, 'domain potentials.svg'))