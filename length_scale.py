import os
import numpy as np

from reader import OceananigansData
from physics import buoyancy
from interpolation import interp1d_axis
from diagnostics import comparison_info

# flags for how to read data
with_halos = False
closure = False
salinity = True
stokes = False

contour_bound = 0.05
name_uni = f'contour-{contour_bound:.2f}'
universal_folder = '/Users/annapauls/Documents/TESLa /Simulations/Oceananigans/dense plume/salinity and temperature/no noise circle inlet/'

# selecting cases to compare
variations = 'all' # 'MLD', 'flux', 'strat', 'all', 'length', 'else'
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

readers[0].load_grid()
readers[0].load_time()
z = readers[0].z
nx = readers[0].nx
lx = readers[0].lx
lx = [lx[0]/2, lx[-1]]
nt = readers[0].nt

nz = np.max(nx[:][2])
x = readers[0].x
y = readers[0].y

# physical parameters
rho0 = 1026
T0 = 25
S0 = 0 
rp = 5

g = 9.80665
S_tol = 10**(-6)
for i, reader in enumerate(readers):
    reader_dTdz = dTdz[i]
    reader_Fs = F_s[i]
    reader_mld = mld[i]
    r, z, time, S_rz, T_fluc_rz, T_rz, ur_rz, w_rz, b_fluc_rz = reader.load_binning()

    # time-average bw at plume center (r=0, smallest bin)
    bw_rz = b_fluc_rz * w_rz                    # element-wise
    
    b_fluc_avg = np.mean(b_fluc_rz[:4, :, 20:], axis=(0, -1))
    w_avg = np.mean(w_rz[:4, :, 20:], axis=(0, -1))

    # zero crossings
    sign_changes_b = np.diff(np.sign(b_fluc_avg))
    sign_changes_w = np.diff(np.sign(w_avg))

    # neutral buoyancy: first positive -> negative crossing
    neg_to_pos_b = np.where(sign_changes_b > 0)[0]
    pos_to_neg_w = np.where(sign_changes_w < 0)[0]
    
    z_max_w = z[pos_to_neg_w[-1] + 1]
    
    z_max_b = z[neg_to_pos_b[0] + 1]

    safety = 2.0
    Lz_required = np.max(np.abs([z_max_w, z_max_b])) * safety

    print(rf"Case MLD={reader_mld}, dTdz={reader_dTdz}, Fs={reader_Fs}: "
        rf"z_max_w={z_max_w:.1f} m, w = {w_avg[pos_to_neg_w[-1]]:.2e} m/s, "
        rf"z_max_b={z_max_b:.1f} m, b = {b_fluc_avg[neg_to_pos_b[0]]:.2e} m/s^2, "
        rf"using {np.max(np.abs([z_max_w, z_max_b]))}, Lz_required={Lz_required:.1f} m")