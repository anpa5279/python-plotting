import os
import numpy as np

from reader import OceananigansData
from physics import buoyancy
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
    readers.append(OceananigansData(folder))

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
coeffs = readers[0].load_equation_of_state(True)
alpha = coeffs['alpha']
beta = coeffs['beta']
g = 9.80665
S_tol = 10**(-6)

for i, reader in enumerate(readers):
    reader_dTdz = dTdz[i]
    reader_Fs = F_s[i]
    reader_mld = mld[i]
    # Load binning from files
    r, z, time, S_rz, T_fluc_rz, T_rz, ur_rz, w_rz, b_fluc_rz = reader.load_binning()
    length_scale_temporal = 0
    print(f"Case MLD = {reader_mld} m, dT/dz = {reader_dTdz} K/m, F_s = {reader_Fs}:")
    reader.load_time()
    for it, save in enumerate(reader.t_save):
        # Calculate buoyancy from T and S
        bs = buoyancy(T_rz[it], coeffs, S_rz[it])
        bC = bs['b_C']
        bT = bs['b_T']
        db = bT - bC
        dbdz = -g * alpha * reader_dTdz

        length_scale_temporal += np.mean(db)/dbdz
        print(f"\t {time[it]} length scale at time {time[it]:.2f} = {np.mean(db)/dbdz:.2f} m")
    length_scale_temporal /= nt
    print(f"Case MLD = {reader_mld} m, dT/dz = {reader_dTdz} K/m, F_s = {reader_Fs}: average length scale = {length_scale_temporal:.2f} m")


for i, reader in enumerate(readers):
    reader_dTdz = dTdz[i]
    reader_Fs = F_s[i]
    reader_mld = mld[i]
    S_value, w_value = reader.load_contour_temporal_averages('interp_temporal_averages.h5')
    dbdz = g * alpha * reader_dTdz
    db = -g * S_value * beta
    print(f"Case MLD = {reader_mld} m, dT/dz = {reader_dTdz} K/m, F_s = {reader_Fs}: \t S contour = {S_value:.7f}, \tlength scale = {db/dbdz:.7f} m")

for i, reader in enumerate(readers):
    reader_dTdz = dTdz[i]
    reader_Fs = F_s[i]
    reader_mld = mld[i]
    S_value, w_value = reader.load_contour_temporal_averages('interp_temporal_averages.h5')
    dbdz = g * alpha * reader_dTdz
    db = g * beta * reader_Fs / w_value
    print(f"Case MLD = {reader_mld} m, dT/dz = {reader_dTdz} K/m, F_s = {reader_Fs}: \t S estimate = {-reader_Fs / w_value:.7f}, \tlength scale = {db/dbdz:.7f} m")


for i, reader in enumerate(readers):
    reader_dTdz = dTdz[i]
    reader_Fs = F_s[i]
    reader_mld = mld[i]
    N = np.sqrt(g * alpha * reader_dTdz)
    length_scale = beta*reader_Fs/(N)
    print(f"Case MLD = {reader_mld} m, dT/dz = {reader_dTdz} K/m, F_s = {reader_Fs}: \t N = {N:.7f}, \tlength scale = {length_scale:.7f} m")


for i, reader in enumerate(readers):
    reader_dTdz = dTdz[i]
    reader_Fs = F_s[i]
    reader_mld = mld[i]
    N = np.sqrt(g * alpha * reader_dTdz)
    jb = g*beta*reader_Fs
    omega = (jb/rp**2)**(1/3)
    lambda_h = 2*np.pi*rp
    lambda_z = 2*np.pi*omega/N #rp*(1/np.sqrt((N/omega)**2 - 1))
    print(rf"Case MLD = {reader_mld} m, dT/dz = {reader_dTdz} K/m, F_s = {reader_Fs}: N = {N:.7f}, Jb = {jb:.7f}, omega = {omega:.7f}, lambda_h = {lambda_h:.7f}, lambda_z = {lambda_z:.7f} m")


for i, reader in enumerate(readers):
    reader_dTdz = dTdz[i]
    reader_Fs = F_s[i]
    reader_mld = mld[i]
    N = np.sqrt(g * alpha * reader_dTdz)
    jb = g*beta*reader_Fs
    H = jb**(1/2)*N**(-3/2) #lx[-1] - reader_mld #
    c = N*H/np.pi
    lambda_h = 2*np.pi*rp
    lambda_z = c/N
    print(rf"Case MLD = {reader_mld} m, dT/dz = {reader_dTdz} K/m, F_s = {reader_Fs}: N = {N:.7f}, Jb = {jb:.7f}, H = {H:.7f}, c = {c:.7f}, lambda_z = {lambda_z:.7f} m")


for i, reader in enumerate(readers):
    reader_dTdz = dTdz[i]
    reader_Fs = F_s[i]
    reader_mld = mld[i]
    N = np.sqrt(g * alpha * reader_dTdz)
    jb = g*beta*reader_Fs
    H = jb**(1/2)*N**(-3/2) #lx[-1] - reader_mld #
    c = N*H/np.pi
    length_scale = beta*reader_Fs/c/N
    print(rf"Case MLD = {reader_mld} m, dT/dz = {reader_dTdz} K/m, F_s = {reader_Fs}: N = {N:.7f}, Jb = {jb:.7f}, H = {H:.7f}, c = {c:.7f}, S = {reader_Fs/c:.7f}, length_scale = {length_scale:.7f} m")

for i, reader in enumerate(readers):
    reader_dTdz = dTdz[i]
    reader_Fs = F_s[i]
    reader_mld = mld[i]
    N = np.sqrt(g * alpha * reader_dTdz)
    jb = g*beta*reader_Fs
    H = jb**(1/2)*N**(-3/2) #lx[-1] - reader_mld #
    c = N*H/np.pi
    length_scale = beta*reader_Fs/c/N
    print(rf"Case MLD = {reader_mld} m, dT/dz = {reader_dTdz} K/m, F_s = {reader_Fs}: N = {N:.7f}, Jb = {jb:.7f}, H = {H:.7f}, c = {c:.7f}, S = {reader_Fs/c:.7f}, length_scale = {length_scale:.7f} m")

for i, reader in enumerate(readers):
    reader_dTdz = dTdz[i]
    reader_Fs = F_s[i]
    reader_mld = mld[i]
    g = 9.81
    N = np.sqrt(g * alpha * reader_dTdz)
    jb = g * beta * reader_Fs
    # buoyancy anomaly scale
    delta_b = (jb*N)**(1/2)
    # buoyancy length scale
    L_b = delta_b / N**2
    # required stratified layer depth (with safety factor)
    safety = 2.0
    Lz_required = reader_mld + safety * L_b
    print(rf"Case MLD={reader_mld}, dTdz={reader_dTdz}: L_b={L_b:.2f} m, Lz_required={Lz_required:.2f} m")