import numpy as np
import dask.array as da
import os 
import h5py

from interpolation import velocities_to_center
from physics import rms

### -------------------------COLLECTING COMPARISON CASE INFO------------------------- ###
def comparison_info(variations, universal_folder = '/Users/annapauls/Documents/TESLa /Simulations/Oceananigans/dense plume/salinity and temperature/no noise circle inlet', ND=False, name_uni = ''):
    wp = -0.001
    if variations == 'strat':
        folder_names =['S0 = 0.1 dTdz = 0.005 MLD = 60', 'S0 = 0.1 dTdz = 0.01 MLD = 60', 'S0 = 0.1 dTdz = 0.05 MLD = 60', 'S0 = 0.1 dTdz = 0.1 MLD = 60'] 
        case_names =[r'dTdz = 0.005', r'dTdz = 0.01', r'dTdz = 0.05', r'dTdz = 0.10']  
        num_cases = len(case_names)
        dTdz = np.array([0.005, 0.01, 0.05, 0.1]) # background temperature gradient in K/m
        mld = 60 * np.ones(num_cases) 
        F_s = wp * 0.1 * np.ones(num_cases) 
    elif variations == 'MLD':
        folder_names =['S0 = 0.1 dTdz = 0.01 MLD = 50', 'S0 = 0.1 dTdz = 0.01 MLD = 60', 'S0 = 0.1 dTdz = 0.01 MLD = 70']
        case_names =[r'MLD = 50m', r'MLD = 60m', r'MLD = 70m']
        num_cases = len(case_names)
        dTdz = 0.01 * np.ones(num_cases) # background temperature gradient in K/m
        mld = np.array([50, 60, 70])
        F_s = wp * 0.1 * np.ones(num_cases) 
    elif variations == 'flux':
        folder_names =['S0 = 0.05 dTdz = 0.01 MLD = 60', 'S0 = 0.1 dTdz = 0.01 MLD = 60', 'S0 = 0.15 dTdz = 0.01 MLD = 60', 'S0 = 0.2 dTdz = 0.01 MLD = 60']
        case_names =[r'F$_{\text{C}} = -5.0\cdot 10^{-5}$', r'F$_{\text{C}} = -1.0\cdot 10^{-4}$', r'F$_{\text{C}} = -1.5\cdot 10^{-4}$', r'F$_{\text{C}} = - 2.0\cdot 10^{-4}$']
        num_cases = len(case_names)
        dTdz = 0.01 * np.ones(num_cases) # background temperature gradient in K/m
        mld = 60 * np.ones(num_cases)  
        F_s = wp * np.array([0.05, 0.1, 0.15, 0.2])
    elif variations == 'all':
        folder_names =['S0 = 0.1 dTdz = 0.01 MLD = 60', 
                    'S0 = 0.1 dTdz = 0.01 MLD = 50', 'S0 = 0.1 dTdz = 0.01 MLD = 70', 
                    'S0 = 0.1 dTdz = 0.005 MLD = 60', 'S0 = 0.1 dTdz = 0.05 MLD = 60', 'S0 = 0.1 dTdz = 0.1 MLD = 60',
                    'S0 = 0.05 dTdz = 0.01 MLD = 60', 'S0 = 0.15 dTdz = 0.01 MLD = 60', 'S0 = 0.2 dTdz = 0.01 MLD = 60']
        case_names =[r'F$_{\text{C}} = -1.0\cdot 10^{-4}$, MLD = 60m, dTdz = 0.01', 
                    r'F$_{\text{C}} = -1.0\cdot 10^{-4}$, MLD = 50m, dTdz = 0.01', r'F$_{\text{C}} = -1.0\cdot 10^{-4}$, MLD = 70m, dTdz = 0.01', 
                    r'F$_{\text{C}} = -1.0\cdot 10^{-4}$, MLD = 60m, dTdz = 0.005', r'F$_{\text{C}} = -1.0\cdot 10^{-4}$, MLD = 60m, dTdz = 0.05', r'F$_{\text{C}} = -1.0\cdot 10^{-4}$, MLD = 60m, dTdz = 0.1', 
                    r'F$_{\text{C}} = -5.0\cdot 10^{-5}$, MLD = 60m, dTdz = 0.01', r'F$_{\text{C}} = -1.5\cdot 10^{-4}$, MLD = 60m, dTdz = 0.01', r'F$_{\text{C}} = - 2.0\cdot 10^{-4}$, MLD = 60m, dTdz = 0.01']
        num_cases = len(case_names)
        mld = np.array([60, 50, 70, 60, 60, 60, 60, 60, 60]) # mld in m
        dTdz = np.array([0.01, 0.01, 0.01, 0.005, 0.05, 0.1, 0.01, 0.01, 0.01]) # background temperature gradient in K/m
        F_s = wp * np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.15, 0.2])
    elif variations == 'Lz160m':
        folder_names =['Lz = 160m/S0 = 0.1 dTdz = 0.01 MLD = 60', 'Lz = 160m/S0 = 0.1 dTdz = 0.01 MLD = 70', 'Lz = 160m/S0 = 0.2 dTdz = 0.01 MLD = 60']
        case_names =[r'F$_{\text{C}} = -1.0\cdot 10^{-4}$, MLD = 60m, dTdz = 0.01', 
                     r'F$_{\text{C}} = -1.0\cdot 10^{-4}$, MLD = 70m, dTdz = 0.01', 
                     r'F$_{\text{C}} = - 2.0\cdot 10^{-4}$, MLD = 60m, dTdz = 0.01']
        num_cases = len(case_names)
        dTdz = 0.01 * np.ones(num_cases) # background temperature gradient in K/m
        mld = [60, 70, 60]
        F_s = wp * 0.1 * np.ones(num_cases) 
        F_s[2] = wp * 0.2
    elif variations == 'vertical length':
        folder_names =['nz = 77 z = 96.25 m', 'nz = 128 z = 160 m', 'nz = 192 z = 240 m']
        case_names =[r'L$_{\text{z}}$ = 96.25 m', r'L$_{\text{z}}$ = 160 m', r'L$_{\text{z}}$ = 240 m']
        num_cases = len(case_names)
        dTdz = 0.01 * np.ones(num_cases) # background temperature gradient in K/m
        mld = 60 * np.ones(num_cases)
        F_s = wp * 0.1 * np.ones(num_cases) 
    elif variations == 'vertical resolution':
        folder_names =['nz = 64', 'nz = 128', 'nz = 192', 'nz = 256']
        case_names =[r'$\Delta z = 1.5$m', r'$\Delta z = 0.75$m' r'$\Delta z = 0.5$m', r'$\Delta z = 0.375$m']
        num_cases = len(case_names)
        dTdz = 0.01 * np.ones(num_cases) # background temperature gradient in K/m
        mld = 60 * np.ones(num_cases)
        F_s = wp * 0.1 * np.ones(num_cases) 
    elif variations == 'horizontal resolution':
        folder_names =['coarse2/mod grid', 
                       'coarse1/mod grid', 
                       'mod-default/mod grid', 
                       'fine1/mod grid', 
                       'fine2/mod grid',
                       'fine3/mod grid']
        case_names =[r'$\Delta x = 2.0$m', r'$\Delta x = 1.67$m', r'$\Delta x = 1.25$m', r'$\Delta x = 1.0$m', r'$\Delta x = 0.5$m', r'$\Delta x = 0.25$m']
        num_cases = len(case_names)
        dTdz = 0.01 * np.ones(num_cases) # background temperature gradient in K/m
        mld = 60 * np.ones(num_cases)
        F_s = wp * 0.1 * np.ones(num_cases) 
    elif variations == 'WENO':
        folder_names = ['S0 = 0.1 dTdz = 0.01 MLD = 60', 'S0 = 0.1 dTdz = 0.01 MLD = 60 WENO mod', 'S0 = 0.1 dTdz = 0.01 MLD = 60 WENO mod callback']
        case_names = [r'Default', r'WENO modified', r'WENO modified with callback 0 function']
        num_cases = len(case_names)
        dTdz = 0.01 * np.ones(num_cases) # background temperature gradient in K/m
        mld = 60 * np.ones(num_cases)
        F_s = wp * 0.1 * np.ones(num_cases)
    else:
        print("Variation type not recognized.")
        return None # user defined specific case not defined here
    folder_names = [os.path.join(universal_folder, folder) for folder in folder_names]
    # Set up folder and simulation parameters
    if ND:
        vars_exps = np.array([ # columns: Ri, Fr, MLD
            [0, -1/3, -1/2], # w_rms
            [-1/2, -1/3, 1/3], # b_center
            [-1/3, -3/4, -1], # bw_fluc_avg
            [-1/4, -1/4, -1/2], # r_profile
            [-1/2, -1/4, 3/4], # T_fluc_center
            [-1/3, -3/4, 3/4] # S_avg
        ]) # manually manipulate
        fig_folder = os.path.join(universal_folder, 'ND analysis', 'interpolation', variations, name_uni)
        os.makedirs(fig_folder, exist_ok=True)
        case_info = {
            "folder_names": folder_names,
            "fig_folder": fig_folder,
            "case_names": case_names,
            "num_cases": num_cases,
            "dTdz": dTdz,
            "mld": mld,
            "F_s": F_s,
            "vars_exps": vars_exps
        }
    else:
        fig_folder = os.path.join(universal_folder, 'comparison figures', variations + ' comparison figures', 'interpolated')
        case_info = {
            "folder_names": folder_names,
            "fig_folder": fig_folder,
            "case_names": case_names,
            "num_cases": num_cases,
            "dTdz": dTdz,
            "mld": mld,
            "F_s": F_s
        }
    return case_info
### -------------------------CALCULATING 1D AVERAGES------------------------- ###
def compute_temporal_averages(reader, center=(0.0, 0.0), start=10):
    x, y = reader.x, reader.y
    t_save = reader.t_save[start:]
    x0, y0 = center

    # Load constants once
    reader.load_equation_of_state()
    g = 9.80665
    T0 = 25
    alpha = reader.alpha
    beta  = reader.beta

    # Pre-cache spatial indices
    ix = np.argmin(np.abs(x - x0))
    iy = np.argmin(np.abs(y - y0))

    # Load all fields lazily — shape (nt - start, nx, ny, nz)
    T = reader.lazy_field('T')
    T = T[start:, :, :, :]
    S = reader.lazy_field('S')
    S = S[start:, :, :, :]
    w = reader.lazy_field('w')
    w = w[start:, :, :, :]

    # Center velocities (still lazy)
    w = velocities_to_center(w, axis=-1)

    # Buoyancy (still lazy)
    b = g * alpha * (T - T0) - (g * beta * S)

    # Horizontal means over (nx, ny) → shape (nt - start, nz)
    b_xy = da.mean(b, axis=(1, 2))

    # Fluctuation and flux
    b_fluc = b - b_xy[:, np.newaxis, np.newaxis, :]
    bw     = da.mean(b_fluc * w, axis=(1, 2))   # (nt - start, nz)

    # bw_idx per timestep — argmax not lazy, so compute bw now
    bw_np  = bw.compute()                        # (nt - start, nz) — small, cheap
    bw_idx = np.argmax(bw_np, axis=1)            # (nt,)

    # Point values at (ix, iy, bw_idx[it]) per timestep
    nt     = len(t_save)
    it_idx = np.arange(nt)
    # extract the (nt - start, nx, ny, nz) arrays only at needed points
    S_pts  = S[:, ix, iy, :].compute()
    w_pts  = w[:, ix, iy, :].compute() 
    S_value = np.mean(S_pts[it_idx, bw_idx])
    w_value = np.mean(w_pts[it_idx, bw_idx])

    return {
        "S_value":  S_value,
        "w_value":  w_value,
    }
def compute_fluct_averages(reader):
    # Load constants once
    reader.load_equation_of_state()
    g = 9.80665
    T0 = 25
    alpha = reader.alpha
    beta  = reader.beta

    # Load all fields lazily — shape (nt, nx, ny, nz)
    T = reader.load_binning_var('T')
    S = reader.load_binning_var('S')
    u = reader.load_binning_var('horizontal velocity')
    v = reader.load_binning_var('rotation velocity')
    w = reader.load_binning_var('w')

    # Buoyancy (still lazy)
    b = g * alpha * (T - T0) - (g * beta * S)

    # Horizontal means over (nx, ny) → shape (nt, nz)
    T_xy = da.mean(T, axis=0)
    S_xy = da.mean(S, axis=0)
    u_xy = da.mean(u, axis=0)
    v_xy = da.mean(v, axis=0)
    w_xy = da.mean(w, axis=0)
    b_xy = da.mean(b, axis=0)

    # Fluctuation
    T_fluc = T - T_xy
    S_fluc = S - S_xy
    u_fluc = u - u_xy
    v_fluc = v - v_xy
    w_fluc = w - w_xy
    b_fluc = b - b_xy

    #Flux fluctuations
    bu_fluc = da.mean(b_fluc * w, axis=0)
    bv_fluc = da.mean(b_fluc * w, axis=0)
    bw_fluc = da.mean(b_fluc * w, axis=0)

    # averages of fluctuations
    T_fluc_avg = da.mean(T_fluc, axis=0)
    S_fluc_avg = da.mean(S_fluc, axis=0)
    u_fluc_avg = da.mean(u_fluc, axis=0)
    v_fluc_avg = da.mean(v_fluc, axis=0)
    w_fluc_avg = da.mean(w_fluc, axis=0)
    b_fluc_avg = da.mean(b_fluc, axis=0)
    bu_fluc_avg = da.mean(bu_fluc, axis=0)
    bv_fluc_avg = da.mean(bv_fluc, axis=0)
    bw_fluc_avg = da.mean(bw_fluc, axis=0)

    return {'T_fluc': T_fluc_avg,
            'S_fluc': S_fluc_avg,
            'ur_fluc': u_fluc_avg,
            'utheta_fluc': v_fluc_avg,
            'w_fluc': w_fluc_avg,
            'b_fluc': b_fluc_avg,
            'bu_fluc': bu_fluc_avg,
            'bv_fluc': bv_fluc_avg,
            'bw_fluc': bw_fluc_avg}
def compute_rms(reader):
    u_rms = np.empty((reader.nt, reader.nx[2]))
    v_rms = np.empty((reader.nt, reader.nx[2]))
    w_rms = np.empty((reader.nt, reader.nx[2]))
    for it, t in enumerate(reader.t_save):
        u = reader.lazy_field('u', steps=t)
        v = reader.lazy_field('v', steps=t)
        w = reader.lazy_field('w', steps=t)
        u = velocities_to_center(u, axis=-3)
        v = velocities_to_center(v, axis=-2) 
        w = velocities_to_center(w, axis=-1)
        u_rms[it, :] = rms(u)
        v_rms[it, :] = rms(v)
        w_rms[it, :] = rms(w)
    return {'u_rms': u_rms,
            'v_rms': v_rms,
            'w_rms': w_rms}
### -------------------------WRITING TEMPORAL AVERAGES------------------------- ###
def write_temporal_averages(file_path, data):
    folder_contour = f"contour temporal averages"

    with h5py.File(file_path, "w") as f:
        f.create_group(f'{folder_contour}')
        f.create_dataset(f'{folder_contour}/S', data=data['S_value'])
        f.create_dataset(f'{folder_contour}/w', data=data['w_value'])
    f.close()