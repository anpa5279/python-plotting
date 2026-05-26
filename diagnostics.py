import numpy as np
import dask.array as da
import os 
import h5py

from interpolation import velocities_to_center, vertical_line, point
from dense_plume import PlumeAnalysis
from physics import rms, buoyancy

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
    elif variations == 'length':
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
        folder_names =['domain resolution testing/horizontal resolution/sj0.1-mld60-dTdz0.01-lx320-nx192', 'domain resolution testing/proposed vertical resolution/S0 = 0.1 dTdz = 0.01 MLD = 60', 'domain resolution testing/horizontal resolution/sj0.1-mld60-dTdz0.01-lx320-nx320', 'domain resolution testing/horizontal resolution/sj0.1-mld60-dTdz0.01-lx320-nx384', 'domain resolution testing/horizontal resolution/sj0.1-mld60-dTdz0.01-lx320-nx640']
        case_names =[r'$\Delta x = 1.67$m', r'$\Delta x = 1.25$m' r'$\Delta x = 1.0$m', r'$\Delta x = 0.833$m', r'$\Delta x = 0.5$m']
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
        print("Variation type not recognized. Please choose from 'MLD', 'flux', 'strat', 'all', 'length', 'WENO', 'vertical resolution', or define your own case info in the comparison_info function.")
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
### -------------------------CALCULATING TEMPORAL AVERAGES------------------------- ###
def compute_temporal_averages(reader, center=(0.0, 0.0), start=10):
    x, y, z = reader.x, reader.y, reader.z
    t_save = reader.t_save[start:]
    x0, y0 = center

    # Load constants once
    reader.load_equation_of_state()
    g = 9.80665
    T0 = 25
    alpha = reader.alpha
    beta  = reader.beta if reader.salinity else None

    # Pre-cache spatial indices
    ix = np.argmin(np.abs(x - x0))
    iy = np.argmin(np.abs(y - y0))

    # Load all fields lazily — shape (nt - start, nx, ny, nz)
    T = reader.lazy_field('T')
    S = reader.lazy_field('S')
    u = reader.lazy_field('u')
    v = reader.lazy_field('v')
    w = reader.lazy_field('w')

    # Center velocities (still lazy)
    u = velocities_to_center(u, axis=-3)
    v = velocities_to_center(v, axis=-2)
    w = velocities_to_center(w, axis=-1)

    # Buoyancy (still lazy)
    b = g * alpha * (T - T0) - (g * beta * S if beta is not None else 0)

    # Horizontal means over (nx, ny) → shape (nt - start, nz)
    w_xy = da.mean(w[start:, :, :, :], axis=(1, 2))
    b_xy = da.mean(b[start:, :, :, :], axis=(1, 2))

    # Fluctuation and flux
    b_fluc = b[start:, :, :, :] - b_xy[:, np.newaxis, np.newaxis, :]
    bw     = da.mean(b_fluc * w[start:, :, :, :], axis=(1, 2))   # (nt - start, nz)

    # Temporal means of horizontal means → shape (nz,)
    S_avg = da.mean(S[start:, :, :, :], axis=(0, 1, 2))
    T_avg = da.mean(T[start:, :, :, :], axis=(0, 1, 2))
    w_avg = da.mean(w_xy, axis=0)

    # RMS then mean over time
    u_rms = da.mean(rms(u[start:, :, :, :]), axis=0)
    v_rms = da.mean(rms(v[start:, :, :, :]), axis=0)
    w_rms = da.mean(rms(w[start:, :, :, :]), axis=0)

    # bw_idx per timestep — argmax not lazy, so compute bw now
    bw_np  = bw.compute()                        # (nt - start, nz) — small, cheap
    bw_idx = np.argmax(bw_np, axis=1)            # (nt,)

    # Point values at (ix, iy, bw_idx[it]) per timestep
    nt     = len(t_save)
    it_idx = np.arange(nt)
    # extract the (nt - start, nx, ny, nz) arrays only at needed points
    S_pts  = S[start:, ix, iy, :].compute()           # (nt - start, nz)
    w_pts  = w[start:, ix, iy, :].compute()           # (nt - start, nz)
    S_value = np.mean(S_pts[it_idx, bw_idx])
    w_value = np.mean(w_pts[it_idx, bw_idx])

    # Center profiles — shape (nt - start, nz), mean over time → (nz,)
    S_center = da.mean(S[start:, ix, iy, :], axis=0)
    T_center = da.mean(T[start:, ix, iy, :], axis=0)
    w_center = da.mean(w[start:, ix, iy, :], axis=0)

    # Single compute call for everything still lazy
    (S_avg, T_avg, w_avg,
     u_rms, v_rms, w_rms,
     S_center, T_center, w_center) = da.compute(
        S_avg, T_avg, w_avg,
        u_rms, v_rms, w_rms,
        S_center, T_center, w_center,
    )

    return {
        "S_avg":    S_avg,
        "T_avg":    T_avg,
        "w_avg":    w_avg,
        "u_rms":    u_rms,
        "v_rms":    v_rms,
        "w_rms":    w_rms,
        "S_center": S_center,
        "T_center": T_center,
        "w_center": w_center,
        "S_value":  S_value,
        "w_value":  w_value,
    }

def compute_temporal_radius_avg(reader, tracer0, contour_bound = 0.05, start = 10):
    dense_plume = PlumeAnalysis(tracer0*contour_bound)
    x, y = reader.x, reader.y
    nx = reader.nx
    nt = reader.nt
    t_save = reader.t_save
    # ---------------- initializing arrays ---------------- #
    r_avg = np.zeros(nx[2])
    n = 0
    # ---------------- time loop ---------------- #
    for it in range(start, nt):
        S = reader.lazy_field('S'[it])
        dense_plume.input_info(S)
        r = dense_plume.plume_tracer_radius(x = x, y = y)
        r_avg += r
        n += 1

    return r_avg / n
### -------------------------WRITING TEMPORAL AVERAGES------------------------- ###
def write_temporal_averages(file_path, data, contour_bound = 0.05):
    folder_avg = "1D temporal averages"
    folder_centerline = "centerline temporal averages"
    folder_contour = f"contour temporal averages"
    folder_plume = f"plume statistics/contour {contour_bound}"

    with h5py.File(file_path, "w") as f:
        f.create_group(f'{folder_avg}')
        f.create_dataset(f'{folder_avg}/S', data=data['S_avg'])
        f.create_dataset(f'{folder_avg}/T', data=data['T_avg'])

        f.create_group(f'{folder_centerline}')
        f.create_dataset(f'{folder_centerline}/S', data=data['S_center'])
        f.create_dataset(f'{folder_centerline}/T', data=data['T_center'])
        f.create_dataset(f'{folder_centerline}/w', data=data['w_center'])

        f.create_group(f'{folder_contour}')
        f.create_dataset(f'{folder_contour}/S', data=data['S_value'])
        f.create_dataset(f'{folder_contour}/w', data=data['w_value'])