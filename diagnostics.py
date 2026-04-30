import numpy as np
import os 
import h5py

from interpolation import velocities_to_center, vertical_line, point
from dense_plume import PlumeAnalysis
from physics import rms, buoyancy

### -------------------------COLLECTING COMPARISON CASE INFO------------------------- ###
def comparison_info(variations, universal_folder, ND=False):
    if variations == 'strat':
        folder_names =['S0 = 0.1 dTdz = 0.005 MLD = 60', 'S0 = 0.1 dTdz = 0.01 MLD = 60', 'S0 = 0.1 dTdz = 0.05 MLD = 60', 'S0 = 0.1 dTdz = 0.1 MLD = 60'] 
        case_names =[r'dTdz = 0.005', r'dTdz = 0.01', r'dTdz = 0.05', r'dTdz = 0.10']  
        num_cases = len(case_names)
        dTdz = np.array([0.005, 0.01, 0.05, 0.1]) # background temperature gradient in K/m
        mld = 60 * np.ones(num_cases) 
        F_s = 0.001 * 0.1 * np.ones(num_cases) 
    elif variations == 'MLD':
        folder_names =['S0 = 0.1 dTdz = 0.01 MLD = 50', 'S0 = 0.1 dTdz = 0.01 MLD = 60', 'S0 = 0.1 dTdz = 0.01 MLD = 70']
        case_names =[r'MLD = 50m', r'MLD = 60m', r'MLD = 70m']
        num_cases = len(case_names)
        dTdz = 0.01 * np.ones(num_cases) # background temperature gradient in K/m
        mld = np.array([50, 60, 70])
        F_s = 0.001 * 0.1 * np.ones(num_cases) 
    elif variations == 'flux':
        folder_names =['S0 = 0.05 dTdz = 0.01 MLD = 60', 'S0 = 0.1 dTdz = 0.01 MLD = 60', 'S0 = 0.15 dTdz = 0.01 MLD = 60', 'S0 = 0.2 dTdz = 0.01 MLD = 60']
        case_names =[r'F$_{\text{C}} = -5.0\cdot 10^{-5}$', r'F$_{\text{C}} = -1.0\cdot 10^{-4}$', r'F$_{\text{C}} = -1.5\cdot 10^{-4}$', r'F$_{\text{C}} = - 2.0\cdot 10^{-4}$']
        num_cases = len(case_names)
        dTdz = 0.01 * np.ones(num_cases) # background temperature gradient in K/m
        mld = 60 * np.ones(num_cases)  
        F_s = 0.001 * np.array([0.05, 0.1, 0.15, 0.2])
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
        F_s = 0.001 * np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.15, 0.2])
    elif variations == 'length':
        folder_names =['nz = 77 z = 96.25 m', 'nz = 128 z = 160 m', 'nz = 192 z = 240 m']
        case_names =[r'L$_{\text{z}}$ = 96.25 m', r'L$_{\text{z}}$ = 160 m', r'L$_{\text{z}}$ = 240 m']
        num_cases = len(case_names)
        dTdz = 0.01 * np.ones(num_cases) # background temperature gradient in K/m
        mld = 60 * np.ones(num_cases)
        F_s = 0.001 * 0.1 * np.ones(num_cases) 
    else:
        print("Variation type not recognized. Please choose from 'MLD', 'flux', 'strat', 'all', 'length', or define your own case info in the comparison_info function.")
        return None # user defined specific case not defined here
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
        if variations == 'length':
            universal_folder = '/Users/annapauls/Library/CloudStorage/OneDrive-UCB-O365/CU-Boulder/TESLa/Carbon Sequestration/Simulations/Oceananigans/NBP/salinity and temperature/no noise circle inlet/vertical domain increase/dTdz = 0.01'
        else:
            universal_folder = '/Users/annapauls/Library/CloudStorage/OneDrive-UCB-O365/CU-Boulder/TESLa/Carbon Sequestration/Simulations/Oceananigans/NBP/salinity and temperature/no noise circle inlet'
        
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
def compute_temporal_averages(reader, center=(0.0, 0.0), start=10, T0 = 25.0, rho0 = 1026.0):
    x, y, z = reader.x, reader.y, reader.z
    nt = reader.nt
    t_save = reader.t_save
    nx = reader.nx
    x0, y0 = center

    coeffs = reader.load_equation_of_state(True)

    n = 0
    # ---------------- initializing arrays ---------------- #
    S_avg = np.zeros(nx[2])
    T_avg = np.zeros(nx[2])
    b_avg = np.zeros(nx[2])
    w_avg = np.zeros(nx[2])

    T_fluc_avg = np.zeros(nx[2])
    b_fluc_avg = np.zeros(nx[2])

    Tw_avg = np.zeros(nx[2])
    Sw_avg = np.zeros(nx[2])
    bw_avg = np.zeros(nx[2])

    u_rms = np.zeros(nx[2])
    v_rms = np.zeros(nx[2])
    w_rms = np.zeros(nx[2])

    S_value = 0
    w_value = 0

    S_center = np.zeros(nx[2])
    T_center = np.zeros(nx[2])
    T_fluc_center = np.zeros(nx[2])
    b_center = np.zeros(nx[2])
    b_fluc_center = np.zeros(nx[2])
    w_center = np.zeros(nx[2])

    # ---------------- time loop ---------------- #
    for it in range(start, nt):
        u = reader.lazy_field('u', t_save[it])
        v = reader.lazy_field('v', t_save[it])
        w = reader.lazy_field('w', t_save[it])
        T = reader.lazy_field('T', t_save[it])
        S = reader.lazy_field('S', t_save[it])
        # center velocities
        u, v, w = velocities_to_center(u, v, w)
        # buoyancy
        b_temp = buoyancy(T, rho0, coeffs, T0, S)
        b = b_temp['b_total']

        # ---------------- horizontal means ---------------- #
        S_xy = np.mean(S, axis=(-3, -2))
        T_xy = np.mean(T, axis=(-3, -2))
        b_xy = np.mean(b, axis=(-3, -2))
        w_xy = np.mean(w, axis=(-3, -2))
        # ---------------- fluctuations ---------------- #
        T_fluc = T - T_xy
        b_fluc = b - b_xy
        T_fluc_xy = np.mean(T_fluc, axis=(-3, -2))
        b_fluc_xy = np.mean(b_fluc, axis=(-3, -2))
        # ---------------- fluxes ---------------- #
        Tw = np.mean(T_fluc * w, axis=(-3, -2))
        Sw = np.mean(S * w, axis=(-3, -2))
        bw = np.mean(b_fluc * w, axis=(-3, -2))
        # ---------------- RMS ---------------- #
        u_rms += rms(u)
        v_rms += rms(v)
        w_rms += rms(w)
        # ---------------- contour values ---------------- #
        bw_idx = np.where(bw==np.max(bw))[0][0]
        S_value += point(S, x, y, z, x0, y0, z[bw_idx])
        w_value += point(w, x, y, z, x0, y0, z[bw_idx])
        # ---------------- accumulation ---------------- #
        S_avg += S_xy
        T_avg += T_xy
        b_avg += b_xy
        w_avg += w_xy
        T_fluc_avg += T_fluc_xy
        b_fluc_avg += b_fluc_xy
        Tw_avg += Tw
        Sw_avg += Sw
        bw_avg += bw
        S_center += vertical_line(S, x, y, x0, y0)
        T_center += vertical_line(T, x, y, x0, y0)
        T_fluc_center += vertical_line(T_fluc, x, y, x0, y0)
        b_center += vertical_line(b, x, y, x0, y0)
        b_fluc_center += vertical_line(b_fluc, x, y, x0, y0)
        w_center += vertical_line(w, x, y, x0, y0)

        n += 1

    return {
        "S_avg": S_avg / n,
        "T_avg": T_avg / n,
        "b_avg": b_avg / n,
        "w_avg": w_avg / n,

        "T_fluc_avg": T_fluc_avg / n,
        "b_fluc_avg": b_fluc_avg / n,

        "Tw_avg": Tw_avg / n,
        "Sw_avg": Sw_avg / n,
        "bw_avg": bw_avg / n,

        "u_rms": u_rms / n,
        "v_rms": v_rms / n,
        "w_rms": w_rms / n,

        "S_center": S_center / n,
        "T_center": T_center / n,
        "T_fluc_center": T_fluc_center / n,
        "b_center": b_center / n,
        "b_fluc_center": b_fluc_center / n,
        "w_center": w_center / n,

        "S_value": S_value / n,
        "w_value": w_value / n
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
        S = reader.lazy_field('S', t_save[it])
        dense_plume.input_info(S)
        r = dense_plume.plume_tracer_radius(x, y)
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
        f.create_dataset(f'{folder_avg}/b', data=data['b_avg'])
        f.create_dataset(f'{folder_avg}/w', data=data['w_avg'])
        f.create_dataset(f'{folder_avg}/T\'', data=data['T_fluc_avg'])
        f.create_dataset(f'{folder_avg}/b\'w', data=data['bw_fluc_avg'])

        f.create_group(f'{folder_centerline}')
        f.create_dataset(f'{folder_centerline}/S', data=data['S_center'])
        f.create_dataset(f'{folder_centerline}/T', data=data['T_center'])
        f.create_dataset(f'{folder_centerline}/T\'', data=data['T_fluc_center'])
        f.create_dataset(f'{folder_centerline}/b', data=data['b_center'])
        f.create_dataset(f'{folder_centerline}/b\'', data=data['b_fluc_center'])
        f.create_dataset(f'{folder_centerline}/w', data=data['w_center'])

        f.create_group(f'{folder_contour}')
        f.create_dataset(f'{folder_contour}/S', data=data['S_value'])
        f.create_dataset(f'{folder_contour}/w', data=data['w_value'])

        f.create_group(f'{folder_plume}')
        f.create_dataset(f'{folder_plume}/plume tracer radius with depth', data=data['radius_tracer'])


