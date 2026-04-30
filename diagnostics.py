import numpy as np
import os 
import h5py

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
def compute_temporal_averages(reader, time_indices, grid, physics, interp, center=(0.0, 0.0)):
    x, y, z = grid["x"], grid["y"], grid["z"]
    nx = grid["nx"]
    x0, y0 = center
    n = 0
    # ---------------- initializing arrays ---------------- #
    S_avg = np.zeros(nx[2])
    T_avg = np.zeros(nx[2])
    b_avg = np.zeros(nx[2])
    w_avg = np.zeros(nx[2])

    Tw_avg = np.zeros(nx[2])
    Sw_avg = np.zeros(nx[2])
    bw_avg = np.zeros(nx[2])

    u_rms = np.zeros(nx[2])
    v_rms = np.zeros(nx[2])
    w_rms = np.zeros(nx[2])

    S_center = np.zeros(nx[2])
    T_center = np.zeros(nx[2])
    b_center = np.zeros(nx[2])
    w_center = np.zeros(nx[2])

    # ---------------- time loop ---------------- #
    for it in time_indices:
        u, v, w, T, S, _, _ = reader(it)
        # center velocities
        u = interp.fcc_ccc(u)
        v = interp.cfc_ccc(v)
        w = interp.ccf_ccc(w)

        # buoyancy
        b = (
            physics["g"] * physics["alpha"] * (T - physics["T0"])
            - physics["g"] * physics["beta"] * (S - physics["S0"])
        )

        # ---------------- horizontal means ---------------- #
        S_h = np.mean(S, axis=(-3, -2))
        T_h = np.mean(T, axis=(-3, -2))
        b_h = np.mean(b, axis=(-3, -2))
        w_h = np.mean(w, axis=(-3, -2))
        # ---------------- fluctuations ---------------- #
        S_fluc = S - S_h
        T_fluc = T - T_h
        b_fluc = b - b_h
        w_fluc = w - w_h
        # ---------------- fluxes ---------------- #
        Tw = T_fluc * w_fluc
        Sw = S_fluc * w_fluc
        bw = b_fluc * w_fluc
        # ---------------- RMS ---------------- #
        u_rms += np.mean((u - np.mean(u, axis=(-3, -2)))**2, axis=(-3, -2))
        v_rms += np.mean((v - np.mean(v, axis=(-3, -2)))**2, axis=(-3, -2))
        w_rms += np.mean(w_fluc**2, axis=(-3, -2))
        # ---------------- accumulation ---------------- #
        S_avg += S_h
        T_avg += T_h
        b_avg += b_h
        w_avg += w_h
        Tw_avg += np.mean(Tw, axis=(-3, -2))
        Sw_avg += np.mean(Sw, axis=(-3, -2))
        bw_avg += np.mean(bw, axis=(-3, -2))
        # ---------------- centerline ---------------- #
        S_center += interp.xy_plane(S, z, z0=0.0)
        T_center += interp.xy_plane(T, z, z0=0.0)
        b_center += interp.xy_plane(b, z, z0=0.0)
        w_center += interp.xy_plane(w, z, z0=0.0)

        n += 1

    inv_n = 1.0 / n

    return {
        "S_avg": S_avg * inv_n,
        "T_avg": T_avg * inv_n,
        "b_avg": b_avg * inv_n,
        "w_avg": w_avg * inv_n,

        "Tw_avg": Tw_avg * inv_n,
        "Sw_avg": Sw_avg * inv_n,
        "bw_avg": bw_avg * inv_n,

        "u_rms": np.sqrt(u_rms * inv_n),
        "v_rms": np.sqrt(v_rms * inv_n),
        "w_rms": np.sqrt(w_rms * inv_n),

        "S_center": S_center * inv_n,
        "T_center": T_center * inv_n,
        "b_center": b_center * inv_n,
        "w_center": w_center * inv_n,
    }
### -------------------------WRITING TEMPORAL AVERAGES------------------------- ###
def write_temporal_averages(file_path, data):
    with h5py.File(file_path, "w") as f:
        grp1 = f.create_group("1D averages")
        grp2 = f.create_group("centerline")
        for k, v in data.items():
            if "center" in k:
                grp2.create_dataset(k, data=v)
            else:
                grp1.create_dataset(k, data=v)
