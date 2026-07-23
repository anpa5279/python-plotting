import numpy as np
import dask.array as da
import os 
import h5py

from interpolation import velocities_to_center
from physics import rms

### -------------------------COLLECTING COMPARISON CASE INFO----------------- ###
def comparison_info(variations, universal_folder = '/Users/annapauls/Documents/TESLa /Simulations/Oceananigans/dense plume/salinity and temperature/no noise circle inlet', ND=False, name_uni = ''):
    wp = -0.001
    if variations == 'strat':
        folder_names =['S0 = 0.1 dTdz = 0.005 MLD = 60', 'S0 = 0.1 dTdz = 0.01 MLD = 60', 'S0 = 0.1 dTdz = 0.05 MLD = 60', 'S0 = 0.1 dTdz = 0.1 MLD = 60'] 
        case_names =[r'dTdz = 0.005', r'dTdz = 0.01', r'dTdz = 0.05', r'dTdz = 0.10']  
        num_cases = len(folder_names)
        dTdz = np.array([0.005, 0.01, 0.05, 0.1]) # background temperature gradient in K/m
        mld = 60 * np.ones(num_cases) 
        F_s = wp * 0.1 * np.ones(num_cases) 
    elif variations == 'MLD':
        folder_names =['S0 = 0.1 dTdz = 0.01 MLD = 50', 'S0 = 0.1 dTdz = 0.01 MLD = 60', 'S0 = 0.1 dTdz = 0.01 MLD = 70']
        case_names =[r'MLD = 50m', r'MLD = 60m', r'MLD = 70m']
        num_cases = len(folder_names)
        dTdz = 0.01 * np.ones(num_cases) # background temperature gradient in K/m
        mld = np.array([50, 60, 70])
        F_s = wp * 0.1 * np.ones(num_cases) 
    elif variations == 'flux':
        folder_names =['S0 = 0.05 dTdz = 0.01 MLD = 60', 'S0 = 0.1 dTdz = 0.01 MLD = 60', 'S0 = 0.15 dTdz = 0.01 MLD = 60', 'S0 = 0.2 dTdz = 0.01 MLD = 60']
        case_names =[r'F$_{\text{C}} = -5.0\cdot 10^{-5}$', r'F$_{\text{C}} = -1.0\cdot 10^{-4}$', r'F$_{\text{C}} = -1.5\cdot 10^{-4}$', r'F$_{\text{C}} = - 2.0\cdot 10^{-4}$']
        num_cases = len(folder_names)
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
        num_cases = len(folder_names)
        mld = np.array([60, 50, 70, 60, 60, 60, 60, 60, 60]) # mld in m
        dTdz = np.array([0.01, 0.01, 0.01, 0.005, 0.05, 0.1, 0.01, 0.01, 0.01]) # background temperature gradient in K/m
        F_s = wp * np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.15, 0.2])
    elif variations == 'Lz160m':
        folder_names =[#'S0 = 0.0 dTdz = 0.01 MLD = 60', 
                       'S0 = 0.1 dTdz = 0.01 MLD = 60', 
                       'S0 = 0.1 dTdz = 0.01 MLD = 70', 
                       'S0 = 0.2 dTdz = 0.01 MLD = 60']
        case_names =[#r'F$_{\text{C}} = 0.0, MLD = 60m, dTdz = 0.01', 
                     r'F$_{\text{C}} = -1.0\cdot 10^{-4}$, MLD = 60m, dTdz = 0.01', 
                     r'F$_{\text{C}} = -1.0\cdot 10^{-4}$, MLD = 70m, dTdz = 0.01', 
                     r'F$_{\text{C}} = - 2.0\cdot 10^{-4}$, MLD = 60m, dTdz = 0.01']
        num_cases = len(folder_names)
        dTdz = 0.01 * np.ones(num_cases) # background temperature gradient in K/m
        mld = [60, 70, 60]
        F_s = wp * 0.1 * np.ones(num_cases) 
        #F_s[0] = 0.0
        F_s[2] = wp * 0.2
    elif variations == 'vertical length':
        folder_names =['nz = 77 z = 96.25 m', 'nz = 128 z = 160 m', 'nz = 192 z = 240 m']
        case_names =[r'L$_{\text{z}}$ = 96.25 m', r'L$_{\text{z}}$ = 160 m', r'L$_{\text{z}}$ = 240 m']
        num_cases = len(folder_names)
        dTdz = 0.01 * np.ones(num_cases) # background temperature gradient in K/m
        mld = 60 * np.ones(num_cases)
        F_s = wp * 0.1 * np.ones(num_cases) 
    elif variations == 'vertical resolution':
        folder_names =['dz05', 
                       'dz025',
                       'dz0125',
                       'dz00625'
                       ]
        case_names =[r'$\Delta z = 0.5$m', r'$\Delta z = 0.25$m', r'$\Delta z = 0.125$m', r'$\Delta z = 0.0625$m']
        num_cases = len(folder_names)
        dTdz = 0.01 * np.ones(num_cases) # background temperature gradient in K/m
        mld = 60 * np.ones(num_cases)
        F_s = wp * 0.1 * np.ones(num_cases)
        #folder_names =['vertical/dz2', 
        #            'vertical/dz1', 
        #            'ground0', 
        #            'vertical/dz025'
        #            ]
        #case_names =[r'$\Delta z = 2.0$m', r'$\Delta z = 1.0$m', r'$\Delta z = 0.5$m', r'$\Delta z = 0.25$m']
        #num_cases = len(folder_names)
        #dTdz = 0.01 * np.ones(num_cases) # background temperature gradient in K/m
        #mld = 60 * np.ones(num_cases)
        #F_s = wp * 0.1 * np.ones(num_cases)
    elif variations == 'AR=1':
        folder_names =[
                    'dz05',
                    'AR=1/dx025', 
                    'AR=1/dx0125', 
                    #'AR=1/dx00625', 
                       ]
        case_names =[r'$\Delta x = 0.5$m', r'$\Delta x = 0.25$m', r'$\Delta x = 0.125$m']#, r'$\Delta x = 0.0625$m',]
        num_cases = len(folder_names)
        dTdz = 0.01 * np.ones(num_cases) # background temperature gradient in K/m
        mld = 60 * np.ones(num_cases)
        F_s = wp * 0.1 * np.ones(num_cases)    
    elif variations == 'closure':
        folder_names =[
                    'dx05',
                    'dx025', 
                    'dx0125', 
                    #'dx00625', 
                       ]
        case_names =[r'$\Delta x = 0.5$m', r'$\Delta x = 0.25$m', r'$\Delta x = 0.125$m']#, r'$\Delta x = 0.0625$m',]
        num_cases = len(folder_names)
        dTdz = 0.01 * np.ones(num_cases) # background temperature gradient in K/m
        mld = 60 * np.ones(num_cases)
        F_s = wp * 0.1 * np.ones(num_cases)
    elif variations == 'horizontal resolution':
        folder_names =['dz05', 
                       'dx025', 
                       'dx0125', 
                       'dx00625', 
                       ]
        case_names =[r'$\Delta x = 0.5$m', r'$\Delta x = 0.25$m', r'$\Delta x = 0.125$m', r'$\Delta x = 0.0625$m']
        num_cases = len(folder_names)
        dTdz = 0.01 * np.ones(num_cases) # background temperature gradient in K/m
        mld = 60 * np.ones(num_cases)
        F_s = wp * 0.1 * np.ones(num_cases) 
    elif variations == 'vertical erf':
        folder_names =[
                       'ground0-erf',
                       'vertical/dz-025', 
                       #'vertical/dz-0125',
                       ]
        case_names =[r'$\Delta z = 0.5$m', r'$\Delta z = 0.25$m']#, r'$\Delta z = 0.125$m']
        num_cases = len(folder_names)
        dTdz = 0.01 * np.ones(num_cases) # background temperature gradient in K/m
        mld = 60 * np.ones(num_cases)
        F_s = wp * 0.1 * np.ones(num_cases)
    elif variations == 'visc':
        folder_names =[
                       'dz-05',
                       'dz-025', 
                       'dz-0125',
                       ]
        case_names =[r'$\Delta z = 0.5$m', r'$\Delta z = 0.25$m', r'$\Delta z = 0.125$m']
        num_cases = len(folder_names)
        dTdz = 0.01 * np.ones(num_cases) # background temperature gradient in K/m
        mld = 60 * np.ones(num_cases)
        F_s = wp * 0.1 * np.ones(num_cases)
    elif variations == 'w BC':
        folder_names = ['open w BC', 
                        'open w top BC', 
                        'open w BC without scheme', 
                        'open w top BC without scheme', 
                        'open w BC bottom adjusted',
                        'testing gaussian top/open w gauss top BC 0 scheme'
                        ]
        case_names = ['Open BC with PA top and \nOpen BC with nothing bottom', 
                      'Open BC with PA top', 
                      'Open BC top and \nOpen BC with nothing bottom', 
                      'Open BC top',
                      'Open BC with PA top & \nOpen BC with PA bottom', 
                      'Open gaussian BC with PA top & \nOpen BC with nothing bottom', 
                      
                      ]
        num_cases = len(folder_names)
        dTdz = 0.01 * np.ones(num_cases) # background temperature gradient in K/m
        mld = 60 * np.ones(num_cases)
        F_s = wp * 0.1 * np.ones(num_cases)
    elif variations == 'w timescale BC':
        folder_names = [#'open w BC', 
                        'testing timescale/open w BC bottom adjusted default scheme', 
                        'open w BC bottom adjusted', 
                        ]
        case_names = [ 
                      'Open BC with default PA top \n& Open BC with default PA bottom scaled', 
                      'Open BC with 0.0 PA top \n& Open BC with 0.0 PA bottom scaled', 
                      ]
        num_cases = len(folder_names)
        dTdz = 0.01 * np.ones(num_cases) # background temperature gradient in K/m
        mld = 60 * np.ones(num_cases)
        F_s = wp * 0.1 * np.ones(num_cases)
    elif variations == 'w gaus BC':
        folder_names = [
                        'open w gauss top BC default scheme',
                        'open w gauss top BC 0 scheme',
                        'open w 2rp gauss top BC default scheme',
                        'open w 2rp gauss BC 0 scheme',
                        ]
        case_names = [
                      r'Open BC $\sigma$ = 4m top w/ default PA' + os.linesep +'& Open BC w/ nothing bottom',
                      r'Open BC $\sigma$ = 4m top w/ 0.0 PA' + os.linesep +'& Open BC w/ nothing bottom',
                      r'Open BC $\sigma$ = 8m top w/ default PA' + os.linesep +'& Open BC w/ nothing bottom',
                      r'Open BC $\sigma$ = 8m BC top w/ 0.0 PA' + os.linesep +'& Open BC w/ nothing bottom',
                      ]
        num_cases = len(folder_names)
        dTdz = 0.01 * np.ones(num_cases) # background temperature gradient in K/m
        mld = 60 * np.ones(num_cases)
        F_s = wp * 0.1 * np.ones(num_cases)
    elif variations == 'WENO':
        folder_names = ['S0 = 0.1 dTdz = 0.01 MLD = 60', 'S0 = 0.1 dTdz = 0.01 MLD = 60 WENO mod', 'S0 = 0.1 dTdz = 0.01 MLD = 60 WENO mod callback']
        case_names = [r'Default', r'WENO modified', r'WENO modified with callback 0 function']
        num_cases = len(folder_names)
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
        fig_folder = os.path.join(universal_folder, 'comparison figures', variations + ' comparison figures')
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
### -------------------------BINNING----------------------------------------- ###
def azimuthal_avg(var, X, Y, dx_scale = None, return_r = False):
    if dx_scale is None:
        X_bin = X
        dx_scale = max([np.diff(X)[0, 0], np.diff(Y, axis=0)[0, 0]]) 
    else:
        x_bin = np.arange(X.min(), X.max()+dx_scale, dx_scale)
        y_bin = np.arange(Y.min(), Y.max()+dx_scale, dx_scale)
        X_bin, Y_bin = np.meshgrid(x_bin, y_bin)

    nx = X_bin.shape
    ncirc = min(nx[0], nx[1])
    if ncirc % 2 == 0:
        ncirc = ncirc//2      # full circular shells
    else:
        ncirc = ncirc//2+1      # full circular shells
    r_bin = np.sqrt((X/dx_scale)**2 + (Y/dx_scale)**2).astype(int)
    counts = np.bincount(r_bin.flat)  # number of points in each radial shell, including corners
    bin_var = np.bincount(r_bin.flat, weights=var.flat)

    # cut off the corners that aren't full circles.
    bin_var = (1 / counts[:ncirc]) * bin_var[:ncirc]
    if return_r:
        r = np.arange(np.min(np.abs([X_bin, Y_bin]))/2, ncirc*dx_scale, dx_scale)
        return r, bin_var
    return bin_var
def binning_oc(reader, center=(0.0, 0.0)):
    nx = reader.nx
    t_save = reader.t_save
    nt = len(t_save)

    dx_scale = np.max(reader.dx[:-1]) # not including dz
    x = reader.x - center[0]
    y = reader.y - center[1]
    z = reader.z
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    nr = np.min(nx[:-1])//2
    S_rz = np.empty((nr, nx[2], nt))
    T_rz = np.empty((nr, nx[2], nt)) 
    ur_rz = np.empty((nr, nx[2], nt))
    utheta_rz = np.empty((nr, nx[2], nt))
    w_rz = np.empty((nr, nx[2], nt))

    for it, t in enumerate(t_save):
        # Load data from files
        T = reader.lazy_field('T', t).compute()
        u = reader.lazy_field('u', t).compute()
        v = reader.lazy_field('v', t).compute()
        w = reader.lazy_field('w', t).compute()
        if reader.salinity:
            S = reader.lazy_field('S', t).compute()

        u = velocities_to_center(u, axis=-3)
        v = velocities_to_center(v, axis=-2)
        w = velocities_to_center(w, axis=-1)

        # u and v to cylindrical polar coordinates
        ur, utheta = vel_to_cylindrical(u, v, X, Y)

        for k in range(nx[2]):
            if reader.salinity:
                S_rz[:, k, it] = azimuthal_avg(S[:, :, k], X[:, :, k], Y[:, :, k], dx_scale=dx_scale)
            T_rz[:, k, it] = azimuthal_avg(T[:, :, k], X[:, :, k], Y[:, :, k], dx_scale=dx_scale)
            utheta_rz[:, k, it] = azimuthal_avg(utheta[:, :, k], X[:, :, k], Y[:, :, k], dx_scale=dx_scale)
            ur_rz[:, k, it] = azimuthal_avg(ur[:, :, k], X[:, :, k], Y[:, :, k], dx_scale=dx_scale)
            w_rz[:, k, it] = azimuthal_avg(w[:, :, k], X[:, :, k], Y[:, :, k], dx_scale=dx_scale)
        del T, u, v, w, ur, utheta # to be memory efficient
        if reader.salinity:
            del S # to be memory efficient
    return S_rz, T_rz, ur_rz, utheta_rz, w_rz

### -------------------------CYLINDRICAL POLAR COORDINATES------------------- ###
def vel_to_cylindrical(u, v, X, Y):
    dist = np.sqrt(X**2 + Y**2)
    ur = u*X/dist + v*Y/dist
    utheta = -u*Y/dist + v*X/dist
    return ur, utheta
### -------------------------CALCULATING 1D AVERAGES------------------------- ###
def compute_temporal_averages(reader, start=10):
    # Load constants once
    reader.load_equation_of_state()

    # Load all fields lazily — shape (nt - start, nx, ny, nz)
    S = reader.field_centerline('S')
    S = S[start:, :]
    w = reader.field_centerline('w')
    w = w[start:, :]

    #loading in vertical 1D info
    b_xy, b_rms, b_centerline, b_fluc_centerline = reader.load_buoyancy()
    
    # Fluctuation and flux
    bw = b_fluc_centerline[start:, :] * w   # (nt - start, nz)

    # bw_idx per timestep — argmax not lazy, so compute bw now
    bw_idx = np.argmax(bw, axis = 1)            # (nt,)

    # Point values per timestep
    it_idx = np.arange(bw.shape[0])
    # extract the (nt - start, nx, ny, nz) arrays only at needed points
    S_value = np.mean(S[it_idx, bw_idx])
    w_value = np.mean(w[it_idx, bw_idx])

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

    # Load all fields lazily — shape (nt, nx, ny, nz)
    dims = (-3, -2)
    T = reader.lazy_field('T').compute()
    u = reader.lazy_field('u').compute()
    v = reader.lazy_field('v').compute()
    w = reader.lazy_field('w').compute()
    if reader.salinity:
        S = reader.lazy_field('S').compute()
        # Buoyancy (still lazy)
        beta  = reader.beta
        b = g * alpha * (T - T0) - (g * beta * S)
    else:
        b = g * alpha * (T - T0)
    
    # Center velocities (still lazy)
    u = velocities_to_center(u, axis=-3)
    v = velocities_to_center(v, axis=-2)
    w = velocities_to_center(w, axis=-1)

    # convert u and v to cylindrical polar coordinates (still lazy)
    X, Y, Z = np.meshgrid(reader.x, reader.y, reader.z, indexing='ij')
    ur, utheta = vel_to_cylindrical(u, v, X, Y)

    # Horizontal means over (nx, ny) → shape (nt, nz)
    T_xy = da.mean(T, axis=dims)
    ur_xy = da.mean(ur, axis=dims)
    utheta_xy = da.mean(utheta, axis=dims)
    w_xy = da.mean(w, axis=dims)
    b_xy = da.mean(b, axis=dims)

    # Fluctuation
    T_fluc = T - T_xy[:, np.newaxis, np.newaxis, :]
    ur_fluc = ur - ur_xy[:, np.newaxis, np.newaxis, :]
    utheta_fluc = utheta - utheta_xy[:, np.newaxis, np.newaxis, :]
    w_fluc = w - w_xy[:, np.newaxis, np.newaxis, :]
    b_fluc = b - b_xy[:, np.newaxis, np.newaxis, :]

    # averages of fluctuations
    T_fluc_avg = da.mean(T_fluc, axis=dims)
    ur_fluc_avg = da.mean(ur_fluc, axis=dims)
    utheta_fluc_avg = da.mean(utheta_fluc, axis=dims)
    w_fluc_avg = da.mean(w_fluc, axis=dims)
    b_fluc_avg = da.mean(b_fluc, axis=dims)
    bur_fluc_avg = da.mean(b_fluc * ur, axis=dims)
    butheta_fluc_avg = da.mean(b_fluc * utheta, axis=dims)
    bw_fluc_avg = da.mean(b_fluc * w, axis=dims)
    if reader.salinity:
        S_xy = da.mean(S, axis=dims)
        S_fluc = S - S_xy[:, np.newaxis, np.newaxis, :]
        S_fluc_avg = da.mean(S_fluc, axis=dims)
        return {'T_fluc': T_fluc_avg,
                'S_fluc': S_fluc_avg,
                'ur_fluc': ur_fluc_avg,
                'utheta_fluc': utheta_fluc_avg,
                'w_fluc': w_fluc_avg,
                'b_fluc': b_fluc_avg,
                'bur_fluc': bur_fluc_avg,
                'butheta_fluc': butheta_fluc_avg,
                'bw_fluc': bw_fluc_avg}
    else:
        return {'T_fluc': T_fluc_avg,
                'ur_fluc': ur_fluc_avg,
                'utheta_fluc': utheta_fluc_avg,
                'w_fluc': w_fluc_avg,
                'b_fluc': b_fluc_avg,
                'bur_fluc': bur_fluc_avg,
                'butheta_fluc': butheta_fluc_avg,
                'bw_fluc': bw_fluc_avg}

def compute_rms(reader):
    u_rms = np.empty((reader.nt, reader.nx[2]))
    v_rms = np.empty((reader.nt, reader.nx[2]))
    w_rms = np.empty((reader.nt, reader.nx[2]))
    for it, t in enumerate(reader.t_save):
        u = reader.lazy_field('u', steps=t).compute()
        v = reader.lazy_field('v', steps=t).compute()
        w = reader.lazy_field('w', steps=t).compute()
        u = velocities_to_center(u, axis=-3)
        v = velocities_to_center(v, axis=-2) 
        w = velocities_to_center(w, axis=-1)
        u_rms[it, :] = rms(u)
        v_rms[it, :] = rms(v)
        w_rms[it, :] = rms(w)
    return {'u_rms': u_rms,
            'v_rms': v_rms,
            'w_rms': w_rms}
