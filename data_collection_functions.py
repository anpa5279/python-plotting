import os
import numpy as np
import h5py

### -------------------------EXTRACTING DATA------------------------- ###
## collecting model information
def collect_time_outputs(file, Nranks, stokes=False):
    with h5py.File(file, 'r') as f:
        timeseries_group = [g for g in f.keys() if 'timeseries' in g][0]
        t_group = f[timeseries_group + '/t']
        t_save = sorted([float(k) for k in t_group.keys()])
        t_save = np.array(t_save)
        time = np.array([t_group[str(int(k))][()] for k in t_save])
        nx = [f['grid/Nx'][()] * Nranks, f['grid/Ny'][()], f['grid/Nz'][()]]
        hx = [f['grid/Hx'][()], f['grid/Hy'][()], f['grid/Hz'][()]]
        lx = [f['grid/Lx'][()] * Nranks, f['grid/Ly'][()], f['grid/Lz'][()]]
        dx = [f['grid/Δxᶜᵃᵃ'][()], f['grid/Δyᵃᶜᵃ'][()], f['grid/z/Δᵃᵃᶜ'][()]]
        visc = f['closure/ν'][()]
        diff= f['closure/κ']
        x = dx[0]/2 + np.arange(nx[0]) * dx[0]
        y = f['grid/yᵃᶜᵃ'][hx[1]:-hx[1]]
        z = f['grid/z/cᵃᵃᶜ'][hx[2]:-hx[2]]
        xf = np.arange(nx[0]) * dx[0]
        yf = f['grid/yᵃᶠᵃ'][hx[1]:-hx[1]]
        zf = f['grid/z/cᵃᵃᶠ'][hx[2]:-hx[2]]
    if stokes:
        u_f =np.array(f["IC/"]["friction_velocity"])
        u_s = np.array(f["IC/"]["stokes_velocity"])
        return time, t_save, nx, hx, lx, x, y, z, xf, yf, zf, dx, visc, diff, u_f, u_s
    else:
        return time, t_save, nx, hx, lx, x, y, z, xf, yf, zf, dx, visc, diff
# collecting thermal expansion and haline contraction coefficients
def collect_temp_and_sal(file, salinity=False):
    with h5py.File(file, 'r') as f:
        alpha = f['buoyancy/formulation/equation_of_state/thermal_expansion'][()]
        if salinity:
            beta = f['buoyancy/formulation/equation_of_state/haline_contraction'][()]
            return alpha, beta
    return alpha
## collecting 3D fields
## -------------------------EXTRACTING 3D FIELDS------------------------- ###
def collect_fields(folder, dtn, t_save, hx, temperature=True, salinity=False, with_halos=False):
    # Load data from files
    fname = os.path.join(folder, dtn)
    if temperature:
        with h5py.File(fname, 'r') as f:
            if with_halos:
                u = (f['timeseries/u'][f'{int(t_save)}'])[hx[0]:-hx[0], hx[1]:-hx[1], hx[2]:-hx[2]].transpose(2, 1, 0)
                v = (f['timeseries/v'][f'{int(t_save)}'])[hx[0]:-hx[0], hx[1]:-hx[1], hx[2]:-hx[2]].transpose(2, 1, 0)
                w = (f['timeseries/w'][f'{int(t_save)}'])[hx[0]:-hx[0], hx[1]:-hx[1], hx[2]:-hx[2]].transpose(2, 1, 0)
                T = (f['timeseries/T'][f'{int(t_save)}'])[hx[0]:-hx[0], hx[1]:-hx[1], hx[2]:-hx[2]].transpose(2, 1, 0) 
                Pdynamic = (f['timeseries/P_dynamic'][f'{int(t_save)}'][hx[0]:-hx[0], hx[1]:-hx[1], hx[2]:-hx[2]]).transpose(2, 1, 0) 
                Pstatic = (f['timeseries/P_static'][f'{int(t_save)}'][hx[0]:-hx[0], hx[1]:-hx[1], hx[2]:-hx[2]]).transpose(2, 1, 0) 
                if salinity:
                    S = (f['timeseries/S'][f'{int(t_save)}'])[hx[0]:-hx[0], hx[1]:-hx[1], hx[2]:-hx[2]].transpose(2, 1, 0)
                    return u, v, w, T, S, Pdynamic, Pstatic
            else:
                u = (f['timeseries/u'][f'{int(t_save)}'][:, :, :]).transpose(2, 1, 0)
                v = (f['timeseries/v'][f'{int(t_save)}'][:, :, :]).transpose(2, 1, 0) 
                w = (f['timeseries/w'][f'{int(t_save)}'][:, :, :]).transpose(2, 1, 0) 
                T = (f['timeseries/T'][f'{int(t_save)}'][:, :, :]).transpose(2, 1, 0) 
                Pdynamic = (f['timeseries/P_dynamic'][f'{int(t_save)}'][:, :, :]).transpose(2, 1, 0) 
                Pstatic = (f['timeseries/P_static'][f'{int(t_save)}'][:, :, :]).transpose(2, 1, 0) 
                if salinity:
                    S = (f['timeseries/S'][f'{int(t_save)}'][:, :, :]).transpose(2, 1, 0)
                    S[S<10*(-16)] = 0.0
                    return u, v, w, T, S, Pdynamic, Pstatic
        return u, v, w, T, Pdynamic, Pstatic
    else:
        with h5py.File(fname, 'r') as f:
            if with_halos:
                u = (f['timeseries/u'][f'{int(t_save)}'])[hx[0]:-hx[0], hx[1]:-hx[1], hx[2]:-hx[2]].transpose(2, 1, 0)
                v = (f['timeseries/v'][f'{int(t_save)}'])[hx[0]:-hx[0], hx[1]:-hx[1], hx[2]:-hx[2]].transpose(2, 1, 0)
                w = (f['timeseries/w'][f'{int(t_save)}'])[hx[0]:-hx[0], hx[1]:-hx[1], hx[2]:-hx[2]].transpose(2, 1, 0)
                b = (f['timeseries/b'][f'{int(t_save)}'])[hx[0]:-hx[0], hx[1]:-hx[1], hx[2]:-hx[2]].transpose(2, 1, 0) 
                Pdynamic = (f['timeseries/P_dynamic'][f'{int(t_save)}'][hx[0]:-hx[0], hx[1]:-hx[1], hx[2]:-hx[2]]).transpose(2, 1, 0) 
                Pstatic = (f['timeseries/P_static'][f'{int(t_save)}'][hx[0]:-hx[0], hx[1]:-hx[1], hx[2]:-hx[2]]).transpose(2, 1, 0) 
            else:
                u = (f['timeseries/u'][f'{int(t_save)}'][:, :, :]).transpose(2, 1, 0)
                v = (f['timeseries/v'][f'{int(t_save)}'][:, :, :]).transpose(2, 1, 0) 
                w = (f['timeseries/w'][f'{int(t_save)}'][:, :, :]).transpose(2, 1, 0) 
                b = (f['timeseries/b'][f'{int(t_save)}'][:, :, :]).transpose(2, 1, 0) 
                Pdynamic = (f['timeseries/P_dynamic'][f'{int(t_save)}'][:, :, :]).transpose(2, 1, 0) 
                Pstatic = (f['timeseries/P_static'][f'{int(t_save)}'][:, :, :]).transpose(2, 1, 0) 
        return u, v, w, b, Pdynamic, Pstatic
## distribued
def collect_fields_distributed(Nranks, folder, dtn, t_save, hx, nx, temperature=True, salinity=False, with_halos=False):
    # Initialize arrays
    u = np.zeros((nx[0], nx[1], nx[2]))
    v = np.zeros((nx[0], nx[1], nx[2]))
    w = np.zeros((nx[0], nx[1], nx[2] + 1))
    Pdynamic = np.zeros((nx[0], nx[1], nx[2]))
    Pstatic = np.zeros((nx[0], nx[1], nx[2]))
    xrange = range(0, nx[0] // Nranks)
    if temperature and not salinity:
        T = np.zeros((nx[0], nx[1], nx[2]))
        for r in range(Nranks):
            u_dist, v_dist, w_dist, T_dist, P_d_dist, P_s_dist = collect_fields(folder, dtn[r], t_save, hx, temperature, salinity, with_halos)
            u[xrange, :, :] = u_dist
            v[xrange, :, :] = v_dist
            w[xrange, :, :] = w_dist
            T[xrange, :, :] = T_dist
            Pdynamic[xrange, :, :] = P_d_dist
            Pstatic[xrange, :, :] = P_s_dist
            new_range = range(xrange.start + (nx[0] // Nranks), xrange.stop + (nx[0] // Nranks))
            xrange = new_range
        return u, v, w, T, Pdynamic, Pstatic
    elif temperature and salinity:
        T = np.zeros((nx[0], nx[1], nx[2]))
        S = np.zeros((nx[0], nx[1], nx[2])) 
        for r in range(Nranks):
            u_dist, v_dist, w_dist, T_dist, S_dist, P_d_dist, P_s_dist = collect_fields(folder, dtn[r], t_save, hx, temperature, salinity, with_halos)
            u[xrange, :, :] = u_dist
            v[xrange, :, :] = v_dist
            w[xrange, :, :] = w_dist
            T[xrange, :, :] = T_dist
            S[xrange, :, :] = S_dist
            Pdynamic[xrange, :, :] = P_d_dist
            Pstatic[xrange, :, :] = P_s_dist
            new_range = range(xrange.start + (nx[0] // Nranks), xrange.stop + (nx[0] // Nranks))
            xrange = new_range
        return u, v, w, T, S, Pdynamic, Pstatic
    else:  
        b = np.zeros((nx[0], nx[1], nx[2]))
        for r in range(Nranks):
            u_dist, v_dist, w_dist, b_dist, P_d_dist, P_s_dist = collect_fields(folder, dtn[r], t_save[r], hx, temperature, salinity, with_halos)
            u[xrange, :, :] = u_dist
            v[xrange, :, :] = v_dist
            w[xrange, :, :] = w_dist
            b[xrange, :, :] = b_dist
            Pdynamic[xrange, :, :] = P_d_dist
            Pstatic[xrange, :, :] = P_s_dist
            new_range = range(xrange.start + (nx[0] // Nranks), xrange.stop + (nx[0] // Nranks))
            xrange = new_range
        return u, v, w, b, Pdynamic, Pstatic

