import os
import numpy as np
import h5py

### -------------------------EXTRACTING DATA------------------------- ###
## collecting model information
def collect_time_outputs(file, stokes=False, closure=True):
    with h5py.File(file, 'r') as f:
        timeseries_group = [g for g in f.keys() if 'timeseries' in g][0]
        t_group = f[timeseries_group + '/t']
        t_save = sorted([float(k) for k in t_group.keys()])
        t_save = np.array(t_save)
        time = np.array([t_group[str(int(k))][()] for k in t_save])
        if closure:
            visc = f['closure/ν'][()]
            diff= f['closure/κ']
        else:
            visc = 0.0
            diff = 0.0
    if stokes:
        u_f =np.array(f["IC/"]["friction_velocity"])
        u_s = np.array(f["IC/"]["stokes_velocity"])
    else:
        u_f = None
        u_s = None
    return time, t_save, visc, diff, u_f, u_s
# collecting grid information
def collect_grid(folder, file, Nranks):
    with h5py.File(os.path.join(folder, file), 'r') as f:
        nx = [f['grid/Nx'][()] * Nranks, f['grid/Ny'][()], f['grid/Nz'][()]]
        hx = [f['grid/Hx'][()], f['grid/Hy'][()], f['grid/Hz'][()]]
        lx = [f['grid/Lx'][()] * Nranks, f['grid/Ly'][()], f['grid/Lz'][()]]
        dx = [f['grid/Δxᶜᵃᵃ'][()], f['grid/Δyᵃᶜᵃ'][()], f['grid/z/Δᵃᵃᶜ'][()]]
        x = f['grid/xᶜᵃᵃ'][hx[0]] + np.arange(nx[0]) * dx[0]
        y = f['grid/yᵃᶜᵃ'][hx[1]:-hx[1]]
        z = f['grid/z/cᵃᵃᶜ'][hx[2]:-hx[2]]
        zf = f['grid/z/cᵃᵃᶠ'][hx[2]:-hx[2]]
    return nx, hx, lx, x, y, z, zf
# collecting thermal expansion and haline contraction coefficients
def collect_temp_and_sal(file, salinity=False):
    with h5py.File(file, 'r') as f:
        alpha = f['buoyancy/formulation/equation_of_state/thermal_expansion'][()]
        if salinity:
            beta = f['buoyancy/formulation/equation_of_state/haline_contraction'][()]
            return alpha, beta
    return alpha
## -------------------------EXTRACTING 3D FIELDS------------------------- ###
## collecting 3D fields
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
            #print(dtn[r])
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
## -------------------------Writing grid info------------------------- ###
def writing_grid(folder, dtn, nx, lx, hx, rank=None):
    if rank is None:
        dtn_new = 'fields_with_grid.jld2'
    else:
        dtn_new = f'fields_with_grid_rank{rank}.jld2'
    dx = lx / nx
    d_min = -hx*dx
    d_max = hx*dx + lx
    xf = np.arange(d_min[0]-lx[0]/2, d_max[0]-lx[0]/2, dx[0])
    yf = np.arange(d_min[1]-lx[1]/2, d_max[1]-lx[1]/2, dx[1])
    zf = np.arange(-lx[2]-dx[2]*hx[2], dx[2]*(hx[2]+1), dx[2])
    x = xf + 0.5*dx[0]
    y = yf + 0.5*dx[1]
    z = zf[:-1] + 0.5*dx[2]
    if rank is not None:
        xf = xf[rank*nx[0]:(rank+1)*nx[0]]
    new_file_directory = os.path.join(folder, dtn_new)
    with h5py.File(os.path.join(folder, dtn), 'r') as src, h5py.File(new_file_directory, 'w') as dst:
        for key in src.keys():
            if key == 'grid':
                continue
            try:
                src.copy(key, dst)
            except Exception as e:
                print(f"Skipping {key}: {e}")

    file = h5py.File(new_file_directory, 'a')
    file.create_group('grid')
    file.create_dataset('grid/Nx', data=nx[0])
    file.create_dataset('grid/Ny', data=nx[1])
    file.create_dataset('grid/Nz', data=nx[2])

    file.create_dataset('grid/Lx', data=lx[0])
    file.create_dataset('grid/Ly', data=lx[1])
    file.create_dataset('grid/Lz', data=lx[2])

    file.create_dataset('grid/Hx', data=hx[0])
    file.create_dataset('grid/Hy', data=hx[1])
    file.create_dataset('grid/Hz', data=hx[2])

    file.create_dataset('grid/Δxᶜᵃᵃ', data=dx[0])
    file.create_dataset('grid/Δyᵃᶜᵃ', data=dx[1])
    file.create_dataset('grid/z/Δᵃᵃᶜ', data=dx[2])

    file.create_dataset('grid/xᶜᵃᵃ', data=x)
    file.create_dataset('grid/yᵃᶜᵃ', data=y)
    file.create_dataset('grid/z/cᵃᵃᶜ', data=z)
    file.create_dataset('grid/z/cᵃᵃᶠ', data=zf)
    file.close()

    os.remove(os.path.join(folder, dtn))

    return x, y, z, zf
## -------------------------Collecting temporal averages------------------------- ###
def collect_temporal_averages(folder, dtn, temperature=True, salinity=False):
    rms_list = {}
    b_and_w_list = {}
    # Load data from files
    fname = os.path.join(folder, dtn)
    with h5py.File(fname, 'r') as f:
        b_and_w_list['w_avg'] = f['1D temporal averages/w'][()] 
        b_and_w_list['b_avg'] = f['1D temporal averages/b'][()] 
        b_and_w_list['w_fluc_avg'] = f['1D temporal averages/w\''][()] 
        b_and_w_list['b_fluc_avg'] = f['1D temporal averages/b\''][()] 
        rms_list['u_rms'] = f['1D temporal averages/urms'][()] 
        rms_list['v_rms'] = f['1D temporal averages/vrms'][()] 
        rms_list['w_rms'] = f['1D temporal averages/wrms'][()] 
        b_and_w_list['b_centerline_avg'] = f['centerline temporal averages/b'][()] 
        b_and_w_list['b_fluc_centerline_avg'] = f['centerline temporal averages/b\''][()] 
        b_and_w_list['bw_fluc_avg'] = f['1D temporal averages/b\'w\''][()]
        if temperature and salinity:
            T_list = {}
            S_list = {}
            T_list['T_avg'] = f['1D temporal averages/T'][()] 
            S_list['S_avg'] = f['1D temporal averages/S'][()] 
            T_list['T_fluc_avg'] = f['1D temporal averages/T\''][()] 
            S_list['S_fluc_avg'] = f['1D temporal averages/S\''][()] 
            T_list['Tw_fluc_avg'] = f['1D temporal averages/T\'w\''][()] 
            S_list['Sw_fluc_avg'] = f['1D temporal averages/S\'w\''][()] 
            T_list['T_centerline_avg'] = f['centerline temporal averages/T'][()] 
            S_list['S_centerline_avg'] = f['centerline temporal averages/S'][()] 
            T_list['T_fluc_centerline_avg'] = f['centerline temporal averages/T\''][()] 
            S_list['S_fluc_centerline_avg'] = f['centerline temporal averages/S\''][()] 
        elif temperature and not salinity:
            T_list = {}
            T_list['T_avg'] = f['1D temporal averages/T'][()] 
            T_list['T_fluc_avg'] = f['1D temporal averages/T\''][()] 
            T_list['Tw_fluc_avg'] = f['1D temporal averages/T\'w\''][()] 
            T_list['T_centerline_avg'] = f['centerline temporal averages/T'][()] 
            T_list['T_fluc_centerline_avg'] = f['centerline temporal averages/T\''][()] 
            S_list = None
        else:
            T_list = {}
            S_list = {}
    return rms_list, b_and_w_list, T_list, S_list

def collect_contour_val(folder, dtn):
    # Load data from files
    fname = os.path.join(folder, dtn)
    with h5py.File(fname, 'r') as f:
        S = f['contour temporal averages/S'][()] 
        w = f['contour temporal averages/w'][()] 
    return S, w

def collect_plume_stats(folder, dtn, percent_contour):
    # Load data from files
    fname = os.path.join(folder, dtn)
    with h5py.File(fname, 'r') as f:
        dset_path = f"plume statistics/contour {percent_contour}/plume tracer radius with depth"
        rp_profile = f[dset_path][()]

    return rp_profile