import numpy as np
from interpolation import horizontal_line, velocities_to_center
# ------------------------- TURBULENT STATISTICS ------------------------- #
# reynolds stress function (fluctuations of two variables)
def reynolds_stress(a, b, a_avg, b_avg):
    """
    Computes the mean fluctuations in x and y directions.

    Parameters:
        a: ndarray of shape (nx, ny, nz)
        b: ndarray of shape (nx, ny, nz)
        a_avg: 1D array of shape (nz,) or (1, 1, nz)
        b_avg: 1D array of shape (nz,) or (1, 1, nz)
    Returns:
        ab_fluct_avg: 1D array of shape (nz,)
    """
    # Compute squared fluctuation a'*b' = (a-a_avg)(b-b_avg)
    ab_fluc = (a - a_avg)*(b - b_avg)

    return ab_fluc
# Root mean square error function)
def rms(var, reader = None, t = None):
    if reader is not None:
        if t is not None:
            a = np.array(reader.lazy_field(var, steps = t))
        else:
            a = np.array(reader.lazy_field(var))
        avg = np.mean(a, axis=(-3, -2))
    else:
        if var.ndim == 4: # all time stesp
            a = np.array(var)
            avg = np.mean(a, axis=(-3, -2))
            avg = avg[:, np.newaxis, np.newaxis, :]
        else:
            a = var
            avg = np.mean(a, axis=(-3, -2))
    return np.mean((a-avg)**2, axis=(-3, -2))**0.5
# fluctuations of one variable 
def a_fluc_b(a, b, a_avg=None):
    if a_avg is None:
        a_avg = np.mean(a, axis=(-3, -2))
    return (a - a_avg) * b

# ------------------------- BUOYANCY ANALYSIS ------------------------- #
# calculate buoyancy 
def buoyancy(reader, T0 = 25):
    g = 9.80665
    reader.load_equation_of_state()
    alpha = reader.alpha
    if not reader.salinity:
        T = reader.lazy_field('T')
        b = g * alpha * (T - T0)
    else:
        S = reader.lazy_field('S')
        T = reader.lazy_field('T')
        beta = reader.beta
        b = np.squeeze(g * alpha * (T - T0) - g * beta * S)
    b_avg = np.mean(b, axis=(-3, -2))
    return {'b':b, 'b_avg':b_avg}
# calculate buoyancy flux
def buoyancy_flux_avg_1d(reader, T0=25):
    """
    Computes <b'u'>, <b'v'>, <b'w'> vs depth (Nz, Nt) 
    without ever loading the full 4D field.
    Processes one timestep and one rank at a time.
    """
    g = 9.80665
    reader.load_equation_of_state()
    alpha = reader.alpha
    beta = reader.beta if reader.salinity else None
    nx_local = reader.nx[0] // reader.Nranks

    bu_avg = np.zeros((reader.nx[2], reader.nt))
    bv_avg = np.zeros((reader.nx[2], reader.nt))
    bw_avg = np.zeros((reader.nx[2], reader.nt))

    for i, t in enumerate(reader.t_save):
        # --- accumulate xy-mean of each field across ranks ---
        T_sum  = np.zeros(reader.nx[2])
        S_sum  = np.zeros(reader.nx[2]) if reader.salinity else None
        u_sum  = np.zeros(reader.nx[2])
        v_sum  = np.zeros(reader.nx[2])
        w_sum  = np.zeros(reader.nx[2])

        n_pts = reader.nx[0] * reader.nx[1]  # total xy points

        for file in reader.files:
            fname = os.path.join(reader.folder, file)
            with h5py.File(fname, 'r') as f:
                def load(name):
                    d = f[f'timeseries/{name}/{int(t)}']
                    if reader.halos:
                        d = d[reader.hx[2]:-reader.hx[2] or None,
                              reader.hx[1]:-reader.hx[1] or None,
                              reader.hx[0]:-reader.hx[0] or None]
                    return np.asarray(d)  # (z, y, x_local)

                T_sum += load('T').mean(axis=(1, 2)) * (nx_local * reader.nx[1])
                u_sum += load('u').mean(axis=(1, 2)) * (nx_local * reader.nx[1])
                v_sum += load('v').mean(axis=(1, 2)) * (nx_local * reader.nx[1])
                w_sum += load('w').mean(axis=(1, 2)) * (nx_local * reader.nx[1])
                if reader.salinity:
                    S_sum += load('S').mean(axis=(1, 2)) * (nx_local * reader.nx[1])

        # xy-means, shape (Nz,)
        T_avg = T_sum / n_pts
        u_avg = u_sum / n_pts
        v_avg = v_sum / n_pts
        w_avg = w_sum / n_pts
        b_avg = g * alpha * (T_avg - T0)
        if reader.salinity:
            S_avg = S_sum / n_pts
            b_avg -= g * beta * S_avg

        # --- second pass: accumulate <b'u'>, <b'v'>, <b'w'> ---
        bu_sum = np.zeros(reader.nx[2])
        bv_sum = np.zeros(reader.nx[2])
        bw_sum = np.zeros(reader.nx[2])
        count  = 0

        for file in reader.files:
            fname = os.path.join(reader.folder, file)
            with h5py.File(fname, 'r') as f:
                def load(name):
                    d = f[f'timeseries/{name}/{int(t)}']
                    if reader.halos:
                        d = d[reader.hx[2]:-reader.hx[2] or None,
                              reader.hx[1]:-reader.hx[1] or None,
                              reader.hx[0]:-reader.hx[0] or None]
                    return np.asarray(d)  # (z, y, x_local)

                T = load('T')   # (z, y, x_local)
                u = load('u')
                v = load('v')
                w = load('w')
                if reader.salinity:
                    S = load('S')

                # buoyancy fluctuation (z, y, x_local)
                b = g * alpha * (T - T0)
                if reader.salinity:
                    b -= g * beta * S
                b_fluc = b - b_avg[:, None, None]   # broadcast (Nz,) → (z,y,x)

                u_fluc = u - u_avg[:, None, None]
                v_fluc = v - v_avg[:, None, None]
                w_fluc = w - w_avg[:, None, None]

                nx_loc = T.shape[2]
                ny_loc = T.shape[1]

                bu_sum += (b_fluc * u_fluc).sum(axis=(1, 2))
                bv_sum += (b_fluc * v_fluc).sum(axis=(1, 2))
                bw_sum += (b_fluc * w_fluc).sum(axis=(1, 2))
                count  += nx_loc * ny_loc

        bu_avg[:, i] = bu_sum / count
        bv_avg[:, i] = bv_sum / count
        bw_avg[:, i] = bw_sum / count

    return bu_avg, bv_avg, bw_avg   # (Nz, Nt)
def buoyancy_flux_line(reader, z0, x0 = None, y0 = None):
    """
    Returns b'u', b'v', b'w' along a horizontal line at fixed (y0, z0)
    as a function of x, for each timestep — shape (Nx, Nt).
    Never loads full 4D field.
    """
    if x0 is not None:
        h0 = x0 
        h = reader.x
        dir = 'x'
    elif y0 is not None:
        h0 = y0
        h = reader.y
        dir = 'y'

    reader.load_equation_of_state()
    b = reader.load_buoyancy()

    bu_out = np.zeros((reader.nx[0], reader.nt))
    bv_out = np.zeros((reader.nx[0], reader.nt))
    bw_out = np.zeros((reader.nx[0], reader.nt))

    for it, t in enumerate(reader.t_save):
        u = reader.lazy_field['u']
        v = reader.lazy_field['v']
        w = reader.lazy_field['w']

        u = velocities_to_center(u, axis=0)
        v = velocities_to_center(v, axis=1)
        w = velocities_to_center(w, axis=2)

        # extract horizontal line at (y0, z0) for each flux
        bu_out[:, it] = horizontal_line(a_fluc_b(b, u), h, reader.z, h0, z0, axis=dir)
        bv_out[:, it] = horizontal_line(a_fluc_b(b, v), h, reader.z, h0, z0, axis=dir)
        bw_out[:, it] = horizontal_line(a_fluc_b(b, w), h, reader.z, h0, z0, axis=dir)

    return bu_out, bv_out, bw_out   # (Nx, Nt)
# Richardson number
def richardson_number(dbdz, z, u, v):
    du_dz = np.gradient(u, z, axis=-1)
    dv_dz = np.gradient(v, z, axis=-1)
    shear_squared = du_dz**2 + dv_dz**2
    ri = dbdz/shear_squared
    return ri
# Atwood number 
def atwood_number(rho_tracer, rho_background):
    return (rho_tracer - rho_background)/(rho_tracer + rho_background)
# Ozmidov length scale, length in which buoyancy is negigible
def ozmidov_length(epsilon, dbdz):
    return (epsilon/(np.abs(dbdz)**(3/2)))**0.5
# lamb vectors
def lamb_vector(u, v, w, x, y, z):
    omega_x = np.gradient(w, y, axis=-2) - np.gradient(v, z, axis=-1)
    omega_y = np.gradient(u, z, axis=-1) - np.gradient(w, x, axis=-3)
    omega_z = np.gradient(v, x, axis=-3) - np.gradient(u, y, axis=-2)
    lamb_x = v * omega_z - w * omega_y
    lamb_y = w * omega_x - u * omega_z
    lamb_z = u * omega_y - v * omega_x
    lamb_x_avg = np.mean(lamb_x, axis=(-3, -2))
    lamb_y_avg = np.mean(lamb_y, axis=(-3, -2))
    lamb_z_avg = np.mean(lamb_z, axis=(-3, -2))
    return lamb_x_avg, lamb_y_avg, lamb_z_avg
### -------------------------IMPORTANT DEPTHS------------------------- ###
# mixed layer depth information
def mld_info(w, bw_fluc, rho_perturbed, z, mld): # inputs are 1d arrays
    # info at mixed layer depth
    dz_ml = np.abs(z + mld)/mld
    mld_idx = np.where(dz_ml==dz_ml.min())[0][-1]
    mld_w = w[mld_idx]
    mld_bw_fluc = bw_fluc[mld_idx]
    mld_rho_perturbed = rho_perturbed[mld_idx]
    return mld_idx, mld_w, mld_bw_fluc, mld_rho_perturbed
