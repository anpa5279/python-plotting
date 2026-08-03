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
def buoyancy(reader, type = 'field'):
    """
    calculates buoyancy relative to type
    type = 'plane' --> from reader.load_plane_var, all time steps
         = 'field' --> from reader.lazy_field, one time step at a time
         = 'bi' --> from reader.load_binning_var, all time steps
    """
    if type == 'plane':
        T = reader.load_plane_var('T')
        if reader.salinity:
            S = reader.load_plane_var('S')
    elif type == 'field':
        T = reader.lazy_field('T')
        if reader.salinity:
            S = reader.lazy_field('S')
    elif type == 'bin':
        T = reader.load_binning_var('T')
        if reader.salinity:
            S = reader.load_binning_var('S')
    g = 9.80665
    
    alpha = reader.alpha
    if not reader.salinity:
        b = g * alpha * (T - reader.T0)
    else:
        beta = reader.beta
        b = np.squeeze(g * alpha * (T - reader.T0) - g * beta * S)
    return b
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
