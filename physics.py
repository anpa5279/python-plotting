import numpy as np
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
def rms(a):
    avg = np.mean(a, axis=(-3, -2))
    return np.mean((a-avg)**2, axis=(-3, -2))**0.5
# fluctuations of one variable 
def a_fluc_b(a, b):
    a_avg = np.mean(a, axis=(-3, -2))
    return (a - a_avg)*b

# ------------------------- BUOYANCY ANALYSIS ------------------------- #
# calculate buoyancy 
def buoyancy(reader, T, S = [], rho0 = 1026, T0 = 25):
    g = 9.80665
    reader.load_equation_of_state()
    alpha = reader.alpha
    if not reader.salinity:
        b = g * alpha * (T - T0)
        rho = rho0 * alpha * (T - T0)
        bs = {'b_total':b, 'rho':rho}
    else:
        beta = reader.beta
        rho = rho0 - rho0 * alpha * (T - T0) + rho0 * beta * S
        bs ={'b_total':g * alpha * (T - T0) - g * beta * S,
            'b_T':g * alpha * (T - T0),
            'b_C':-g * beta * S,
            'rho':rho}
    return bs

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
