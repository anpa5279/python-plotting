import numpy as np
# ------------------------- GENERAL BUOYANCY ------------------------- #
def buoyancy(T, rho0, alpha, T0, g, tracer = []):
    if tracer is None:
        b = -g * alpha * (T - T0)
        rho = rho0 * (1 - alpha * (T - T0))
    else:
        beta = tracer['beta']
        C = tracer['C']
        C0 = tracer['C0']
        rho = rho0 - rho0 * alpha * (T - T0) + rho0 * beta * (C - C0)
        bs ={'b_total':-g * alpha * (T - T0) + g * beta * (C - C0),
             'b_T':-g * alpha * (T - T0),
             'b_C':g * beta * (C - C0)}
    return bs, rho

# ------------------------- TURBULENT STATISTICS ------------------------- #
# vertical average fluctuations
def ab_fluc(a, b, a_avg, b_avg):
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
    ab_fluct = (a - a_avg)*(b - b_avg)

    return ab_fluct

# ------------------------- BUOYANCY ANALYSIS ------------------------- #
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