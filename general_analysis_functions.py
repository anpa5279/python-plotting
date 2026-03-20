import numpy as np
from scipy.ndimage import center_of_mass, map_coordinates
### -------------------------indexing functions------------------------- ###
def point_linear_interp(a1, a2, b, b1, b2):
    # linear interpolation to find value of a at b, given a1, a2, b1, b2
    return a1 + (a2 - a1) * (b - b1) / (b2 - b1)
### -------------------------TURBULENT STAT FUNCTIONS------------------------- ###
# strain rate 
def strain_rates(u, v, w, dx):
    e12 = 0.5*(np.gradient(u, dx[1], axis=-2) + np.gradient(v, dx[0], axis=-3))
    e13 = 0.5*(np.gradient(u, dx[2], axis=-1) + np.gradient(w, dx[0], axis=-3))
    e23 = 0.5*(np.gradient(v, dx[2], axis=-1) + np.gradient(w, dx[1], axis=-2))
    return e12, e13, e23
# vertical average fluctuations
def ab_fluc_mean(a, b, a_avg, b_avg):
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
    # Mean over x and y, result shape: (nz,)
    ab_fluct_avg = np.mean(ab_fluct, axis=(-3, -2))

    return ab_fluct, ab_fluct_avg
def a2_fluc_mean(a_fluc):
    """
    Computes the mean fluctuations in x and y directions.

    Parameters:
        a: ndarray of shape (nx, ny, nz)
        a_avg: 1D array of shape (nz,) or (1, 1, nz)
    Returns:
        ab_fluct_avg: 1D array of shape (nz,)
    """
    a_fluc_avg = np.mean(a_fluc, axis=(-3, -2))
    # Compute squared fluctuation a'*b' = (a-a_avg)^2
    a2_fluc = (a_fluc)**2
    # Mean over x and y, result shape: (nz,)
    a2_fluct_avg = np.mean(a2_fluc, axis=(-3, -2))

    return a_fluc_avg, a2_fluc, a2_fluct_avg
def visc_dissipation_rate(visc, u, v, w, dx):
    e12, e13, e23 = strain_rates(u, v, w, dx)
    epsilon_visc = 2*visc*(e12**2 + e13**2 + e23**2)
    return epsilon_visc # m^2/s^3
### -------------------------BUOYANCY FUNCTIONS------------------------- ###
# Richardson number
def richardson_number(dbdz, z, u, v):
    du_dz = np.gradient(u, z, axis=-1)
    dv_dz = np.gradient(v, z, axis=-1)
    shear_squared = du_dz**2 + dv_dz**2
    ri = dbdz/shear_squared
    return ri
def richardson_number_ratio(b, w, lx): # vol_flux^2 * b_flux / momentum_flux^(5/2)
    A = lx[0]*lx[1] # area of horizontal plane
    ri = -b*A**(1/2)/(w**2)
    return ri
# Froude number
def froude_number(u, v, w, dbdz, z):
    N = np.sqrt(np.abs(dbdz))
    velocity_magnitude = np.sqrt(u**2 + v**2 + w**2)
    Fr = velocity_magnitude/N
    return Fr
# Reynolds buoyancy number
def reynolds_buoyancy_number(epsilon, visc, dbdz):
    return epsilon/(visc*dbdz)
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