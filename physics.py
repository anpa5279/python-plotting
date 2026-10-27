import numpy as np
from interpolation import horizontal_line, velocities_to_center
# ------------------------- TURBULENT STATISTICS ------------------------- #
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

### -------------------------IMPORTANT DEPTHS------------------------- ###
