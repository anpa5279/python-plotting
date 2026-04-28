import numpy as np
# ------------------------- TO CENTER OF GRID INTERPOLATION ------------------------- #
def face_to_center(f, axis, periodic=True):
    if periodic:
        pad = [(0, 0)] * f.ndim
        pad[axis] = (0, 1)
        f = np.concatenate([f, np.take(f, [0], axis=axis)], axis=axis)

    sl0 = [slice(None)] * f.ndim
    sl1 = [slice(None)] * f.ndim

    sl0[axis] = slice(0, -1)
    sl1[axis] = slice(1, None)

    return 0.5 * (f[tuple(sl0)] + f[tuple(sl1)])

def velocities_to_center(u, v, w):
    return (
        face_to_center(u, axis=2),
        face_to_center(v, axis=1),
        face_to_center(w, axis=0),
    )

# ------------------------- GENERAL INTERPOLATION ------------------------- #
def interp1d_axis(f, coord, coord_new, axis=-1):
    """
    Linear interpolation along a given axis.

    Parameters
    ----------
    f : array-like (NumPy or Dask)
        Data array
    coord : 1D array
        Coordinates along interpolation axis
    coord_new : float
        Desired coordinate
    axis : int
        Axis along which to interpolate

    Returns
    -------
    Interpolated array with that axis removed
    """

    coord = np.asarray(coord)

    # find index below target
    idx = np.searchsorted(coord, coord_new) - 1
    idx = np.clip(idx, 0, len(coord) - 2)

    c0 = coord[idx]
    c1 = coord[idx + 1]

    # slice helpers
    sl0 = [slice(None)] * f.ndim
    sl1 = [slice(None)] * f.ndim

    sl0[axis] = idx
    sl1[axis] = idx + 1

    f0 = f[tuple(sl0)]
    f1 = f[tuple(sl1)]

    w = (coord_new - c0) / (c1 - c0)

    return (1 - w) * f0 + w * f1
# ------------------------- PLANE SLICES ------------------------- #
def xy_plane(f, z, z0):
    return interp1d_axis(f, z, z0, axis=2)

def yz_plane(f, x, x0):
    return interp1d_axis(f, x, x0, axis=0)

def xz_plane(f, y, y0):
    return interp1d_axis(f, y, y0, axis=1)

def vertical_line(f, x, y, x0, y0):
    # interpolate in x
    fx = interp1d_axis(f, x, x0, axis=0)

    # interpolate in y
    fxy = interp1d_axis(fx, y, y0, axis=0)

    return fxy  # shape: (Nz,)

def horizontal_line(f, hor, z, hor0, z0, axis='y'):
    if axis == 'y':
        fh = interp1d_axis(f, hor, hor0, axis=1)
    else:
        fh = interp1d_axis(f, hor, hor0, axis=0)

    return interp1d_axis(fh, z, z0, axis=-1)

# ------------------------- GRID POINT ------------------------- #
def point(f, x, y, z, x0, y0, z0):
    fx = interp1d_axis(f, x, x0, axis=0)
    fxy = interp1d_axis(fx, y, y0, axis=0)
    fxyz = interp1d_axis(fxy, z, z0, axis=0)
    return fxyz

