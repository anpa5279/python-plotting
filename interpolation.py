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
        face_to_center(u, axis=0),
        face_to_center(v, axis=1),
        face_to_center(w, axis=2, periodic=False),
    )

# ------------------------- GENERAL INTERPOLATION ------------------------- #
def interp1d_axis(f, coord, f_new = None, coord_new = None, axis=-1):
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
    if f_new is not None:
        coord = np.asarray(coord)

        # find index below target
        if f_new == 0:
            idx = np.where(np.diff(np.sign(f))!=0)[0][0]
        else:
            idx = np.searchsorted(f, f_new) - 1
            idx = np.clip(idx, 0, len(f) - 2)

        f0 = f[idx]
        f1 = f[idx + 1]

        # slice helpers
        sl0 = [slice(None)] * coord.ndim
        sl1 = [slice(None)] * coord.ndim

        sl0[axis] = idx
        sl1[axis] = idx + 1

        c0 = coord[tuple(sl0)]
        c1 = coord[tuple(sl1)]

        w = (f_new - f0) / (f1 - f0)

        return (1 - w) * c0 + w * c1
    if coord_new is not None:
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
    return interp1d_axis(f, z, coord_new = z0, axis=2)

def yz_plane(f, x, x0):
    return interp1d_axis(f, x, coord_new = x0, axis=0)

def xz_plane(f, y, y0):
    return interp1d_axis(f, y, coord_new = y0, axis=1)

def vertical_line(f, x, y, x0, y0):
    # interpolate in x
    fx = interp1d_axis(f, x, coord_new = x0, axis=0)

    # interpolate in y
    fxy = interp1d_axis(fx, y, coord_new = y0, axis=0)

    return fxy  # shape: (Nz,)

def horizontal_line(f, hor, z, hor0, z0, axis='y'):
    if axis == 'y':
        fh = interp1d_axis(f, hor, coord_new = hor0, axis=1)
    else:
        fh = interp1d_axis(f, hor, coord_new = hor0, axis=0)

    return interp1d_axis(fh, z, coord_new = z0, axis=-1)

# ------------------------- GRID POINT ------------------------- #
def point(f, z, f0 = None, z0 = None, x = None, x0 = 0.0, y = None, y0 = 0.0):
    if f0 is not None:
        znew = interp1d_axis(f, z, f_new = f0, axis=-1) 
        new = znew
    if z0 is not None:
        fnew = interp1d_axis(f, z, coord_new = z0, axis=-1)
        new = fnew
    if x is not None and y is not None:
        fnewyz = interp1d_axis(fnew, y, coord_new = y0, axis=-1)
        new = interp1d_axis(fnewyz, x, coord_new = x0, axis=-1)
    else:
        return new


