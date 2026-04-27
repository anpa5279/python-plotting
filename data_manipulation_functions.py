import numpy as np

### -------------------------INTERPOLATION------------------------- ###
## interpolation for center of grid
def fcc_ccc(f, x = np.array([]), x_want = np.array([]), periodic = True):
    """ 
    f = 3D array of the variable with location face, center, center
    x = 1D array of the x face values
    x_want = 1D array of x the center values 
    """
    if periodic:
        f = np.concatenate((f, f[0:1, :, :]), axis=0)
    if x.size == 0: #uniform grid 
        return 0.5*(f[:-1, :, :] + f[1:, :, :])
    elif x.size !=0:
        for i in range(np.size(x)-1):
            frac = (f[i+1, :, :]-f[i, :, :])/(x[i+1] - x[i])
            f[i, :, :] = f[i, :, :] + frac*(x_want[i] - x[i])
        return f
def cfc_ccc(f, y = np.array([]), y_want = np.array([]), periodic = True):
    """ 
    f = 3D array of the variable with location center, face, center
    y = 1D array of the y face values
    y_want = 1D array of y the center values 
    """
    if periodic:
        f = np.concatenate((f, f[:, 0:1, :]), axis=1)
    if y.size == 0: #uniform grid 
        return 0.5*(f[:, :-1, :] + f[:, 1:, :])
    elif y.size !=0:
        for i in range(np.size(y)-1):
            frac = (f[:, i+1, :]-f[:, i, :])/(y[i+1] - y[i])
            f[:, i, :] = f[:, i, :] + frac*(y_want[i] - y[i])
        return f
def ccf_ccc(f, z = np.array([]), z_want = np.array([])):
    """ 
    f = 3D array of the variable with location center, center, face
    z = 1D array of the z face values
    z_want = 1D array of z the center values 
    """
    if z.size == 0: #uniform grid 
        return 0.5*(f[:, :, :-1] + f[:, :, 1:])
    elif z.size !=0:
        for i in range(np.size(z)-1):
            frac = (f[:, :, i+1]-f[:, :, i])/(z[i+1] - z[i])
            f[:, :, i] = f[:, :, i] + frac*(z_want[i] - z[i])
        return f
# interpolation to a certain plane slice
def xy_plane_interpolation(f, z, z_desired): # which z location for the xy plane slice?
    """
    f = 3D array of the variable to be interpolated
    z = 1D array of the vertical coordinates corresponding to f
    z_desired = the z location at which we want to find the corresponding xy plane of f
    """
    # getting mld index location 
    dz = np.abs(z - z_desired)
    if dz.min() == 0:
        pos_loc_idx = np.where(dz==0)[0][0]
        f_interp = f[:, :, pos_loc_idx]
    else:
        pos_loc_idx = np.where(dz==dz.min())[0]
        if pos_loc_idx.size > 1:
            pos_loc_idx_1 = pos_loc_idx[1]
            pos_loc_idx = pos_loc_idx[0]
        else:
            pos_loc_idx = pos_loc_idx[0]
            pos_loc_idx_1 = np.where(dz==np.min([dz[pos_loc_idx - 1], dz[pos_loc_idx + 1]]))[0][0]
        min_idx = np.min([pos_loc_idx, pos_loc_idx_1])
        max_idx = np.max([pos_loc_idx, pos_loc_idx_1])
        frac = (f[:, :, max_idx]-f[:, :, min_idx])/(z[max_idx] - z[min_idx])
        f_interp = f[:, :, max_idx] + frac *(z_desired - z[max_idx])
    return f_interp
def z_plane_interpolation(f, hor, hor_desired, planeslice='yz'): # which x or y location for the vertical plane slice?
    """
    f = 3D array of the variable to be interpolated
    hor = 1D array of the x or y coordinates corresponding to f
    hor_desired = the x or y location at which we want to find the corresponding vertical line of f
    planeslice = the plane slice to interpolate ('yz' or 'xz')
    """
    # getting mld index location 
    dhor = np.abs(hor - hor_desired)
    if dhor.min() == 0:
        pos_loc_idx = np.where(dhor==0)[0][0]
        f_interp = f[pos_loc_idx, :, :]
    else:
        pos_loc_idx = np.where(dhor==dhor.min())[0]
        if pos_loc_idx.size > 1:
            pos_loc_idx_1 = pos_loc_idx[1]
            pos_loc_idx = pos_loc_idx[0]
        else:
            pos_loc_idx = pos_loc_idx[0]
            pos_loc_idx_1 = np.where(dhor==np.min([dhor[pos_loc_idx - 1], dhor[pos_loc_idx + 1]]))[0][0]
    
    min_idx = np.min([pos_loc_idx, pos_loc_idx_1])
    max_idx = np.max([pos_loc_idx, pos_loc_idx_1])
    if planeslice == 'yz':
        frac = (f[max_idx, :, :]-f[min_idx, :, :])/(hor[max_idx] - hor[min_idx])
        f_interp = f[max_idx, :, :] + frac *(hor_desired - hor[max_idx])
    elif planeslice == 'xz':
        if dhor.min() == 0:
            pos_loc_idx = np.where(dhor==0)[0][0]
            f_interp = f[:, pos_loc_idx, :]
        else:
            frac = (f[:, max_idx, :]-f[:, min_idx, :])/(hor[max_idx] - hor[min_idx])
            f_interp = f[:, max_idx, :] + frac *(hor_desired - hor[max_idx])
    return f_interp
# intperpolation to a vertical line
def z_line_interpolation(f, x, y, x_desired, y_desired):
    """
    f = 3D array of the variable to be interpolated
    x = 1D array of the x coordinates corresponding to f
    y = 1D array of the y coordinates corresponding to f
    x_desired = the x location at which we want to find the corresponding vertical line of f
    y_desired = the y location at which we want to find the corresponding vertical line
    """
    # getting mld index location 
    dx = np.abs(x - x_desired)
    dy = np.abs(y - y_desired)
    if dx.min() == 0:
        maxx_idx = np.where(dx==0)[0][0]
        fracx = np.zeros(f.shape[1:])
    else:
        pos_loc_idx = np.where(dx==dx.min())[0]
        if pos_loc_idx.size > 1:
            pos_loc_idx_1 = pos_loc_idx[1]
            pos_loc_idx = pos_loc_idx[0]
        else:
            pos_loc_idx = pos_loc_idx[0]
            pos_loc_idx_1 = np.where(dx==np.min([dx[pos_loc_idx - 1], dx[pos_loc_idx + 1]]))[0][0]
        minx_idx = np.min([pos_loc_idx, pos_loc_idx_1])
        maxx_idx = np.max([pos_loc_idx, pos_loc_idx_1])
        fracx = (f[maxx_idx, :, :]-f[minx_idx, :, :])/(x[maxx_idx] - x[minx_idx])
    if dy.min() == 0:
        maxy_idx = np.where(dy==0)[0][0]
        f_interp = f[maxx_idx, maxy_idx, :] + fracx[maxy_idx, :] * (x_desired - x[maxx_idx])
    else:
        pos_loc_idx = np.where(dy==dy.min())[0]
        if pos_loc_idx.size > 1:
            pos_loc_idx_1 = pos_loc_idx[1]
            pos_loc_idx = pos_loc_idx[0]
        else:
            pos_loc_idx = pos_loc_idx[0]
            pos_loc_idx_1 = np.where(dy==np.min([dy[pos_loc_idx - 1], dy[pos_loc_idx + 1]]))[0][0]
        miny_idx = np.min([pos_loc_idx, pos_loc_idx_1])
        maxy_idx = np.max([pos_loc_idx, pos_loc_idx_1])
        fracy = (f[maxx_idx, maxy_idx, :]-f[miny_idx, maxx_idx, :])/(y[maxy_idx] - y[miny_idx])
        f_interp = f[maxx_idx, maxy_idx, :] + fracy *(y_desired - y[maxy_idx]) + fracx[maxy_idx, :] *(x_desired - x[maxx_idx])
    return f_interp
# find desired location based on wanted value
def vert_point_interpolation(f, f_desired, z):
    """
    f = 1D array of the variable to be interpolated
    f_desired = the value of f at which we want to find the corresponding z location
    z = 1D array of the vertical coordinates corresponding to f
    """
    # getting mld index location 
    df = np.abs(f - f_desired)
    if df.min() == 0:
        pos_loc_idx = np.where(df==0)[0][0]
        z_interp = z[pos_loc_idx]
    else:
        pos_loc_idx = np.where(df==df.min())[0]
        if pos_loc_idx.size > 1:
            pos_loc_idx_1 = pos_loc_idx[1]
            pos_loc_idx = pos_loc_idx[0]
        else:
            pos_loc_idx = pos_loc_idx[0]
            pos_loc_idx_1 = np.where(df==np.min([df[pos_loc_idx [0] - 1], df[pos_loc_idx [0]+ 1]]))[0][0]
        min_idx = np.min([pos_loc_idx, pos_loc_idx_1])
        max_idx = np.max([pos_loc_idx, pos_loc_idx_1])
        frac = (z[max_idx] - z[min_idx])/(f[max_idx, :, :]-f[min_idx, :, :])
        z_interp = z[max_idx] + frac *(f_desired - f[max_idx, :, :])
    return z_interp