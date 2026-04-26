import numpy as np

### -------------------------INTERPOLATION------------------------- ###
## interpolation for center of grid
def fcc_ccc(f, x = np.array([]), x_want = np.array([]), periodic = True):
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
    if z.size == 0: #uniform grid 
        return 0.5*(f[:, :, :-1] + f[:, :, 1:])
    elif z.size !=0:
        for i in range(np.size(z)-1):
            frac = (f[:, :, i+1]-f[:, :, i])/(z[i+1] - z[i])
            f[:, :, i] = f[:, :, i] + frac*(z_want[i] - z[i])
        return f
# interpolation to a certain plane slice
def xy_plane_interpolation(f, z, z_desired):
    # getting mld index location 
    dz = np.abs(z + z_desired)/z_desired
    if dz.min() == 0:
        pos_loc_idx = np.where(dz==0)[0][0]
        f_interp = f[:, :, pos_loc_idx]
    else:
        pos_loc_idx = np.where(dz==dz.min())[0][0]
        pos_loc_idx_1 = np.where(dz==np.min([dz[pos_loc_idx -1], dz[pos_loc_idx + 1]]))[0][0]
        min_idx = np.min([pos_loc_idx, pos_loc_idx_1])
        max_idx = np.max([pos_loc_idx, pos_loc_idx_1])
        frac = (f[:, :, max_idx]-f[:, :, min_idx])/(z[max_idx] - z[min_idx])
        f_interp = f[:, :, max_idx] + frac *(z_desired - z[max_idx])
    return f_interp
# intperpolation to a vertical line
def vertical_line_interpolation(f, x, y, x_desired, y_desired):
    # getting mld index location 
    dx = np.abs(x - x_desired)/x_desired
    dy = np.abs(y - y_desired)/y_desired
    if dx.min() == 0:
        pos_loc_idx_x = np.where(dx==0)[0][0]
    else:
        pos_loc_idx_x = np.where(dx==dx.min())[0][0]
    if dy.min() == 0:
        pos_loc_idx_y = np.where(dy==0)[0][0]
    else:
        pos_loc_idx_y = np.where(dy==dy.min())[0][0]
    f_interp = f[pos_loc_idx_x, pos_loc_idx_y, :]
    return f_interp