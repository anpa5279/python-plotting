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