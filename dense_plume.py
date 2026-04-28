import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy.spatial import ConvexHull

from interpolation import interp1d_axis
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

### -------------------------TRACER CALCULATIONS------------------------- ###
def plume_tracer_radius(x, y, nx, tracer, tracer_contour):
    plume_contour = tracer >= tracer_contour
    xi, yi, zi = np.where(plume_contour)
    plume_index = (xi, yi, zi)

    centerx = np.mean(x)
    centery = np.mean(y)

    r = np.sqrt((x[xi] - centerx)**2 + (y[yi] - centery)**2)
    counts = np.bincount(zi, minlength=nx[2])
    sums   = np.bincount(zi, weights=r, minlength=nx[2])

    rp_profile = np.zeros(nx[2])
    mask = counts > 0
    rp_profile[mask] = sums[mask] / counts[mask]
    return rp_profile, plume_index

### -------------------------MOMENTUM ANALYSIS------------------------- ###
def plume_momentum_analysis(nx, w, b, b_fluc, rho_fluc, X, Y, w_mag_tol):
    # checking magnitude of values to help define bounds
    w_mag = np.abs(w)
    w_mag_order = np.floor(np.log10(w_mag))
    w_mag_cl = w_mag_order[centerline_index[0, :], centerline_index[1, :], centerline_index[2, :]]

    if np.any(w_mag_cl == w_mag_tol):
        # index of plume points of interest
        rho_cl = rho_fluc[centerline_index[0, :], centerline_index[1, :], centerline_index[2, :]] 
        rho_cl_sign = np.sign(rho_cl)
        rho_cl_sign_change = np.diff(rho_cl_sign)
        w_cl = w[centerline_index[0, :], centerline_index[1, :], centerline_index[2, :]]
        idx_max =np.where(np.diff(np.sign(w_cl)) < 0)[0]
        if np.size(idx_max) == 0: # early stages of plume development
            idx_max = nx[2]-1
            idx_neutral = idx_max
            Q = np.zeros(nx[2])
            M = np.zeros(nx[2])
            F = np.zeros(nx[2])
            B = np.zeros(nx[2])
            wm = np.zeros(nx[2])
            dm = np.zeros(nx[2])
            bm = np.zeros(nx[2])
            Ri = np.zeros(nx[2])
            area_idx = np.zeros_like(rho_fluc).astype(bool)
            return Q, M, F, B, wm, dm, bm, Ri, area_idx, idx_max, idx_neutral
        else:
            idx_max =idx_max[-1] +1 
            idx_rho_max = np.where(rho_cl_sign_change < 0)[0]
            idx_diff = np.abs(idx_rho_max - idx_max)
            if np.size(idx_rho_max) == 0:
                idx_max = idx_max
            else:
                idx_max_2 = idx_rho_max[idx_diff.argmin()] + 1 
                idx_max = np.max([idx_max, idx_max_2])
    else: # early stages of plume development
        idx_max = nx[2]-1
        idx_neutral = idx_max
        Q = np.zeros(nx[2])
        M = np.zeros(nx[2])
        F = np.zeros(nx[2])
        B = np.zeros(nx[2])
        wm = np.zeros(nx[2])
        dm = np.zeros(nx[2])
        bm = np.zeros(nx[2])
        Ri = np.zeros(nx[2])
        area_idx = np.zeros_like(rho_fluc).astype(bool)
        return Q, M, F, B, wm, dm, bm, Ri, area_idx, idx_max, idx_neutral
    # initializing arrays 
    area_idx = np.zeros_like(rho_fluc).astype(bool)
    area = np.zeros(nx[2])
    w_xy_avg = np.zeros(nx[2])
    b_fluc_xy_avg = np.zeros(nx[2])
    Q = np.zeros(nx[2])
    M = np.zeros(nx[2])
    F = np.zeros(nx[2])
    B = np.zeros(nx[2])
    wm = np.zeros(nx[2])
    dm = np.zeros(nx[2])
    bm = np.zeros(nx[2])
    Ri = np.zeros(nx[2])
    # horizontal area 
    for k in range(idx_max, nx[2]):
        #collecting values of interest at each horizontal plane
        wk = w[:, :, k].reshape(nx[0], nx[1])
        wmagk = w_mag_order[:, :, k]
        b_fluc_k = b_fluc[:, :, k].reshape(nx[0], nx[1])
        #area_bk = (np.abs(b_fluc_k) >= b_tol).astype(float)
        area_wmag = (wmagk >= w_mag_tol).astype(float)
        area_opt = area_wmag#area_bk + 
        area_opt = area_opt>0
        if np.sum(area_opt) < 3:
            idx_max = idx_max + 1
            continue
        area_opt = binary_fill_holes(area_opt)
        if np.all(wk[area_opt]>0): # if there is no negative w, then we are not in the plume yet
            idx_max = idx_max + 1
            continue
        area_idx[:, :, k] = area_opt
        # compute area 
        x_idx, y_idx = np.where(area_opt)
        Xloc = X[x_idx, y_idx, k]
        Yloc = Y[x_idx, y_idx, k]
        points = np.stack([Xloc, Yloc], axis=1)
        hull = ConvexHull(points)
        area[k] = hull.volume
        # compute horizontal averages
        w_xy_avg[k] = np.mean(wk[area_opt])
        b_fluc_xy_avg[k] = np.mean(b_fluc_k[area_opt])
        b_xy_avg = np.mean(b[:, :, k][area_opt])
        # volume flux
        Q[k] = area[k]*w_xy_avg[k]
        # momentum flux
        M[k] = area[k]*w_xy_avg[k]**2 
        # buoyancy flux
        F[k] = area[k]*b_xy_avg*w_xy_avg[k]
        # the buoyancy integral
        B[k] = area[k]*b_xy_avg
        # characteristic w, wm[k] = M[k]/Q[k]
        wm[k] = w_xy_avg[k] 
        # characteristic width of plume, dm = Q / (M**0.5)
        dm[k] = np.sqrt(area[k])
        # characteristic buoyancy
        bm[k] = B[k]*M[k]/(Q[k]**2)
        # Richardson
        Ri[k] = B[k]*Q[k]/(M[k]**1.5)
    Q_sign = np.sign(Q)
    Q_sign_change = np.diff(Q_sign)
    idx_neutral = np.where(Q_sign_change < 0)[0]+1
    if np.size(idx_neutral) > 1:
        idx_diff = np.abs(idx_neutral - idx_max)
        if np.min(idx_diff) < 5:
            idx_neutral = np.delete(idx_neutral, np.where(idx_diff==idx_diff.argmin())[0])
        if np.size(idx_neutral) > 1:
            Ri_sign = np.sign(Ri)
            Ri_sign_change = np.diff(Ri_sign)
            idx_Ri_neutral = np.where(Ri_sign_change < 0)[0] + 1
            idx_diff = np.abs(idx_Ri_neutral - idx_max)
            if np.min(idx_diff) < 5:
                idx_Ri_neutral = np.delete(idx_Ri_neutral, np.where(idx_diff==idx_diff.argmin())[0])
            idx_neutral_2D = np.tile(idx_neutral, (np.size(idx_Ri_neutral), 1)).T
            idx_diff = np.abs(idx_Ri_neutral - idx_neutral_2D)
            idx_neutral = idx_neutral_2D[np.where(idx_diff==idx_diff.min())]
            if np.size(idx_neutral) > 0:
                idx_neutral = idx_neutral[-1]
    area_idx = np.where(area_idx)
    return Q, M, F, B, wm, dm, bm, Ri, area_idx, int(idx_max), int(idx_neutral)
