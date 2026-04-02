import numpy as np
from scipy.ndimage import center_of_mass
from general_analysis_functions import point_linear_interp
from scipy.ndimage import binary_fill_holes
from scipy.spatial import ConvexHull
### -------------------------NBJ FUNCTIONS------------------------- ###
# mixed layer depth information
def mld_info(w, bw_fluc, rho_perturbed, z, mld): # inputs are 1d arrays
    # info at mixed layer depth
    dz_ml = np.abs(z + mld)/mld
    mld_index = np.where(dz_ml==dz_ml.min())[0][-1]
    mld_w = w[mld_index]
    mld_bw_fluc = bw_fluc[mld_index]
    mld_rho_perturbed = rho_perturbed[mld_index]
    return mld_index, mld_w, mld_bw_fluc, mld_rho_perturbed
# depth at which plume is neutrally buoyant
def z_s_analytical(rhoB, rho0, dbdz, g, mld):
    dbdz_norm = dbdz/g
    top = rhoB/rho0-1-dbdz_norm*mld
    return top/dbdz_norm
# max depth of plume intrusion
def z_m_analytical(rhoB, rho_z, g, w):
    At = (rho_z - rhoB)/(rho_z + rhoB)
    return w**2/(g*At)
# centerline analysis with no stokes/ shear influencing flow
def centerline_analysis_buoyancy(bw_fluc_center, dbdz_center, z, nx):
    # finding max height of plume and neutral buoyancy height (perturbed vertical buoyancy flux should be 0)
    bw_fluc_sign = np.sign(bw_fluc_center)
    sign_change_z = bw_fluc_sign[0:(nx[2]-1)] * bw_fluc_sign[1:(nx[2])] < 0
    sign_change_z = np.insert(sign_change_z, 0, False) # to align with original array length (we skipped the bottom, which is first in the array)
    b_fluc_w_sign_change = bw_fluc_center[sign_change_z]
    if len(b_fluc_w_sign_change) < 2: # early stages of plume development
        max_index = np.where(bw_fluc_center==b_fluc_w_sign_change[-1])[0][0]
        neutral_index = max_index
    else:
        neutral_index = np.where(bw_fluc_center==b_fluc_w_sign_change[-1])[0][0]
        max_index = np.where(bw_fluc_center==b_fluc_w_sign_change[-2])[0][0]
    
    z_max = z[max_index]
    z_neutral = z[neutral_index]

    # finding Ozmidov length scale of plume
    dbdz_plume_avg = np.mean(dbdz_center[max_index:])

    return neutral_index, max_index, dbdz_plume_avg

def plume_bw_anlaysis(w, tracer, b_perturbed, bw_perturbed, rho_perturbed, nx, contour, centerline_index, r_max_profile, plume_index):
    # getting centerline values of interest
    rho_perturbed_centerline = rho_perturbed[centerline_index[0], centerline_index[1], centerline_index[2]]
    bw_centerline = bw_perturbed[centerline_index[0], centerline_index[1], centerline_index[2]] # finding b'w' relative to centerline of plume 
    bfluc_centerline = b_perturbed[centerline_index[0], centerline_index[1], centerline_index[2]] # finding b' relative to centerline of plume 
    w_centerline = w[centerline_index[0], centerline_index[1], centerline_index[2]] # finding vertical velocity relative to centerline of plume
    
    # finding sign changes of b', b'w', rho'
    # sign_change = 2: negative to positive, sign_change = -2: positive to negative, 0: no change
    bfluc_sign = np.sign(bfluc_centerline)
    bfluc_sign_change = np.diff(bfluc_sign) # indicates neutral layer: (+ to -), from bottom of domain
    bw_sign = np.sign(bw_centerline)
    bw_sign_change = np.diff(bw_sign) # indicates neutral (- to +) and intrusion layer (+ to -):  from bottom of domain
    rho_sign = np.sign(rho_perturbed_centerline)
    rho_sign_change = np.diff(rho_sign) # indicates neutral layer: (- to +), from bottom of domain
    w_sign = np.sign(w_centerline)
    w_sign_change = np.diff(w_sign) # indicates intrusion layer: (+ to -), from bottom of domain
    
    # finding plume bounds via contour on the centerline of the tracer
    plume_contour = tracer >= contour
    plume_index = np.where(plume_contour)
    if np.sum(plume_index[2])==0: # if there is no tracer in the domain
        plume_bottom_index = nx[2]-1
    else:
        plume_bottom_index = np.min(plume_index[2])

    # checking potential intrusion indices
    idx_bw_intrusion = np.where(bw_sign_change < 0)[0]
    in_plume_test = np.where(idx_bw_intrusion >= plume_bottom_index)[0]
    idx_bw_intrusion_in_plume = idx_bw_intrusion[in_plume_test]
    idx_w_intrusion = np.where(w_sign_change < 0)[0]
    in_plume_test = np.where(idx_w_intrusion >= plume_bottom_index)[0]
    idx_w_intrusion_in_plume = idx_w_intrusion[in_plume_test]
    idx_intrusion_in_plume = np.intersect1d(idx_bw_intrusion_in_plume, idx_w_intrusion_in_plume)

    # checking potential neutral buoyancy indices
    idx_rho_neutral = np.where(rho_sign_change > 0)[0] 
    in_plume_test = np.where(idx_rho_neutral > plume_bottom_index)[0]
    idx_rho_neutral_in_plume = idx_rho_neutral[in_plume_test]
    idx_bw_neutral = np.where(bw_sign_change > 0)[0]
    in_plume_test = np.where(idx_bw_neutral > plume_bottom_index)[0]
    idx_bw_neutral_in_plume = idx_bw_neutral[in_plume_test]
    idx_bfluc_neutral = np.where(bfluc_sign_change < 0)[0]
    in_plume_test = np.where(idx_bfluc_neutral > plume_bottom_index)[0]
    idx_bfluc_neutral_in_plume = idx_bfluc_neutral[in_plume_test]
    idx_neutral_in_plume = np.intersect1d(idx_rho_neutral_in_plume, idx_bfluc_neutral_in_plume)
    idx_neutral_in_plume = np.intersect1d(idx_neutral_in_plume, idx_bw_neutral_in_plume)

    # ensuring there are values for the indices
    if len(idx_intrusion_in_plume)==0:
        idx_intrusion_in_plume = plume_bottom_index
    idx_intrusion_in_plume = int(np.atleast_1d(idx_intrusion_in_plume)[0])

    if len(idx_neutral_in_plume) == 0: # early stages of plume development
        idx_neutral_in_plume = idx_intrusion_in_plume
    else:
        idx_neutral_in_plume = idx_neutral_in_plume[0]
    
    # find the max radius of plume on xy plane
    rp_max = np.max(r_max_profile)
    # ensuring the max radius location is close to the neutral buoyancy height
    idx_rp_max = np.where(r_max_profile == rp_max)[0]
    neural_vs_rp = idx_rp_max - idx_neutral_in_plume
    idx_rp_max = idx_rp_max[np.where(neural_vs_rp==np.min(neural_vs_rp))] 
    return idx_rp_max, idx_neutral_in_plume, idx_neutral_in_plume

def plume_tracer_radius(x, y, nx, centerline_index, tracer, tracer_contour=None, idx=None, contour=None):

    # --- contour threshold ---
    if tracer_contour is None or np.size(tracer_contour) == 0:
        tracer_contour = np.max(tracer[
            centerline_index[0, :],
            centerline_index[1, :],
            idx
        ]) * contour

    plume_contour = tracer >= tracer_contour
    xi, yi, zi = np.where(plume_contour)
    plume_index = (xi, yi, zi)

    centerx = x[centerline_index[0, zi]]
    centery = y[centerline_index[1, zi]]

    r = np.sqrt((x[xi] - centerx)**2 + (y[yi] - centery)**2)
    counts = np.bincount(zi, minlength=nx[2])
    sums   = np.bincount(zi, weights=r, minlength=nx[2])

    rp_profile = np.zeros(nx[2])
    mask = counts > 0
    rp_profile[mask] = sums[mask] / counts[mask]
    return rp_profile, plume_index, tracer_contour

def plume_tracer_analysis(x, y, z, lx, nx, tracer, tracer_contour = [], idx = [], contour = [], calc_option='middle domain'):
    if calc_option == 'middle domain':
        # finding centerline of plume 
        centerline_index = np.zeros((3, nx[2])).astype(int)
        center_xy_loc = np.zeros((3, nx[2]))
        center_xy_loc[0, :] = lx[0]/2
        center_xy_loc[1, :] = lx[1]/2
        center_xy_loc[2, :] = z
        centerline_index[0, :] = nx[0]//2 - 1
        centerline_index[1, :] = nx[1]//2 - 1
        centerline_index[2, :] = np.arange(nx[2]).astype(int)
    elif calc_option == 'center of mass':
        # finding centerline of plume 
        centerline_index = np.zeros((3, nx[2]))
        center_xy_loc = np.zeros((3, nx[2]))
        for k in range(nx[2]-1, -1, -1): # nx[2] = top of the domain 
            center_xy = center_of_mass(tracer[:, :, k])
            if ((np.isnan(center_xy[0]) or np.isnan(center_xy[1]))) and (k < nx[2]-1):
                center_xy = centerline_index[:, k + 1]
            elif ((np.isnan(center_xy[0]) or np.isnan(center_xy[1]))) and (k == nx[2]-1):
                center_xy = [nx[0]//2, nx[1]//2]
            centerline_index[:, k] = [(center_xy[0]), (center_xy[1]), k]
            center_int = [int(center_xy[0]), int(center_xy[1])]
            center_xy_loc[0, k] = point_linear_interp(x[center_int[0]], x[center_int[0]+1], centerline_index[0][k], center_int[0], center_int[0]+1)
            center_xy_loc[1, k] = point_linear_interp(y[center_int[1]], y[center_int[1]+1], centerline_index[1][k], center_int[1], center_int[1]+1)
            center_xy_loc[2, k] = z[k]
        centerline_index = np.round(centerline_index).astype(int)
    # finding plume bounds via contour on the centerline of the tracer
    if np.size(tracer_contour) == 0:
        tracer_contour = np.max(tracer[centerline_index[0, :], centerline_index[1, :], idx])*contour
    plume_contour = tracer >= tracer_contour
    plume_index = np.where(plume_contour)
    edge_mask = plume_contour & (
        ~np.roll(plume_contour, 1, axis=0)
        | ~np.roll(plume_contour,-1, axis=0)
        | ~np.roll(plume_contour, 1, axis=1)
        | ~np.roll(plume_contour,-1, axis=1)
    )
    edge_index = np.where(edge_mask)
    # find the radius of plume on xy plane
    X, Y = np.meshgrid(x, y, indexing='ij')
    rp_profile = np.zeros(nx[2])
    for k in np.arange(nx[2]):
        hor_plane = np.where(edge_index[2]==k)[0]
        if len(hor_plane)==0:
            rp_profile[k] = 0.0
        else:
            r = np.zeros(len(hor_plane))
            for i in range(len(hor_plane)):
                rx = np.abs(x[edge_index[0][hor_plane[i]]]) - center_xy_loc[0, k]
                ry = np.abs(y[edge_index[1][hor_plane[i]]]) - center_xy_loc[1, k]
                r[i] = np.sqrt(rx**2 + ry**2)
            rp_profile[k] = np.mean(r)
    return center_xy_loc, centerline_index, rp_profile, plume_index, tracer_contour
def plume_momentum_analysis(centerline_index, center_xy_loc, nx, x, y, z, w, b, b_fluc, rho_fluc, X, Y, dbdz_tol, b_tol, w_mag_tol):
    # checking magnitude of values to help define bounds
    w_mag = np.abs(w)
    w_mag_order = np.floor(np.log10(w_mag))
    w_mag_cl = w_mag_order[centerline_index[0, :], centerline_index[1, :], centerline_index[2, :]]

    if np.sum(w_mag_cl == w_mag_tol) > 0:
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
            idx_max_2 = idx_rho_max[idx_diff.argmin()] + 1 
            if (idx_max > idx_max_2):
                idx_max = idx_max_2 
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
        if np.sum(area_opt) == 0:
            idx_max = idx_max + 1
            continue
        area_opt = binary_fill_holes(area_opt)
        if np.all(wk[area_opt]>0): # if there is no negative w, then we are not in the plume yet
            idx_max = idx_max + 1
            continue
        area_idx[:, :, k] = area_opt
        # compute area 
        x_idx, y_idx = np.where(area_opt)
        Xloc = X[x_idx, y_idx, k] - center_xy_loc[0, k]
        Yloc = Y[x_idx, y_idx, k] - center_xy_loc[1, k]
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
    return Q, M, F, B, wm, dm, bm, Ri, area_idx, idx_max, idx_neutral

# neutral buoyancy calculation 
def neutral_buoyancy_loc(b_fluc, plume_index, centerline_index):
    if np.size(plume_index)==0:
        neutral_index = np.array([])
    else:
        # finding sign changes of b', b'w', rho'
        # sign_change = 2: negative to positive, sign_change = -2: positive to negative, 0: no change
        bfluc_sign = np.sign(b_fluc[centerline_index[0], centerline_index[1], centerline_index[2]])
        bfluc_sign_change = np.diff(bfluc_sign) # indicates neutral layer: (+ to -), from bottom of domain

        # finding index location of neutral depth 
        idx_bfluc_neutral = np.where(bfluc_sign_change < 0)[0]
        plume_bottom_index = np.min(plume_index[2])
        in_plume_test = np.where(idx_bfluc_neutral > plume_bottom_index)[0]
        neutral_index = idx_bfluc_neutral[in_plume_test]
        if np.size(neutral_index)>=1:
            neutral_index = neutral_index[-1]
    return neutral_index