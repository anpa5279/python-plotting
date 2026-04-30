import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy.spatial import ConvexHull

from interpolation import vertical_line
class PlumeAnalysis:
    def __init__(self, tracer_contour = None):
        self.tracer = None 
        self.tracer_contour = tracer_contour

        self.b_tracer = None
        self.b_background = None
        self.b_fluc = None
        self.b_fluc_w = None

        self.nz = None
        self.plume_index = None
        self.radius_tracer = None

        self.neutral_depth = None
        self.max_depth = None


        self.area_idx =None
        self.area = None
        self.wm = None
        self.dm = None
        self.bm = None
        self.Ri = None
        self.Q = None
        self.M = None
        self.F = None
        self.B = None

        self._contour_cache = {}

    ### -------------------------TRACER INFORMATION------------------------- ###
    def input_info(self, tracer, b_tracer = None, b_background = None, bw_fluc = None):
        self.tracer = tracer
        self.b_tracer = b_tracer
        self.b_background = b_background
        if b_tracer is not None and b_background is not None:
            self.b_fluc = b_background - b_tracer
        self.b_fluc_w = bw_fluc
        self.nz = self.tracer.shape[-1]

    ### -------------------------IMPORTANT DEPTHS------------------------- ###
    def neutral_layer(self, z):
        i_idx, j_idx, k_idx = self.plume_index
        values = self.b_fluc_w[i_idx, j_idx, k_idx]
        # sum per k
        sum_per_k = np.bincount(k_idx, weights=values)
        # count per k
        count_per_k = np.bincount(k_idx)
        # average per k
        bw_fluc_plume_avg = sum_per_k / count_per_k
        bw_fluc_plume_avg[np.isnan(bw_fluc_plume_avg)] = 0
        self.neutral_depth = z[np.where(np.diff(np.sign(bw_fluc_plume_avg))>0)][-1]
        return self.neutral_depth
    def max_penetration(self, z):

        return self.max_depth
    ### -------------------------TRACER CALCULATIONS------------------------- ###
    def plume_tracer_radius(self, x, y):
        plume_contour = self.tracer >= self.tracer_contour
        xi, yi, zi = np.where(plume_contour)
        self.plume_index = [xi, yi, zi]

        x0 = np.mean(x)
        centery = np.mean(y)

        r = np.sqrt((x[xi] - x0)**2 + (y[yi] - centery)**2)
        counts = np.bincount(zi, minlength=self.nz)
        sums   = np.bincount(zi, weights=r, minlength=self.nz)

        self.radius_tracer = np.zeros(self.nz)
        mask = counts > 0
        self.radius_tracer[mask] = sums[mask] / counts[mask]
        return self.radius_tracer

    ### -------------------------MOMENTUM ANALYSIS------------------------- ###
    def plume_momentum_analysis(self, w, b):
            w_xy_avg = w[self.area_idx].mean(axis=(0, 1))
            b_xy_avg = b[self.area_idx].mean(axis=(0, 1))
            self.area_idx
            # volume flux
            self.Q = self.area*w_xy_avg
            # momentum flux
            self.M = self.area*w_xy_avg**2 
            # buoyancy flux
            self.F = self.area*b_xy_avg*w_xy_avg
            # the buoyancy integral
            self.B = self.area*b_xy_avg
            # characteristic w, wm = M/Q
            self.wm = self.M/self.Q
            self.wm[np.isnan(self.wm)] = 0
            # characteristic width of plume, dm = Q / (M**0.5)
            self.dm = self.Q / (self.M**0.5)
            self.dm[np.isnan(self.dm)] = 0
            # characteristic buoyancy
            self.bm = self.B*self.M/(self.Q**2)
            self.bm[np.isnan(self.bm)] = 0
            # Richardson
            self.Ri = self.B*self.Q/(self.M**1.5)
            self.Ri[np.isnan(self.Ri)] = 0

    def plume_momentum_area(self, x, y, w, w_mag_tol):
        X, Y = np.meshgrid(x, y)
        # checking magnitude of values to help define bounds
        w_mag = np.abs(w)
        w_mag_order = np.floor(np.log10(w_mag))
        w_mag_cl = w_mag_order[self.plume_index]

        if np.any(w_mag_cl == w_mag_tol):
            # index of plume points of interest
            b_cl = vertical_line(self.b_fluc, x, y, 0.0, 0.0)
            b_cl_sign = np.sign(b_cl)
            b_cl_sign_change = np.diff(b_cl_sign)
            w_cl = vertical_line(w, x, y, 0.0, 0.0)
            idx_max =np.where(np.diff(np.sign(w_cl)) < 0)[0]
            if np.size(idx_max) == 0: # early stages of plume development
                idx_max = self.nz-1
                self.area_idx = np.zeros_like(self.b_fluc).astype(bool)
            else:
                idx_max =idx_max[-1] +1 
                idx_rho_max = np.where(b_cl_sign_change < 0)[0]
                idx_diff = np.abs(idx_rho_max - idx_max)
                if np.size(idx_rho_max) == 0:
                    idx_max = idx_max
                else:
                    idx_max_2 = idx_rho_max[idx_diff.argmin()] + 1 
                    idx_max = np.max([idx_max, idx_max_2])
        # initializing arrays 
        self.area_idx = np.zeros_like(self.b_fluc).astype(bool)
        self.area = np.zeros(self.nz)
        # horizontal area 
        for k in range(idx_max, self.nz):
            #collecting values of interest at each horizontal plane
            wk = w[:, :, k]
            wmagk = w_mag_order[:, :, k]
            #b_fluc_k = self.b_fluc[:, :, k]
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
            self.area_idx[:, :, k] = area_opt
            # compute area 
            Xloc = X[area_opt]
            Yloc = Y[area_opt]
            points = np.stack([Xloc, Yloc], axis=1)
            hull = ConvexHull(points)
            self.area[k] = hull.volume
