import numpy as np

from interpolation import point

class PlumeAnalysis:
    def __init__(self, reader, file = 'binning_rtz.h5', tracer_contours = np.array([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05])):
        self.z = reader.z
        self.w = reader.load_binning_var('w')
        self.c = reader.load_binning_var('S')
        self.background = reader.load_binning_var('T')
        self.c0 = reader.load_S_temporal_avg(file)
        self.tracer_contours = tracer_contours

        reader.load_equation_of_state()
        self.g = 9.80665
        self.beta = reader.beta
        self.b_tracer = -self.g * reader.beta * self.c
        self.b_background = self.g * reader.alpha * (self.background - reader.T0)

        self.nz = reader.nx[2]
        self.nt = reader.nt
        self.tracer_radii = reader.loading_bin_contours(file = file, contour = tracer_contours)

        self.z_n = None
        self.z_p = None
        self.z_c_s = None

        self._contour_cache = {}

    ### -------------------------IMPORTANT DEPTHS------------------------- ###
    def neutral_layer(self):
        bS = -self.g*self.beta*self.c0
        self.z_n = point(np.mean(self.b_background, axis = 0)-bS, self.z, f0 = 0, nt = self.nt)
        return self.z_n
    def max_penetration(self):
        w_centerline = self.w[0, :, :]
        self.z_p = point(w_centerline, self.z, f0 = 0, nt = self.nt)
        return self.z_p
    def max_tracer_depth(self):
        if self.tracer_contours.size == 1:
            contour = self.tracer_contours[0]
            self.z_c_s = point(self.c0, self.z, f0 = self.c0*contour, nt = self.nt)
        else:
            self.z_c_s = np.zeros((len(self.tracer_contours), self.nt))
            for i, contour in enumerate(self.tracer_contours):
                self.z_c_s[i, :] = point(self.c[0, :, :], self.z, f0 = self.c0*contour, nt = self.nt)
        return self.z_c_s
