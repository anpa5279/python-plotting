import os
import numpy as np
import h5py
import dask.array as da

class OceananigansData:
    def __init__(self, folder, files, Nranks):
        self.folder = folder
        self.files = files
        self.Nranks = Nranks

        # Grid-related (set by load_grid)
        self.nx = None
        self.hx = None
        self.lx = None
        self.dx = None
        self.x = None
        self.xf = None
        self.y = None
        self.yf = None
        self.z = None
        self.zf = None

        # contour cache for statistics
        self._contour_cache = {}

    # ------------------------- GRID ------------------------- #
    def load_grid(self):
        with h5py.File(os.path.join(self.folder, self.files[0]), 'r') as f:
            self.nx = [
                f['grid/Nx'][()] * self.Nranks,
                f['grid/Ny'][()],
                f['grid/Nz'][()]
            ]
            self.hx = [
                f['grid/Hx'][()],
                f['grid/Hy'][()],
                f['grid/Hz'][()]
            ]
            self.lx = [
                f['grid/Lx'][()] * self.Nranks,
                f['grid/Ly'][()],
                f['grid/Lz'][()]
            ]
            self.dx = [
                f['grid/Δxᶜᵃᵃ'][()],
                f['grid/Δyᵃᶜᵃ'][()],
                f['grid/z/Δᵃᵃᶜ'][()]
            ]

            self.y = f['grid/yᵃᶜᵃ'][self.hx[1]:-self.hx[1]]
            self.z = f['grid/z/cᵃᵃᶜ'][self.hx[2]:-self.hx[2]]
            self.zf = f['grid/z/cᵃᵃᶠ'][self.hx[2]:-self.hx[2]]

        # Assemble x (distributed if needed)
        if self.Nranks == 1:
            with h5py.File(os.path.join(self.folder, self.files[0]), 'r') as f:
                self.x = f['grid/xᶜᵃᵃ'][self.hx[0]:-self.hx[0]]
        else:
            xrange = self.nx[0] // self.Nranks
            self.x = np.zeros(self.nx[0])

            for i, file in enumerate(self.files):
                with h5py.File(os.path.join(self.folder, file), 'r') as f:
                    self.x[i*xrange:(i+1)*xrange] = \
                        f['grid/xᶜᵃᵃ'][self.hx[0]:-self.hx[0]]
    # ------------------------- EQUATION OF STATE ------------------------- #
    def load_equation_of_state(self, file=None, salinity=False):
        """
        Load thermal expansion (alpha) and optionally
        haline contraction (beta).
        """

        if file is None:
            file = self.files[0]

        fname = os.path.join(self.folder, file)

        with h5py.File(fname, 'r') as f:
            coeffs = {'alpha': f['buoyancy/formulation/equation_of_state/thermal_expansion'][()]}

            if salinity:
                coeffs['beta'] = f['buoyancy/formulation/equation_of_state/haline_contraction'][()]

        return coeffs
    # ------------------------- TIME ------------------------- #
    def load_time(self, stokes=False, closure=True):
        with h5py.File(os.path.join(self.folder, self.files[0]), 'r') as f:
            ts_group = [g for g in f.keys() if 'timeseries' in g][0]
            t_group = f[ts_group + '/t']

            t_save = np.array(sorted(float(k) for k in t_group.keys()))
            time = np.array([t_group[str(int(k))][()] for k in t_save])

            visc = f['closure/ν'][()] if closure else 0.0
            diff = f['closure/κ'] if closure else 0.0

            if stokes:
                u_f = np.array(f["IC/"]["friction_velocity"])
                u_s = np.array(f["IC/"]["stokes_velocity"])
            else:
                u_f, u_s = None, None
        return time, t_save, visc, diff, u_f, u_s
    # ------------------------- INTERNAL UTILS ------------------------- #
    def _slice(self, arr, with_halos):
        if with_halos:
            return arr[
                self.hx[0]:-self.hx[0],
                self.hx[1]:-self.hx[1],
                self.hx[2]:-self.hx[2]
            ]
        return arr[:, :, :]

    def _read_field(self, f, name, t, with_halos):
        data = f[f'timeseries/{name}'][f'{int(t)}']
        return self._slice(data, with_halos).transpose(2, 1, 0)

    # ------------------------- FIELD COLLECTION ------------------------- #
    def lazy_field(self, field, t, with_halos=False):
        """
        Returns a lazy (dask) array of shape (Nx, Ny, Nz)
        without loading into memory.
        """

        arrays = []
        chunk = self.nx[0] // self.Nranks

        for r, file in enumerate(self.files):
            fname = os.path.join(self.folder, file)

            f = h5py.File(fname, 'r')  # keep open!

            dset = f[f'timeseries/{field}'][f'{int(t)}']

            if with_halos:
                dset = dset[
                    self.hx[2]:-self.hx[2],
                    self.hx[1]:-self.hx[1],
                    self.hx[0]:-self.hx[0]
                ]

            # Wrap as dask array
            darr = da.from_array(
                dset,
                chunks=dset.shape  # one chunk per rank (you can tune this)
            )

            # transpose lazily
            darr = darr.transpose(2, 1, 0)

            arrays.append(darr)

        # stitch along x
        return np.array(da.concatenate(arrays, axis=0))
    # ------------------------- TEMPORAL AVERAGES ------------------------- #
    def load_temporal_averages(self, file, temperature=True, salinity=False):
        fname = os.path.join(self.folder, file)

        with h5py.File(fname, 'r') as f:
            rms = {
                'u_rms': f['1D temporal averages/urms'][()],
                'v_rms': f['1D temporal averages/vrms'][()],
                'w_rms': f['1D temporal averages/wrms'][()],
            }

            bw = {
                'w_avg': f['1D temporal averages/w'][()],
                'b_avg': f['1D temporal averages/b'][()],
                'bw_fluc_avg': f['1D temporal averages/b\'w\''][()]
            }

            if temperature:
                T = {
                    'T_avg': f['1D temporal averages/T'][()],
                    'T_fluc_avg': f['1D temporal averages/T\''][()],
                }
            else:
                T = None

            if salinity:
                S = {
                    'S_avg': f['1D temporal averages/S'][()],
                    'S_fluc_avg': f['1D temporal averages/S\''][()],
                }
                contours = {
                    'S_contour': f['plume statistics/']
                }
            else:
                S = None

        return contours, rms, bw, T, S
    def load_contour_temporal_averages(self, file):
        """
        Loads contour temporal averages (cached).
        """

        fname = os.path.join(self.folder, file)

        if file in self._contour_cache:
            return self._contour_cache[file]

        with h5py.File(fname, 'r') as f:
            S = f['contour temporal averages/S'][()]
            w = f['contour temporal averages/w'][()]

        self._contour_cache[file] = (S, w)

        return S, w
