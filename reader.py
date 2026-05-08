import os
import numpy as np
import h5py
import dask.array as da

class OceananigansData:
    def __init__(self, folder, name = 'fields', temperature=True, salinity = False):
        self.folder = folder

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

        # Time-related (set by load_time)
        self.nt = None
        self.time = None
        self.t_save = None

        # possible additional paraemters
        self.f = None           # coriolis
        self.visc = None        # viscosity
        self.diff = None        # diffusivity
        self.u_s = None         # stokes velocity
        self.u_f = None         # friction velocity

        # equation of state information
        self.temperature = temperature
        self.salinity = salinity
        self.alpha = None
        self.beta = None

        # contour cache for statistics
        self._contour_cache = {}

        # ensuring file order
        all_files = [f for f in os.listdir(self.folder) if (f.endswith('.jld2') and f.startswith(f'{name}'))]
        self.Nranks = len(all_files)
        if self.Nranks > 1:
            self.files = [f'{name}_rank{n}.jld2' for n in range(self.Nranks)]
        else:
            self.files = all_files
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
    # ------------------------- TIME ------------------------- #
    def load_time(self):
        with h5py.File(os.path.join(self.folder, self.files[0]), 'r') as f:
            ts_group = [g for g in f.keys() if 'timeseries' in g][0]
            t_group = f[ts_group + '/t']

            self.t_save = np.array(sorted(float(k) for k in t_group.keys()))
            self.time = np.array([t_group[str(int(k))][()] for k in self.t_save])

            self.nt = len(self.time)
            return self.time, self.t_save
    # ------------------------- ADDITIONAL PARAMETERS ------------------------- #
    def load_coriolis(self):
        with h5py.File(os.path.join(self.folder, self.files[0]), 'r') as f:
            self.f = f['serialized/coriolis'][()]
    def load_viscosity(self):
        with h5py.File(os.path.join(self.folder, self.files[0]), 'r') as f:
            self.visc = f['closure/ν'][()]
    def load_diffusivity(self):
        with h5py.File(os.path.join(self.folder, self.files[0]), 'r') as f:
            self.diff = f['closure/κ'][()]
    def load_stokes_velocity(self):
        with h5py.File(os.path.join(self.folder, self.files[0]), 'r') as f:
            self.u_s = f['IC/stokes_velocity'][()]
    def load_friction_velocity(self):
        with h5py.File(os.path.join(self.folder, self.files[0]), 'r') as f:
            self.u_f = f['IC/friction_velocity'][()]
    def load_equation_of_state(self):
        """
        Load thermal expansion (alpha) and optionally
        haline contraction (beta).
        """
        fname = os.path.join(self.folder, self.files[0])
        with h5py.File(fname, 'r') as f:
            self.alpha =  f['buoyancy/formulation/equation_of_state/thermal_expansion'][()]
            if self.salinity:
                self.beta = f['buoyancy/formulation/equation_of_state/haline_contraction'][()]

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
    def load_temporal_averages(self, file, contour_bound = 0.05):
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

            if self.temperature:
                T = {
                    'T_avg': f['1D temporal averages/T'][()],
                    'T_fluc_avg': f['1D temporal averages/T\''][()],
                }
            else:
                T = None

            if self.salinity:
                S = {
                    'S_avg': f['1D temporal averages/S'][()],
                    'S_fluc_avg': f['1D temporal averages/S\''][()],
                }
                r_plume = {'tracer radius': f[f'plume statistics/contour {contour_bound}/plume tracer radius with depth'][()]}
            else:
                S = None
                r_plume = None

        return rms, bw, T, S, r_plume
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
    # ------------------------- BINNING ------------------------- #
    def load_binning(self, file = 'binning_rtz.h5'):
        """
        Loads binning (cached).
        """

        fname = os.path.join(self.folder, 'binning', file)

        if file in self._contour_cache:
            return self._contour_cache[file]

        with h5py.File(fname, 'r') as f:
            r = f['ccc/dimensions/r_bin'][()]
            z = f['ccc/dimensions/z'][()]
            time = f['ccc/dimensions/time'][()]
            S_rz = f['ccc/S_rz'][()]
            T_fluc_rz = f['ccc/T\'_rz'][()]
            T_rz = f['ccc/T_rz'][()]
            ur_rz = f['ccc/horizontal velocity'][()]
            w_rz = f['ccc/w_rz'][()]
            b_fluc_rz = f['ccc/b\'_rz'][()]

        self._contour_cache[file] = (r, z, time, S_rz, T_fluc_rz, T_rz, ur_rz, w_rz, b_fluc_rz)

        return r, z, time, S_rz, T_fluc_rz, T_rz, ur_rz, w_rz, b_fluc_rz