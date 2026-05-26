import os
import numpy as np
import h5py
import dask.array as da
from interpolation import xy_plane, xz_plane, yz_plane, horizontal_line
from physics import buoyancy
class OceananigansData:
    def __init__(self, folder, name = 'fields', temperature=True, salinity = False, with_halos=False):
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
        self.halos = with_halos # does the output data include halos? if True, they will be stripped when loading fields

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
        self.T0 = None
        self.alpha = None
        self.beta = None
        self.b = None

        # contour cache for statistics
        self._contour_cache = {}

        # ensuring file order
        all_files = [f for f in os.listdir(self.folder) if (f.endswith('.jld2') and f.startswith(f'{name}'))]
        self.Nranks = len(all_files)
        if self.Nranks > 1:
            self.files = [f'{name}_rank{n}.jld2' for n in range(self.Nranks)]
        else:
            self.files = all_files
    # ------------------------- GRID ------------------------------------ #
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
    # ------------------------- TIME ------------------------------------ #
    def load_time(self):
        with h5py.File(os.path.join(self.folder, self.files[0]), 'r') as f:
            ts_group = [g for g in f.keys() if 'timeseries' in g][0]
            t_group = f[ts_group + '/t']

            self.t_save = np.array(sorted(float(k) for k in t_group.keys()))
            self.time = np.array([t_group[str(int(k))][()] for k in self.t_save])

            self.nt = len(self.time)
            return self.time, self.t_save
    # ------------------------ ADDITIONAL PARAMETERS -------------------- #
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

    # ------------------------- BUOYANCY INFORMATION -------------------- #
    def load_equation_of_state(self, T0 = 25):
        """
        Load thermal expansion (alpha) and optionally
        haline contraction (beta).
        """
        self.T0 = T0
        fname = os.path.join(self.folder, self.files[0])
        with h5py.File(fname, 'r') as f:
            self.alpha =  f['buoyancy/formulation/equation_of_state/thermal_expansion'][()]
            if self.salinity:
                self.beta = f['buoyancy/formulation/equation_of_state/haline_contraction'][()]
    def load_buoyancy(self):
        if self.temperature or self.salinity:
            b = np.empty((self.nx[0], self.nx[1], self.nx[2], self.nt))
            for i, t in enumerate(self.t_save):
                # load only one timestep into memory at a time
                T = self.lazy_field('T', steps=np.array([t])).compute()  # (Nx, Ny, Nz)
                S = self.lazy_field('S', steps=np.array([t])).compute() if self.salinity else []

                bs = buoyancy(self, T, S=S)

                b[:, :, :, i] = bs['b']
            self.b = b
    # ------------------------- INTERNAL UTILS -------------------------- #
    def _slice(self, arr):
        if self.halos:
            return arr[
                self.hx[0]:-self.hx[0],
                self.hx[1]:-self.hx[1],
                self.hx[2]:-self.hx[2]
            ]
        return arr[:, :, :]
    def _read_field(self, f, name, it):
        data = f[f'timeseries/{name}'][f'{int(it)}']
        return self._slice(data).transpose(2, 1, 0)

    # ------------------------- FIELD COLLECTION ------------------------ #
    def lazy_field(self, field, steps=None):
        """
        Returns a truly lazy dask array of shape (Nt, Nx, Ny, Nz).
        Nothing is loaded until .compute() is called.
        """
        if steps is None:
            steps = self.t_save
        steps = np.atleast_1d(steps)

        time_slabs = []
        for t in steps:
            rank_slabs = []
            for file in self.files:
                fname = os.path.join(self.folder, file)
                f = h5py.File(fname, 'r')               # left open intentionally for dask
                dset = f[f'timeseries/{field}/{int(t)}'] # (z, y, x_local)

                if self.halos:
                    # can't slice an h5py dataset lazily with non-contiguous indexing,
                    # so wrap first then slice as dask
                    darr = da.from_array(dset, chunks=dset.shape)
                    darr = darr[
                        self.hx[2] : -self.hx[2] or None,
                        self.hx[1] : -self.hx[1] or None,
                        self.hx[0] : -self.hx[0] or None,
                    ]
                else:
                    darr = da.from_array(dset, chunks=dset.shape)

                darr = darr.transpose(2, 1, 0)           # → (x_local, y, z)
                rank_slabs.append(darr)

            # concatenate ranks along x — still lazy
            time_slabs.append(da.concatenate(rank_slabs, axis=0))  # (Nx, Ny, Nz)

        out = da.stack(time_slabs, axis=0)               # (Nt, Nx, Ny, Nz)
        return out.squeeze()                             # drop time axis if Nt == 1
    def field_slice(self, field, steps=None, slice='YZ', loc=0.0):
        """
        Returns a 2D slice of the field throughout time.
        'YZ' -> shape (Ny, Nz, nt),  loc is x-position
        'XZ' -> shape (Nx, Nz, nt),  loc is y-position
        'XY' -> shape (Nx, Ny, nt),  loc is z-position
        """
        if steps is None:
            steps = self.t_save
        steps = np.atleast_1d(steps)   # replaces the broken size==1 branch

        # Map slice type to the coordinate axis being sliced through
        slice_cfg = {
            #        coord      axis  plane_fn    files_coord
            'YZ': (self.x,  0, yz_plane,  True),
            'XZ': (self.y,  1, xz_plane,  False),
            'XY': (self.z,  2, xy_plane,  False),
        }

        coord, interp_axis, plane_fn, rank_split = slice_cfg[slice]

        # ------------------------------------------------------------------ #
        # Decide which file(s) to open.                                        #
        # For YZ the x-axis is split across ranks, so we may need 1 or 2 files.
        # For XZ/XY all ranks are needed (y/z are not split).                 #
        # ------------------------------------------------------------------ #
        if rank_split:
            exact = np.where(coord == loc)[0]
            if exact.size:                          # loc sits on a grid point
                file_indices = exact[:1]
                needs_interp = False
            else:                                   # loc lies between two ranks
                nearest = np.argsort(np.abs(coord - loc))[:2]
                file_indices = np.sort(nearest)
                needs_interp = True
            files = self.files[file_indices]
        else:
            files = self.files
            needs_interp = True                     # always interpolate in y/z

        # ------------------------------------------------------------------ #
        # Build per-rank chunk shape for dask                                  #
        # HDF5 layout on disk is (z, y, x_local)                              #
        # ------------------------------------------------------------------ #
        nx_local = self.nx[0] // self.Nranks
        chunk_shape = {
            'YZ': (self.nx[2], self.nx[1], 1),          # we load 1 or 2 x-slabs
            'XZ': (self.nx[2], 1,          nx_local),   # full x, one y at a time
            'XY': (1,          self.nx[1], nx_local),   # full x, one z at a time
        }[slice]

        arrayst = []
        for t in steps:
            slabs = []
            for file in files:
                fname = os.path.join(self.folder, file)
                with h5py.File(fname, 'r') as f:
                    dset = f[f'timeseries/{field}/{int(t)}']   # (z, y, x_local)

                    if self.halos:
                        # HDF5 is (z, y, x) → halo order matches
                        dset = dset[
                            self.hx[2] : -self.hx[2] or None,  # z
                            self.hx[1] : -self.hx[1] or None,  # y
                            self.hx[0] : -self.hx[0] or None,  # x
                        ]

                    darr = da.from_array(dset, chunks=chunk_shape)  # (z, y, x_local)
                    darr = darr.transpose(2, 1, 0)                  # → (x_local, y, z)
                    slabs.append(darr.compute())

            # Concatenate along x to get (x_total, y, z) or (x_slab, y, z)
            block = np.concatenate(slabs, axis=0)

            # ---------------------------------------------------------------- #
            # Apply the appropriate plane function                              #
            # block is always (x, y, z) at this point                          #
            # ---------------------------------------------------------------- #
            if slice == 'YZ':
                x_local = coord[file_indices]           # 1 or 2 x values
                if needs_interp:
                    s = yz_plane(block, x_local, loc)   # → (y, z)
                else:
                    s = block[0]                        # single slab, shape (y, z)
                if field == 'u' and self.u_s is not None:
                    # Subtract stokes velocity if available
                    s = s - self.u_s

            elif slice == 'XZ':
                s = xz_plane(block, self.y, loc)        # → (x, z)
                if field == 'u' and self.u_s is not None:
                    # Subtract stokes velocity if available
                    s = s - self.u_s

            elif slice == 'XY':
                s = xy_plane(block, self.z, loc)        # → (x, y)

            arrayst.append(s)

        out = np.stack(arrayst, axis=-1)    # (..., nt)
        return out.squeeze()                # drop time axis when nt == 1
    def field_line(self, field, steps=None, axis='Y', x0=None, y0=None, z0=None):
        """
        Returns a 1D horizontal line through the field, shape (N, nt).

        axis='X' : line along x at fixed y=y0, z=z0  → shape (Nx, nt)
        axis='Y' : line along y at fixed x=x0, z=z0  → shape (Ny, nt)
        """
        if steps is None:
            steps = self.t_save
        steps = np.atleast_1d(steps)

        if axis == 'X' and (y0 is None or z0 is None):
            raise ValueError("axis='X' requires y0 and z0")
        if axis == 'Y' and (x0 is None or z0 is None):
            raise ValueError("axis='Y' requires x0 and z0")

        nx_local = self.nx[0] // self.Nranks
        chunk_shape = (self.nx[2], self.nx[1], nx_local)

        lines = []
        for t in steps:
            slabs = []
            for file in self.files:
                fname = os.path.join(self.folder, file)
                with h5py.File(fname, 'r') as f:
                    dset = f[f'timeseries/{field}/{int(t)}']
                    if self.halos:
                        dset = dset[
                            self.hx[2] : -self.hx[2] or None,
                            self.hx[1] : -self.hx[1] or None,
                            self.hx[0] : -self.hx[0] or None,
                        ]
                    darr = da.from_array(dset, chunks=chunk_shape)
                    darr = darr.transpose(2, 1, 0)               # → (x_local, y, z)
                    slabs.append(darr.compute())

            block = np.concatenate(slabs, axis=0)                # (Nx, Ny, Nz)

            if axis == 'X':
                # fix y then z, return line along x
                line = horizontal_line(block, self.y, self.z, y0, z0, axis='y')  # (Nx,)
            elif axis == 'Y':
                # fix x then z, return line along y
                line = horizontal_line(block, self.x, self.z, x0, z0, axis='x')  # (Ny,)

            lines.append(line)

        return np.stack(lines, axis=-1).squeeze()                # (N, nt)

    # ------------------------- AVERAGES -------------------------------- #
    def xy_avg_1d(self, field, steps=None, with_halos=False):
        """
        Returns horizontal (xy) average vs depth, shape (Nz, nt).
        Averages on the fly — never loads the full 3D field.
        """
        if steps is None:
            steps = self.t_save
        steps = np.atleast_1d(steps)

        nx_local = self.nx[0] // self.Nranks
        profiles = []

        for t in steps:
            # accumulate mean across ranks without stacking full field
            total = np.zeros(self.nx[2])
            for file in self.files:
                fname = os.path.join(self.folder, file)
                with h5py.File(fname, 'r') as f:
                    dset = f[f'timeseries/{field}/{int(t)}']   # (z, y, x_local)
                    if with_halos:
                        dset = dset[
                            self.hx[2] : -self.hx[2] or None,
                            self.hx[1] : -self.hx[1] or None,
                            self.hx[0] : -self.hx[0] or None,
                        ]
                    # mean over x and y while still on disk shape (z,)
                    total += np.asarray(dset).mean(axis=(1, 2))
            profiles.append(total / self.Nranks)    # average rank contributions
        if field == 'u' and self.u_s is not None:
            # subtract stokes velocity if requested
            profiles = [p - self.u_s for p in profiles]
        return np.stack(profiles, axis=-1).squeeze()   # (Nz, nt)
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

    # ------------------------- BINNING --------------------------------- #
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
            T_rz = f['ccc/T_rz'][()]
            ur_rz = f['ccc/horizontal velocity'][()]
            w_rz = f['ccc/w_rz'][()]

        self._contour_cache[file] = (r, z, time, S_rz, T_rz, ur_rz, w_rz)

        return r, z, time, S_rz, T_rz, ur_rz, w_rz
    def load_binning_var(self, var, file = 'binning_rtz.h5'):
        """
        Loads binning (cached).
        """

        fname = os.path.join(self.folder, file)

        if file in self._contour_cache:
            return self._contour_cache[file]

        with h5py.File(fname, 'r') as f:
            a = f['ccc/'+var+'_rz'][()]

        return a