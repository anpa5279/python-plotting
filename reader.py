import os
import numpy as np
import h5py
import re
import dask.array as da
from interpolation import plane_slice_calc, velocities_to_center
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
        self.dt = None
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
        self._cache = {}

        # ensuring file order
        all_files = [f for f in os.listdir(self.folder) if (f.endswith('.jld2') and f.startswith(f'{name}'))]
        self.Nranks = len(all_files)
        if self.Nranks > 1:
            self.files = [f'{name}_rank{n}.jld2' for n in range(self.Nranks)]
        else:
            self.files = all_files

        self.bin_file = 'binning_rtz.h5'
        self.binning= self.bin_file in os.listdir(self.folder)

        self.load_time()
        self.load_grid()
    
    # ------------------------- GRID ------------------------------------ #
    def load_grid(self, grid_specs = False):
        if 'grid_info.jld2' in os.listdir(self.folder):
            with h5py.File(os.path.join(self.folder, 'grid_info.jld2'), 'r') as f:
                self.Nranks = f['grid/Nranks'][()]
                self.nx = [
                    f['grid/Nx'][()]*self.Nranks if grid_specs else f['grid/Nx'][()],
                    f['grid/Ny'][()],
                    f['grid/Nz'][()]
                ]
                self.hx = [3, 3, 3]
                self.lx = [
                    f['grid/Lx'][()]*self.Nranks if grid_specs else f['grid/Lx'][()],
                    f['grid/Ly'][()],
                    f['grid/Lz'][()]
                ]
                self.dx = [
                    f['grid/Δx'][()],
                    f['grid/Δy'][()],
                    f['grid/Δz'][()]
                ]
                self.x = np.linspace(-self.lx[0]/2 + self.dx[0]/2, self.lx[0]/2 - self.dx[0]/2, self.nx[0])
                self.y = np.linspace(-self.lx[1]/2 + self.dx[1]/2, self.lx[1]/2 - self.dx[1]/2, self.nx[1])# f['grid/y']#[self.hx[1]:-self.hx[1]]
                self.z = np.linspace(-self.lx[2] + self.dx[2]/2, -self.dx[2]/2, self.nx[2])#f['grid/z']#[self.hx[2]:-self.hx[2]]
                self.zf = np.linspace(-self.lx[2], 0, self.nx[2])
        else:
            with h5py.File(os.path.join(self.folder, self.files[-1]), 'r') as f:
                self.nx = [
                    f['grid/Nx'][()] * self.Nranks,
                    f['grid/Ny'][()],
                    f['grid/Nz'][()]
                ]
                self.hx = [3, 3, 3]#[f['grid/Hx'][()], f['grid/Hy'][()], f['grid/Hz'][()]]
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
            self.dt = self.time[1] - self.time[0] if self.nt > 1 else None
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
        fname = os.path.join(self.folder, self.files[-1])
        with h5py.File(fname, 'r') as f:
            self.alpha =  f['buoyancy/formulation/equation_of_state/thermal_expansion'][()]
            if self.salinity:
                self.beta = f['buoyancy/formulation/equation_of_state/haline_contraction'][()]
    def load_buoyancy_small(self, file = 'buoyancy_profile.h5'):
        buoyancy_file = os.path.join(self.folder, file)
        if os.path.exists(buoyancy_file):
            with h5py.File(buoyancy_file, 'r') as f:
                z = f['z'][...]
                b_avg = f['b_avg'][()]
                b_rms = f['b_rms'][...]
                b_centerline = f['centerline/b'][...]
                b_fluc_centerline = f['centerline/b_fluc'][...]
            return b_avg, b_rms, b_centerline, b_fluc_centerline
        else:
            T = self.lazy_field('T').compute()
            if self.salinity:
                S = self.lazy_field('S').compute()
            g = 9.80665
            if not self.salinity:
                b_profile = g * self.alpha * (T - self.T0)
            else:
                b_profile = np.squeeze(g * self.alpha * (T - self.T0) - g * self.beta * S)
            b_avg = np.mean(b_profile)
            b_fluc = b_profile - b_avg
            b_rms = np.mean(b_fluc**2)**0.5
    
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
    def _get_field(self, var):
        key = f'_field_{var}'
        if key not in self._cache:
            self._cache[key] = self.lazy_field(var).compute()
        return self._cache[key]
    
    # ------------------------- FIELD COLLECTION ------------------------ #
    def lazy_field(self, field, steps=None):
        """
        Returns a truly lazy dask array of shape (Nt, Nx, Ny, Nz).
        Nothing is loaded until .compute() is called.
        """
        key = f'_field_{field}'
        if key not in self._cache:
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

                    darr = darr.transpose(2, 1, 0)           # (x_local, y, z)
                    rank_slabs.append(darr)

                # concatenate ranks along x — still lazy
                time_slabs.append(da.concatenate(rank_slabs, axis=0))  # (Nx, Ny, Nz)

            out = da.stack(time_slabs, axis=0)               # (Nt, Nx, Ny, Nz)
        else:
            out = self._cache[key]
        return out.squeeze()                                 # drop time axis if Nt == 1
    def field_slice(self, field, steps=None, slice='YZ', loc=0.0, N=None):
        """
        Returns a 2D slice of the field throughout time.
        'YZ' -> shape (nt, Ny, Nz),  loc is x-position
        'XZ' -> shape (nt, Nx, Nz),  loc is y-position
        'XY' -> shape (nt, Nx, Ny),  loc is z-position
        """
        if steps is None:
            steps = self.t_save
        steps = np.atleast_1d(steps)

        # ------------------------------------------------------------------ #
        # Fast path: N is a grid index — use lazy_field and index directly #
        # ------------------------------------------------------------------ #
        if N is not None:
            lazy = self.lazy_field(field, steps)   # (nt, Nx, Ny, Nz) or (Nx, Ny, Nz) if nt==1
            if lazy.ndim == 3:
                lazy = lazy[np.newaxis]            # (1, Nx, Ny, Nz)

            if slice == 'YZ':
                out = lazy[:, int(N), :, :]        # (nt, Ny, Nz)
            elif slice == 'XZ':
                out = lazy[:, :, int(N), :]        # (nt, Nx, Nz)
            elif slice == 'XY':
                out = lazy[:, :, :, int(N)]        # (nt, Nx, Ny)

            if field == 'u' and self.u_s is not None:
                out = out - self.u_s

            return out.compute().squeeze()

        # ------------------------------------------------------------------ #
        # Slow path: loc is a physical coordinate — interpolate               #
        # ------------------------------------------------------------------ #
        slice_cfg = {
            'YZ': (self.x, True),
            'XZ': (self.y, False),
            'XY': (self.z, False),
        }
        coord, rank_split = slice_cfg[slice]
        nx_local = self.nx[0] // self.Nranks

        if rank_split:
            exact = coord == loc
            if any(exact):
                file_indices = np.where(exact)[0]
                needs_interp = False
                halos_needed = False
            else:
                nearest = np.argsort(np.abs(coord - loc))[:2]
                file_indices = np.array([int(i) for i in np.floor(np.sort(nearest) / self.nx[0] * self.Nranks)])
                # deduplicate in case both nearest points are in the same rank file
                file_indices = np.unique(file_indices)
                needs_interp = True
                halos_needed = self.halos
                if halos_needed:
                    file_indices = file_indices[:1]
            files = [self.files[i] for i in np.atleast_1d(file_indices)]
        else:
            files = self.files
            needs_interp = True
            halos_needed = False

        def _load_slab(fname, t):
            with h5py.File(fname, 'r') as f:
                data = f[f'timeseries/{field}/{int(t)}'][...]   # (z, y, x_local)
            if self.halos and not halos_needed:
                data = data[
                    self.hx[2] : -self.hx[2] or None,   # z
                    self.hx[1] : -self.hx[1] or None,   # y
                    self.hx[0] : -self.hx[0] or None,   # x_local
                ]
            return data.transpose(2, 1, 0)              # (x_local, y, z)

        arrayst = []
        for t in steps:
            slabs = [_load_slab(os.path.join(self.folder, f), t) for f in files]
            block = np.concatenate(slabs, axis=0)        # (x_local_or_total, y, z)

            if slice == 'YZ':
                if needs_interp:
                    # build the actual x-coordinates corresponding to rows of block
                    x_slab_coords = np.concatenate([
                        coord[i * nx_local : (i + 1) * nx_local]
                        for i in np.atleast_1d(file_indices)
                    ])
                    s = plane_slice_calc(block, x_slab_coords, loc)   # (y, z)
                else:
                    # loc sits exactly on a grid point — find its local index
                    global_idx = np.where(coord == loc)[0][0]
                    local_idx = global_idx % nx_local
                    s = block[local_idx]                       # (y, z)

            elif slice == 'XZ':
                s = plane_slice_calc(block, self.y, loc)               # (x, z)

            elif slice == 'XY':
                s = plane_slice_calc(block, self.z, loc)               # (x, y)

            if field == 'u' and self.u_s is not None:
                s = s - self.u_s

            arrayst.append(s)

        out = np.stack(arrayst, axis=0)   # (nt, Ny, Nz) or (nt, Nx, Nz) or (nt, Nx, Ny)
        return out.squeeze()
    def field_centerline(self, field, loc=(0.0, 0.0), steps=None):
        """
        Returns a 1D slice of the field along the centerline throughout time.
        Output shape is (nt,).
        """
        if steps is None:
            steps = self.t_save
        steps = np.atleast_1d(steps)
        s_centerline = np.empty((len(steps), self.nx[2]))

        centerline_files = [f for f in os.listdir(self.folder) if (f.endswith('.jld2') and f.startswith(f'centerline'))] # halos are included in these files vertically
        if any(centerline_files):
            for it, t in enumerate(steps):
                with h5py.File(os.path.join(self.folder, 'centerline'), 'r') as f:
                    data = f[f'timeseries/{field}/{int(t)}']
                s = data.transpose(2, 1, 0)[:, :, self.hx[2]:-self.hx[2]]           # (x_local, y, z)
                # linear interpolation of x and y points is the same as averaging the 4 grid points in file
                if field == 'u': # because raw data is fcc
                    s = s[-1, :, :]
                if field == 'v': # because raw data is cfc
                    s = s[:, -1, :]
                if field == 'w': # because raw data is ccf
                    s = velocities_to_center(s, axis=-1)
                s = np.mean(s, axis=(0, 1))
                s_centerline[it, :] = s
        else:
            for it, t in enumerate(steps):
                s = self.field_slice(field, steps = steps, loc = loc[0])
                # linear interpolation of x and y points is the same as averaging the 4 grid points in file
                if field == 'u': # because raw data is fcc
                    s = s[-1, :, :]
                if field == 'v': # because raw data is cfc
                    s = s[:, -1, :]
                if field == 'w': # because raw data is ccf
                    s = velocities_to_center(s, axis=-1)
                else:
                    s = np.mean(s[0, self.nx[1]//2:self.nx[1]//2+2, :], axis=(0, 1)) # average over x and y dimensions
                s_centerline[it, :] = s

        if field == 'u' and self.u_s is not None:
            s_centerline = s_centerline - self.u_s
        return s_centerline.squeeze()
    # ------------------------- AVERAGES -------------------------------- #
    def load_temporal_averages(self, file, contour_bound = 0.05):
        fname = os.path.join(self.folder, self.bin_file)

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
    def load_S_temporal_avg(self):
        """
        Loads contour temporal averages (cached).
        """

        fname = os.path.join(self.folder, self.bin_file)

        if 'S value' in self._cache:
            return self._cache['S value']

        with h5py.File(fname, 'r') as f:
            S = f['contour temporal averages/S'][()]

        self._cache['S value'] = S

        return S
    
    # ------------------------- BINNING --------------------------------- #
    def load_binning_var(self, var):
        """
        Loads binning (cached). [nr, nz, nt]
        """
        if self.binning:

            fname = os.path.join(self.folder, self.bin_file)
            #if self.bin_file in self._cache:
            #    return self._cache[self.bin_file]
            opt = 'ccc/'+var+'_rz' if len(var) < 3 else 'ccc/'+var

            with h5py.File(fname, 'r') as f:
                a = f[opt][()]

        return a
    def loading_bin_contours(self, contour = 0.05):
        """
        Loads binning radius (cached).
        """

        fname = os.path.join(self.folder, self.bin_file)
        if isinstance(contour, float):
            with h5py.File(fname, 'r') as f:
                r = f[f'r given contour/contour = {contour}'][()]
        else:
            r = np.empty(len(contour), self.nx[2], dtype=object)
            for i, p in enumerate(contour):
                with h5py.File(fname, 'r') as f:
                    r[i, :] = f[f'r given contour/contour = {p}'][()]
        return r
    def loading_bin_radius(self):
        """
        Loads binning radius (cached).
        """

        fname = os.path.join(self.folder, self.bin_file)

        #if self.bin_file in self._cache:
            #return self._cache[self.bin_file]
        
        with h5py.File(fname, 'r') as f:
            r = f[f'ccc/dimensions/r_bin'][()]
        return r
    
    # ------------------------ FLUCTUATIONS ----------------------------- #
    def load_fluc(self, var, file = 'fluctuations.h5'):
        """
        Loads fluctuation variables (cached).
        """

        fname = os.path.join(self.folder, file)

        #if file in self._cache:
            #return self._cache[file]
        opt = 'fluctuations/'+var+'_fluc'
        with h5py.File(fname, 'r') as f:
            a = f[opt][()]

        return a

    # ------------------------ RMS -------------------------------------- #
    def load_rms(self, var, file = 'fluctuations.h5'):
        """
        Loads velocity RMS 
        """

        if var + 'rms' in self._cache:
            return self._cache[var + 'rms']

        fname = os.path.join(self.folder, file)

        #if file in self._cache:
            #return self._cache[file]
        opt = 'rms/'+var
        with h5py.File(fname, 'r') as f:
            a = f[opt][()]
        self._cache[var + 'rms'] = a
        return a

    # ------------------------ PLANE SLICE ------------------------------ #
    def load_plane_var(self, var, loc=0.0, file='plane_slice.h5'):
        """
        Loads plane slice variables (cached).
        """
        fname = os.path.join(self.folder, file)
        
        import re
        pattern = r'x = ' + str(int(loc)) + r'(?:\.\d+)?'
        
        with h5py.File(fname, 'r') as f:
            yz_group = f['YZ']
            # Find the matching key inside YZ group
            matching_key = None
            for key in yz_group.keys():
                if re.fullmatch(pattern, key):
                    matching_key = key
                    break
            
            if matching_key is None:
                raise KeyError(f"No key matching '{pattern}' in YZ group. Available: {list(yz_group.keys())}")
            
            a = yz_group[matching_key][var][()]
        return a