import os
import numpy as np
import h5py
import dask.array as da
from interpolation import plane_slice_calc, velocities_to_center, vertical_line

class OceananigansData:
    def __init__(self, folder, temperature=True, salinity = False, with_halos=False):
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
        self.t = None
        self.dt = None
        self.t_save = None

        # possible additional paraemters
        self.f = None           # coriolis
        self.visc = None        # viscosity
        self.diff = None        # diffusivity
        self.u_s = None         # stokes velocity
        self.u_f = None         # friction velocity

        # equation of state information
        if temperature or salinity:
            self.temperature = temperature
            self.salinity = salinity
            self.T0 = None
            self.alpha = None
            self.beta = None
            self.b = None

        # collecting all jld2 and h5 files names in folder
        self.all_files = [f for f in os.listdir(self.folder) if (f.endswith('.jld2') or f.endswith('.h5'))]

        # ensuring file order for field files
        fields_files = [f for f in self.all_files if (f.endswith('.jld2') and f.startswith('fields') and not f.startswith('fields_pickup'))]

        self.Nranks = len(fields_files)
        if self.Nranks > 1:
            self.files = [f'fields_rank{n}.jld2' for n in range(self.Nranks)]
        else:
            self.files = fields_files

        # checking for binning file
        if 'binning_rtz.h5' in os.listdir(self.folder):
            self.bin_file = 'binning_rtz.h5'
            self.binning = True
        else:
            self.binning = False

        # checking for averaging file
        if any([f for f in self.all_files if (f.endswith('.jld2') and f.startswith('xy_avg'))]):
            self.averaging_file = [f for f in self.all_files if (f.endswith('.jld2') and f.startswith('xy_avg'))][0]
            self.averaging= True
            self.t_save_avg = None
            self.time_avg = None
        else:
            self.averaging= False

        # checking for centerline file
        if any([f for f in self.all_files if ((f.endswith('.jld2') or f.endswith('.h5')) and f.startswith('centerline'))]):
            self.centerline_output = [f for f in self.all_files if (f.endswith('.jld2') and f.startswith('centerline'))][0]
            self.centerline_file = [f for f in self.all_files if (f.endswith('.h5') and f.startswith('centerline'))]
            if self.centerline_file == []:
                self.centerline_file = None
            else: # pickup does not matter because this means that it is already post processed and interpolated to the centerline
                self.centerline_file = self.centerline_file[0]
            self.centerline= True
            self.t_save_center = None
            self.time_center = None
        else:
            self.centerline= False

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
            self.xf = np.linspace(-self.lx[0]/2, self.lx[0]/2, self.nx[0]+1)
            self.y = np.linspace(-self.lx[1]/2 + self.dx[1]/2, self.lx[1]/2 - self.dx[1]/2, self.nx[1])
            self.yf = np.linspace(-self.lx[1]/2, self.lx[1]/2, self.nx[1]+1)
            self.z = np.linspace(-self.lx[2] + self.dx[2]/2, -self.dx[2]/2, self.nx[2])
            self.zf = np.linspace(-self.lx[2], 0, self.nx[2])
            self.hx = [3, 3, 3]
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
                self.yf = f['grid/yᵃᶠᵃ'][self.hx[1]:-self.hx[1]]
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
            self.xf = np.linspace(-self.lx[0]/2, self.lx[0]/2, self.nx[0]+1)
    # ------------------------- TIME ------------------------------------ #
    def load_time(self):
        with h5py.File(os.path.join(self.folder, self.files[0]), 'r') as f:
            t_group = f['timeseries/t']

            self.t_save = np.array(sorted(float(k) for k in t_group.keys()))
            self.t = np.array([t_group[str(int(k))][()] for k in self.t_save])

            self.nt = len(self.t)
            self.dt = self.t[1] - self.t[0] if self.nt > 1 else None
        
        if self.averaging:
            with h5py.File(os.path.join(self.folder, self.averaging_file), 'r') as f:
                t_group = f['timeseries/t']
                self.t_save_avg = np.array(sorted(float(k) for k in t_group.keys()))
                self.time_avg = np.array([t_group[str(int(k))][()] for k in self.t_save_avg])
        if self.centerline:
            with h5py.File(os.path.join(self.folder, self.averaging_file), 'r') as f:
                t_group = f['timeseries/t']
                self.t_save_center = np.array(sorted(float(k) for k in t_group.keys()))
            
                self.time_center = np.array([t_group[str(int(k))][()] for k in self.t_save_center])
        return self.t, self.t_save
    
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

    #--------------------------------------------------------------------#
    #                                                                    #
    #                        FIELD COLLECTION                            #
    #              *load in or calculate if necessary*                   #
    #                                                                    #
    #--------------------------------------------------------------------#
    def lazy_field(self, field, steps=None, transpose=True):
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
                with h5py.File(fname, 'r') as f:
                    dset = f[f'timeseries/{field}/{int(t)}'][...]  # load the small slab eagerly
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
                if darr.ndim == 3:
                    darr = darr.transpose(2, 1, 0)           # (x_local, y, z)
                rank_slabs.append(darr)

            # concatenate ranks along x — still lazy
            time_slabs.append(da.concatenate(rank_slabs, axis=0))  # if 3D, (Nx, Ny, Nz), if 1D (Nz,)

        out = da.stack(time_slabs, axis=0)               # if 3D, (Nt, Nx, Ny, Nz), if 1D (Nt, Nz,)
        if not transpose:
            out = out.transpose(0, 3, 2, 1) if out.ndim == 4 else out.transpose(0, 2, 1)  # (Nt, Nz, Ny, Nx) 
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
                    s = plane_slice_calc(block, x_slab_coords, loc, axis = -3)   # (y, z)
                else:
                    # loc sits exactly on a grid point — find its local index
                    global_idx = np.where(coord == loc)[0][0]
                    local_idx = global_idx % nx_local
                    s = block[local_idx]                       # (y, z)

            elif slice == 'XZ':
                s = plane_slice_calc(block, self.y, loc, axis = -2)               # (x, z)

            elif slice == 'XY':
                s = plane_slice_calc(block, self.z, loc, axis = -1)               # (x, y)

            if field == 'u' and self.u_s is not None:
                s = s - self.u_s

            arrayst.append(s)

        out = np.stack(arrayst, axis=0)   # (nt, Ny, Nz) or (nt, Nx, Nz) or (nt, Nx, Ny)
        return out.squeeze()
    def field_centerline(self, field, steps=None):
        """
        Returns a 1D slice of the field along the centerline throughout time.
        Output shape is (nt,).
        """
        if self.centerline:
            if self.centerline_file is not None: # the interpolated centerline file exists — load from there
                with h5py.File(os.path.join(self.folder, self.centerline_file), 'r') as f:
                    s_centerline = f[f'centerline/{field}'][()] # (z,)
                if steps is not None:
                    time_indices = np.array([np.where(self.time_center == t)[0][0] for t in steps])
                    s_centerline = s_centerline[time_indices]
            elif self.centerline_file is None and self.centerline_output is not None: # the interpolated centerline file does not exist, but the higher frequency output does 
                if steps is None:
                    steps = self.t_save_center
                steps = np.atleast_1d(steps)
                s_centerline = np.empty((len(steps), self.nx[2]))
                for it, t in enumerate(steps):
                    with h5py.File(os.path.join(self.folder, self.centerline_output), 'r') as f:
                        data = f[f'timeseries/{field}/{int(t)}']
                        s = data[self.hx[2]:-self.hx[2], :, :] # (z, y, x_local)
                        s = s.transpose(2, 1, 0)               # (x_local, y, z)
                    # linear interpolation of x and y points is the same as averaging the 4 grid points in file
                    if field == 'u': # because raw data is fcc
                        s = s[-1, :, :]
                    if field == 'v': # because raw data is cfc
                        s = s[:, -1, :]
                    if field == 'w': # because raw data is ccf
                        s = velocities_to_center(s, axis=-1)
                    s = np.mean(s, axis=(0, 1))
                    s_centerline[it, :] = s
        else: # no centerline files at all — need to extract from field files
            if steps is None:
                steps = self.t_save
            steps = np.atleast_1d(steps)
            s_centerline = np.empty((len(steps), self.nx[2]))
            for it, t in enumerate(steps):
                s = self.field_slice(field, steps = t)
                # linear interpolation of x and y points is the same as averaging the 4 grid points in file
                if field == 'u': # because raw data is fcc
                    s = s[:-1, :, :]
                    s = np.mean(s[self.nx[1]//2:self.nx[1]//2+2, :], axis=0) 
                if field == 'v': # because raw data is cfc
                    s = s[:, :-1, :]
                    s = np.mean(s[self.nx[1]//2:self.nx[1]//2+2, :], axis=0) 
                if field == 'w': # because raw data is ccf
                    s = velocities_to_center(s, axis=-1)
                    s = np.mean(s[self.nx[1]//2:self.nx[1]//2+2, :], axis=0) 
                else: # because raw data is ccc
                    s = np.mean(s[self.nx[1]//2:self.nx[1]//2+2, :], axis=0) 
                s_centerline[it, :] = s.squeeze()

        if field == 'u' and self.u_s is not None:
            s_centerline = s_centerline - self.u_s
        return s_centerline.squeeze()

    #--------------------------------------------------------------------#
    #                                                                    #
    #                TURBULENT AND PLUME STATISTICS                      #
    #              *load in or calculate if necessary*                   #
    #                                                                    #
    #--------------------------------------------------------------------#
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
    def load_buoyancy_small(self, file = 'buoyancy_profile.h5', steps=None):
        g = 9.80665
        buoyancy_file = os.path.join(self.folder, file)
        if os.path.exists(buoyancy_file) and not self.centerline and not self.averaging: # the buoyancy file exists and no centerline or averaging files exist
            with h5py.File(buoyancy_file, 'r') as f:
                b_avg = f['b_avg'][()]
                if steps is None:
                    steps = b_avg.shape[0]
                b_avg = b_avg[:steps, :] 
                b_rms = f['b_rms'][:steps, :] 
                b_centerline = f['centerline/b'][:steps, :] 
                b_fluc_centerline = f['centerline/b_fluc'][:steps, :] 
            return b_avg, b_rms, b_centerline, b_fluc_centerline
        if os.path.exists(buoyancy_file) and self.centerline and self.averaging:
            with h5py.File(buoyancy_file, 'r') as f:
                b_rms = f['b_rms'][...]
            if self.centerline_file is None: # that means it was just created in the script
                self.centerline_file = [f for f in self.folder if (f.endswith('.h5') and f.startswith('centerline'))][0]
            with h5py.File(os.path.join(self.folder, self.centerline_file), 'r') as f:
                T_centerline = f[f'centerline/T'][...]
                b_centerline =  g * self.alpha * (T_centerline - self.T0)
                if self.salinity:
                    S_centerline = f[f'centerline/S'][...]
                    b_centerline += - g * self.beta * S_centerline
            T_xy = self.load_averages('T')
            b_avg = g * self.alpha * (T_xy - self.T0)
            if self.salinity:
                S_xy = self.load_averages('S')
                b_avg += - g * self.beta * S_xy
                del S_xy, S_centerline
            del T_xy, T_centerline
            b_fluc_centerline = b_centerline - b_avg
            return b_avg, b_rms, b_centerline, b_fluc_centerline
        else: # collect from field data
            if steps is None:
                steps = self.t_save
            T = self.lazy_field('T', steps=steps).compute()
            b_profile = g * self.alpha * (T - self.T0)
            if self.salinity:
                S = self.lazy_field('S', steps=steps).compute()
                b_profile += - g * self.beta * S
                del S
            b_avg = np.mean(b_profile, axis=(1, 2))
            b_fluc = b_profile - b_avg[:, None, None, :]
            b_rms = np.mean(b_fluc**2, axis=(1, 2))**0.5
            b_centerline = vertical_line(b_profile, x = self.x, y = self.y)
            b_fluc_centerline = vertical_line(b_fluc, x = self.x, y = self.y)
            del T, b_profile, b_fluc
            return b_avg, b_rms, b_centerline, b_fluc_centerline

    # ------------------------- AVERAGES -------------------------------- #
    def load_temporal_averages(self, file_path = None, contour_bound = 0.05):
        if file_path is None:
            file_path = self.bin_file

        fname = os.path.join(self.folder, file_path)

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
    def load_S_temporal_avg(self, file_path = None):
        """
        Loads contour temporal averages.
        """
        if file_path is None:
            file_path = self.bin_file

        fname = os.path.join(self.folder, file_path)

        with h5py.File(fname, 'r') as f:
            S = f['contour temporal averages/S'][()]
        return S
    def load_averages(self, field, steps=None):
        if self.averaging:
            if steps is None:
                steps = self.t_save_avg
            steps = np.atleast_1d(steps)
            field_avg = np.empty((len(steps), self.nx[2])) # (Nt, Nz)
            fname = os.path.join(self.folder, self.averaging_file)
            f = h5py.File(fname, 'r') 
            for it, t in enumerate(steps):
                dset = f[f'timeseries/{field}_avg/{int(t)}']
                if self.halos:
                    dset = dset[self.hx[2] : -self.hx[2]]
                field_avg[it, :] = np.squeeze(dset)/self.Nranks
            f.close()
        else: # calculate from field files
            if steps is None:
                steps = self.t_save
            steps = np.atleast_1d(steps)
            field_data = self.lazy_field(field, steps) # (Nt, Nx, Ny, Nz)
            field_avg = da.mean(field_data, axis=(1, 2)).compute() # (Nt, Nz)
        return field_avg
    # ------------------------- BINNING --------------------------------- #
    def load_binning_var(self, field):
        """
        Loads binning. [nr, nz, nt]
        """
        if self.binning:

            fname = os.path.join(self.folder, self.bin_file)
            opt = 'ccc/'+field

            with h5py.File(fname, 'r') as f:
                a = f[opt][()]

            return a
        else:
            raise FileNotFoundError("Binning file not found in folder. Run oceananigans_setup.py with binning enabled to generate this file.")
    def loading_bin_contours(self, contour = 0.05):
        """
        Loads binning radius.
        """
        if self.salinity:
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
        else:
            raise ValueError("Salinity needs to be a tracer in oreder to have said contour.")
    def loading_bin_radius(self):
        """
        Loads binning radius.
        """

        fname = os.path.join(self.folder, self.bin_file)

        with h5py.File(fname, 'r') as f:
            r = f[f'ccc/dimensions/r_bin'][()]
        return r
    
    # ------------------------ FLUCTUATIONS ----------------------------- #
    def load_fluc(self, field, file = 'fluctuations.h5'):
        """
        Loads fluctuation variables.
        """

        fname = os.path.join(self.folder, file)
        opt = 'fluctuations/'+field+'_fluc'
        with h5py.File(fname, 'r') as f:
            a = f[opt][()]

        return a

    # ------------------------ RMS -------------------------------------- #
    def load_rms(self, field, file = 'fluctuations.h5'):
        """
        Loads velocity RMS 
        """
        fname = os.path.join(self.folder, file)
        opt = 'rms/'+field
        with h5py.File(fname, 'r') as f:
            a = f[opt][()]
        return a

    # ------------------------ PLANE SLICE ------------------------------ #
    def load_plane_var(self, field, loc=0.0, file='plane_slice.h5'):
        """
        Loads plane slice variables.
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
            
            a = yz_group[matching_key][field][()]
        return a