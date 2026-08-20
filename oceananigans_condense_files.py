import os
import numpy as np
import h5py

from reader import OceananigansData

# ==========================================================
# FLAGS
# ==========================================================
field_files_flag = True
high_freq_files_flag = True

salinity = True 
with_halos = False 

# ==========================================================
# READER
# ==========================================================
folder = '/Users/annapauls/Documents/Github repositories/3d_langmuir_gpu/localoutputs/scheme-tests/longer/WENO9/dx1.0/testing'

reader = OceananigansData(folder, salinity = salinity, with_halos = with_halos)

# ==========================================================
# MODEL INFORMATION
# ==========================================================
# grid info
nx = reader.nx
nt = reader.nt
dx = reader.dx
lx = reader.lx
t_last = reader.t_save[-1]

if salinity:
    vars = ['u', 'v', 'w', 'T', 'S']
else:
    vars = ['u', 'v', 'w', 'T']

# combining pickup files into one file for post-processing
pickup_folder = os.path.join(folder, 'pickup')
reader_pickup = OceananigansData(pickup_folder, salinity = salinity, with_halos = with_halos)

# setting up time steps to follow
t_save_update = t_last
t_add = reader.t[-1]
if field_files_flag:
    for t_save in reader_pickup.t_save:
        if t_save != 0: # skip the first saved time step since it is already included in the main files
            t_save_update += t_save
            print(f'Adding pickup time iteration {int(t_save)} to main files as time iteration {int(t_save_update)}')
            for rank, file in enumerate(reader.files):
                with h5py.File(os.path.join(pickup_folder, reader_pickup.files[rank]), 'r') as f:
                    t = f[f'timeseries/t/{int(t_save)}'] + t_add
                with h5py.File(os.path.join(folder, file), "a") as f:
                    f.create_dataset(f'timeseries/t/{int(t_save_update)}', data = t)
                for var in vars:
                    with h5py.File(os.path.join(pickup_folder, reader_pickup.files[rank]), 'r') as f:
                        field = f[f'timeseries/{var}/{int(t_save)}'][:]

                    with h5py.File(os.path.join(folder, file), "a") as f:
                        f.create_dataset(f'timeseries/{var}/{int(t_save_update)}', data = field)

if high_freq_files_flag: # assuming centerline files and averaging files have the same time steps

    if reader.Nranks > 1:
        rank = reader.Nranks//2-1 # only need to copy one rank since averaging and centerline files are the same for all ranks
    else:
        rank = 0
    file_avg = reader.averaging_file
    file_center = reader.centerline_file

    new_file_avg = os.path.join(folder, "xy_avg.h5")
    new_file_center = os.path.join(folder, "centerline_output.h5")
    t_first_only = np.where(reader.t_save_avg <= t_last)

    for t_save in reader.t_save_avg[t_first_only]:
        print(f'For higher frequency files, adding time {int(t_save)} as time iteration {int(t_save)}')
        with h5py.File(os.path.join(pickup_folder, reader.averaging_file), 'r') as f:
            t = f[f'timeseries/t/{int(t_save)}'][()]

        with h5py.File(new_file_avg, "a") as f:
            f.create_dataset(f'timeseries/t/{int(t_save)}', data = t)
        with h5py.File(new_file_center, "a") as f:
            f.create_dataset(f'timeseries/t/{int(t_save)}', data = t)
        for var in vars:
            with h5py.File(os.path.join(pickup_folder, reader_pickup.averaging_file), 'r') as f:
                field = f[f'timeseries/{var}_avg/{int(t_save)}'][:]

            with h5py.File(new_file_avg, "a") as f:
                f.create_dataset(f'timeseries/{var}_avg/{int(t_save)}', data = field)

            with h5py.File(os.path.join(pickup_folder, reader_pickup.centerline_output), 'r') as f:
                field = f[f'timeseries/{var}/{int(t_save)}'][:]

            with h5py.File(new_file_center, "a") as f:
                f.create_dataset(f'timeseries/{var}/{int(t_save)}', data = field)

    t_save_update_freq = t_last

    for t_save in reader_pickup.t_save_avg:
        if t_save != 0: # skip the first saved time step since it is already included in the main files
            t_save_update_freq += t_save
            print(f'For higher frequency files, adding pickup time as time iteration {int(t_save_update_freq)}')
            with h5py.File(os.path.join(pickup_folder, reader_pickup.averaging_file), 'r') as f:
                t = f[f'timeseries/t/{int(t_save)}'][()] + t_add

            with h5py.File(new_file_avg, "a") as f:
                f.create_dataset(f'timeseries/t/{int(t_save_update_freq)}', data = t)
            with h5py.File(new_file_center, "a") as f:
                f.create_dataset(f'timeseries/t/{int(t_save_update_freq)}', data = t)

            for var in vars:
                with h5py.File(os.path.join(pickup_folder, reader_pickup.averaging_file), 'r') as f:
                    field = f[f'timeseries/{var}_avg/{int(t_save)}'][:]

                with h5py.File(new_file_avg, "a") as f:
                    f.create_dataset(f'timeseries/{var}_avg/{int(t_save_update_freq)}', data = field)

                with h5py.File(os.path.join(pickup_folder, reader_pickup.centerline_output), 'r') as f:
                    field = f[f'timeseries/{var}/{int(t_save)}'][:]

                with h5py.File(new_file_center, "a") as f:
                    f.create_dataset(f'timeseries/{var}/{int(t_save_update_freq)}', data = field)

