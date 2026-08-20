import os
import numpy as np
import h5py
import shutil

from reader import OceananigansData

# ==========================================================
# FLAGS
# ==========================================================
field_files_flag = True
high_freq_files_flag = True

salinity = True 
with_halos = True 

# ==========================================================
# READERS
# ==========================================================
folder = '/glade/derecho/scratch/apauls/outputs/version109/square-inlet/open-bottom-BC/AR1/dxi025/weno9'

reader = OceananigansData(folder, salinity = salinity, with_halos = with_halos)
pickup_folder = os.path.join(folder, 'pickup')
reader_pickup = OceananigansData(pickup_folder, salinity = salinity, with_halos = with_halos)

outdir = os.path.join(folder, 'condensed_files')
os.makedirs(outdir, exist_ok = True)

# ==========================================================
# MODEL INFORMATION
# ==========================================================
shutil.copy2(os.path.join(folder, 'grid_info.jld2'), os.path.join(outdir, 'grid_info.jld2'))
# grid info
nx = reader.nx
nt = reader.nt
dx = reader.dx
lx = reader.lx

# setting up time steps to follow
t_last = reader.t_save[-1]
t_add = reader.t[-1]

if salinity:
    vars = ['u', 'v', 'w', 'T', 'S']
else:
    vars = ['u', 'v', 'w', 'T']


# ==========================================================
# COMBINING FILES
# ==========================================================
if field_files_flag:
    print(f'Copying field files to {outdir}')
    new_file_field = os.path.join(outdir, "fields.jld2")
    for file in reader.files:
        shutil.copy2(os.path.join(folder, file), new_file_field)

    for t_save in reader_pickup.t_save:
        if t_save != 0: # skip the first saved time step since it is already included in the main files
            print(f'Adding pickup time iteration {int(t_save)} to main files as time iteration {int(t_save + t_last)}')
            for rank, file in enumerate(reader.files):
                with h5py.File(os.path.join(pickup_folder, reader_pickup.files[rank]), 'r') as f:
                    t = f[f'timeseries/t/{int(t_save)}'] + t_add
                with h5py.File(new_file_field, "a") as f:
                    f.create_dataset(f'timeseries/t/{int(t_save + t_last)}', data = t)
                for var in vars:
                    with h5py.File(os.path.join(pickup_folder, reader_pickup.files[rank]), 'r') as f:
                        field = f[f'timeseries/{var}/{int(t_save)}'][:]

                    with h5py.File(new_file_field, "a") as f:
                        f.create_dataset(f'timeseries/{var}/{int(t_save + t_last)}', data = field)

if high_freq_files_flag: # assuming centerline files and averaging files have the same time steps

    if reader.Nranks > 1:
        rank = reader.Nranks//2-1 # only need to copy one rank since averaging and centerline files are the same for all ranks
    else:
        rank = 0
    file_avg = reader.averaging_file
    file_center = reader.centerline_output

    print(f'Copying {file_avg} and {file_center} to {outdir}')
    new_file_avg = os.path.join(outdir, "xy_avg.jld2")
    new_file_center = os.path.join(outdir, "centerline.jld2")

    t_first_only = np.where(reader.t_save_avg <= t_last)

    for t_save in reader.t_save_avg[t_first_only]:
        with h5py.File(os.path.join(folder, reader.averaging_file), 'r') as f:
            t = f[f'timeseries/t/{int(t_save)}'][()]

        with h5py.File(new_file_avg, "a") as f:
            f.create_dataset(f'timeseries/t/{int(t_save)}', data = t)
        with h5py.File(new_file_center, "a") as f:
            f.create_dataset(f'timeseries/t/{int(t_save)}', data = t)
        for var in vars:
            with h5py.File(os.path.join(folder, reader.averaging_file), 'r') as f:
                field = f[f'timeseries/{var}_avg/{int(t_save)}'][:]

            with h5py.File(new_file_avg, "a") as f:
                f.create_dataset(f'timeseries/{var}_avg/{int(t_save)}', data = field)

            with h5py.File(os.path.join(folder, reader.centerline_output), 'r') as f:
                field = f[f'timeseries/{var}/{int(t_save)}'][:]

            with h5py.File(new_file_center, "a") as f:
                f.create_dataset(f'timeseries/{var}/{int(t_save)}', data = field)

    for t_save in reader_pickup.t_save_avg:
        if t_save != 0: # skip the first saved time step since it is already included in the main files
            print(f'For higher frequency files, adding pickup time iteration {int(t_save)} as time iteration {int(t_save + t_last)}')
            with h5py.File(os.path.join(pickup_folder, reader_pickup.averaging_file), 'r') as f:
                t = f[f'timeseries/t/{int(t_save)}'][()] + t_add

            with h5py.File(new_file_avg, "a") as f:
                f.create_dataset(f'timeseries/t/{int(t_save + t_last)}', data = t)
            with h5py.File(new_file_center, "a") as f:
                f.create_dataset(f'timeseries/t/{int(t_save + t_last)}', data = t)

            for var in vars:
                with h5py.File(os.path.join(pickup_folder, reader_pickup.averaging_file), 'r') as f:
                    field = f[f'timeseries/{var}_avg/{int(t_save)}'][:]

                with h5py.File(new_file_avg, "a") as f:
                    f.create_dataset(f'timeseries/{var}_avg/{int(t_save + t_last)}', data = field)

                with h5py.File(os.path.join(pickup_folder, reader_pickup.centerline_output), 'r') as f:
                    field = f[f'timeseries/{var}/{int(t_save)}'][:]

                with h5py.File(new_file_center, "a") as f:
                    f.create_dataset(f'timeseries/{var}/{int(t_save + t_last)}', data = field)


