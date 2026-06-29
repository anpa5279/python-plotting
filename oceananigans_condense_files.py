import os
import numpy as np
import h5py
import math

from reader import OceananigansData
from interpolation import velocities_to_center

# flags
run_prep = False
post_process_prep = True

salinity = True 

# Set up folder and simulation parameters
folder = '/glade/derecho/scratch/apauls/outputs/version109/square-inlet/erf/dz0125/diff-distributed'

# physical parameters
T0 = 25.0

reader = OceananigansData(folder, salinity = salinity)
# grid info
nx = reader.nx
nt = reader.nt
dx = reader.dx
lx = reader.lx
t_save = reader.load_time()[-1]
t_last = t_save[-1]

# prepping the file to use as IC in pickup run
if run_prep:
    file_name = 'last_time_step.jld2'
    file_path = os.path.join(folder, file_name)

    print(f"Reading data from {folder}")

    # write fields to one file
    u = reader.lazy_field('u', steps = t_last)#, transpose = False)
    u = velocities_to_center(u, axis = -3).transpose(2, 1, 0)
    with h5py.File(file_path, 'a') as f:
        if 'u' in f:
            del f['u']
        f.create_dataset('u', data=u)
    del u

    v = reader.lazy_field('v', steps = t_last)#, transpose = False)
    v = velocities_to_center(v, axis = -2).transpose(2, 1, 0)
    with h5py.File(file_path, 'a') as f:
        if 'v' in f:
            del f['v']
        f.create_dataset('v', data=v)
    del v

    w = reader.lazy_field('w', steps = t_last)#, transpose = False)
    w = velocities_to_center(w, axis = -1).transpose(2, 1, 0)
    with h5py.File(file_path, 'a') as f:
        if 'w' in f:
            del f['w']
        f.create_dataset('w', data=w)
    del w

    T = reader.lazy_field('T', steps = t_last, transpose = False)
    with h5py.File(file_path, 'a') as f:
        if 'T' in f:
            del f['T']
        f.create_dataset('T', data=T)
    del T

    if salinity:
        S = reader.lazy_field('S', steps = t_last, transpose = False)
        with h5py.File(file_path, 'a') as f:
            if 'S' in f:
                del f['S']
            f.create_dataset('S', data=S)

        del S

# combining pickup files into one file for post-processing
if post_process_prep:
    fields_pickup_files = [f for f in os.listdir(folder) if (f.startswith('fields_pickup') and f.endswith('.jld2'))]
    avg_pickup_files = [f for f in os.listdir(folder) if (f.startswith('xy_avg_pickup') and f.endswith('.jld2'))]
    center_pickup_files = [f for f in os.listdir(folder) if (f.startswith('centerline_') and f.endswith('.jld2'))]