import os
import numpy as np
import h5py
import math

from reader import OceananigansData

# ==========================================================
# FLAGS
# ==========================================================
with_halos = True
salinity = True

# ==========================================================
# READER
# ==========================================================
folder = '/glade/derecho/scratch/apauls/outputs/version109/square-inlet/open-bottom-BC/AR1/dxi0125/gpu/outputs/dxi0125/'

reader = OceananigansData(folder, salinity = salinity, with_halos=with_halos, Sval=0.1)
file_path = os.path.join(reader.folder, 'tracer_test.h5')

x = reader.x
y = reader.y
z = reader.z

S = reader.lazy_field('S').compute()

dims = (1, 2, 3)
Smin = np.min(S, axis = dims)
Smax = np.max(S, axis = dims)

for it in range(S.shape[0]):
    min_loc = np.where(S[it, :, :, :] == Smin[it])
    max_loc = np.where(S[it, :, :, :] == Smax[it])
    with h5py.File(file_path, "a") as f:
        f[f'Smin locations/time_step_{it}'] = np.array([x[min_loc[0][0]], y[min_loc[1][0]], z[min_loc[2][0]]])
        f[f'Smax locations/time_step_{it}'] = np.array([x[max_loc[0][0]], y[max_loc[1][0]], z[max_loc[2][0]]])
    print(f"Time step {it}: min S = {Smin[it]} at (x, y, z) = ({x[min_loc[0][0]]}, {y[min_loc[1][0]]}, {z[min_loc[2][0]]})")
    print(f"Time step {it}: max S = {Smax[it]} at (x, y, z) = ({x[max_loc[0][0]]}, {y[max_loc[1][0]]}, {z[max_loc[2][0]]})")