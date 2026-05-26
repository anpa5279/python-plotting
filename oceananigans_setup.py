import os
import numpy as np

from reader import OceananigansData
from diagnostics import compute_temporal_averages, write_temporal_averages, compute_temporal_radius_avg

folder = '/Users/annapauls/Documents/TESLa /Simulations/Oceananigans/dense plume/salinity and temperature/no noise circle inlet/domain resolution testing/horizontal resolution/sj0.1-mld60-dTdz0.01-lx320-nx384'

write_temporal_avg = True
salinity = True

output_file = os.path.join(folder, 'interp_temporal_averages.h5')

g = 9.80665
T0 = 25.0
rho0 = 1026.0

reader = OceananigansData(folder, salinity = salinity)
reader.load_grid()
time, t_save = reader.load_time()

data_temp = compute_temporal_averages(reader)
# compute radius of plume 
#r = compute_temporal_radius_avg(reader, data_temp['S_value'])
data = {
    'S_avg': data_temp['S_avg'],
    'T_avg': data_temp['T_avg'],
    'u_rms': data_temp['u_rms'],
    'v_rms': data_temp['v_rms'],
    'w_rms': data_temp['w_rms'],
    'S_center': data_temp['S_center'],
    'T_center': data_temp['T_center'],
    'w_center': data_temp['w_center'],
    'S_value': data_temp['S_value'],
    'w_value': data_temp['w_value'], 
    #'radius_tracer': r
}
write_temporal_averages(output_file, data)