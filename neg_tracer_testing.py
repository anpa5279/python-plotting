import os
import re
import numpy as np
import math

from plotting_functions import plot_format, plot_ranges, create_video, plot_momentum_plume, plot_tracer_plume, vert_plane_slices, xy_plane_slices, buoyancy_analysis_plot, turb_stats_plot

from reader import OceananigansData
# Set up folder and simulation parameters
folder1 = '/Users/annapauls/Library/CloudStorage/OneDrive-UCB-O365/CU-Boulder/TESLa/Carbon Sequestration/Simulations/Oceananigans/NBP/salinity and temperature/no noise circle inlet/S0 = 0.1 dTdz = 0.01 MLD = 60'
folder2 = '/Users/annapauls/Library/CloudStorage/OneDrive-UCB-O365/CU-Boulder/TESLa/Carbon Sequestration/Simulations/Oceananigans/NBP/salinity and temperature/no noise circle inlet/S0 = 0.1 dTdz = 0.01 MLD = 60 WENO mod'
folder3 = '/Users/annapauls/Library/CloudStorage/OneDrive-UCB-O365/CU-Boulder/TESLa/Carbon Sequestration/Simulations/Oceananigans/NBP/salinity and temperature/no noise circle inlet/S0 = 0.1 dTdz = 0.01 MLD = 60 WENO mod callback'



reader1 = OceananigansData(folder1)
reader2 = OceananigansData(folder2)
reader3 = OceananigansData(folder3)
reader1.load_grid()
reader2.load_grid()
reader3.load_grid()
reader1.load_time()
reader2.load_time()
reader3.load_time()

domain = math.prod(reader1.nx)

for it, t in enumerate(reader1.t_save):
    print('Step:', it)
    # Load data from files
    S = reader1.lazy_field('S', reader1.t_save[it])
    S_sum = np.sum(S<0)
    S_neg = np.mean(S[S<0])
    print('\t default WENO: S mean:', np.mean(S), '\t # negatives:', S_sum, '\t % of domain:', S_sum/domain*100, '\t S min:', np.min(S), '\t S neg mean:', S_neg)

    Smod = reader2.lazy_field('S', t)
    Smod_sum = np.sum(Smod<0)
    Smod_neg = np.mean(Smod[Smod<0])
    print('\t modified WENO: S mean:', np.mean(Smod), '\t # negatives:', Smod_sum, '\t % of domain:', Smod_sum/domain*100, '\t S min:', np.min(Smod), '\t S neg mean:', Smod_neg)

    Scall = reader3.lazy_field('S', t)
    Scall_sum = np.sum(Scall<0)
    Scall_neg = np.mean(Scall[Scall<0])
    print('\t callback with mod WENO: S mean:', np.mean(Scall), '\t # negatives:', Scall_sum, '\t % of domain:', Scall_sum/domain*100, '\t S min:', np.min(Scall), '\t S neg mean:', Scall_neg)
