import os
import imageio.v2 as imageio

from plotting_functions import create_video


folder = '/Users/annapauls/Library/CloudStorage/OneDrive-UCB-O365/CU-Boulder/TESLa/Carbon Sequestration/Simulations/Oceananigans/NBP/salinity and temperature/no noise small square inlet/beta = default S = 20'
outdir = os.path.join(folder, 'NBP buoyancy analysis/')
fig_folder = folder
name = 'Ri-NBP-'
plot_type = 'buoyancy_analysis'
create_video(outdir, fig_folder, name, plot_type)
