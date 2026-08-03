import os
import numpy as np

from reader import OceananigansData
from diagnostics import comparison_info

from plotting_general import (
    create_video,
    plot_ranges, 
    plot_format, 
    comparison_plot_opt)

from plotting_tracer_convergence import (
    plot_tracer_slice_comparison,
    plot_tracer_zoom_comparison,
    plot_turbulence_convergence,
    plot_salinity_convergence,
    plot_temperature_convergence)

# ==========================================================
# FLAGS
# ==========================================================

plot_tracer_slice = True
plot_tracer_zoom = True
plot_bintracer_zoom = False
plot_turbulence = True
plot_salinity = True
plot_temperature = False

video = True

# ==========================================================
# PARAMETERS
# ==========================================================
T0 = 25
contour = 0.005
S_tol = 1e-6

# ==========================================================
# COMPARISON CASES
# ==========================================================
universal_folder = '/Users/annapauls/Documents/TESLa /Simulations/Oceananigans/dense plume/salinity and temperature/version109/res testing/square inlet/open BC/no SGS/default WENO/bottom PA/'
variations = "else"
if variations != "else":
    cases_info = comparison_info(variations, universal_folder=universal_folder)

    case_names = cases_info["case_names"]
    folder_names = cases_info["folder_names"]
    num_cases = cases_info["num_cases"]
    fig_folder = os.path.join(cases_info["fig_folder"], "convergence")
    F_s = cases_info["F_s"]
    mld = cases_info["mld"]
    dTdz = cases_info["dTdz"]
else:
    folder_names = ['dx2', 'horizontal resolution/dx1', 'horizontal resolution/dx05']#['dz2', 'dz1']#, 'dz05']#['dx2', 'dx1']#, 'dx05']#, 'dx025']#

    case_names =[r'$\Delta x = \Delta y = \Delta z = 2.0$', r'$\Delta x = \Delta y = 1.0$ $ \Delta z = 2.0$', r'$\Delta x = \Delta y = 0.5$ $ \Delta z = 2.0$']#[r'$\Delta x = 2.0$', r'$\Delta x = 1.0$', r'$\Delta x = 0.5$']#, r'$\Delta x = 0.25$']#


    num_cases = len(folder_names)
    fig_folder = os.path.join(universal_folder, 'horizontal resolution', 'comparison figures', 'convergence')
    F_s = 0.1 * np.ones(num_cases)
    mld = 60 * np.ones(num_cases)
    dTdz = 0.01 * np.ones(num_cases)

os.makedirs(fig_folder, exist_ok=True)
# ==========================================================
# READERS
# ==========================================================
readers = []

for n, folder in enumerate(folder_names):
    has_salinity = F_s[n] != 0
    readers.append(OceananigansData(os.path.join(universal_folder, folder), salinity=has_salinity, Sval=0.1))

# ==========================================================
# MODEL INFORMATION
# ==========================================================
if plot_bintracer_zoom:
    r = []
x = []
y = []
z = []
z_plot = []
lx = np.empty((3, num_cases), dtype=object)
nx = np.empty((3, num_cases), dtype=object)

time = []

nt_min = np.inf

for n, reader in enumerate(readers):
    x.append(reader.x)
    y.append(reader.y)
    z.append(reader.z)
    z_plot.append(reader.zf[1:])
    lx[:, n] = reader.lx
    nx[:, n] = reader.nx
    if plot_bintracer_zoom:
        r.append(reader.r)

    time.append(reader.t)

    nt_min = min(nt_min, reader.nt)

nt_min = int(nt_min)

if plot_salinity or plot_temperature or plot_turbulence:
    color_opt, line_opt, marker_opt = comparison_plot_opt(num_cases, markers = True)
    nz = np.array([reader.nx[2] for reader in readers])
    marker_iter = [int(nz_loc/min(nz)*5) for nz_loc in nz]
    plot_line_opt = [color_opt, marker_opt, marker_iter]

# ==========================================================
# DATA STORAGE
# ==========================================================

S_plane = []
S_bin = []

u_rms = []
v_rms = []
w_rms = []

bw_fluc = []

S_avg = []
S_center = []

r_tracer = []

T_avg = []
T_center = []

T_fluc_center = []

# ==========================================================
# LOAD DATA
# ==========================================================

for n, reader in enumerate(readers):

    print(f"Loading case: {case_names[n]}")

    # ------------------------------------------------------
    # vertical tracer slice
    # ------------------------------------------------------

    if plot_tracer_slice or plot_tracer_zoom:

        Stemp = reader.load_plane_var("S")

        Stemp[Stemp < S_tol] = S_tol

        S_plane.append(Stemp)

    if plot_bintracer_zoom:
        S_rz = reader.load_binning_var('S')
        S_bin.append(S_rz)
    # ------------------------------------------------------
    # turbulence statistics
    # ------------------------------------------------------

    if plot_turbulence:

        u_rms.append(reader.load_rms("u"))
        v_rms.append(reader.load_rms("v"))
        w_rms.append(reader.load_rms("w"))

        bw_fluc.append(reader.load_fluc("bw"))

    # ------------------------------------------------------
    # salinity statistics
    # ------------------------------------------------------

    if plot_salinity:
        if reader.centerline:
            S_center.append(reader.field_centerline("S")[::100, :])
        else:
            S_center.append(reader.field_centerline("S"))
        if reader.averaging:
            S_avg.append(reader.load_averages("S")[::100, :])
        else:
            S_avg.append(reader.load_averages("S"))

        r_tracer.append(reader.loading_bin_contours(contour=contour))

    # ------------------------------------------------------
    # temperature statistics
    # ------------------------------------------------------

    if plot_temperature:
        if reader.centerline:
            Tcenter = reader.field_centerline("T")[::100, :]
        else:
            Tcenter = reader.field_centerline("T")
        if reader.averaging:
            Tavg = reader.load_averages("T")[::100, :]
        else:
            Tavg = reader.load_averages("T")

        T_avg.append(Tavg)
        T_center.append(Tcenter)
        T_fluc_center.append(Tcenter - Tavg)

# ==========================================================
# PLOTTING
# ==========================================================

plot_format()
ranges = plot_ranges(lz = 96, mld = np.max(mld), T0 = T0, dTdz = np.max(dTdz), C = 0.1, C_tol = S_tol)
ranges['Tracer'] =[S_tol, 0.1]
ranges['S_avg'] = [0, 1*10**(-3)]
ranges['T'] = [T0-0.7, T0 + 0.05]
ranges['T_fluc'] = [-0.5, 0.5]
ranges['vel_rms'] = [0, 0.8*10**(-2)]
ranges['bw_fluc'] = [-5*10**(-8), 5*10**(-8)]
ranges['plume_radius'] = [0, np.max(lx[:1, :])/2]

time_min = min(time, key=len)

for it in range(nt_min):
    # ------------------------------------------------------
    # tracer slice
    # ------------------------------------------------------

    if plot_tracer_slice:

        tracer_slice_dir = (plot_tracer_slice_comparison(time_min[it], it, case_names, ranges,
                y, z_plot, [S_plane[n][it] for n in range(num_cases)], 
                readers[0].Sval,
                fig_folder))

    # ------------------------------------------------------
    # tracer zoom
    # ------------------------------------------------------

    if plot_tracer_zoom:

        tracer_zoom_dir = (plot_tracer_zoom_comparison(time_min[it], it, case_names, ranges,
                y, z_plot, 
                [S_plane[n][it] for n in range(num_cases)], 
                readers[0].Sval,
                fig_folder, ylim = (-10, 10), zlim = (-20, 0), ))
    # ------------------------------------------------------
    # tracer bin zoom
    # ------------------------------------------------------

    if plot_bintracer_zoom:
        bin_zoom_dir = (plot_tracer_zoom_comparison(time_min[it], it, case_names, ranges,
                r, z_plot, 
                [S_bin[n][:, :, it] for n in range(num_cases)], 
                readers[0].Sval,
                fig_folder, ylim = (0, 10), zlim = (-10, 0), 
                binning = True))

    # ------------------------------------------------------
    # turbulence
    # ------------------------------------------------------

    if plot_turbulence:

        turbulence_dir = (plot_turbulence_convergence(time_min[it], it, case_names, ranges, plot_line_opt, z_plot, [u_rms[n][it] for n in range(num_cases)],[v_rms[n][it] for n in range(num_cases)],[w_rms[n][it] for n in range(num_cases)],[bw_fluc[n][it] for n in range(num_cases)],
                fig_folder))

    # ------------------------------------------------------
    # salinity
    # ------------------------------------------------------

    if plot_salinity:

        salinity_dir = (plot_salinity_convergence(time_min[it], it, case_names, ranges, plot_line_opt, z_plot, [S_avg[n][it] for n in range(num_cases)],[S_center[n][it] for n in range(num_cases)],[r_tracer[n][:, it] for n in range(num_cases)],
                contour,
                fig_folder))

    # ------------------------------------------------------
    # temperature
    # ------------------------------------------------------

    if plot_temperature:

        temperature_dir = (plot_temperature_convergence(time_min[it], it, case_names, ranges, plot_line_opt, z_plot, [T_avg[n][it] for n in range(num_cases)],[T_fluc_center[n][it] for n in range(num_cases)],
                fig_folder))

    print(f"Plotted frame {it+1}/{nt_min}")

# ==========================================================
# VIDEOS
# ==========================================================

if video:
    if plot_tracer_slice:
        create_video(tracer_slice_dir, fig_folder, "comparison", "tracer_slice")

    if plot_tracer_zoom:
        create_video(tracer_zoom_dir, fig_folder, "comparison", "tracer_zoom")

    if plot_bintracer_zoom:
        create_video(bin_zoom_dir, fig_folder, "comparison", "tracer_bin_zoom")

    if plot_turbulence:
        create_video(turbulence_dir, fig_folder, "comparison", "turbulence")

    if plot_salinity:
        create_video(salinity_dir, fig_folder, "comparison", f"salinity-{contour}S0")

    if plot_temperature:
        create_video(temperature_dir, fig_folder, "comparison", "temperature")
