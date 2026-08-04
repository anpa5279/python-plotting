import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from reader import OceananigansData
from plotting_general import plot_format, create_video, comparison_plot_opt, plot_ranges
from diagnostics import azimuthal_avg
"""
    what happens when you set neagtive tracer values to 0?
        tracer distribution?
        buoyancy profiles?
"""
# ==========================================================
# FLAGS
# ==========================================================
plot_tracer_profile = True
plot_buoyancy_profile = True
video = True

salinity = True
with_halos = False

# ==========================================================
# PARAMETERS
# ==========================================================
hml = 60
g = 9.80665
Sval = 0.1
contours = np.array([0.001, 0.005, 0.01, 0.05])

# ==========================================================
# MODEL INFORMATION
# ==========================================================
folder = '/Users/annapauls/Documents/TESLa /Simulations/Oceananigans/dense plume/salinity and temperature/version109/square inlet/open BC/no SGS/default WENO/bottom PA/dx0125'

outdir = os.path.join(folder, 'figures')
reader = OceananigansData(folder, salinity = salinity, with_halos = with_halos, Sval = Sval)

# collecting model information for all cases
x = reader.x
y = reader.y
z = reader.z
nx = reader.nx
lx = reader.lx
dx = reader.dx
nt = reader.nt
# video or not setup
if video:
    time = reader.t
    if reader.averaging:
        time1 = reader.time_avg
else:
    time = reader.t[-1]

# ==========================================================
# LOAD DATA
# ==========================================================
S = reader.lazy_field('S').compute()
S_pos = S.copy()
S_pos[S_pos < 0] = 0
dims = (1, 2, 3)
if plot_tracer_profile:
    S_avg = np.mean(S, axis = (1, 2))
    S_pos_avg = np.mean(S_pos, axis = (1, 2))
    ncirc = min(nx[0], nx[1])//2      # full circular shells
    dx_scale = np.max(reader.dx[:-1]) # not including dz
    r = np.arange(dx[0]/2, lx[0]/2, dx_scale)
    X, Y = np.meshgrid(x, y, indexing='ij')
    S_pos_rz = np.zeros((nx[0]//2, nx[2], nt))
    for it in range(nt):
        for k in range(nx[2]):
            S_pos_rz[:, k, it] = azimuthal_avg(S_pos[it, :, :, k], X[:, :], Y[:, :], dx_scale=dx_scale)
    r_tracer = np.zeros((len(contours), nx[2], len(time)))
    r_pos_tracer = np.zeros((len(contours), nx[2], len(time)))
    for n, contour in enumerate(contours):
        r_tracer[n, :, :] = reader.loading_bin_contours(contour=contour)
        level = Sval * contour
        for it in range(nt):
            radius_tracer = np.zeros(nx[2])
            for k in range(nx[2]):
                S_radial = S_pos_rz[:ncirc, k, it]

                # Guard 1: level not reached at this depth/time
                if np.max(S_radial) < level:
                    continue

                # Orient so r is ascending and S trends downward outward
                if S_radial[0] < S_radial[-1]:
                    S_radial = S_radial[::-1]
                    r_search = r[::-1]
                else:
                    r_search = r

                # Guard 2: trim to the bracketing region around the crossing
                above = np.where(S_radial >= level)[0]
                if len(above) == 0:
                    continue
                i_last = above[-1]
                i_end = min(i_last + 2, len(S_radial))
                S_trimmed = S_radial[:i_end]
                r_trimmed = r_search[:i_end]

                if len(S_trimmed) < 2:
                    radius_tracer[k] = r_trimmed[-1] if len(S_trimmed) else 0.0
                    continue

                # If we never drop below `level` in the trimmed window,
                # take the last (outermost) sample as the best estimate
                above_mask = S_trimmed >= level
                if above_mask.all():
                    radius_tracer[k] = r_trimmed[-1]
                    continue

                # Find the first index where S drops below level; the
                # crossing is bracketed by (i1, i2) = (last above, first below)
                idx_below = np.where(~above_mask)[0]
                i2 = idx_below[0]
                i1 = i2 - 1

                if i1 < 0:
                    # Level exceeded already at the first trimmed point
                    radius_tracer[k] = r_trimmed[0]
                    continue

                S1, S2 = S_trimmed[i1], S_trimmed[i2]
                r1, r2 = r_trimmed[i1], r_trimmed[i2]

                if S1 == S2:
                    radius_tracer[k] = r1
                else:
                    frac = (level - S1) / (S2 - S1)
                    radius_tracer[k] = r1 + frac * (r2 - r1)

            # Sanity clip: radius can never be negative or exceed grid extent
            radius_tracer = np.clip(radius_tracer, 0.0, r.max())
            r_pos_tracer[n, :, it] = radius_tracer
    del S_pos_rz
if plot_buoyancy_profile:
    T = reader.lazy_field('T').compute()
    b = g * reader.alpha * (T - reader.T0) - g * reader.beta * S
    b_pos = g * reader.alpha * (T - reader.T0) - g * reader.beta * S_pos
    del T, S, S_pos
    b_avg = np.mean(b, axis = (1, 2))
    b_pos_avg = np.mean(b_pos, axis = (1, 2))
    b_rms = np.mean((b - b_avg[:, None, None, :])**2, axis = (1, 2))**0.5
    b_pos_rms = np.mean((b_pos - b_pos_avg[:, None, None, :])**2, axis = (1, 2))**0.5
    del b, b_pos
print("finished loading data")

# ==========================================================
# PLOTTING
# ==========================================================
ranges = plot_ranges()
ranges['S_avg'] = [0.0, 6*10**-4]
ranges['b_avg'] = [-1.5*10**(-3), 1.0*10**(-5)]
ranges['b_rms'] = [0, 4*10**(-5)]
ranges['percent'] = [-1, 1]
color_opt, line_opt  = comparison_plot_opt(2)
plot_format()
os.makedirs(outdir, exist_ok=True)
if plot_tracer_profile:
    tracer_dir = os.path.join(outdir, 'tracer_profile')
    os.makedirs(tracer_dir, exist_ok=True)
    for it in range(nt):
        fig, axes = plt.subplots(2, len(contours)+1, figsize=(4*(len(contours)+1), 8), sharey =True)
        axes[0, 0].plot(S_avg[it, :], z, color=color_opt[0], label=r'S$_{data}$')
        axes[0, 0].plot(S_pos_avg[it, :], z, color=color_opt[1], label=r'S$_{positive}$')
        axes[0, 0].set_title(r'S$_{xy}$')
        axes[0, 0].set_xlabel('[g/kg]')
        axes[0, 0].legend(loc = 'lower right')
        axes[0, 0].set_xlim(ranges['S_avg'])

        diff = (S_pos_avg[it, :] - S_avg[it, :])/Sval
        axes[1, 0].plot(diff, z, color=color_opt[1])
        axes[1, 0].set_title(f'Percent Difference Tracer contour {contour}')
        axes[1, 0].set_xlabel(r'100$\cdot \frac{S_{positive} - S_{data}}{S_{0}}$[%]')
        axes[1, 0].set_xlim(ranges['percent'])

        for n, contour in enumerate(contours):
            axes[0, n + 1].plot(r_tracer[n, :, it], z, color=color_opt[0], label=r'S$_{data}$')
            axes[0, n + 1].plot(r_pos_tracer[n, :, it], z, color=color_opt[1], label=r'S$_{positive}$')
            axes[0, n + 1].set_title(rf'Tracer contour {contour}$\cdot \text{{S}}_{{0}}$')
            axes[0, n + 1].set_xlabel('Radius [m]')
            axes[0, n + 1].set_xlim([0, r.max()])

            diff = (r_pos_tracer[n, :, it] - r_tracer[n, :, it])/r_tracer[n, :, it]
            axes[1, n + 1].plot(diff, z, color=color_opt[1])
            axes[1, n + 1].set_title(f'Percent Difference Tracer contour {contour}')
            axes[1, n + 1].set_xlabel(r'100$\cdot \frac{r_{positive} - r_{data}}{r_{data}}$[%]')
            axes[1, n + 1].set_xlim(ranges['percent'])
        axes[0, 0].set_ylabel('Depth [m]')
        axes[1, 0].set_ylabel('Depth [m]')
        frame_path = os.path.join(tracer_dir, f'tracer_profile_{it:04d}.png')
        plt.savefig(frame_path)
        plt.close(fig)
if plot_buoyancy_profile:
    b_dir = os.path.join(outdir, 'b_profile')
    os.makedirs(b_dir, exist_ok=True)
    for it in range(nt):
        fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharey =True)
        axes[0, 0].plot(b_avg[it, :], z, color=color_opt[0], label=r'S$_{data}$')
        axes[0, 0].plot(b_pos_avg[it, :], z, color=color_opt[1], label=r'S$_{positive}$')
        axes[0, 0].set_title('Buoyancy Profile')
        axes[0, 0].set_xlabel(r'[$\text{m}/\text{s}^2$]')
        axes[0, 0].set_ylabel('Depth [m]')
        axes[0, 0].legend(loc = 'lower right')
        axes[0, 0].set_xlim(ranges['b_avg'])

        diff = (b_pos_avg[it, :] - b_avg[it, :])/(g*reader.beta*Sval)
        axes[1, 0].plot(diff, z, color=color_opt[1])
        axes[1, 0].set_title(r'b$_{xy}$ Percent Difference')
        axes[1, 0].set_xlabel(r'100$\cdot \frac{b_{positive} - b_{data}}{b_{0}}$[%]')
        axes[1, 0].set_ylabel('Depth [m]')
        axes[1, 0].set_xlim(ranges['percent'])

        axes[0, 1].plot(b_rms[it, :], z, color=color_opt[0])
        axes[0, 1].plot(b_pos_rms[it, :], z, color=color_opt[1])
        axes[0, 1].set_title('RMS Buoyancy Profile')
        axes[0, 1].set_xlabel(r'[$\text{m}/\text{s}^2$]')
        #axes[0, 1].set_ylabel('Depth [m]')
        axes[0, 1].set_xlim(ranges['b_rms'])

        diff = (b_pos_rms[it, :] - b_rms[it, :])/(g*reader.beta*Sval)
        axes[1, 1].plot(diff, z, color=color_opt[1])
        axes[1, 1].set_title(r"b$_{rms}$ Percent Difference")
        axes[1, 1].set_xlabel(r"100$\cdot \frac{b_{positive} - b_{data}}{b_{0}}$[%]")
        #axes[1, 1].set_ylabel('Depth [m]')
        axes[1, 1].set_xlim(ranges['percent'])

        frame_path = os.path.join(b_dir, f'b_profile_{it:04d}.png')
        plt.savefig(frame_path)
        plt.close(fig)

# creating videos
if video:
    if plot_tracer_profile:
        create_video(tracer_dir, outdir, '', 'plot_tracer_profile')
    if plot_buoyancy_profile:
        create_video(b_dir, outdir, '', 'plot_buoyancy_profile')
