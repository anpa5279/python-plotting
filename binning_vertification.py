import os
import numpy as np
import matplotlib.pyplot as plt

from diagnostics import azimuthal_avg
from plotting_general import plot_format, comparison_plot_opt

outdir = 'figures and videos/'
os.makedirs(outdir, exist_ok=True)

# domain information
lx = 200, 200
dx = 0.5, 0.5 
x = np.linspace(-lx[0]/2, lx[0]/2, int(lx[0]/dx[0])+1)
y = np.linspace(-lx[1]/2, lx[1]/2, int(lx[1]/dx[1])+1)

X, Y = np.meshgrid(x, y)
n_gauss = [len(x), len(y)]

# create gaussian 
def gaussian(a, x, y, sigma=20):
    return a*np.exp(-((x**2 + y**2) / (2 * sigma**2)))
mag = 1
gaus = gaussian(mag, X, Y)
# binning 
d_min = np.max(dx)
dr_bins = [2*d_min, 4*d_min, 8*d_min, 16*d_min]
r = []
binned = []
for dr in dr_bins:
    r_temp, binned_temp = azimuthal_avg(gaus, X, Y, dx_scale=dr, return_r = True)
    r.append(r_temp)
    binned.append(binned_temp)

# plotting
plot_format()
color_opt, line_opt = comparison_plot_opt(len(binned)+1)
fig, axes = plt.subplots(1, 3, figsize=(22, 7), gridspec_kw={'width_ratios': [0.75, 0.75, 1]})
axes = axes.ravel()
for i, bin in enumerate(binned):
    axes[0].scatter(r[i], bin, color=color_opt[i + 1], label=f'binning, dr = {dr_bins[i]} m', marker='x')
axes[0].plot(x[n_gauss[0]//2:], gaus[n_gauss[0]//2:, n_gauss[1]//2], color=color_opt[0], label=r'A$\cdot$exp$(-\frac{(x^2+y^2)}{2\sigma^2})$')
axes[0].set_title("Field vs Binning")
axes[0].set_xlim(0, x[-1])
axes[0].legend()
axes[0].set_xlabel("r [m]")
axes[0].set_ylim(0, mag*1.1)
axes[0].set_ylabel(r'gaussian, A$\cdot$exp$(-\frac{(x^2+y^2)}{2\sigma^2})$')

for i, bin in enumerate(binned):
    error = np.abs(bin - gaussian(mag, r[i], 0)) / mag * 100
    axes[1].scatter(r[i], error, color=color_opt[i + 1], label=f'binning, dr = {dr_bins[i]} m', marker='x')
axes[1].set_title("Error")
axes[1].set_xlim(0, x[-1])
axes[1].legend()
axes[1].set_xlabel("r [m]")
axes[1].set_ylim(0, 10)
axes[1].set_ylabel(r"$\frac{\text{bin - gaussian}}{A}$[%]")

im = axes[2].imshow(gaus, extent=[x[0], x[-1], y[0], y[-1]], interpolation ='none', aspect='auto')
fig.colorbar(im, ax = axes[2], label='Variable magnitude', shrink=0.8)
axes[2].set_title(r'A$\cdot$exp$(-\frac{(x^2+y^2)}{2\sigma^2})$')
axes[2].set_xlim(x[0], x[-1])
axes[2].set_xlabel("x [m]")
axes[2].set_ylim(y[0], y[-1])
axes[2].set_ylabel("y [m]")
axes[2].set_aspect('equal')

# --- Save Frame ---
frame_path = os.path.join(outdir, f"binning_verification_gaus.svg")
plt.savefig(frame_path)
plt.close(fig)
