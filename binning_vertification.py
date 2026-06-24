import os
import numpy as np
import scipy
import matplotlib.pyplot as plt

from diagnostics import azimuthal_avg
from plotting_general import plot_format, comparison_plot_opt
show_field = False
outdir = 'figures and videos/'
os.makedirs(outdir, exist_ok=True)

# domain information
lx = 200, 200
dx = 0.125, 0.125 
x = np.linspace(-lx[0]/2, lx[0]/2, int(lx[0]/dx[0])+1)
y = np.linspace(-lx[1]/2, lx[1]/2, int(lx[1]/dx[1])+1)

X, Y = np.meshgrid(x, y)
n_gauss = [len(x), len(y)]

# binning 
d_min = np.max(dx)
dr_bins = [2*d_min, 4*d_min, 8*d_min, 16*d_min, 32*d_min, 64*d_min, 128*d_min, 256*d_min]

# create gaussian 
def gaussian(a, x, y, sigma=20):
    return a*np.exp(-((x**2 + y**2) / (2 * sigma**2)))
mag = 1
s =128*d_min
gaus = gaussian(mag, X, Y, sigma=s)
r = []
binned = []
int_area = np.zeros(len(dr_bins))
for i, dr in enumerate(dr_bins):
    r_temp, binned_temp = azimuthal_avg(gaus, X, Y, dx_scale=dr, return_r = True)
    r.append(r_temp)
    binned.append(binned_temp)
    int_area[i] = np.sum(binned_temp)*dr 
    1/2+1/(np.sqrt(2*np.pi))*np.sum(binned[-1])

# plotting
plot_format()
color_opt, line_opt = comparison_plot_opt(len(binned)+1)
if show_field:
    ncols = 4
    field = 'with_field_'
    width_opt = [0.75, 0.75, 0.75, 1]
else:
    ncols = 3
    field = ''
    width_opt = np.ones(ncols)
fig, axes = plt.subplots(1, 3, figsize=(5*ncols+1, 5), gridspec_kw={'width_ratios': width_opt})
axes = axes.ravel()
for i, bin in enumerate(binned):
    axes[0].scatter(r[i], bin, color=color_opt[i + 1], label=rf'dr/$\sigma$ = {dr_bins[i]/s}', marker='x')
axes[0].plot(x[n_gauss[0]//2:], gaus[n_gauss[0]//2:, n_gauss[1]//2], color=color_opt[0], label=r'A$\cdot$exp$(-\frac{(x^2+y^2)}{2\sigma^2})$')
axes[0].set_title("Field vs Binning")
axes[0].set_xlim(0, x[-1])
axes[0].legend()
axes[0].set_xlabel("r [m]")
axes[0].set_ylim(0, mag*1.1)
axes[0].set_ylabel(r'gaussian, A$\cdot$exp$(-\frac{(x^2+y^2)}{2\sigma^2})$')

er_max = 0.0
for i, bin in enumerate(binned):
    error = np.abs(bin - gaussian(mag, r[i], 0)) / mag * 100
    axes[1].scatter(r[i], error, color=color_opt[i + 1], label=rf'dr/$\sigma$ = {dr_bins[i]/s}', marker='x')
    er_max = max(er_max, np.max(error))
axes[1].set_title("Error")
axes[1].set_xlim(0, x[-1])
#axes[1].legend()
axes[1].set_xlabel("r [m]")
axes[1].set_ylim(0, er_max * 1.1)
axes[1].set_ylabel(r"$\frac{\text{bin - gaussian}}{A}$[%]")

"""
plotting log-log to get the order of accuracy of the error
area = C * dr^p
log(area) = log(C)+p*log(dr) --> equivalent to y = mx + b
p is order of accuracy
"""
area_true = mag*np.sqrt(2*np.pi*s**2)/2 # because we are only looking at half of the gaussian
def quadratic_fit(x, a, c):
    return a*x**2 + c
#dr_bins = np.insert(dr_bins, 0, 0)/s
r_opt = np.linspace(0, max(dr_bins)*2, 1000)
coef = np.polyfit(dr_bins, int_area/area_true, 1)
axes[2].plot(r_opt, np.poly1d(coef)(r_opt), color=color_opt[0], label=rf'a={coef[0]:.2e}(dr/$\sigma$) + {coef[1]:.2f}', linestyle=line_opt[1], linewidth = 0.6)
coef = scipy.optimize.curve_fit(quadratic_fit, dr_bins, int_area/area_true)[0]
axes[2].plot(r_opt, quadratic_fit(r_opt, *coef), color=color_opt[0], label=rf'a={coef[0]:.2e}(dr/$\sigma$)$^2$ + {coef[1]:.2f}', linestyle=line_opt[2], linewidth = 0.6)

for i, dr in enumerate(dr_bins):
    axes[2].scatter(dr, int_area[i]/area_true, color=color_opt[i+1], marker='x')
axes[2].set_title("Integrated Area")
axes[2].set_xlim(min(dr_bins)*0.9, max(dr_bins)*1.1)
axes[2].set_xlabel(r'dr/$\sigma$')
axes[2].set_ylim(0.1, 1.1)#(min(int_area/area_true)*0.9, max(int_area/area_true)*1.05)
axes[2].legend()
axes[2].set_ylabel(r"ND Area, $\frac{\sum_{n=1}^{N} (\text{bin}(n)\cdot\text{dr})}{\frac{\text{A}}{2}\sqrt{2\pi\sigma^2}}$")
axes[2].set_xscale('log')#, base=2)
axes[2].set_yscale('log')#, base=2)

if show_field:
    im = axes[3].imshow(gaus, extent=[x[0], x[-1], y[0], y[-1]], interpolation ='none', aspect='auto')
    fig.colorbar(im, ax = axes[3], label='Variable magnitude', shrink=0.8)
    axes[3].set_title(r'A$\cdot$exp$(-\frac{(x^2+y^2)}{2\sigma^2})$')
    axes[3].set_xlim(x[0], x[-1])
    axes[3].set_xlabel("x [m]")
    axes[3].set_ylim(y[0], y[-1])
    axes[3].set_ylabel("y [m]")
    axes[3].set_aspect('equal')

# --- Save Frame ---
frame_path = os.path.join(outdir, f"{field}binning_verification_gaus.svg")
plt.savefig(frame_path)
plt.close(fig)
