import os
import numpy as np
import scipy
import matplotlib.pyplot as plt

from diagnostics import azimuthal_avg
from plotting_general import plot_format, comparison_plot_opt
# ==========================================================
# FLAGS
# ==========================================================
show_field = True

outdir = 'figures and videos/'
os.makedirs(outdir, exist_ok=True)

# ==========================================================
# PARAMETERS
# ==========================================================
lx = 200, 200
dx = 0.125, 0.125 
x = np.arange(-lx[0]/2+dx[0]/2, lx[0]/2, dx[0])
y = np.arange(-lx[1]/2+dx[1]/2, lx[1]/2, dx[1])

X, Y = np.meshgrid(x, y)
n_gauss = [len(x), len(y)]

# ==========================================================
# ANALYSIS
# ==========================================================
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

# ==========================================================
# PLOTTING
# ==========================================================
plot_format()
case_opt = comparison_plot_opt(len(binned)+1)
ncols = 3
field = ''
width_opt = np.ones(ncols)
fig, axes = plt.subplots(1, ncols, figsize=(12, 5), gridspec_kw={'width_ratios': width_opt})
axes = axes.ravel()
for i, bin in enumerate(binned):
    axes[0].scatter(r[i], bin, color = case_opt['color'][i], label=rf'dr/$\sigma$ = {dr_bins[i]/s}', marker='x')
axes[0].plot(x[n_gauss[0]//2:], gaus[n_gauss[0]//2:, n_gauss[1]//2], color = case_opt['color'][0], label=r'A$\cdot$exp$(-\frac{(x^2+y^2)}{2\sigma^2})$')
axes[0].set_title("Field vs Binning")
axes[0].set_xlim(0, x[-1])
axes[0].legend()
axes[0].set_xlabel("r [m]")
axes[0].set_ylim(0, mag*1.1)
axes[0].set_ylabel(r'gaussian, A$\cdot$exp$(-\frac{(x^2+y^2)}{2\sigma^2})$')

area_true = mag*np.sqrt(2*np.pi*s**2)/2 # because we are only looking at half of the gaussian
def quadratic_fit(x, a, c):
    return a*x**2 + c

dr_nd = dr_bins/s
r_opt = np.linspace(0, max(dr_nd)*2, 1000)
coef = np.polyfit(dr_nd, int_area/area_true, 1)
axes[1].plot(r_opt, np.poly1d(coef)(r_opt), color = case_opt['color'][0], label=rf'a={coef[0]:.2e}(dr/$\sigma$) + {coef[1]:.2f}', linestyle=case_opt['line_styles'][1], linewidth = 0.6)
coef = scipy.optimize.curve_fit(quadratic_fit, dr_nd, int_area/area_true)[0]
axes[1].plot(r_opt, quadratic_fit(r_opt, *coef), color = case_opt['color'][0], label=rf'a={coef[0]:.2e}(dr/$\sigma$)$^2$ + {coef[1]:.2f}', linestyle=case_opt['line_styles'][2], linewidth = 0.6)

for i, dr in enumerate(dr_nd):
    axes[1].scatter(dr, int_area[i]/area_true, color = case_opt['color'][i+1], marker='x')
axes[1].set_title("Integrated Area")
axes[1].set_xlim(min(dr_nd)*0.9, max(dr_nd)*1.1)
axes[1].set_xlabel(r'dr/$\sigma$')
axes[1].set_ylim(0.7, 1.1)#(min(int_area/area_true)*0.9, max(int_area/area_true)*1.05)
axes[1].legend()
axes[1].set_ylabel(r"ND Area, $\frac{\sum_{n=1}^{N} (\text{bin}(n)\cdot\text{dr})}{\frac{\text{A}}{2}\sqrt{2\pi\sigma^2}}$")
axes[1].set_xscale('log')#, base=2)
axes[1].set_yscale('log')#, base=2)

"""
plotting log-log of the integrated error to get the order of accuracy of the error
area = C * dr^p
log(area) = log(C)+p*log(dr) --> equivalent to y = mx + b
p is order of accuracy
"""
def log_10_fit(x, m, b):
    return x**m * 10**b
area_err = np.abs(int_area-area_true)/area_true
m = np.log(area_err[0]/area_err[-1])/np.log(dr_nd[0]/dr_nd[-1])
coef = scipy.optimize.curve_fit(log_10_fit, dr_nd, area_err)[0]
axes[2].plot(r_opt, log_10_fit(r_opt, *coef), color = case_opt['color'][0], label=rf'error=(dr/$\sigma)^{{{coef[0]:.3f}}}$ 10$^{{{coef[1]:.3f}}}$', linestyle=case_opt['line_styles'][2], linewidth = 0.6)
#axes[2].plot(r_opt, r_opt**m, color = case_opt['color'][0], label=rf'error={coef[0]:.2e}(dr/$\sigma$)$^2$ + {coef[1]:.2e}', linestyle=case_opt['line_styles'][2], linewidth = 0.6)

for i, dr in enumerate(dr_nd):
    axes[2].scatter(dr, area_err[i], color = case_opt['color'][i+1], marker='x')
axes[2].set_title("Error of Integrated Area")
axes[2].set_xlim(min(dr_nd)*0.9, max(dr_nd)*1.1)
axes[2].set_xlabel(r'dr/$\sigma$')
#axes[2].set_ylim(0.1, 1.1)#(min(int_area/area_true)*0.9, max(int_area/area_true)*1.05)
axes[2].legend()
axes[2].set_ylabel(r"Error of Integrated Area")
axes[2].set_xscale('log')#, base=2)
axes[2].set_yscale('log')#, base=2)

# --- Save Frame ---
frame_path = os.path.join(outdir, f"{field}binning_verification_gaus.svg")
plt.savefig(frame_path)
plt.close(fig)

if show_field:
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    im = ax.imshow(gaus, extent=[x[0], x[-1], y[0], y[-1]], interpolation ='none', aspect='auto')
    fig.colorbar(im, ax = ax, label='Variable magnitude', shrink=0.8)
    ax.set_title(r'A$\cdot$exp$(-\frac{(x^2+y^2)}{2\sigma^2})$')
    ax.set_xlim(x[0], x[-1])
    ax.set_xlabel("x [m]")
    ax.set_ylim(y[0], y[-1])
    ax.set_ylabel("y [m]")
    ax.set_aspect('equal')
    # --- Save Frame ---
    frame_path = os.path.join(outdir, f"binning_field_gaus.svg")
    plt.savefig(frame_path)
    plt.close(fig)