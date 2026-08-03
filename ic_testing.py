import os
import numpy as np
import math
import matplotlib.pyplot as plt

from plotting_general import plot_format, comparison_plot_opt
outdir = 'figures and videos/'
os.makedirs(outdir, exist_ok=True)

# domain information
hml = 60
T0 = 25
dTdz = 0.01
lz = 128

dzs = [1*10**(-6), 0.25, 0.5, 1, 2]


z = []
range = [-hml-2.0, -hml+2]
for dz in dzs:
    z.append(np.arange(range[0], range[1], dz))

z_true = np.linspace(range[0], range[1], 900)

# error function
a = dTdz*np.sqrt(np.pi)/2

T_erf = np.empty(len(z_true))   
for i, z_opt in enumerate(z_true):
    T_erf[i] = a*math.erf(z_opt+hml) + T0 - a

dT_erfdz = np.gradient(T_erf, z_true)

# error function applied to grid 
T1 = a*math.erf(-hml+hml) + T0 - a
T = []
for z_opt in z:
    T_grid = np.empty(len(z_opt))
    for i, z_val in enumerate(z_opt):
        if z_val < -hml:
            T_grid[i] = T1 + dTdz*(z_val+hml)
        else:
            T_grid[i] = a*math.erf(z_val+hml) + T0 - a
    T.append(T_grid)

# plotting
plot_format()
color_opt, line_opt = comparison_plot_opt(len(dzs))
ncols = 3
nrows = 1
width_opt = np.ones(ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4), sharey = True)
axes = axes.ravel()
axes[0].plot(T_erf, z_true, color='k')
axes[0].plot([-100, 100], -hml* np.ones(2), linestyle='dashed', linewidth=0.5, color='k')
axes[0].set_title("Error Function")
axes[0].set_xlim(T0-0.02, T0+0.005)
axes[0].set_ylabel("z [m]")
axes[0].set_ylim(min(z_true), max(z_true))

axes[1].plot(dT_erfdz, z_true, color='k')
axes[1].plot([-100, 100], -hml* np.ones(2), linestyle='dashed', linewidth=0.5, color='k')
axes[1].set_title("Gradient of Error Function")
axes[1].set_xlim(0.0, dTdz*1.05)
axes[1].set_ylim(min(z_true), max(z_true))


axes[2].plot([-100, 100], -hml* np.ones(2), linestyle='dashed', linewidth=0.5, color='k')
for i, dz in enumerate(dzs):
    if i == 0:
        axes[2].plot(T[i], z[i], color=color_opt[i])
    else:
        axes[2].scatter(T[i], z[i], color=color_opt[i], marker='x', label=rf'dz = {dz}')
axes[2].set_title("Temperature Profile")
axes[2].set_xlim(T0-0.03, T0+0.001)
axes[2].set_xlabel("Temperature [C]")
axes[2].set_ylim(range[0], range[1])
axes[2].legend(loc='upper left')
axes[2].ticklabel_format(axis='x', style='sci', scilimits=(-3,2), useMathText=True, useOffset=False)

# --- Save Frame ---
frame_path = os.path.join(outdir, f"erf_test.svg")
plt.savefig(frame_path)
plt.close(fig)
