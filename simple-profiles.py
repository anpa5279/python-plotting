import os
import numpy as np
import matplotlib.pyplot as plt
from plotting_functions import stratification_profile
from plotting_comparisons import plot_format
# flags
plot_ics = False
plot_exp_profile = False
variations = True

# output information
fig_folder = "figures and videos/simple_profiles/"
variations = "strat"
# parameters
Nz = 256
Lz = -96.0
N = 256
L = 320.0
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
z = np.linspace(Lz, 0, Nz)
X, Y, Z = np.meshgrid(x, y, z)
num_cases = 3
alpha = 2.0e-4
g = -9.80665 # m/s^2
mld = 30 * np.ones(num_cases) # np.array([20, 30, 40]) #
dTdz = np.array([0.01, 0.05, 0.1]) # 0.01 * np.ones(num_cases) # 
T0 = 25.0
b0 = -5.0e-2
############ PLOTTING ############
# font for plotting 
plt.rcParams['font.family'] = 'serif' # or 'sans-serif' or 'monospace'
plt.rcParams['font.serif'] = 'cmr10'
plt.rcParams['font.sans-serif'] = 'cmss10'
plt.rcParams['font.monospace'] = 'cmtt10'
plt.rcParams["axes.formatter.use_mathtext"] = True # to fix the minus signs
plt.rcParams['font.size'] = 12
# varied initial conditions
if variations:
    ic_profile = np.zeros((num_cases, Nz))
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    color_opts, line_opts = plot_format(num_cases)
    for i in range(num_cases):
        ic_profile[i, :] = stratification_profile(z, T0, dTdz[i], -mld[i])
        ax.plot(ic_profile[i, :], z, color=color_opts[i])
    ax.set_ylabel('Depth (m)')
    ax.grid(False)
    ax.set_xticklabels([])
    ax.set_xlim(T0-1.0, T0+0.05)
    ax.set_ylim(Lz, 0)
    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
    os.makedirs(fig_folder, exist_ok=True)
    plt.savefig(os.path.join(fig_folder, f'initial_condition_profiles_{variations}.svg'))

# initial conditions
if plot_ics:
    ic_profile = np.zeros(Nz)
    izi = int(mld/Lz * Nz)
    dbdz = alpha * g * dTdz 
    print(izi)
    #for k in range(izi, Nz):
    #    ic_profile[k] = b0*norm_3d(L/2, L/2, z[k])
    for k in range(0, Nz-izi-1):
        ic_profile[k] = -dbdz * (z[k] - mld)
    plt.figure(figsize=(4, 4))
    plt.plot(ic_profile, z, color='black')
    plt.ylabel('Depth (m)')
    plt.title('Initial Condition Profile')
    plt.grid()
    os.makedirs(fig_folder, exist_ok=True)
    plt.savefig(os.path.join(fig_folder, 'initial_condition_profile.svg'))
    plt.close()


# exponential decay profile
exp_profile = np.exp(z / 4.0)
if plot_exp_profile:
    plt.figure(figsize=(4, 4))
    plt.plot(exp_profile, z, color='black')
    plt.ylabel('Depth (m)')
    plt.title('Random Noise Distribution')
    plt.grid()
    plt.savefig(os.path.join(fig_folder, 'exponential_decay_profile.svg'))
    plt.close()