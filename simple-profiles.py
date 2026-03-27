import os
import numpy as np
import matplotlib.pyplot as plt

def norm_3d(x, y, z):
    sigma = 10.0
    return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-z**2 / (2 * sigma**2)) * np.exp(-(x-L/2)**2 / (2 * sigma**2)) * np.exp(-(y-L/2)**2 / (2 * sigma**2)) 
# flags
plot_ics = True
plot_exp_profile = True

# output information
fig_folder = "figures and videos/simple_profiles/"
# parameters
Nz = 64
Lz = -96.0
N = 32
L = 320.0
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
z = np.linspace(Lz, 0, Nz)
X, Y, Z = np.meshgrid(x, y, z)
alpha = 2.0e-4
g = -9.80665 # m/s^2
mld = -30 # m depth of mixed layer
dTdz = 0.01
b0 = -5.0e-2
############ PLOTTING ############
# font for plotting 
plt.rcParams['font.family'] = 'serif' # or 'sans-serif' or 'monospace'
plt.rcParams['font.serif'] = 'cmr10'
plt.rcParams['font.sans-serif'] = 'cmss10'
plt.rcParams['font.monospace'] = 'cmtt10'
plt.rcParams["axes.formatter.use_mathtext"] = True # to fix the minus signs
plt.rcParams['font.size'] = 12
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