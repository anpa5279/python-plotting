import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# Simulation runs
# --------------------------
# runs = list of dicts, each dict contains variables and parameters
# Example: runs = [{'b': b1, 'T': T1, 'S': S1, 'u': u1, 'v': v1, 'wc': wc1, 'z': z1, 'dx': dx1, 'nx': nx1}, ...]
# Each run can have different F_s, rj, dTdz, etc.

variables = ['b', 'T', 'S', 'u', 'v', 'wc']

# --------------------------
# Set up plot
# --------------------------
plt.rcParams.update({'font.size': 16})
fig, ax = plt.subplots(2, 3, dpi=150, figsize=(30, 12))
ax = ax.ravel()

# --------------------------
# Loop over variables
# --------------------------
importance_scores = {}

for i, var_name in enumerate(variables):
    nd_profiles_all_runs = []

    for run in runs:
        var = run[var_name]
        z = run['z']  # shape (nz, ny) or (nz,)
        dx = run['dx']
        nx_run = run['nx']

        # Remove horizontal mean
        var_prime = var - np.mean(var, axis=(0,1))[..., np.newaxis]

        nz = var.shape[2]
        rms_list = []
        L_list = []

        for k in range(nz):
            var_hat = np.fft.fft2(var_prime[:,:,k])
            E = 0.5 * np.abs(var_hat)**2
            var_rms = np.sqrt(E.sum())
            rms_list.append(var_rms)

            # FFT wavenumbers
            kx = np.fft.fftfreq(nx_run[0], dx[0]) * 2*np.pi
            ky = np.fft.fftfreq(nx_run[1], dx[1]) * 2*np.pi
            k_h = np.sqrt(kx[:,None]**2 + ky[None,:]**2)
            k_mean = (E.ravel() * k_h.ravel()).sum() / E.sum()
            L_list.append(2*np.pi / k_mean)

        var_rms = np.array(rms_list)
        L_dom = np.array(L_list)

        # Nondimensionalize
        z_nd = z[:, 0] / L_dom       # pick first horizontal slice
        var_nd = np.mean(var, axis=(0,1)) / var_rms
        nd_profiles_all_runs.append(var_nd)

        # Plot
        ax[i].plot(var_nd, z_nd, linewidth=2, alpha=0.7)

    # Compute simple importance score: mean spread across runs
    nd_profiles_all_runs = np.array(nd_profiles_all_runs)
    spread = np.std(nd_profiles_all_runs, axis=0)
    importance_scores[var_name] = np.mean(spread)

    ax[i].set_xlabel(f'{var_name} / RMS', fontsize=18)
    ax[i].set_ylabel('z / L_dom', fontsize=18)
    ax[i].grid(True)
    ax[i].set_title(f'{var_name}', fontsize=20)

plt.tight_layout()
plt.show()

# --------------------------
# Print importance scores
# --------------------------
print("Variable importance (higher = more sensitive to input changes):")
for var, score in importance_scores.items():
    print(f"{var}: {score:.3f}")
