import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm

plt.rcParams.update({'font.size': 8})

fig, ax = plt.subplots(1,4, figsize=(45, 6))  # taller figure
ax = ax.ravel()
#(np.min(w_mag_order[:, :, k]), np.max(w_mag_order[:, :, k]))
fields = [wk[:, :], w_mag_order[:, :, k], b_fluc_k[:, :], area_opt[:, :]]
titles = ['w', 'w_mag_order', 'b_fluc', 'area_opt']
ranges = [(-0.01, 0.01), (-5, -2), (-6.5*10**(-5), 6.5*10**(-5)), (0, 1)]
levels = [100, np.unique(w_mag_order[:, :, k]), 100, 2]

for i, field in enumerate(fields):
    field.astype(float)
    norm = mcolors.Normalize(vmin=ranges[i][0], vmax=ranges[i][-1])
    mappable = cm.ScalarMappable(norm=norm, cmap='RdBu_r')
    cf = ax[i].contourf(X[:, :, 0], Y[:, :, 0], field, levels = levels[i], norm=norm, cmap='RdBu_r')
    #ax[i].set_ylim(0, lx[1])
    #ax[i].set_xlim(0, lx[0])
    ax[i].set_title(titles[i])
    cbar = fig.colorbar(mappable, ax=ax[i], shrink=0.8, orientation='horizontal')
    cbar.formatter.set_powerlimits((-3, 2))
    cbar.update_ticks()
    ax[i].set_aspect('equal')

plt.show()

#fields = [area_idx[:, :, idx_max], area_idx[:, :, idx_neutral], area_idx[:, :, mld_index], area_idx[:, :, mld_index+2], area_idx[:, :, idx_neutral-1], area_idx[:, :, mld_index+50]]
#titles = ['max', 'neutral', 'mld', 'mld+2', 'neutral-1', 'mld+50']

plt.rcParams.update({'font.size': 8})

fig, ax = plt.subplots(1,3, figsize=(45, 4))  # taller figure
ax = ax.ravel()

fields = [b_fluc[127, 127, :], db_flucdz[127, 127, :], wc[127, 127, :]]
titles = ['b_fluc', 'db_flucdz', 'wc']

for i, field in enumerate(fields):
    span = [field.min(), field.max()]
    ax[i].plot(span, -mld*np.ones(2), linestyle='--', linewidth = 0.5, color = 'black')#, label = "MLD")
    ax[i].plot(span, z[idx_max]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'cornflowerblue')#, label = "Intrusion Depth")
    ax[i].plot(span, z[idx_neutral]*np.ones(2), linestyle='--', linewidth = 0.5, color = 'mediumblue')#, label = "Neutral Buoyancy")
    field.astype(float)
    ax[i].plot(field, z, linewidth = 0.5)
    ax[i].set_ylim(-lx[-1], 0)
    ax[i].set_title(titles[i])

plt.show()

# ND testing
z_nd = (z)# + (F_s /F_s[0] - 1)*mld) / rj
testS = S_avg[:, 0]
testS2 = S_avg[:, 2]
testT = T_fluc_center[:, 0]
testT2 = T_fluc_center[:, 2]
testz = (z[:, 0] - z[182, 0])
testz2 = (z[:, 2] - z[158, 2])
plt.rcParams.update({'font.size': 8})

fig, ax = plt.subplots(1,2, figsize=(45, 4))  # taller figure
ax = ax.ravel()

fields = [testT, testT2]
zs = [testz, testz2]
titles = [case_names[0], case_names[2]]

for i, field in enumerate(fields):
    span = [field.min(), field.max()]
    field.astype(float)
    ax[0].plot(field, zs[i], linewidth = 0.5, label = titles[i])

ax[0].set_ylim(ymin = np.min(zs), ymax = np.max(zs))
ax[0].legend(loc='lower right')

plt.show()

# FFT testing 
plt.rcParams.update({'font.size': 16})

fig, ax = plt.subplots(2, 3, dpi=150, figsize=(30, 12))
ax = ax.ravel()
titles = ['b', 'T', 'S', 'u', 'v', 'wc']

for i, var in enumerate([b, T, S, u, v, wc]):
    var_prime = var - np.mean(var, axis=(0,1))[..., np.newaxis]
    rms_list = []
    L_list = []
    for k in range(nx[2]):
        var_hat = np.fft.fft2(var_prime[:,:,k])
        E = 0.5 * np.abs(var_hat)**2
        var_rms = np.sqrt(E.sum())
        rms_list.append(var_rms)
        kx = np.fft.fftfreq(nx[0], dx[0]) * 2*np.pi
        ky = np.fft.fftfreq(nx[1], dx[1]) * 2*np.pi
        k_h = np.sqrt(kx[:,None]**2 + ky[None,:]**2)
        k_mean = (E.ravel() * k_h.ravel()).sum() / E.sum()
        L_list.append(2*np.pi / k_mean)
    var_rms = np.array(rms_list)
    L_dom = np.array(L_list)
    var_nd = np.mean(var, axis=(0,1)) / var_rms
    z_temp = z[:, -1] / L_dom
    ax[i].plot(var_nd, z_temp, linewidth=2)
    ax[i].set_ylabel('z / L_dom', fontsize=18)
    ax[i].grid(True)
    ax[i].set_title(titles[i])

plt.tight_layout()
plt.show()


