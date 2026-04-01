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
