plt.rcParams.update({'font.size': 8})

fig, ax = plt.subplots(2,3, figsize=(45, 6))  # taller figure
ax = ax.ravel()

fields = [area_idx[:, :, idx_max], area_idx[:, :, idx_neutral], area_idx[:, :, mld_index], area_idx[:, :, bw_idx], area_idx[:, :, bw_idx+10]]
titles = ['max', 'neutral', 'mld', 'max bw', 'max bw + 10']

for i, field in enumerate(fields):
    field.astype(float)
    cf = ax[i].contourf(X[:, :, 0], Y[:, :, 0], field)
    ax[i].set_ylim(0, lx[1])
    ax[i].set_xlim(0, lx[0])
    ax[i].set_title(titles[i])
    cbar = fig.colorbar(cf, ax=ax[i], shrink=0.8, orientation='horizontal')
    cbar.formatter.set_powerlimits((-3, 2))
    cbar.set_ticks([field.min(), field.max()])
    cbar.update_ticks()
    ax[i].set_aspect('equal')

plt.show()



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
