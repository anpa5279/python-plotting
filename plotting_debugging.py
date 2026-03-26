plt.rcParams.update({'font.size': 8})

fig, ax = plt.subplots(2,3, figsize=(45, 6))  # taller figure
ax = ax.ravel()

fields = [area_rho_opt, area_w_opt, area_combo_option, area_combo_overlay]
titles = ['area_rho_opt', 'area_w_opt', 'area_combo_option', 'area_combo_overlay']

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
