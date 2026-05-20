import os
import numpy as np
import matplotlib.pyplot as plt

from reader import OceananigansData
from dense_plume import PlumeAnalysis
from diagnostics import comparison_info
from physics import rms, a_fluc_b, buoyancy
from plotting_general import plot_format, plot_ranges, create_video, comparison_plot_opt, plot_plume_vertical_spatial, plot_plume_horizontal_spatial, plot_variable_vert_slice, plot_variable_xy_slice, plot_combo_exponents, plot_rig_exponents, plot_Fr_exponents, plot_mld_exponents
from interpolation import velocities_to_center, vertical_line, horizontal_line, yz_plane, xy_plane, xz_plane

# flags for what to plot
plot_variables = True
plot_1d_z = True
plot_1d_y = False
transient_mld = False
temporal_averages_flag = False
video = True

ND = False
bin = True

# flags for how to read data
with_halos = False
closure = False
salinity = True
stokes = False

contour_bound = 0.05
name_uni = f'contour-{contour_bound:.2f}'
universal_folder = '/Users/annapauls/Documents/TESLa /Simulations/Oceananigans/dense plume/salinity and temperature/no noise circle inlet/'#/Lz = 160m'#resolution testing'#vertical domain increase/dTdz = 0.01'
#harddrive: '/Volumes/Anna External/Oceananigans/dense plume with stratification/salinity and temperature /no noise circle inlet/resolution testing'#

if ND:
    combo_flag = False
    morton_znd_flag = True
    exponents = [] # for plotting reference lines with different exponents, set to empty array to not plot any, -4/3, -1, -3/4, -2/3, -1/2, 1/2, 2/3, 3/4, 1, 4/3
    if temporal_averages_flag:
        title = 'Temporal averages'
        name_uni += '_temporal_averages'

# selecting cases to compare
variations = 'else' # 'MLD', 'flux', 'strat', 'all', 'length', 'WENO', 'resolution', 'else'
if variations != 'else':
    cases_info = comparison_info(variations, universal_folder = universal_folder, ND = ND)
else:
    folder_names = ['proposed resolution/S0 = 0.1 dTdz = 0.01 MLD = 70', 'Lz = 160m/S0 = 0.1 dTdz = 0.01 MLD = 70']
    num_cases = len(folder_names)
    fig_folder = os.path.join(universal_folder, 'comparison figures', '96m vs 160m' + ' comparison figures', 'interpolated', 'MLD = 70m')
    case_names =[r'L$_z = 96$m', r'L$_z = 160$m']#r'$\Delta z = 0.5$m', r'$\Delta z = 0.375$m'#[r'F$_{\text{C}} = -1.0\cdot 10^{-4}$, MLD = 60m, dTdz = 0.01', r'F$_{\text{C}} = -1.0\cdot 10^{-4}$, MLD = 70m, dTdz = 0.01', r'F$_{\text{C}} = - 2.0\cdot 10^{-4}$, MLD = 60m, dTdz = 0.01']
    cases_info = {
            "folder_names": folder_names,
            "fig_folder": fig_folder,
            "case_names": case_names,
            "num_cases": num_cases,
            "dTdz": 0.01*np.ones(num_cases),
            "mld": np.array([70, 70]),
            "F_s": np.array([-1.0*10**(-4), -1.0*10**(-4)])
        }

dTdz = cases_info['dTdz']
case_names = cases_info['case_names']
num_cases = cases_info['num_cases']
fig_folder = cases_info['fig_folder']
mld = cases_info['mld']
if ND:
    F_s = cases_info['F_s']

readers = []
for folder in cases_info["folder_names"]:
    folder = os.path.join(universal_folder, folder)
    readers.append(OceananigansData(folder, salinity = salinity))

# collecting model information for all cases
t_save = []
mld_idx = []
z = []
nx = []
lx = []
if salinity:
    dense_plume = []

for i, reader in enumerate(readers):

    reader.load_time()
    t_save.append(reader.t_save)
    reader.load_grid()
    z.append(reader.z)
    nx.append(reader.nx)
    lx.append(reader.lx)
    if i == 0:
        nt = reader.nt
        nz = reader.nx[2]
    else:
        nt = np.min([nt, reader.nt])
        nz = np.max([nz, reader.nx[2]])
    if salinity and plot_1d_z:
        S_value, w_value = reader.load_contour_temporal_averages('interp_temporal_averages.h5')
        dense_plume.append(PlumeAnalysis(S_value*contour_bound))

x = readers[0].x
y = readers[0].y

# physical parameters
x0 = 0.0
centery = 0.0
rj = 5 # m, radius of salinity flux circle at the surface
g = 9.80665  # gravity in m/s^2
rho0 = 1026
T0 = 25
S0 = 0 

# video or not setup
if video:
    nt = np.arange(0, nt)
    time = readers[0].time
else:
    nt = [nt,] # last time step
    time = readers[0].time[-1]

# plotting prep
plot_format()
if plot_variables:
    if salinity:
        var_names = ['Tracer', 'Temperature', 'Density', 'u', 'v', 'w']#, 'Perturbed Vertical Buoyancy Flux', 'Perturbed Density']
        range_names = ['Tracer', 'T', 'rho', 'u', 'v', 'w']#, 'bw_fluc', 'rho_fluc']
    else:
        var_names = ['Temperature', 'Density', 'u', 'v', 'w']#, 'Perturbed Vertical Buoyancy Flux', 'Perturbed Density']
        range_names = ['T', 'rho', 'u', 'v', 'w']#, 'bw_fluc', 'rho_fluc']
    planeslice = 'vertical' # 'vertical' or 'horizontal'
    variable_dir = {}
    if planeslice == 'horizontal':
        name_uni += '_horizontal_slice'
        loc = 'z' # 'cell' or 'z'
        if loc == 'z':
            loc_z = -mld
            name_uni += '_at_mld'
        else:
            n = 254
            loc_z = z[:, n]

S_tol = 10**(-6)
ranges = plot_ranges(lz = 96, mld = np.max(mld), rho0 = rho0, T0 = T0, dTdz = np.max(dTdz), C_tol = S_tol)
ranges['rho'] = [rho0, rho0+0.15]
ranges['rho_fluc'] = [-0.025, 0.025]
ranges['Tracer'] =[S_tol, 0.15]
ranges['Tracer_fluc'] = [-0.2, 0.2]
ranges['Tracer_avg'] = [0, 1.2*10**(-3)]
ranges['T'] = [T0-1.0, T0 + 0.01]
ranges['w'] = [-1.5*10**(-1), 1.5*10**(-1)]
ranges['u'] = [-1.2*10**(-2), 1.2*10**(-2)]
ranges['v'] = [-2*10**(-2), 2*10**(-2)]
ranges['vel_rms'] = [0, 4*10**-3]
ranges['bw_fluc'] = [-5*10**(-9), 5*10**(-9)]
if plot_1d_z or plot_1d_y:
    color_opt, line_opt = comparison_plot_opt(num_cases)

if plot_1d_y:
    ranges_hor = ranges.copy()
    ranges_hor['Tracer'] = [S_tol, 3*10**(-2)]
    ranges_hor['vel_rms'] = [0, 4*10**-3]
    ranges_hor['bw_fluc'] = [-2*10**(-5), 2*10**(-5)]
    ranges_hor['b_flux'] = [-4*10**(-6), 4*10**(-6)]
    ranges_hor['b_fluc'] = [-2*10**(-4), 2*10**(-4)]
    ranges_hor['w'] = [-0.15, 0.15]
    ranges_hor['T'] = [T0 - 0.2, T0 + 0.2]
    loc_z = -mld
    hor_str = ' '.join([f"{depth} m" for depth in loc_z])
    name_xy = name_uni + f"at z = {hor_str}"

############ NONDIMENSIONALIZATION ############
if ND:
    area = (2*rj)**2 
    N2 = g * alpha * dTdz 
    Ri_g = N2/(g/rj)
    Fr_flux = F_s * beta / np.sqrt(rj * g)
    vel_scale = np.sqrt(rj * g)
    b_scale = g
    F_b_scale = b_scale * vel_scale
    T_scale = 1/alpha
    S_scale =  1/beta
    F_T_scale = beta * F_s / alpha
    F_S_scale = F_s
    hor_scale = rj
    if morton_znd_flag:
        name_uni += '_morton_scaling'
        F0 = area * beta * g * F_s
        Ln =(F0/N2**(3/2))**(1/4)
        z_nd = (z+mld)*(mld)**(1/3)/(Ln**(4/3))
    else:
        name_uni += '_simple_znd'
        F0 = area * beta * g * F_s
        Ln =(F0/N2**(3/2))**(1/4)
        z_nd = (z+mld)/(Ln)
        z_str = r'$(z + h$_{ML}$)/L_M$'
        #for i in range(num_cases):
        #    z_nd[0:mld_idx[i], i] = (z[0:mld_idx[i], i]+mld[i])*(mld[i])**(1/3)/(Ln[i]**(4/3))

    if temporal_averages_flag:
        T_avg = np.zeros((nx[2], num_cases))
        b_avg = np.zeros((nx[2], num_cases))
        w_fluc_avg = np.zeros((nx[2], num_cases))
        bw_fluc_avg = np.zeros((nx[2], num_cases))
        w_rms = np.zeros((nx[2], num_cases))
        r_profile = np.zeros((nx[2], num_cases))
        b_center = np.zeros((nx[2], num_cases))
        T_fluc_center = np.zeros((nx[2], num_cases))
        S_fluc_center = np.zeros((nx[2], num_cases))
        S_avg = np.zeros((nx[2], num_cases))
        for i, reader in enumerate(readers):
            vel_rms_avgt, bw_avgt, T_avgt, S_avgt, r_plume_avgt = reader.load_temporal_averages('interp_temporal_averages.h5', temperature=True, salinity=salinity)
            w_rms[:, i] = vel_rms_avgt['w_rms']
            b_avg[:, i] = bw_avgt['b_avg']
            bw_fluc_avg[:, i] = bw_avgt['bw_fluc_avg']
            b_center[:, i] = bw_avgt['b_centerline_avg']
            T_fluc_center[:, i] = T_avgt['T_fluc_centerline_avg']
            S_avg[:, i] = S_avgt['S_avg']
            r_profile[:, i] = r_plume_avgt['tracer radius']

            if transient_mld:
                    dbdz = np.gradient(b_avg[:, i], z[:, i])
                    dbdz_tol = dbdz <= (5.0*10**(-7))
                    if np.any(dbdz_tol):
                        mld_idx[i] = np.min(np.where(dbdz_tol))
                    else:
                        mld_idx[i] = nx[2] - 1
                    mld[i] = -z[mld_idx[i], i]
        if transient_mld:
            Ln =(F0/N2**(3/2))**(1/4)
            z_nd = (z+mld)*(mld)**(1/3)/(Ln**(4/3))
        bw_fluc_avg = bw_fluc_avg/F_b_scale
        S_avg = S_avg/S_scale
        b_avg = b_avg/b_scale
        b_center = b_center/b_scale
        T_fluc_center = T_fluc_center / T_scale
        S_fluc_center = S_fluc_center / S_scale
        r_profile = r_profile / hor_scale
        w_rms = w_rms/vel_scale
        ############ PLOTTING ############
        if variations == 'all' or combo_flag:
            plot_combo_exponents(color_opt, title, name_uni, fig_folder, w_rms, b_center, bw_fluc_avg, r_profile, T_fluc_center, S_avg, z_nd, cases_info['vars_exps'], Ri_g, Fr_flux, mld/rj, case_names)
        if np.size(exponents)==0 and (variations == 'strat' or variations == 'flux' or variations == 'MLD'):
            if variations == 'strat' :
                plot_rig_exponents(color_opt, title, name_uni, fig_folder, w_rms, b_center, bw_fluc_avg, r_profile, T_fluc_center, S_avg, z_nd, Ri_g, case_names)
            if variations == 'flux' :
                plot_Fr_exponents(color_opt, title, name_uni, fig_folder, w_rms, b_center, bw_fluc_avg, r_profile, T_fluc_center, S_avg, z_nd, Fr_flux, case_names)
            if variations == 'MLD':
                plot_mld_exponents(color_opt, title, name_uni, fig_folder, w_rms, b_center, bw_fluc_avg, r_profile, T_fluc_center, S_avg, z_nd, mld/rj, case_names)
        elif np.size(exponents)!=0 and (variations == 'strat' or variations == 'flux' or variations == 'MLD'):
            if variations == 'strat' :
                plot_rig_exponents(color_opt, title, name_uni, fig_folder, w_rms, b_center, bw_fluc_avg, r_profile, T_fluc_center, S_avg, z_nd, Ri_g, case_names, exponents = exponents)
            if variations == 'flux' :
                plot_Fr_exponents(color_opt, title, name_uni, fig_folder, w_rms, b_center, bw_fluc_avg, r_profile, T_fluc_center, S_avg, z_nd, Fr_flux, case_names, exponents = exponents)
            if variations == 'MLD':
                plot_mld_exponents(color_opt, title, name_uni, fig_folder, w_rms, b_center, bw_fluc_avg, r_profile, T_fluc_center, S_avg, z_nd, mld/rj, case_names, exponents = exponents)
else:
    for it in nt:
        if salinity:
            S_avg = []
            S_fluc_center = []
            S_hor = []
            S_plane = []
        if plot_1d_z:
            T_avg = []
            b_avg = []
            u_rms = []
            v_rms = []
            w_rms = []
            u_fluc_avg = []
            v_fluc_avg = []
            w_fluc_avg = []
            bu_fluc_avg = []
            bv_fluc_avg = []
            bw_fluc_avg = []
            r_profile = []
            b_center = []
            T_fluc_center = []
        if plot_1d_y:
            centery = 0.0
            u_hor = []
            v_hor = []
            w_hor = []
            b_fluc_hor = []
            bu_fluc_hor = []
            bv_fluc_hor = []
            bw_fluc_hor = []
            T_hor = []
        if plot_variables:
            T_plane = []
            u_plane = []
            v_plane = []
            w_plane = []
            rho_plane = []
            bw_plane = []
            rho_fluc_plane = []
        for i, reader in enumerate(readers):
            # Load data from files
            u = reader.lazy_field('u', steps = reader.t_save[it])
            v = reader.lazy_field('v', steps = reader.t_save[it])
            w = reader.lazy_field('w', steps = reader.t_save[it])
            T = reader.lazy_field('T', steps = reader.t_save[it])
            if salinity:
                S = reader.lazy_field('S', steps = reader.t_save[it])
            if stokes:
                u = u - reader.u_s
            # interpolate velocities to cell centers
            u, v, w = velocities_to_center(u, v, w)
            # convert temperature and salinity to buoyancy 
            bs = buoyancy(reader)
            b = bs['b']
            rho = bs['rho']
            rho_fluc = rho - np.mean(rho, axis=(-3, -2))

            # calcualte buoyancy fluxes
            bu_fluc = a_fluc_b(b, u)
            bv_fluc = a_fluc_b(b, v)
            bw_fluc = a_fluc_b(b, w)

            # vertical lines to save for plotting
            if plot_1d_z:
                # rms fluctuations
                u_rms.append(rms(reader, 'u'))
                v_rms.append(rms(reader, 'v'))
                w_rms.append(rms(reader, 'w'))
                bu_fluc_avg.append(np.mean(bu_fluc, axis=(-3, -2)))
                bv_fluc_avg.append(np.mean(bv_fluc, axis=(-3, -2)))
                bw_fluc_avg.append(np.mean(bw_fluc, axis=(-3, -2)))
                # calculate means
                b_avg.append(np.mean(b, axis=(-3, -2)))
                T_avg.append(np.mean(T, axis=(-3, -2)))
                # dense plume analysis
                if salinity:
                    S_avg.append(np.mean(S, axis=(-3, -2)))
                    dense_plume[i].input_info(S, b_tracer = bs['b_C'], b_background = bs['b_T'], bw_fluc = bw_fluc)
                    r_profile.append(dense_plume[i].plume_tracer_radius(x = x, y = y))
                    b_center.append(vertical_line(b, x, y, x0, centery))
                    T_fluc_center.append(vertical_line(T-T_avg[i], x, y, x0, centery))
                    S_fluc_center.append(vertical_line(S-S_avg[i], x, y, x0, centery))
            # horizontal lines to save for plotting
            if plot_1d_y:
                u_hor.append(horizontal_line(u, y, z[i, :], centery, loc_z[i]))
                v_hor.append(horizontal_line(v, y, z[i, :], centery, loc_z[i]))
                w_hor.append(horizontal_line(w, y, z[i, :], centery, loc_z[i]))
                b_fluc_hor.append(horizontal_line(b-b_avg[i], y, z[i, :], centery, loc_z[i]))
                bu_fluc_hor.append(horizontal_line(bu_fluc, y, z[i, :], centery, loc_z[i]))
                bv_fluc_hor.append(horizontal_line(bv_fluc, y, z[i, :], centery, loc_z[i]))
                bw_fluc_hor.append(horizontal_line(bw_fluc, y, z[i, :], centery, loc_z[i]))
                T_hor.append(horizontal_line(T, y, z[i, :], centery, loc_z[i]))
                S_hor.append(horizontal_line(S, y, z[i, :], centery, loc_z[i]))
            # plane slices to save for plotting
            if plot_variables and planeslice == 'vertical':
                T_plane.append(yz_plane(T, x, x0))
                u_plane.append(yz_plane(u, x, x0))
                v_plane.append(yz_plane(v, x, x0))
                w_plane.append(yz_plane(w, x, x0))
                rho_plane.append(yz_plane(rho, x, x0))
                if salinity:
                    S_plane.append(yz_plane(S, x, x0))
            elif plot_variables and planeslice == 'horizontal':
                if loc == 'z':
                    T_plane.append(xy_plane(T, z, loc_z[i]))
                    u_plane.append(xy_plane(u, z, loc_z[i]))
                    v_plane.append(xy_plane(v, z, loc_z[i]))
                    w_plane.append(xy_plane(w, z, loc_z[i]))
                    rho_plane.append(xy_plane(rho, z, loc_z[i]))
                    if salinity:
                        S_plane.append(xy_plane(S, z, loc_z[i]))
                else:
                    T_plane.append(T[:, :, n])
                    u_plane.append(u[:, :, n])
                    v_plane.append(v[:, :, n])
                    w_plane.append(w[:, :, n])
                    rho_plane.append(rho[:, :, n])
                    if salinity:
                        S_plane.append(S[:, :, n])

    ############ PLOTTING ############
    for it in nt:
        if plot_variables:
            if salinity: #'Tracer', 'T', 'Density', 'u', 'v', 'w'
                variables = [S_plane, T_plane, rho_plane, u_plane, v_plane, w_plane]
                colorbar_labels = [r"g/kg", r"$^\circ$C", r"kg/m$^3$", r"m/s", r"m/s", r"m/s"]
                cmaps = ['viridis', 'viridis', 'viridis', 'RdBu_r', 'RdBu_r', 'RdBu_r']
            else: #'T', 'Density', 'u', 'v', 'w'
                variables = [T_plane, rho_plane, u_plane, v_plane, w_plane]
                colorbar_labels = [r"$^\circ$C", r"kg/m$^3$", r"m/s", r"m/s", r"m/s"]
                cmaps = ['viridis', 'viridis', 'RdBu_r', 'RdBu_r', 'RdBu_r', 'RdBu_r', 'RdBu_r']
            if planeslice == 'vertical':
                for dir, var in enumerate(variables):
                    variable_dir[var_names[dir]] = plot_variable_vert_slice(time[it], it, ranges, fig_folder, lx[-1], y, z, var, case_names, var_names[dir], range_names[dir], colorbar_label = colorbar_labels[dir], cmap = cmaps[dir], plane='YZ')
            elif planeslice == 'horizontal':
                for dir, var in enumerate(variables):
                    variable_dir[var_names[dir]] = plot_variable_xy_slice(time[it], it, ranges, fig_folder, lx[-1], x, y, var, case_names, var_names[dir], range_names[dir], colorbar_label = colorbar_labels[dir], cmap = cmaps[dir])
        if plot_1d_z:
            buoyancy_dir_z = plot_plume_vertical_spatial(time[it], it, ranges, color_opt, fig_folder, case_names, name_uni, lx[-1], z, S_avg, u_rms, v_rms, w_rms, b_avg, b_center, r_profile, bu_fluc_avg, bv_fluc_avg, bw_fluc_avg, T_avg, T_fluc_center, S_fluc_center)
        if plot_1d_y:
            buoyancy_dir_y = plot_plume_horizontal_spatial(time[it], it, ranges_hor, color_opt, fig_folder, case_names, name_xy, lx[-1], y, u_hor, v_hor, w_hor, b_fluc_hor, bu_fluc_hor, bv_fluc_hor, bw_fluc_hor, T_hor, S_hor)
        if ND:
            if variations == 'all' or combo_flag:
                plot_combo_exponents(color_opt, title, name_uni, fig_folder, w_rms, b_center, bw_fluc_avg, r_profile, T_fluc_center, S_avg, z_nd, cases_info['vars_exps'], Ri_g, Fr_flux, mld/rj, case_names)
            if np.size(exponents)==0 and (variations == 'strat' or variations == 'flux'or variations == 'MLD'):
                if variations == 'strat':
                    plot_rig_exponents(color_opt, title, name_uni, fig_folder, w_rms, b_center, bw_fluc_avg, r_profile, T_fluc_center, S_avg, z_nd, Ri_g, case_names)
                if variations == 'flux':
                    plot_Fr_exponents(color_opt, title, name_uni, fig_folder, w_rms, b_center, bw_fluc_avg, r_profile, T_fluc_center, S_avg, z_nd, Fr_flux, case_names)
                if variations == 'MLD':
                    plot_mld_exponents(color_opt, title, name_uni, fig_folder, w_rms, b_center, bw_fluc_avg, r_profile, T_fluc_center, S_avg, z_nd, mld/rj, case_names)
            elif np.size(exponents)!=0 and (variations == 'strat' or variations == 'flux'or variations == 'MLD'):
                if variations == 'strat':
                    plot_rig_exponents(color_opt, title, name_uni, fig_folder, w_rms, b_center, bw_fluc_avg, r_profile, T_fluc_center, S_avg, z_nd, Ri_g, case_names, exponents = exponents)
                if variations == 'flux':
                    plot_Fr_exponents(color_opt, title, name_uni, fig_folder, w_rms, b_center, bw_fluc_avg, r_profile, T_fluc_center, S_avg, z_nd, Fr_flux, case_names, exponents = exponents)
                if variations == 'MLD':
                    plot_mld_exponents(color_opt, title, name_uni, fig_folder, w_rms, b_center, bw_fluc_avg, r_profile, T_fluc_center, S_avg, z_nd, mld/rj, case_names, exponents = exponents)
    print("All frames created.")
    # creating videos
    if video:
        if plot_variables:
            for dir, name in enumerate(var_names):
                create_video(variable_dir[var_names[dir]], fig_folder, name_uni, name)
        if plot_1d_z:
            create_video(buoyancy_dir_z, fig_folder, name_uni, 'vertical profile')
        if plot_1d_y:
            create_video(buoyancy_dir_y, fig_folder, name_uni, 'horizontal profile')

