import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
### ----------------------------------PROFILES------------------------------- ###
## stratification profile
def stratification_profile(z, a0, dadz, mld):
    """Returns a linear stratification profile."""
    a = a0 * np.ones(len(z))
    a[z<mld] = a0 + dadz * (z[z<mld] - mld)
    return a
### -------------------------PLOTTING PREP FUNCTIONS------------------------- ###
## default plot formatting 
def plot_format(fontsize = 12):
    plt.rcParams['font.family'] = 'serif' # or 'sans-serif' or 'monospace'
    plt.rcParams['font.serif'] = 'cmr10'
    plt.rcParams['font.sans-serif'] = 'cmss10'
    plt.rcParams['font.monospace'] = 'cmtt10'
    plt.rcParams["axes.formatter.use_mathtext"] = True 
    plt.rcParams['font.size'] = fontsize
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'DejaVu Serif'
    plt.rcParams['mathtext.it'] = 'DejaVu Serif:italic'
    plt.rcParams['mathtext.bf'] = 'DejaVu Serif:bold'
## defining ranges for plotting
def plot_ranges(lz = 96, mld = 60, rho0 = 1026, T0 = 25, dTdz = 0.01, C = 0.04, C_tol = 10**(-7)):
    ranges = {}
    list_pqr = ['u', 'v', 'w', 'b', 'T', 'Tracer', 'Pdynamic', 'Pstatic', 'rho', 
                'b_flux', 
                'vel_rms', 'b_rms', 
                'b_avg', 'T_avg', 'vel_avg', 'lamb_avg', 'Tracer_avg', 
                'vel_restress', 'vel_flux', 'Ri', 
                'u_fluc', 'v_fluc', 'w_fluc', 'b_fluc', 'vel_fluc', 'bw_fluc', 'Tw_fluc', 'rho_fluc', 'T_fluc', 'Tracer_fluc',
                'lengthscale', 'gradb', 'alphas', 
                'Q', 'F', 'M', 'B']
    for i in range(0,len(list_pqr),1):
        ranges[list_pqr[i]] = list()
    ranges['u'] = [-0.002, 0.002]
    ranges['v'] = [-0.002, 0.002]
    ranges['w'] = [-0.002, 0.002]
    ranges['u_fluc'] = [-0.002, 0.002]
    ranges['v_fluc'] = [-0.002, 0.002]
    ranges['w_fluc'] = [-0.002, 0.002]
    ranges['b'] = [-1.5*10**(-3), 10**(-5)]
    ranges['T'] = [T0-(dTdz*(lz-mld))+0.1, T0 + 0.02]
    ranges['Tracer'] = [C_tol, C]
    ranges['vel'] = [-0.00035, 0.00035]
    ranges['vel_rms'] = [0, 0.004]
    ranges['vel_flux'] = [-1*10**(-2), 1*10**(-2)]
    ranges['restress'] = [-3*10**(-6), 3*10**(-6)]
    ranges['Ri'] = [0, 5*10**(-3)]
    ranges['Pdynamic'] = [-0.005, 0.005]
    ranges['Pstatic'] = [-0.05, 0.05]
    ranges['b_avg'] = [-1.5*10**(-3), 1.0*10**(-5)]
    ranges['T_avg'] = ranges['T']
    ranges['lamb_avg'] = [-4*10**(-6), 4*10**(-6)]
    ranges['b_rms'] = [0, 2*10**(-5)]
    ranges['bw_fluc'] = [-1*10**(-8), 1*10**(-8)]
    ranges['b_flux'] = [-1*10**(-8), 1*10**(-8)]
    ranges['b_fluc'] = [-1*10**(-4), 1*10**(-4)]
    ranges['lengthscale'] = [0, 0.4]
    ranges['rho'] = [rho0-0.02, rho0+0.15]
    ranges['rho_fluc'] = [-0.02, 0.02]
    ranges['brms_sign'] = [0, 5*10**(-5)]
    ranges['bflux_rms'] = [-1*10**(-5), 1*10**(-5)]
    ranges['z_sign'] = [-1*lz, 0]
    ranges['alphas'] = [0, 0.2]
    ranges['gradb'] = [-3*10**(-5), 3*10**(-5)]
    ranges['T_fluc'] = [-5*10**(-1), 5*10**(-1)]
    ranges['Tracer_fluc'] = [-1*10**(-1), 1*10**(-1)]
    ranges['Tw_fluc'] = [-1.6*10**(-4), 1.6*10**(-4)]
    ranges['Cw'] = [-1*10**(-5), 1*10**(-5)]
    for key in ranges:
        ranges[key] = np.array(ranges[key])
    return ranges
## for multiple case comparison plotting
def comparison_plot_opt(ncases):
    colors = ['black', 'red', 'blue', 'green', 'orange', 'purple', 'pink', 'gray', 'olive', 'cyan', 'magenta']
    line_styles = ['solid', 'dashed', 'dotted', 'dashdot', 'dashdotted']

    return colors[:ncases], line_styles[:ncases]
### -------------------------SAVING FRAMES AND MAKING VIDEOS------------------------- ###
def create_video(outdir, fig_folder, name, plot_type):
    print("Creating video...")
    filenames = sorted([f for f in os.listdir(outdir) if f.endswith(".png")])
    vid_name = os.path.join(fig_folder, name + plot_type + '.mp4')
    with imageio.get_writer(vid_name, fps=10) as writer:
        for filename in filenames:
            image = imageio.imread(f"{outdir}/{filename}")
            writer.append_data(image)
    print(f"Video saved as {vid_name}")
