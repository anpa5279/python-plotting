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
def plot_format(fontsize = 20):
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
## for multiple case comparison plotting
def comparison_plot_opt(cases, case_opt = 'coolwarm', distinct_groups = [1, ]):
    if isinstance(cases, int):
        ncases = cases
        cases = [f'case_{i}' for i in range(ncases)]
    else:
        ncases = len(cases)
    case_opts = {}
    line_styles = ['dashed', 'dotted', 'dashdot', 'dashdotted', 'solid']
    markers = ['o', 's', '^', 'D', 'v', '*', 'x', '+', 'p', 'h']
    if len(distinct_groups)>1: # defining distinct groups by colors
        case_opt = ['Reds', 'Blues', 'Purples', 'Greens', 'Oranges', 'OrRd', 'BuPu'][0:len(distinct_groups)]
        for n, group in enumerate(distinct_groups):
            cmap = plt.get_cmap(case_opt[n], group+1)
            group_cases = cases[sum(distinct_groups[:n]):sum(distinct_groups[:n+1])]
            for i, case in enumerate(group_cases):
                case_opts[case] = {'color': cmap(i+1), 'linestyle': line_styles[i], 'marker': markers[i%len(markers)]}
    else:
        if case_opt == 'coolwarm' or case_opt == 'viridis' or case_opt == 'plasma' or case_opt == 'cividis':
            cmap = plt.get_cmap(case_opt, ncases)
            for i, case in enumerate(cases):
                case_opts[case] = {'color': cmap(i), 'linestyle': line_styles[i%len(line_styles)], 'marker': markers[i%len(markers)]}
        else:
            cmap = plt.get_cmap(case_opt, ncases+1)
            for i in range(ncases+1):
                if i < (ncases):
                    case_opts[cases[i]] = {'color': cmap(i), 'linestyle': line_styles[i%len(line_styles)], 'marker': markers[i%len(markers)]}
                else:
                    cmap(i)
    #case_opts[cases[-1]]['linestyle'] = 'solid' # make last case solid line
    return case_opts
    # if you want cases separated by line
### -------------------------SAVING FRAMES AND MAKING VIDEOS------------------------- ###
def round_up_to_multiple(px, m=16):
    return int(np.ceil(px / m) * m)
def save_frame(fig, outdir, it, size_in, file_name='', dpi=200):
    w_px = round_up_to_multiple(size_in[0] * dpi)
    h_px = round_up_to_multiple(size_in[1] * dpi)
    fig.set_size_inches(w_px / dpi, h_px / dpi)
    frame_path = os.path.join(outdir, file_name + f'{it:04d}.png')
    plt.savefig(frame_path, dpi=dpi)
    plt.close(fig)
def create_video(outdir, fig_folder, name, plot_type):
    print("Creating video...")
    filenames = sorted([f for f in os.listdir(outdir) if f.endswith(".png")])
    vid_name = os.path.join(fig_folder, name + plot_type + '.mp4')
    with imageio.get_writer(vid_name, fps=10) as writer:
        for filename in filenames:
            image = imageio.imread(f"{outdir}/{filename}")
            writer.append_data(image)
            print(f"Appending frame: {filename}")
    print(f"Video saved as {vid_name}")
