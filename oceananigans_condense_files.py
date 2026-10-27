"""
Condense an Oceananigans run and ALL of its nested pickup runs
(folder/pickup/pickup/...) into a single folder.

The output follows the rank layout of the ORIGINAL (top-level) run. If a pickup
level was run on more ranks than the original, groups of consecutive pickup
ranks are stitched together along x to fit the original layout.

Whether the number of processors differs is detected automatically from the
number of field files each reader finds (len(reader.files)).
"""
import os
import shutil
from contextlib import ExitStack
from dataclasses import dataclass

import h5py
import numpy as np

from reader import OceananigansData

# ==========================================================
# SETTINGS
# ==========================================================
folder = '/glade/derecho/scratch/apauls/outputs/version109/square-inlet/open-bottom-BC/AR1/dxi00625/WENO9/'

field_files_flag = True
high_freq_files_flag = False

salinity = True
with_halos = False

# The xy-average file stores a sum over ranks (OceananigansData.load_averages
# divides by Nranks). If a pickup run used a different number of ranks than the
# original, its averages must be rescaled so that dividing by the ORIGINAL
# Nranks still gives the right mean.
RESCALE_XY_AVG = True

PICKUP_NAME = 'pickup'
OUTDIR_NAME = 'condensed_files'


# ==========================================================
# HELPERS
# ==========================================================
@dataclass
class Level:
    folder: str
    reader: OceananigansData
    iter_offset: int      # added to this level's saved iteration numbers
    time_offset: float    # added to this level's saved times

    @property
    def nranks(self):
        return len(self.reader.files)

    @property
    def halo(self):
        """x-halo size on each side of the saved field arrays (0 if no halos)."""
        return int(self.reader.hx[0]) if with_halos else 0


def find_pickup_chain(top_folder):
    """[folder, folder/pickup, folder/pickup/pickup, ...] for every folder that exists."""
    chain = [top_folder]
    while True:
        nxt = os.path.join(chain[-1], PICKUP_NAME)
        if not os.path.isdir(nxt):
            return chain
        chain.append(nxt)


def rank_ratio(level_nranks, main_nranks):
    """How many pickup ranks make up one original rank."""
    if level_nranks == main_nranks:
        return 1
    if level_nranks > main_nranks and level_nranks % main_nranks == 0:
        return level_nranks // main_nranks
    raise ValueError(
        f'Pickup run has {level_nranks} ranks and the original has {main_nranks}; '
        f'only an equal or integer-multiple rank count is supported.'
    )


def merge_along_x(pieces, halo):
    """
    Stitch per-rank arrays together along x. Raw field arrays are stored (z, y, x),
    so x is the LAST axis. Interior halos (right halo of every piece but the last,
    left halo of every piece but the first) are dropped.
    """
    if len(pieces) == 1:
        return pieces[0]
    n = len(pieces)
    trimmed = []
    for i, p in enumerate(pieces):
        lo = 0 if i == 0 else halo
        hi = p.shape[-1] if i == n - 1 else p.shape[-1] - halo
        trimmed.append(p[..., lo:hi])
    return np.concatenate(trimmed, axis=-1)


# ==========================================================
# FIELD FILES
# ==========================================================
def append_field_level(level, main_files, outdir, ratio, variables):
    """Append one pickup level's field files to the condensed (main-layout) files."""
    r = level.reader
    save_iters = [t for t in r.t_save if t != 0]  # first save duplicates the end of the previous run
    print(f'  Adding {len(save_iters)} field saves '
          f'(iteration offset {level.iter_offset}, rank ratio {ratio})')

    for g, out_name in enumerate(main_files):
        src_names = r.files[g * ratio:(g + 1) * ratio]
        with ExitStack() as stack:
            srcs = [stack.enter_context(h5py.File(os.path.join(level.folder, n), 'r'))
                    for n in src_names]
            dst = stack.enter_context(h5py.File(os.path.join(outdir, out_name), 'a'))

            for t_save in save_iters:
                k = int(t_save)
                new = int(t_save + level.iter_offset)
                t = srcs[0][f'timeseries/t/{k}'][()] + level.time_offset
                dst.create_dataset(f'timeseries/t/{new}', data=t)
                for var in variables:
                    pieces = [s[f'timeseries/{var}/{k}'][:] for s in srcs]
                    dst.create_dataset(f'timeseries/{var}/{new}',
                                       data=merge_along_x(pieces, level.halo))


# ==========================================================
# HIGH-FREQUENCY FILES (xy average + centerline)
# ==========================================================
def append_highfreq_level(level, main_nranks, is_first, is_last, f_avg, f_cen, variables):
    r = level.reader
    t_saves = np.asarray(r.t_save_avg)
    keep = np.ones(len(t_saves), dtype=bool)
    if not is_first:
        keep &= t_saves != 0                 # duplicate of the previous run's final save
    if not is_last:
        keep &= t_saves <= r.t_save[-1]      # drop output past the point the next run restarts from
    sel = t_saves[keep]

    scale = 1.0
    if RESCALE_XY_AVG and level.nranks != main_nranks:
        scale = main_nranks / level.nranks
    print(f'  Adding {len(sel)} high-frequency saves '
          f'(iteration offset {level.iter_offset}, xy_avg scale {scale:g})')

    with h5py.File(os.path.join(level.folder, r.averaging_file), 'r') as src_avg, \
         h5py.File(os.path.join(level.folder, r.centerline_output), 'r') as src_cen:
        for t_save in sel:
            k = int(t_save)
            new = int(t_save + level.iter_offset)
            t = src_avg[f'timeseries/t/{k}'][()] + level.time_offset
            f_avg.create_dataset(f'timeseries/t/{new}', data=t)
            f_cen.create_dataset(f'timeseries/t/{new}', data=t)
            for var in variables:
                avg = src_avg[f'timeseries/{var}_avg/{k}'][:]
                if scale != 1.0:
                    avg = avg * scale
                f_avg.create_dataset(f'timeseries/{var}_avg/{new}', data=avg)
                f_cen.create_dataset(f'timeseries/{var}/{new}',
                                     data=src_cen[f'timeseries/{var}/{k}'][:])


# ==========================================================
# MAIN
# ==========================================================
def main():
    outdir = os.path.join(folder, OUTDIR_NAME)
    os.makedirs(outdir, exist_ok=True)
    shutil.copy2(os.path.join(folder, 'grid_info.jld2'), os.path.join(outdir, 'grid_info.jld2'))

    variables = ['u', 'v', 'w', 'T', 'S'] if salinity else ['u', 'v', 'w', 'T']
 
    # ---- build the chain of runs: folder, folder/pickup, folder/pickup/pickup, ...
    chain = find_pickup_chain(folder)
    print(f'Found {len(chain)} run(s) to condense:')
    levels = []
    iter_off, time_off = 0, 0.0
    for i, fld in enumerate(chain):
        file_name = 'fields_pickup' if i > 0 else 'fields'
        print(f'  [{i}] {fld} with file name prefix "{file_name}"')
        reader = OceananigansData(fld, salinity=salinity, with_halos=with_halos, file_name=file_name)
        lv = Level(fld, reader, iter_off, time_off)
        levels.append(lv)
        print(f'  [{i}] {fld}  (ranks = {lv.nranks}, x-halo = {lv.halo}, '
              f'last saved iteration = {int(reader.t_save[-1])})')
        iter_off += int(reader.t_save[-1])
        time_off += float(reader.t[-1])

    main_level = levels[0]
    main_nranks = main_level.nranks
    ratios = [rank_ratio(lv.nranks, main_nranks) for lv in levels[1:]]
    same_nprocs = all(r == 1 for r in ratios)
    print(f'Same number of processors across all runs: {same_nprocs}')

    if any(lv.halo != main_level.halo for lv in levels[1:]):
        raise ValueError('Halo size differs between runs; cannot combine field files.')

    # ---- field files
    if field_files_flag:
        print(f'Copying field files to {outdir}')
        main_files = list(main_level.reader.files)
        for name in main_files:
            shutil.copy2(os.path.join(folder, name), os.path.join(outdir, name))
        for i, (lv, ratio) in enumerate(zip(levels[1:], ratios), start=1):
            print(f'Level {i}: {lv.folder}')
            append_field_level(lv, main_files, outdir, ratio, variables)

    # ---- high-frequency files
    if high_freq_files_flag:
        avg_out = os.path.join(outdir, 'xy_avg.jld2')
        cen_out = os.path.join(outdir, 'centerline.jld2')
        print(f'Combining averaging and centerline files into {outdir}')
        with h5py.File(avg_out, 'w') as f_avg, h5py.File(cen_out, 'w') as f_cen:
            for i, lv in enumerate(levels):
                print(f'Level {i}: {lv.folder}')
                append_highfreq_level(lv, main_nranks, i == 0, i == len(levels) - 1,
                                      f_avg, f_cen, variables)

            # buoyancy information from the original run
            with h5py.File(os.path.join(folder, main_level.reader.averaging_file), 'r') as f:
                alpha = f['buoyancy/formulation/equation_of_state/thermal_expansion'][()]
                beta = f['buoyancy/formulation/equation_of_state/haline_contraction'][()]
            f_avg.create_dataset('buoyancy/formulation/equation_of_state/thermal_expansion', data=alpha)
            f_avg.create_dataset('buoyancy/formulation/equation_of_state/haline_contraction', data=beta)

    print(f'Condensed files saved to {outdir}')


if __name__ == '__main__':
    main()