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
folder = '/scratch/alpine/anpa5279/dxi00625/weno9/testing'

field_files_flag = False
high_freq_files_flag = False
time_step_debugging = True   # also save the very last time step as one global 3D file

salinity = True
with_halos = True

# Raw xy_avg files store a sum over ranks, so each run's averages are divided by
# THAT run's own number of ranks before being combined. The result is written to
# a file ending in '_condensed.jld2' so the reader knows not to divide by Nranks again.
AVG_OUT_NAME = 'xy_avg_condensed.jld2'
DEBUG_OUT_NAME = 'last_time_step.jld2'

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


def has_field_files(level_folder):
    """Same criterion OceananigansData uses to find field files."""
    return any(f.endswith('.jld2') and f.startswith('fields')
               for f in os.listdir(level_folder))


def ensure_grid_info(level_folder, top_folder):
    """The reader wants a grid_info.jld2; borrow the original run's if the pickup has none."""
    gi = os.path.join(level_folder, 'grid_info.jld2')
    if not os.path.exists(gi):
        print(f'  No grid_info in {level_folder}; copying from {top_folder}')
        shutil.copy2(os.path.join(top_folder, 'grid_info.jld2'), gi)


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
# LAST TIME STEP (DEBUGGING)
# ==========================================================
def strip_halos(arr, hx):
    """Drop halos from a raw (z, y, x) array and return it as (x, y, z), like OceananigansData.lazy_field."""
    hx_, hy, hz = hx
    arr = arr[hz:-hz or None, hy:-hy or None, hx_:-hx_ or None]
    return arr.transpose(2, 1, 0)


def save_last_time_step(level, outdir, variables):
    """
    Assemble every rank's data for the final saved time step of `level` into one
    global (Nx, Ny, Nz) array per variable. Reads straight from the last run's own
    field files, so it does not depend on the condensed field files existing.
    """
    r = level.reader
    t_last = r.t_save[-1]
    k = int(t_last)
    iteration = int(t_last + level.iter_offset)
    time = float(r.t[-1]) + level.time_offset
    hx = [int(h) for h in r.hx] if with_halos else [0, 0, 0]
    out_path = os.path.join(outdir, DEBUG_OUT_NAME)
    print(f'Saving last time step (iteration {iteration}, from {level.folder}, '
          f'{level.nranks} rank files) to {out_path}')

    dsets, nx_local = {}, {}
    with h5py.File(out_path, 'w') as out:
        out.create_dataset('iteration', data=iteration)
        out.create_dataset('t', data=time)
        out.attrs['source_folder'] = str(level.folder)
        out.attrs['array_order'] = 'x, y, z (halos removed)'
        for rank, fname in enumerate(r.files):
            with h5py.File(os.path.join(level.folder, fname), 'r') as f:
                for var in variables:
                    block = strip_halos(f[f'timeseries/{var}/{k}'][...], hx)
                    if var not in dsets:
                        nx_local[var], ny, nz = block.shape
                        dsets[var] = out.create_dataset(
                            var, shape=(nx_local[var] * level.nranks, ny, nz), dtype=block.dtype)
                    n = nx_local[var]
                    dsets[var][rank * n:(rank + 1) * n] = block


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
def append_highfreq_level(level, is_first, is_last, f_avg, f_cen, variables):
    r = level.reader
    t_saves = np.asarray(r.t_save_avg)
    keep = np.ones(len(t_saves), dtype=bool)
    if not is_first:
        keep &= t_saves != 0                 # duplicate of the previous run's final save
    if not is_last:
        keep &= t_saves <= r.t_save[-1]      # drop output past the point the next run restarts from
    sel = t_saves[keep]

    norm = level.nranks
    print(f'  Adding {len(sel)} high-frequency saves '
          f'(iteration offset {level.iter_offset}, xy_avg divided by {norm} ranks)')

    with h5py.File(os.path.join(level.folder, r.averaging_file), 'r') as src_avg, \
         h5py.File(os.path.join(level.folder, r.centerline_output), 'r') as src_cen:
        for t_save in sel:
            k = int(t_save)
            new = int(t_save + level.iter_offset)
            t = src_avg[f'timeseries/t/{k}'][()] + level.time_offset
            f_avg.create_dataset(f'timeseries/t/{new}', data=t)
            f_cen.create_dataset(f'timeseries/t/{new}', data=t)
            for var in variables:
                avg = src_avg[f'timeseries/{var}_avg/{k}'][:] / norm
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
        if not has_field_files(fld):
            found = sorted(f for f in os.listdir(fld) if f.endswith(('.jld2', '.h5')))
            msg = (f'No field files (fields*.jld2, excluding fields_pickup*) in {fld}. '
                   f'.jld2/.h5 files present: {found}')
            if i == 0:
                raise FileNotFoundError(msg)
            print(f'  WARNING: {msg}\n  Stopping the chain here; later pickups (if any) are ignored.')
            break
        if i > 0:
            ensure_grid_info(fld, folder)
        rdr = OceananigansData(fld, salinity=salinity, with_halos=with_halos)
        lv = Level(fld, rdr, iter_off, time_off)
        levels.append(lv)
        print(f'  [{i}] {fld}  (ranks = {lv.nranks}, x-halo = {lv.halo}, '
              f'last saved iteration = {int(rdr.t_save[-1])})')
        iter_off += int(rdr.t_save[-1])
        time_off += float(rdr.t[-1])

    if time_step_debugging:
        save_last_time_step(levels[-1], outdir, variables)

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
        avg_out = os.path.join(outdir, AVG_OUT_NAME)
        cen_out = os.path.join(outdir, 'centerline.jld2')
        print(f'Combining averaging and centerline files into {outdir}')
        with h5py.File(avg_out, 'w') as f_avg, h5py.File(cen_out, 'w') as f_cen:
            for i, lv in enumerate(levels):
                print(f'Level {i}: {lv.folder}')
                append_highfreq_level(lv, i == 0, i == len(levels) - 1,
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