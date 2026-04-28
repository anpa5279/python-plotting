import numpy as np
import h5py

### -------------------------IMPORTANT DEPTHS------------------------- ###
# mixed layer depth information
def mld_info(w, bw_fluc, rho_perturbed, z, mld): # inputs are 1d arrays
    # info at mixed layer depth
    dz_ml = np.abs(z + mld)/mld
    mld_idx = np.where(dz_ml==dz_ml.min())[0][-1]
    mld_w = w[mld_idx]
    mld_bw_fluc = bw_fluc[mld_idx]
    mld_rho_perturbed = rho_perturbed[mld_idx]
    return mld_idx, mld_w, mld_bw_fluc, mld_rho_perturbed

### -------------------------CALCULATING TEMPORAL AVERAGES------------------------- ###
def compute_temporal_averages(reader, time_indices, grid, physics, interp, center=(0.0, 0.0)):
    x, y, z = grid["x"], grid["y"], grid["z"]
    nx = grid["nx"]
    x0, y0 = center
    n = 0
    # ---------------- initializing arrays ---------------- #
    S_avg = np.zeros(nx[2])
    T_avg = np.zeros(nx[2])
    b_avg = np.zeros(nx[2])
    w_avg = np.zeros(nx[2])

    Tw_avg = np.zeros(nx[2])
    Sw_avg = np.zeros(nx[2])
    bw_avg = np.zeros(nx[2])

    u_rms = np.zeros(nx[2])
    v_rms = np.zeros(nx[2])
    w_rms = np.zeros(nx[2])

    S_center = np.zeros(nx[2])
    T_center = np.zeros(nx[2])
    b_center = np.zeros(nx[2])
    w_center = np.zeros(nx[2])

    # ---------------- time loop ---------------- #
    for it in time_indices:
        u, v, w, T, S, _, _ = reader(it)
        # center velocities
        u = interp.fcc_ccc(u)
        v = interp.cfc_ccc(v)
        w = interp.ccf_ccc(w)

        # buoyancy
        b = (
            physics["g"] * physics["alpha"] * (T - physics["T0"])
            - physics["g"] * physics["beta"] * (S - physics["S0"])
        )

        # ---------------- horizontal means ---------------- #
        S_h = np.mean(S, axis=(-3, -2))
        T_h = np.mean(T, axis=(-3, -2))
        b_h = np.mean(b, axis=(-3, -2))
        w_h = np.mean(w, axis=(-3, -2))
        # ---------------- fluctuations ---------------- #
        S_fluc = S - S_h
        T_fluc = T - T_h
        b_fluc = b - b_h
        w_fluc = w - w_h
        # ---------------- fluxes ---------------- #
        Tw = T_fluc * w_fluc
        Sw = S_fluc * w_fluc
        bw = b_fluc * w_fluc
        # ---------------- RMS ---------------- #
        u_rms += np.mean((u - np.mean(u, axis=(-3, -2)))**2, axis=(-3, -2))
        v_rms += np.mean((v - np.mean(v, axis=(-3, -2)))**2, axis=(-3, -2))
        w_rms += np.mean(w_fluc**2, axis=(-3, -2))
        # ---------------- accumulation ---------------- #
        S_avg += S_h
        T_avg += T_h
        b_avg += b_h
        w_avg += w_h
        Tw_avg += np.mean(Tw, axis=(-3, -2))
        Sw_avg += np.mean(Sw, axis=(-3, -2))
        bw_avg += np.mean(bw, axis=(-3, -2))
        # ---------------- centerline ---------------- #
        S_center += interp.xy_plane(S, z, z0=0.0)
        T_center += interp.xy_plane(T, z, z0=0.0)
        b_center += interp.xy_plane(b, z, z0=0.0)
        w_center += interp.xy_plane(w, z, z0=0.0)

        n += 1

    inv_n = 1.0 / n

    return {
        "S_avg": S_avg * inv_n,
        "T_avg": T_avg * inv_n,
        "b_avg": b_avg * inv_n,
        "w_avg": w_avg * inv_n,

        "Tw_avg": Tw_avg * inv_n,
        "Sw_avg": Sw_avg * inv_n,
        "bw_avg": bw_avg * inv_n,

        "u_rms": np.sqrt(u_rms * inv_n),
        "v_rms": np.sqrt(v_rms * inv_n),
        "w_rms": np.sqrt(w_rms * inv_n),

        "S_center": S_center * inv_n,
        "T_center": T_center * inv_n,
        "b_center": b_center * inv_n,
        "w_center": w_center * inv_n,
    }
def write_temporal_averages(file_path, data):
    with h5py.File(file_path, "w") as f:
        grp1 = f.create_group("1D averages")
        grp2 = f.create_group("centerline")
        for k, v in data.items():
            if "center" in k:
                grp2.create_dataset(k, data=v)
            else:
                grp1.create_dataset(k, data=v)
