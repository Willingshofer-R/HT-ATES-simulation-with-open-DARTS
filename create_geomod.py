import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from plot_func import plot_grid


def create_ZGR(soil_df, dz_mult=1.7, print_stat=False):
    """
    Create vertical grid from soil profile DataFrame.

    Returns
    -------
    dZ_array : np.ndarray
        Cell thicknesses (top → bottom)
    n_ly : np.ndarray
        Number of layers per model unit (cap, res, bot)
    depth_to_top : float
        Depth to top of model
    """

    n_units = len(soil_df)

    n_ly = soil_df['n_layers'].to_numpy(dtype=int)

    cap_idx = soil_df.index[soil_df['type'] == 'cap'].to_numpy()
    bot_idx = soil_df.index[soil_df['type'] == 'bot'].to_numpy()

    first_cap = cap_idx[0] if len(cap_idx) > 0 else None
    last_bot = bot_idx[-1] if len(bot_idx) > 0 else None

    dZ_list = []

    for i in range(n_units):
        z_top = soil_df.loc[i, 'z_top']
        z_bot = soil_df.loc[i, 'z_bot']
        H = z_bot - z_top
        n = n_ly[i]

        if i == first_cap:
            H_next = soil_df.loc[i + 1, 'z_top'] - soil_df.loc[i + 1, 'z_bot']
            n_next = n_ly[i + 1]
            d_vert = H_next / n_next

            array = np.zeros((n))

            for i in range(n):
                array[i] = d_vert * (dz_mult ** i)

            coeff = H / np.sum(array)
            dz = np.flip(array * coeff)

            print("The cell closest to the reservoir with a variable cell size in the cap is of dz size:", dz[-1],
                  "m")

        elif i == last_bot:
            H_prev = soil_df.loc[i - 1, 'z_top'] - soil_df.loc[i - 1, 'z_bot']
            n_prev = n_ly[i - 1]
            d_vert = H_prev / n_prev

            array = np.zeros((n))

            for i in range(n):
                array[i] = d_vert * (dz_mult ** i)

            coeff = H / np.sum(array)
            dz = array * coeff
            print("The cell closest to the reservoir with a variable cell size in the bot is of dz size:", dz[0],
                  "m")

        else:
            dz = np.ones(n) * H / n

        dZ_list.append(dz)

    dZ_array = np.concatenate(dZ_list)

    n_ly_cap_res_bot = np.zeros((3))

    for i in range(n_units):
        if soil_df.loc[i, 'type'] == 'cap':
            n_ly_cap_res_bot[0] += n_ly[i]
        elif soil_df.loc[i, 'type'] == 'res':
            n_ly_cap_res_bot[1] += n_ly[i]
        elif soil_df.loc[i, 'type'] == 'bot':
            n_ly_cap_res_bot[2] += n_ly[i]

    depth_to_top = soil_df.loc[0, 'z_top']

    if print_stat:
        print(f"Z grid: {len(dZ_array)} layers")
        print(f"dz min / max: {dZ_array.min():.3f} / {dZ_array.max():.3f} m")
        print("Depth to the top layer =", depth_to_top, "m")

    return dZ_array, n_ly_cap_res_bot, depth_to_top


def build_axis_grid(
        coord_hot,
        coord_cold,
        d_min,
        d_med,
        d_max,
        d_min_bound,
        d_med_bound,
        mult_fact,
        n_max_bound,
        axis_name='x'):
    """
    Helper function: Builds 1D grid along one axis based on hot well coordinates

    Returns
    -------
    grid : np.ndarray
        Monotonically increasing grid node coordinates
    """

    h_min, h_max = coord_hot.min(), coord_hot.max()
    c_min, c_max = coord_cold.min(), coord_cold.max()

    h_lo = h_min - d_min_bound - 0.5 * d_min
    h_hi = h_max + d_min_bound + 0.5 * d_min
    c_lo = c_min - d_min_bound - 0.5 * d_min
    c_hi = c_max + d_min_bound + 0.5 * d_min

    fine_hot = np.arange(h_lo, h_hi + d_min, d_min)
    fine_cold = np.arange(c_lo, c_hi + d_min, d_min)

    if h_hi >= c_lo and c_hi >= h_lo:
        fine_lo = min(h_lo, c_lo)
        fine_hi = max(h_hi, c_hi)
        fine = np.arange(fine_lo, fine_hi + d_min, d_min)

    elif c_hi >= h_lo and c_lo <= h_hi:
        fine_lo = min(h_lo, c_lo)
        fine_hi = max(h_hi, c_hi)
        fine = np.arange(fine_lo, fine_hi + d_min, d_min)

    else:
        start_p = min(h_hi, c_hi)
        end_p = max(h_lo, c_lo)
        N_cells = len(np.arange(start_p, end_p, d_med))
        d_intermed = (end_p - start_p) / N_cells
        intermed_cells = np.arange(start_p, end_p + d_intermed, d_intermed)
        fine = np.sort(np.unique(np.concatenate([fine_hot, fine_cold, intermed_cells])))

        print("Grid alligned in direction:", axis_name, " with cells of size:", d_intermed, "m")
        print("Target was:", d_med, "m")

    fine_min = fine.min()
    fine_max = fine.max()

    med_lo = fine_min - d_med_bound
    med_hi = fine_max + d_med_bound

    med_left = np.arange(med_lo, fine_min, d_med)
    med_right = np.arange(fine_max + d_med, med_hi + d_med, d_med)
    grid = np.unique(np.concatenate([med_left, fine, med_right]))

    # Filter in case rounding errors have produced ultra fine grid cells
    grid_round = np.round(grid, 3)
    grid = grid_round[grid_round != 0]

    dx = d_med
    x = grid[0]
    left_nodes = []

    while dx < d_max:
        dx = min(dx * mult_fact, d_max)
        x -= dx
        left_nodes.append(x)

    dx = d_med
    x = grid[-1]
    right_nodes = []

    while dx < d_max:
        dx = min(dx * mult_fact, d_max)
        x += dx
        right_nodes.append(x)

    grid = np.unique(np.concatenate([left_nodes[::-1], grid, right_nodes]))

    left_pad = grid[0] - d_max * np.arange(n_max_bound, 0, -1)
    right_pad = grid[-1] + d_max * np.arange(1, n_max_bound + 1)

    grid = np.unique(np.concatenate([left_pad, grid, right_pad]))

    print(f"{axis_name.upper()} grid built: {len(grid) - 1} cells")

    return grid


def create_XY_grid(
        wells_df,
        d_min,
        d_min_bound,
        d_med,
        d_med_bound,
        d_max,
        mult_fact,
        n_max_bound,
        XY_GR_plot=True,
        image_loc = r'C:\Users',
        image_name = 'XY_grid_image'):

    """
    Function to construct the X and Y grid based on well-coordinates and user-specified settings

    Returns
    -------
    dX_array : np.ndarray
        array containing all cell widths in the x-direction
    dY_array: np.ndarray
        array containing all cell widths in the y-direction
    well_indices:
        data Frame containing the indices and well types of all wells
    """

    hot = wells_df[wells_df['type'] == 'H']
    cold = wells_df[wells_df['type'] == 'C']

    XGR = build_axis_grid(
        hot['x'].to_numpy(),
        cold['x'].to_numpy(),
        d_min, d_med, d_max,
        d_min_bound, d_med_bound, mult_fact, n_max_bound, 'x'
    )

    YGR = build_axis_grid(
        hot['y'].to_numpy(),
        cold['y'].to_numpy(),
        d_min, d_med, d_max,
        d_min_bound, d_med_bound, mult_fact, n_max_bound, 'y'
    )

    well_idx_records = []

    for _, w in wells_df.iterrows():
        ix = int(np.argmin(np.abs(XGR - w['x'])))
        iy = int(np.argmin(np.abs(YGR - w['y'])))

        well_idx_records.append({
            'well_index_x': ix,
            'well_index_y': iy,
            'type': w['type']
        })

    well_idx_df = pd.DataFrame(well_idx_records)

    print("Domain X from", min(XGR), "to", max(XGR), "m")
    print("Domain X from", min(YGR), "to", max(YGR), "m")

    if XY_GR_plot:
        plot_grid(XGR, YGR, image_loc, image_name, wells_df, show=True, save=True)

    return np.diff(XGR), np.diff(YGR), well_idx_df

def property_fill(
    dX_array,
    dY_array,
    soil_df,
    vertical_perm=False,
    mday_2_mD=True,
):

    nx = len(dX_array)
    ny = len(dY_array)

    n_ly = soil_df['n_layers'].to_numpy(dtype=int)
    nz = n_ly.sum()

    perm_h = np.zeros((nx, ny, nz))
    perm_v = np.zeros((nx, ny, nz))
    poro = np.zeros((nx, ny, nz))
    Tcond = np.zeros((nx, ny, nz))
    Cv = np.zeros((nx, ny, nz))

    kh = soil_df['kh'].to_numpy(float)

    if vertical_perm:
        kv = soil_df['ani'].to_numpy(float)
    else:
        kv = kh / soil_df['ani'].to_numpy(float)

    if mday_2_mD:
        kh = hydraulic_con_2_mD(15.0, kh)
        kv = hydraulic_con_2_mD(15.0, kv)

    k = 0
    for i in range(len(soil_df)):
        sl = slice(k, k + n_ly[i])

        perm_h[:, :, sl] = kh[i]
        perm_v[:, :, sl] = kv[i]
        poro[:, :, sl] = soil_df.loc[i, 'poro']
        Tcond[:, :, sl] = soil_df.loc[i, 'tcond']
        Cv[:, :, sl] = soil_df.loc[i, 'hcap']

        k += n_ly[i]

    print("Property fill summary:")
    print(f"  Grid size (nx, ny, ny) = {perm_h.shape}")
    print(f"  kh range: {perm_h.min():.2e} – {perm_h.max():.2e} mD")
    print(f"  kv range: {perm_v.min():.2e} – {perm_v.max():.2e} mD")

    return perm_h, perm_v, poro, Tcond, Cv


def hydraulic_con_2_mD(T, Kh):
    """
    In Beernink 2022 the relationships for mu and temperature are given.
    The temperature at which the flow experiment is conducted must be provided,
    to get accurate results for intrinsic permeability

    T in degrees C
    Kh in m/day
    """
    mu = (2.494 * 1e-5) * 10 ** (248.73 / (T + 133.15))
    mu_sec = mu / (24 * 3600)

    rho = 1000 - ((T ** 2 - 4) / 207)

    k_m2 = ((Kh * mu_sec) / (rho * 9.81))
    k_mD = k_m2 / (1e-15)
    return k_mD


def get_well_rates_block(V_in_hot, V_out_hot, daysprofile, storage_periods, wells_df):
    """
    Compute per-well injection and production rates for a block-type operation.

    Rates are computed by identifying Charge and Discharge periods from storage_periods
    """

    hot = wells_df[wells_df['type'] == 'H']
    cold = wells_df[wells_df['type'] == 'C']

    n_hot = len(hot)
    n_cold = len(cold)

    if n_hot == 0 and n_cold == 0:
        raise ValueError("No wells detected.")

    daysprofile = np.asarray(daysprofile)

    charge_idx = [i for i, p in enumerate(storage_periods) if p == 'Charge']
    discharge_idx = [i for i, p in enumerate(storage_periods) if p == 'Discharge']

    # Well rates [m3/day]
    Q_hot_inj = V_in_hot / (daysprofile[charge_idx] * n_hot)
    Q_hot_prod = V_out_hot / (daysprofile[discharge_idx] * n_hot)

    Q_cold_prod = V_in_hot / (daysprofile[charge_idx] * n_cold)
    Q_cold_inj = V_out_hot / (daysprofile[discharge_idx] * n_cold)

    return n_hot, n_cold, Q_hot_inj, Q_hot_prod, Q_cold_prod, Q_cold_inj