import numpy as np

def simparams_2DARTS_input(dx, dy, dz_top, dz_res, dz_bot,
                           dom_X, dom_Y, H_top, H_res, H_bot,
                           HW_X, HW_Y, CW_X, CW_Y,
                           k_h_top, k_h_res, k_h_bot, ani_top, ani_res, ani_bot,
                           por_top, por_res, por_bot, Cv_top, Cv_res, Cv_bot,
                           lam_top, lam_res, lam_bot):
    """
    Convert Run_Darts_Model input parameters to valid DARTS model initialization.
    Grid ordering is (nz, ny, nx). -> Double check!!
    """

    # Number of vertical layers
    n_ly_top = int(H_top / dz_top)
    n_ly_res = int(H_res / dz_res)
    n_ly_bot = int(H_bot / dz_bot)

    n_ly = np.array([n_ly_top, n_ly_res, n_ly_bot])

    dz_array = np.concatenate([
        np.ones(n_ly_top) * dz_top,
        np.ones(n_ly_res) * dz_res,
        np.ones(n_ly_bot) * dz_bot
    ])

    # Number of grid cells
    n_x = int(dom_X / dx)
    n_y = int(dom_Y / dy)
    n_z = np.sum(n_ly)

    n_cells = np.array([n_x, n_y, n_z])

    n_cells_tot = n_x * n_y * n_z
    print("Total number of cells in reservoir and cap layers =", n_cells_tot)

    # Well indices (x, y) Assumes grid starts at X = 0,Y = 0!
    idx_HW_X = int(HW_X / dx)
    idx_HW_Y = int(HW_Y / dy)
    idx_CW_X = int(CW_X / dx)
    idx_CW_Y = int(CW_Y / dy)

    well_indices = np.array([
        [idx_HW_X, idx_HW_Y],
        [idx_CW_X, idx_CW_Y]
    ])

    # Grid cell property arrays (nx, ny, nz)
    top = np.ones((n_x, n_y, n_ly_top))
    reservoir = np.ones((n_x, n_y, n_ly_res))
    bot = np.ones((n_x, n_y, n_ly_bot))

    # Permeability
    perm_h_top = top * k_h_top
    perm_v_top = perm_h_top / ani_top

    perm_h_res = reservoir * k_h_res
    perm_v_res = perm_h_res / ani_res

    perm_h_bot = bot * k_h_bot
    perm_v_bot = perm_h_bot / ani_bot

    perm_h_full = np.concatenate(
        [perm_h_top, perm_h_res, perm_h_bot], axis = 2)

    perm_v_full = np.concatenate(
        [perm_v_top, perm_v_res, perm_v_bot], axis =2 )

    # Porosity
    porosity_top = top * por_top
    porosity_res = reservoir * por_res
    porosity_bot = bot * por_bot

    porosity_full = np.concatenate(
        [porosity_top, porosity_res, porosity_bot], axis = 2)

    # Volumetric heat capacity
    Cv_top_arr = top * Cv_top
    Cv_res_arr = reservoir * Cv_res
    Cv_bot_arr = bot * Cv_bot

    Cv_full = np.concatenate(
        [Cv_top_arr, Cv_res_arr, Cv_bot_arr], axis = 2)

    # Thermal conductivity
    lam_top_arr = top * lam_top
    lam_res_arr = reservoir * lam_res
    lam_bot_arr = bot * lam_bot

    lam_full = np.concatenate(
        [lam_top_arr, lam_res_arr, lam_bot_arr], axis = 2)

    return (
        n_ly,
        n_cells,
        dz_array,
        well_indices,
        perm_h_full,
        perm_v_full,
        porosity_full,
        Cv_full,
        lam_full
    )
