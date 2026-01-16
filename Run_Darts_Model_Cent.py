#Load model creation
from Basic_Ates_Model_Cent import Model
#Load helper functions
from simparams_2_DARTS_input import simparams_2DARTS_input
from simparams_2_DARTS_input import geomod_xarray_2DARTS_input
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from Read_well_h5_opendarts import read_well_h5, write_well_perforation_id

### =========== Simulation settings ==============
simulation_name = 'sim_1W_V300_T70' # Can be any string, used for naming the output Excel file and vtk data directory
# Specify the path to a manually inputted geomodel
# If nothing is specified, the reservoir characteristics later in the code are used
input_geomod_path = None #Path(r"C:\path\to\your\geomodel.nc")

# -------------- Output name and location -------
output_directory = f"vtk_data_{simulation_name}" #Directory name for spatial and pressure cell data
output_well_data_excel = f"well_data_{simulation_name}.xlsx" #Name of output Excel file

# --------------- Initial conditions reservoir -------
depth_to_top = 120 #[m], the depth to the top boundary of the upper clay layer (== top of geomodel)
geothermal_grad = 18 #[K / km], the geothermal gradient for the initial condition of the reservoir

# -------------- Operational parameters -----------------------
# Well temperatures [K]
InjT = 273.15 + 90
TCutOff = 273.15 + 50 # Serves as the injection temperature of the warm well

# Annual total (i.e. over all wells) storage volumes [m3/year] (from the hot well perspective)
V_in_hot = 200000
V_out_hot = 180000

# Block-function Operational profile
daysprofile = [120, 60, 120, 65] #[days]
storage_periods = ['Charge', 'Rest', 'Discharge', 'Rest']
op_profile = [1, 0, 1, 0]

# Well rates [m3/day]
Q_hot_inj   = V_in_hot / daysprofile[0]
Q_hot_prod  = V_out_hot / daysprofile[2]

Q_cold_prod = V_in_hot / daysprofile[0]
Q_cold_inj  = V_out_hot  / daysprofile[2]

# --------------- Simulation time and space settings  --------
set_run_years = 1 #Total simulation length [years]
max_ts = 30 #Maximum time step size [days]
dt_mult = 8 #Time step upscaling [-]
set_transition_runtime = 1e-3 #dt after an operational period change [days]

# ------------- Reservoir model characteristics --------

if geomod is not None and geomod.exists():
    print(f"Using geomodel: {input_geomodel_path.name}")
    n_ly, n_cells, dz_array, wells, perm_h, perm_v, poro, hcap, tcond = geomod_xarray_2DARTS_input(geomod_path)
else:
    print("Geomodel not found : using default 3 layer model")
    # ------------- Default 3 layer model  ---------
    # Thickness [m]
    H_top = 25
    H_res = 60
    H_bot = 20

    # Horizontal Permeability [mD]
    k_h_top = 0.1
    k_h_res = 1000
    k_h_bot = 0.1

    # Anisotropy ratio [-]
    ani_top = 5
    ani_res = 5
    ani_bot = 5

    # Porosity [-]
    por_top = 0.3
    por_res = 0.3
    por_bot = 0.3

    # Volumetric Heat Capacity [kJ/m3/K)]
    Cv_top = 2000
    Cv_res = 2000
    Cv_bot = 2000

    # Thermal Conductivity [kJ/m/day/K]
    lam_top = 200
    lam_res = 200
    lam_bot = 200

    # Spatial Discretization cell size [m]
    dx = 10
    dy = 10
    ## NOTE: H_top, H_res, H_bot need to be exactly divisible by dz_top, dz_res, dz_bot in the current version of the code
    dz_top = 5
    dz_res = 5
    dz_bot = 5

    # The horizontal domain size in which the HT-ATES simulation takes place [m]
    ## NOTE: dom_X and dom_Y need to be exactly divisible by dx and dy in the current version of the code
    dom_X = 600
    dom_Y = 1000

    # The spatial coordinates of the well in the reservoir [m], must be within [0 and dom_X]
    HW_X = 300
    HW_Y = 300
    CW_X = 300
    CW_Y = 700

    n_ly, n_cells, dz_array, wells, perm_h, perm_v, poro, hcap, tcond = simparams_2DARTS_input(dx, dy,
                                                                                               dz_top, dz_res, dz_bot,
                                                                                               dom_X, dom_Y,
                                                                                               H_top, H_res, H_bot,
                                                                                               HW_X, HW_Y, CW_X, CW_Y,
                                                                                               k_h_top, k_h_res,
                                                                                               k_h_bot,
                                                                                               ani_top, ani_res,
                                                                                               ani_bot,
                                                                                               por_top, por_res,
                                                                                               por_bot,
                                                                                               Cv_top, Cv_res, Cv_bot,
                                                                                               lam_top, lam_res,
                                                                                               lam_bot)



### ============== Run Simulation =====================
#Input model params here
m = Model(n_ly, n_cells, dx, dy, dz_array,
          perm_h, perm_v, poro, hcap, tcond,
          wells, depth_to_top, geothermal_grad, dt_mult, max_ts)
m.init()
m.set_output()

iterr = 1
for k in range(set_run_years):
    for i, runtime in enumerate(daysprofile):
        if storage_periods[i] == 'Charge':

            m.set_rate_hot(Q_hot_inj, temp=InjT, func='inj')
            m.set_rate_cold(Q_cold_prod, func='prod')
            print('Operation: Charge')

        elif storage_periods[i] == 'Discharge':

            m.set_rate_hot(Q_hot_prod, func='prod')
            m.set_rate_cold(Q_cold_inj, temp=TCutOff, func='inj')
            print('Operation: Discharge')

        elif storage_periods[i] == 'Rest':

            m.set_rate_hot(0, func='prod')
            m.set_rate_cold(0, func='prod')
            print('Operation: Rest')

        m.run(runtime, restart_dt=set_transition_runtime)

        print("\nIterr :", iterr, "\tYear :", k, "\tRun Time :", runtime)
        print("\n")
        iterr += 1
m.print_timers()
m.print_stat()
m.output.store_well_time_data()
# %%-----------------Write Results to Excel-----------------
# output well information to Excel file
td = pd.DataFrame.from_dict(m.physics.engine.time_data)
writer = pd.ExcelWriter(output_well_data_excel)
td.to_excel(writer, 'Sheet1')
writer.close()

# %%-----------------Plot Well Data-----------------
# plot temperature at production well and technological limit

for w in m.reservoir.wells:
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 9), dpi=300, sharex=True)
    fig.suptitle(f'{w.name}', weight='bold', fontsize=18)

    ax[0].plot(td['time'], td[f'{w.name} : temperature (K)'] - 273.15, color='red', marker='o', fillstyle='none',
               linestyle='-', markersize=8)
    ax[0].tick_params(labelsize=14)
    ax[0].set_ylabel('Temperature (C)', fontsize=14)
    ax[0].grid(True, linestyle='--', alpha=0.7)

    ax[1].plot(td['time'], td[f'{w.name} : BHP (bar)'], color='blue', marker='', fillstyle='none', linestyle='-',
               markersize=8)
    ax[1].tick_params(labelsize=14)
    ax[1].set_ylabel('BHP (bar)', fontsize=14)
    ax[1].grid(True, linestyle='--', alpha=0.7)

    ax[2].plot(td['time'], td[f'{w.name} : water rate (m3/day)'], color='green', marker='', fillstyle='none',
               linestyle='-', markersize=8)
    ax[2].tick_params(labelsize=14)
    ax[2].set_xlabel('Days', fontsize=14)
    ax[2].set_ylabel('Flow Rate (m3/day)', fontsize=14)
    ax[2].grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'20250414_{w.name}_5000mD.png', dpi=300, bbox_inches='tight')

# %%-----------------read H5-----------------
well_id, well_depth = write_well_perforation_id(m)
r = read_well_h5(well_block_id=well_id, well_block_depth=well_depth)
r.draw_combined_well_data()  # All wells on same subplots
