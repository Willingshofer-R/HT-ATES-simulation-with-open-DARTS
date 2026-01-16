#Load model creation
from Basic_Ates_Model_v2 import Model
#Load helper functions
from create_geomod import create_ZGR
from create_geomod import create_XY_grid
from create_geomod import property_fill
from create_geomod import get_well_rates_block

import matplotlib.pyplot as plt
import pandas as pd
from Read_well_h5_opendarts import read_well_h5, write_well_perforation_id

### =========== Simulation settings ==============
simulation_name = 's_dmin2_bound_10_dmed_10_bound_100' # Can be any string, used for naming the output Excel file and vtk data directory

# -------------- Output name and location -------
# 2D grid plot name and save location, if safe == True
image_loc = r'C:\Users\robin\Documents\TU_Delft_GTEcon'
image_name = 'test_XY_grid'
# Output vtk and xlsx data
output_directory = f"vtk_data_{simulation_name}" #Directory name for spatial and pressure cell data
output_well_data_excel = f"well_data_{simulation_name}.xlsx" #Name of output Excel file

# -------------- Input soil and well data -------
soil = pd.DataFrame({
    'z_top':  [75., 123, 143, 145, 151, 157, 163, 167, 183],
    'z_bot':  [123, 143, 145, 151, 157, 163, 167, 183, 209],
    'type':   ['cap','res','res','res','res','res','res','res','bot'],
    'n_layers':[5, 20, 2, 6, 6, 6, 4, 16, 5],
    'kh':     [0.05, 10.5, 0.05, 11.6, 0.044, 17.7,0.057,12.30,0.050],
    'ani':    [10, 4, 4, 4, 4, 4, 4, 4, 10],
    'poro':   [0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35],
    'tcond':  [155.5, 190.0, 164.2, 190.0, 164.2, 190.0, 164.2, 190.0, 155.5],
    'hcap':   [2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000]
})

wells = pd.DataFrame({
    'x': [-20.0, 0.0, 20.0, 0.0, 0.0, 0.0],
    'y': [0.0, 0.0, 0.0, 300.0, 318.0, 336.0],
    'type': ['H', 'H', 'H', 'C', 'C', 'C']
})

# --------------- Initial conditions reservoir -------
geothermal_grad = 18 #(K / km), the geothermal gradient for the initial condition of the reservoir

# -------------- Operational parameters -----------------------
# Well temperatures
InjT = 273.15 + 90 # (K)
TCutOff = 273.15 + 50 # (K), Serves as the injection temperature of the warm well

# Annual total (i.e. over all wells) storage volumes (from the hot well perspective)
V_in_hot = 500000  # (m3/year)
V_out_hot = 450000 # (m3/year)
# Block-function Operational profile
daysprofile = [120, 60, 120, 65] # (days)
storage_periods = ['Charge', 'Rest', 'Discharge', 'Rest']
op_profile = [1, 0, 1, 0]

# --------------- Simulation time and space settings  --------
set_run_years = 1 #Total simulation length (years)
max_ts = 30 #Maximum time step size (days)
dt_mult = 8 #Time step upscaling (-)
set_transition_runtime = 1e-3 #dt after an operational period change [days]

# XY plane discretization settings
d_min = 2.5 # Cell dx and dy of the smallest cells surrounding the wells (m)
d_min_bound = 5 # Domain-width surrounding the wells with cell size d_min (m)
d_med = 10 # Cell dx and dy of the grid around the d_min_bound finest grid (m)
d_med_bound = 200 # Domain-width surrounding the wells with cell size d_med (m)
d_max = 100 # The maximum cell size in the coarsest, outermost part of the grid (m)
mult_fact = 1.5 # How aggressively d_med is upscaled to d_max (recommended: 1 < mult_fact < 2) (-)
n_max_bound = 3 # How many (buffer) cells of size d_max should bound the terrain (-)

# ------------- Run geomod creation functions  --------
dZ_array, n_ly, depth_to_top = create_ZGR(soil_uni, dz_mult = 1.8, print_stat = True)
dX_array, dY_array, well_indices = create_XY_grid (wells_mono, d_min, d_min_bound, d_med, d_med_bound, d_max, mult_fact, n_max_bound,
                                                   XY_GR_plot = True, image_loc = image_loc, image_name = image_name)
perm_h, perm_v, poro, tcond, hcap = property_fill(dX_array, dY_array, soil_uni, vertical_perm=False, mday_2_mD=True)
n_hot, n_cold, Q_hot_inj, Q_hot_prod, Q_cold_prod, Q_cold_inj = get_well_rates_block(V_in_hot, V_out_hot, daysprofile, storage_periods, wells_mono)


### ============== Run Simulation =====================
#Input model params here
m = Model(n_ly, dX_array, dY_array, dZ_array,
          perm_h, perm_v, poro, hcap, tcond,
          well_indices, depth_to_top, geothermal_grad, dt_mult, max_ts)
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
    plt.savefig(f'M_dmin_05_{d_min}_{d_min_bound}_{w.name}.png', dpi=300, bbox_inches='tight')

# %%-----------------read H5-----------------
well_id, well_depth = write_well_perforation_id(m)
r = read_well_h5(well_block_id=well_id, well_block_depth=well_depth)
r.draw_combined_well_data()  # All wells on same subplots
