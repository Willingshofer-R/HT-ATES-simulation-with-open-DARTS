import pickle

from Basic_Ates_Model import Model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Read_well_h5_opendarts import read_well_h5, write_well_perforation_id

# --------------scenario 2--------------
Q_hot = 300  # m3/day
Q_cold = 300  # m3/day

storage_periods = np.concatenate([np.array(["Charge"] * 4),
                                  np.array(["Rest"] * 12),
                                  np.array(["Discharge"] * 18),
                                  np.array(["Rest"] * 5)])

op_profile = np.concatenate([np.ones(4),
                             np.ones(12) * 0,
                             np.ones(18),
                             np.ones(5) * 0])

daysprofile = np.concatenate([np.ones(4) * 30,
                              np.ones(12) * 5,
                              np.ones(12) * 5,
                              np.ones(6) * 10,
                              np.ones(4) * 15,
                              np.array([5])])

# %%#-----------------Run Model-------------------
output_directory = 'vtk_data'
output_well_data_excel = "well_data.xlsx"


m = Model()
m.init()


m.set_output()

set_transition_runtime = 1e-3
set_run_years = 1

flw_rates = [x * Q_hot for x in op_profile]
# coldrates = [x * Q_cold for x in op_profile]

InjT = 273.15 + 90
TCutOff = 273.15 + 50

iterr = 1
for k in range(set_run_years):
    for i, runtime in enumerate(daysprofile):
        if storage_periods[i] == 'Charge':

            m.set_rate_hot(flw_rates[i], temp=InjT, func='inj')
            m.set_rate_cold(flw_rates[i], func='prod')
            print('Operation: Charge')

        elif storage_periods[i] == 'Discharge':

            m.set_rate_hot(flw_rates[i], func='prod')
            m.set_rate_cold(flw_rates[i], temp=TCutOff, func='inj')
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
