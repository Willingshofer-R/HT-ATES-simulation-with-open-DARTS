import pandas as pd
from iapws import IAPWS97
import h5py
import matplotlib.pyplot as plt
import numpy as np

class read_well_h5:
    """
    A class to read and process HDF5 well data, get pressure and enthalpy data, and compute temperature values,
    and export the processed data to Excel files. It also provides visualization capabilities for temperature and pressure data.
    """
    def __init__(self, file_path: str = "output/well_data.h5", well_block_id: dict = None, well_block_depth: dict = None):
        """
        Initializes the read_well_h5 class by reading HDF5 data and processing it.
        
        :param file_path: Path to the HDF5 file containing well data.
        :param well_block_id: Dictionary mapping well names to block IDs.
        """
        if file_path is None or well_block_id is None:
            raise ValueError("File path and well block ID dictionary must be provided.")

        self.well_block_id = well_block_id
        self.well_block_depth = well_block_depth
        self.time = None
        self.cell_id = None
        self.X = None
        self.var_names = None
        self.temp = dict()
        self.pres = dict()
        self.enthalpy = dict()
        self.df_temp = pd.DataFrame()
        self.df_pres = pd.DataFrame()
        self.df_enthalpy = pd.DataFrame()

        # Function calls
        self.read_specific_data(filename=file_path)
        self.create_empty_P_T_dict()
        self.process_and_save_well_data()
    
    def read_specific_data(self, filename: str, timestep: int = None, is_return: bool = False):
        """
        Reads data from the specified HDF5 file and stores it in the class attributes.
        Original DartsModel function was embedded here!
        
        :param filename: Path to the HDF5 file.
        :param timestep: Specific time step to extract data from. If None, all time steps are read.
        :param is_return: If True, returns the read data.
        :return: time, cell_id, X, var_names if is_return is True.
        """
        with h5py.File(filename, 'r') as file:
            if timestep is None:
                cell_id = file['dynamic/cell_id'][:]
                var_names = file['dynamic/variable_names'][:]
                time = file['dynamic/time'][:]
                X = file['dynamic/X'][:]
            else:
                cell_id = file['dynamic/cell_id'][:]
                var_names = file['dynamic/variable_names'][:]
                time = file['dynamic/time'][timestep].reshape(1)
                X = file['dynamic/X'][timestep].reshape(1, len(cell_id), len(var_names))
        
        self.time = time
        self.cell_id = cell_id
        self.X = X
        self.var_names = [name.decode() for name in var_names]

        if is_return:
            return time, cell_id, X, var_names

    def create_empty_P_T_dict(self):
        """
        Initializes empty dictionaries for temperature, pressure, and enthalpy values.
        """
        self.temp['time'] = []
        self.pres['time'] = []
        self.enthalpy['time'] = []

        # for well_name, block_ids in self.well_block_id.items():
        #     for block_id in block_ids:
        #         header = f"{well_name}_{block_id}"
        #         self.temp[header] = []
        #         self.pres[header] = []
        #         self.enthalpy[header] = []
        
        for well_name, block_depth in self.well_block_depth.items():
            for block_depth in block_depth:
                header = f"{well_name}_{block_depth}"
                self.temp[header] = []
                self.pres[header] = []
                self.enthalpy[header] = []

    def process_and_save_well_data(self):
        """
        Processes well data by computing pressure, enthalpy, and temperature values and stores them.
        """        
        try:
            pressure_index = self.var_names.index("pressure")
            enthalpy_index = self.var_names.index("enthalpy")
        except ValueError:
            raise ValueError("Pressure or enthalpy variables not found!")

        self.temp['time'] = self.time.tolist()
        self.pres['time'] = self.time.tolist()
        self.enthalpy['time'] = self.time.tolist()

        # Her kuyunun block_id ve karşılık gelen derinlik (depth) bilgilerini aynı indeks sırasına göre eşleştiriyoruz.
        for well_name, block_ids in self.well_block_id.items():
            for idx, block_id in enumerate(block_ids):
                if block_id in self.cell_id:
                    cell_index = np.where(self.cell_id == block_id)[0][0]
                else:
                    print(f"Warning: Block ID {block_id} not found in data for {well_name}, skipping.")
                    continue

                pressure_values_bar = self.X[:, cell_index, pressure_index]
                enthalpy_values_kJ_kmol = self.X[:, cell_index, enthalpy_index]

                pressure_values_MPa = pressure_values_bar * 0.1
                enthalpy_values_kJ_kg = enthalpy_values_kJ_kmol / 18.015

                temperature_values = []
                for p, h in zip(pressure_values_MPa, enthalpy_values_kJ_kg):
                    try:
                        steam = IAPWS97(P=p, h=h)
                        temperature_values.append(steam.T - 273.15)
                    except ValueError:
                        temperature_values.append(None)

                # İlgili kuyunun derinlik bilgisine ulaşmak için well_block_depth'ten aynı indeksli öğeyi kullanıyoruz.
                depth_value = self.well_block_depth[well_name][idx]
                header = f"{well_name}_{depth_value}"
                self.pres[header] = pressure_values_bar.tolist()
                self.enthalpy[header] = enthalpy_values_kJ_kmol.tolist()
                self.temp[header] = temperature_values

        self.df_temp = pd.DataFrame.from_dict(self.temp)
        self.df_pres = pd.DataFrame.from_dict(self.pres)
        self.df_enthalpy = pd.DataFrame.from_dict(self.enthalpy)

    def write_df_to_excel(self):
        """
        Exports temperature, pressure, and enthalpy data to Excel files.
        """
        self.df_temp.to_excel("Well_Temperatures.xlsx", index=False)
        self.df_pres.to_excel("Well_Pressures.xlsx", index=False)
        self.df_enthalpy.to_excel("Well_Enthalpy.xlsx", index=False)
        print("Data successfully saved to Excel files.")

    def define_column_groups(self):
        """
        Groups columns based on well prefixes (H1, H2, H3, etc.) for easier visualization.
        """
        self.column_groups = {key: [] for key in self.well_block_id.keys()}
        for col in self.df_temp.columns:
            for prefix in self.column_groups.keys():
                if col.startswith(prefix):
                    self.column_groups[prefix].append(col)

    def draw_well_blocks_data(self, data_type="T"):
        """
        Plots the well block data for the specified data type (temperature, pressure, or enthalpy).
        
        :param data_type: Type of data to plot ('T' for temperature, 'P' for pressure, 'E' for enthalpy).
        """
        if data_type == "T":
            df = self.df_temp
            y_label = 'Temperature [°C]'
        elif data_type == "P":
            df = self.df_pres
            y_label = 'Pressure [bar]'
        elif data_type == "E":
            df = self.df_enthalpy
            y_label = 'Enthalpy [kJ/kmol]'
        else:
            print("Error: Invalid data type. Use 'T', 'P', or 'E'.")
            return
        
        self.define_column_groups()
        fig, axes = plt.subplots(4, 2, figsize=(15, 15), dpi=300)
        fig.tight_layout(pad=5.0)

        axes = axes.flatten()
        for ax, (prefix, columns) in zip(axes, self.column_groups.items()):
            for col in columns:
                ax.plot(df['time'], df[col], "--o", label=col, markersize=4)
            ax.set_title(f'{prefix} Well')
            ax.set_ylabel(y_label)
            ax.set_xlabel('Time [day]')
            ax.legend(loc='best', fontsize='small')

        for ax in axes[len(self.column_groups):]:
            ax.axis('off')

        plt.show()
        
    
    def draw_combined_well_data(self):
        """
        Tek bir figürde 3 satır (sırasıyla sıcaklık, entalpi, basınç) ve kuyu sayısı kadar sütun oluşturur.
        Her sütun, ilgili kuyunun tüm bloklarına ait zaman serilerini çizer.
        """
        # Kuyu isimlerine göre sütun gruplarını oluşturur (self.column_groups sözlüğü, well_block_id anahtarlarını kullanır)
        self.define_column_groups()
        well_names = list(self.column_groups.keys())
        n_wells = len(well_names)

        # 3 satır ve n_wells sütunlu bir figür oluşturuyoruz.
        fig, axes = plt.subplots(3, n_wells, figsize=(n_wells * 9, 12), dpi=300, sharex='col')
        
        # Eğer sadece tek bir kuyu varsa, axes 1D geleceği için 2D forma çeviriyoruz.
        if n_wells == 1:
            axes = np.expand_dims(axes, axis=1)

        for j, well in enumerate(well_names):
            cols = self.column_groups[well]  # İlgili kuyunun blok sütunları

            # 1. Satır: Sıcaklık verileri
            ax_temp = axes[0, j]
            for col in cols:
                ax_temp.plot(self.df_temp['time'], self.df_temp[col], marker='o', linestyle='--', label=col)
            ax_temp.set_title(f"{well} - Temperature")
            ax_temp.set_ylabel("Temperature [°C]")
            ax_temp.legend(fontsize='small')

            # 2. Satır: Enthalpy verileri
            ax_enthalpy = axes[1, j]
            for col in cols:
                ax_enthalpy.plot(self.df_enthalpy['time'], self.df_enthalpy[col], marker='o', linestyle='--', label=col)
            ax_enthalpy.set_title(f"{well} - Enthalpy")
            ax_enthalpy.set_ylabel("Enthalpy [kJ/kmol]")
            ax_enthalpy.legend(fontsize='small')

            # 3. Satır: Pressure verileri
            ax_pres = axes[2, j]
            for col in cols:
                ax_pres.plot(self.df_pres['time'], self.df_pres[col], marker='o', linestyle='--', label=col)
            ax_pres.set_title(f"{well} - Pressure")
            ax_pres.set_ylabel("Pressure [bar]")
            ax_pres.set_xlabel("Time [day]")
            ax_pres.legend(fontsize='small')


        plt.tight_layout()
        plt.savefig('20250414_well_grid_ptv_20250512.png', dpi=300, bbox_inches='tight')
        # plt.show()


#%%
def write_well_perforation_id(m, write_to_excel: bool = False):
    """
    Extracts perforation IDs and depths for each well from the reservoir model.

    Parameters:
        m : Reservoir model object. The wells and their perforation information 
            are accessible via m.reservoir.wells and m.reservoir.mesh.depth.
        write_to_excel (bool): If True, the resulting DataFrame is saved as an Excel file named "Wells_ID_Depth.xlsx".

    Returns:
        dict: A dictionary with two keys:
              "well_id"   - contains a dictionary with perforation IDs for each well.
              "well_depth" - contains a dictionary with perforation depths for each well.

    In case of an error, the exception message is printed to the console.
    """
    well_id = {}
    well_depth = {}
    
    try:
        # Create an empty DataFrame
        df_ID = pd.DataFrame()
        
        # Iterate through each well
        for w in m.reservoir.wells:
            # Retrieve perforation IDs and depths for each well
            perforation_ids = [i[1] for i in w.perforations]
            perforation_depth = [m.reservoir.mesh.depth[i[1]] for i in w.perforations]
            
            # Add the lists to the corresponding dictionaries
            well_id[w.name] = perforation_ids
            well_depth[w.name] = perforation_depth
            
            # Add the data to the DataFrame using the well name as a prefix for column names
            df_ID[w.name + "_ID"] = pd.Series(perforation_ids)
            df_ID[w.name + "_Depth"] = pd.Series(perforation_depth)
        
        # If write_to_excel is True, save the DataFrame to an Excel file
        if write_to_excel:
            df_ID.to_excel("Wells_ID_Depth.xlsx", index=False)
            print("Wells_ID_Depth.xlsx was created!")
    
    except Exception as e:
        print("An error occurred:", e)
    
    # Return the dictionaries containing well perforation IDs and depths
    return well_id, well_depth
        
        
#%%-----Usage Example-----

# filename="h5_well_data/well_data.h5"
# # filename="h5_well_data/V_1254_10_Y_well_data.h5"
# # filename="h5_well_data/V_Sinu_10_Y_well_data.h5"

# well_id, well_depth=write_well_perforation_id(m)
# r=read_well_h5(well_block_id=well_id,well_block_depth=well_depth)
# r.draw_combined_well_data()  # Tek grafik üzerinde tüm kuyu verilerini çizmek için

# #-----Draw Graphs-----
# # for s in ["T","P","E"]:
# #     r.draw_well_blocks_data(data_type=s)
    
# r.draw_combined_well_data()  # Tek grafik üzerinde tüm kuyu verilerini çizmek için

#Alternative usage
# # Kuyu blok ID'leri sözlüğü
# well_id = {
#     "SH-4": [105175, 120800, 136425, 152050, 167675, 183300, 198925, 214550, 230175, 245800, 261425, 277050],
#     "SH-5": [175946, 191571, 207196, 222821, 238446, 254071]}

# # Kuyu blok derinlik sözlüğü
# well_depth = {
#     "SH-4": [408, 418, 428, 438, 448, 458, 468, 478, 488, 498, 508, 518],
#     "SH-5": [458, 468, 478, 488, 498, 508]}

# r = read_well_h5(file_path=filename, well_block_id=well_id, well_block_depth=well_depth)
# r.write_df_to_excel()

