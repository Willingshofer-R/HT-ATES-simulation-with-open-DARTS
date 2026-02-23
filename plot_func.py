import numpy as np
import matplotlib.pyplot as plt
import os
import pyvista as pv
import pandas as pd

def plot_grid(
        XGR,
        YGR,
        image_loc,
        image_name,
        wells_df=None,
        show=True,
        save=True
):
    """
    Plot XY grid and wells (if specified).

    Parameters
    ----------
    XGR, YGR : np.ndarray
        Grid node coordinates
    image_loc : str
        Output directory
    image_name : str
        File name (without extension)
    wells_df : pandas.DataFrame, optional
        Must contain columns ['x', 'y', 'type']
    """

    full_path = f"{image_loc}/{image_name}.png"
    fig, ax = plt.subplots(figsize=(10, 8))

    for x in XGR:
        ax.plot([x, x], [YGR[0], YGR[-1]],
                color='lightgray', linewidth=0.5, zorder = 1)

    for y in YGR:
        ax.plot([XGR[0], XGR[-1]], [y, y],
                color='lightgray', linewidth=0.5, zorder = 1)

    if wells_df is not None:
        if not {'x', 'y', 'type'}.issubset(wells_df.columns):
            raise ValueError("wells_df must contain columns: x, y, type")

        hot = wells_df[wells_df['type'] == 'H']
        cold = wells_df[wells_df['type'] == 'C']

        if not hot.empty:
            ax.scatter(hot['x'], hot['y'], color='red', marker='o', s=25, edgecolors='k',
                       linewidths = 0.5, label='Hot wells', zorder = 3)

        if not cold.empty:
            ax.scatter(cold['x'], cold['y'], color='cyan', marker='o', s=25,
                       edgecolors='k', linewidths = 0.5, label='Cold wells', zorder=3)

        if not hot.empty or not cold.empty:
            ax.legend()

    ax.set_xlim(XGR.min(), XGR.max())
    ax.set_ylim(YGR.min(), YGR.max())
    ax.set_aspect('equal')
    ax.set_title('Map XY Grid', fontsize=18)
    ax.set_xlabel('X [m]', fontsize=14)
    ax.set_ylabel('Y [m]', fontsize=14)

    if wells_df is not None:
        ax.legend()
    if save:
        plt.savefig(full_path)
    if show:
        plt.show()

def plot_YZ_section (d_X_array, d_Y_array, d_Z_array, depth_to_top,
                     X_slice_index, center_Y_index, plot_domain,
                     var_name, V_max, V_min,
                     vts_location, time_step, aspect = 3,  contour = True,
                     safe_loc = None):
    """
    Function for creating a cross-section of the pressure or temperature data.

    Inputs:
        d_X_array: numpy array of size nx
            Array containing the cell sizes in the X-direction
        d_Y_array: numpy array of size ny
            Array containing the cell sizes in the Y-direction
        d_Z_array: numpy array of size nz
            Array containing the cell sizes in the Z-direction
        depth_to_top: int or float
            The depth to the top of the model in meters
        X_slice_index: int
            The X-index where the slice should be taken. The XGR-cell closest to it will be used
        center_Y_index: int
            Integer containing the Y-cell-index where the plot should be centered.
        plot_domain: int or float
            Number describing the total horizontal width of the plot.
        var_name: str
            Either 'temperature[K]' or 'pressure[Bar]'
        V_max: int or float
            The maximum value of the plotted variable (used in the colormap)
        V_min: int or float
            The maximum value of the plotted variable (used in the colormap)
        vts_location: str
            The path to the vts data_files
        time_step: int
            The time step at which the plot should be visualized
        safe_loc: str
            The path where the resulting figure should be plotted, if None the plot is not safed
    """

    # Obtain plotting dimensions and initialize coordinates
    nx = len(d_X_array)
    ny = len(d_Y_array)
    nz = len(d_Z_array)

    #XGR = np.insert(np.cumsum(d_X_array), 0, 0)
    YGR = np.insert(np.cumsum(d_Y_array), 0, 0)
    ZGR = np.insert(np.cumsum(d_Z_array), 0 ,0) + depth_to_top

    half_width = plot_domain / 2
    ymin = np.argmin(np.abs(YGR - (YGR[center_Y_index] - half_width)))
    ymax = np.argmin(np.abs(YGR - (YGR[center_Y_index] + half_width))) + 1
    # Center around the center coordinate
    YGR_crop = YGR[ymin:ymax + 1] - YGR[center_Y_index]
    #  For pcolormesh (needs +1 edges)
    Y_edge, Z_edge = np.meshgrid(YGR_crop, ZGR)
    # For contour (needs same shape as data)
    Y_center, Z_center = np.meshgrid(0.5 * (YGR_crop[:-1] + YGR_crop[1:]), 0.5 * (ZGR[:-1] + ZGR[1:]))

    # Extract the vts data
    sol_ts = f'solution_ts{time_step}.vts'
    vts_file = os.path.join(vts_location, sol_ts)
    grid = pv.read(vts_file)
    var_array = grid.cell_data[var_name]
    var_np = np.array(var_array).reshape((nz, ny, nx)) #, order='F')
    var_2D_YZ = var_np[:, ymin:ymax, X_slice_index]

    if var_name == 'temperature[K]':
        data_plot = var_2D_YZ - 273.15
    else:
        data_plot = var_2D_YZ

    # Plotting lines
    fig, ax = plt.subplots(figsize=(10, 6))

    if var_name == 'temperature[K]':
        pcm = ax.pcolormesh(Y_edge, Z_edge, data_plot, cmap="coolwarm", vmin=V_min, vmax=V_max, shading="flat")

    elif var_name == 'pressure[Bar]':
        pcm = ax.pcolormesh(Y_edge, Z_edge, data_plot, cmap="viridis", vmin=V_min, vmax=V_max, shading="flat")

    if contour:
        levels = np.linspace(V_min, V_max, 10)
        cs = ax.contour(
            Y_center, Z_center, data_plot,
            levels=levels, colors="k",
            linewidths=0.5, linestyles="dashed"
        )

    ax.set_aspect(aspect, adjustable='box')
    # Check if this is necessary -> origin is upper is used in the plotting lines
    ax.invert_yaxis()

    if var_name == 'temperature[K]':
        ax.set_title('Reservoir Temperature (°C)')
        cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
        cbar.set_label("Temperature [°C]")

    elif var_name == 'pressure[Bar]':
        ax.set_title('Reservoir Pressure (bar)')
        cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
        cbar.set_label("Pressure [bar]")

    ax.set_ylabel('Depth [m]')
    ax.set_xlabel('Y [m]')

    plt.tight_layout()

    if safe_loc is not None:
        os.makedirs(safe_loc, exist_ok=True)
        save_path = os.path.join(safe_loc, f"{var_name}_YZ_ts{time_step}.png")
        plt.savefig(save_path, dpi=300)

    plt.show()

