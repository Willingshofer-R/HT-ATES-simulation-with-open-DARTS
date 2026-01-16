import numpy as np
import matplotlib.pyplot as plt
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
    fig, ax = plt.subplots(figsize=(10, 10))

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
