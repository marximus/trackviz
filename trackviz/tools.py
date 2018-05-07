import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, LocatableAxes, Size


class FigureAxes:
    """
    Create a figure around a fixed size Axes with a dimension defined in pixels.

    Parameters
    ----------
    ax_size : tuple, (width, height)
        Width and height of Axes (in pixels).
    wspace : int
        Space between colorbar and main Axes (in pixels). Ignored if `cbar` is False.
    cbar : bool
        If True, create an Axes for a colorbar.
    cbar_width : int
        Width of colorbar (in pixels). Ignored if `cbar` is False.
    dpi : int
        Dots per inch.
    left, bottom, right, top : int
        Margin between plot elements and figure (in pixels).
    fig_func : None or class derived from matplotlib.figure.Figure.
        Call fig_func to create new figure. If None, create new figure using pyplot.figure().

    Attributes
    ----------
    fig : matplotlib Figure
        Figure instance.
    ax : matplotlib Axes
        Main axes in figure.
    cax : matplotlib Axes
        Axes for colorbar. Will be None if `cbar` was False.
    """
    def __init__(self, ax_size, wspace, dpi, cbar, cbar_width, left, bottom, right, top, fig_func=None):
        if not cbar:
            wspace = 0
            cbar_width = 0

        # convert arguments from pixels to inches
        ax_size = np.asarray(ax_size) / dpi
        wspace, cbar_width, left, bottom, right, top = np.array((wspace, cbar_width, left, bottom, right, top)) / dpi

        horiz = [left, ax_size[0], wspace, cbar_width, right]
        vert = [bottom, ax_size[1], top]

        figsize = (sum(horiz), sum(vert))
        fig_func = plt.figure if fig_func is None else fig_func
        fig = fig_func(figsize=figsize, dpi=dpi)

        horiz, vert = list(map(Size.Fixed, horiz)), list(map(Size.Fixed, vert))
        divider = Divider(fig, (0.0, 0.0, 1., 1.), horiz, vert, aspect=False)

        ax = LocatableAxes(fig, divider.get_position())
        ax.set_axes_locator(divider.new_locator(nx=1, ny=1))
        fig.add_axes(ax)

        if cbar:
            cax = LocatableAxes(fig, divider.get_position())
            cax.set_axes_locator(divider.new_locator(nx=3, ny=1))
            fig.add_axes(cax)

        self.fig = fig
        self.ax = ax
        self.cax = cax if cbar else None


def tracks_to_lines(tracks, dims=3, return_trackid=False):
    """Return sequence of trajectory lines.

    Each array in the returned sequence contains the points of a single trajectory. The array will be a N x 3
    array of (x, y, t) points if `dims` is 3 or a N x 2 array of (x, y) points if `dims` is 2.

    Parameters
    ----------
    tracks : pandas.DataFrame
        Trajectory data. Must have columns x, y, t, and trackid.
    dims : {3, 2}
        Return (x, y) coordinates if `dims` is 2 and return (x, y, t) coordinates if `dims` is 3 (default is 3).
    return_trackid : bool, optional
        If True, also return the trackid of each trajectory (default is False).

    Returns
    -------
    lines : list of N x 3 or N x 2 arrays
        Trajectories as lines.
    trackids : array, optional
        The trackids of the trajectories returned in `lines`. Only provided if `return_trackid` is True.
    """
    if dims not in (2, 3):
        raise ValueError('dims must be either 2 or 3')
    cols = ('x', 'y') if dims == 2 else ('x', 'y', 't')

    trackids, lines = zip(*[(trackid, df.loc[:, cols].values) for trackid, df in tracks.groupby('trackid')])
    trackids = list(trackids)
    lines = list(lines)

    if return_trackid:
        return lines, trackids
    return lines


def line_to_segments(line):
    """Return line segments between each consecutive pair of points in line.

    For a line with points [(x1,y1), (x2,y2), (x3,y3)], the return value will be [[(x1,y1), (x2,y2)], [(x2,y2), (x3,y3)]].

    Parameters
    ----------
    line : array, (N, 3)
        Points in a line.

    Returns
    -------
    segments : array, shape (N-1, 2, 3)
        Line segments.
    """
    points = line[:, None, :]
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    return segments
