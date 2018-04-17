import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, LocatableAxes, Size
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d import Axes3D

import trackviz.tools


class FigureAxes:
    """
    Create a figure around a fixed size Axes with a dimension defined in pixels.

    Parameters
    ----------
    ax_size : tuple, (width, height)
        Width and height of Axes (in pixels).
    wspace : int
        Space between colorbar and main Axes (in pixels).
    cbar : bool
        If True, create an Axes for a colorbar.
    cbar_width : int
        Width of colorbar (in pixels).
    dpi : int
        Dots per inch.
    left, bottom, right, top : int
        Margin between plot elements and figure (in pixels).

    Attributes
    ----------
    fig : matplotlib Figure
        Figure instance.
    ax : matplotlib Axes
        Main axes in figure.
    cax : matplotlib Axes
        Axes for colorbar. Will be None if `cbar` was False.
    """
    def __init__(self, ax_size, wspace, dpi, cbar, cbar_width, left, bottom, right, top):
        if not cbar:
            cbar_width = 0

        # convert arguments from pixels to inches
        ax_size = np.asarray(ax_size) / dpi
        wspace, cbar_width, left, bottom, right, top = np.array((wspace, cbar_width, left, bottom, right, top)) / dpi

        horiz = [left, ax_size[0], wspace, cbar_width, right]
        vert = [bottom, ax_size[1], top]

        figsize = (sum(horiz), sum(vert))
        fig = plt.figure(figsize=figsize, dpi=dpi, frameon=True)

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


# TODO: Rather than tracks being a (N, L, 3) array where all the tracks must have the same
# TODO: length, use a pandas.DataFrame with columns (x, y, frame_num, track_id) or (x, y, z, track_id).
def trajectory_2d(
    tracks,
    image=None,
    labels=None,
    color='z',
    xlim=None, ylim=None,
    cmap=None,
    cbar=False,
    cbar_width=30,
    show_points=False,
    line_kws=None,
    scat_kws=None,
    dpi=None,
    scale=1.
):
    """
    2d plot of trajectories.

    Parameters
    ----------
    tracks : ndarray, shape (N, L, 3)
        Trajectory coordinates. There are N tracks, where each track has L (x, y, z) points.
    image : ndarray, shape (H, W) or (H, W, 3) or (H, W, 4), optional
        Image to plot in background.
    labels : ndarray, shape (N,), optional
        Trajectory labels. Will be used to color trajectories.
    color : {'z', 'label', None}
        Value(s) used to color trajectories. If `color` == 'z', color each line segment of each trajectory according
        to its z value. If `color == 'label', color each trajectory based on its label. If `color` == None, do not
        color the trajectories.
    xlim, ylim : array_like, shape (2,) optional
        Data limits for the x and y axis. If no limits are given, they will be computed from the data.
    cmap : matplotlib colormap name or object, optional
        Colormap used to map z coordinates to colors.
    cbar : bool, optional
        Whether to draw a colorbar. Must be False if `color` == None, because no colormapping is used.
    cbar_width : int
        Width of colorbar (in pixels).
    show_points : bool, optional
        Whether to plot each trajectory as a point.
    line_kws : dict, optional
        Keyword arguments for matplotlib.collection.LineCollection.
    scat_kws : dict, optional
        Keyword arguments for ax.scatter.
    dpi : int or None, optional
        Figure dots per inch. If None, default to matplotlib.rcParams['figure.dpi'].
    scale : float, optional
        Output size scale.

    Returns
    -------
    figure : matplotlib Figure
        Figure containing trajectory plot.
    axes : matplotlib Axes
        Axes on which the trajectories are plotted.
    """
    if not (tracks.ndim == 3 and tracks.shape[-1] == 3):
        raise ValueError('tracks must be a (N, L, 3) array')
    if tracks.shape[0] < 1:
        raise ValueError('tracks has zero length')
    if image is not None:
        if not (image.ndim == 2 or image.ndim == 3):
            raise ValueError('image must be (H, W), (H, W, 3) or (H, W, 4)')
        if image.ndim == 4 and not (image.shape[-1] == 3 or image.shape[-1] == 4):
            raise ValueError('image must be (H, W), (H, W, 3) or (H, W, 4)')
    if labels is not None and labels.shape != (len(tracks),):
        raise ValueError('labels must have shape ({},)'.format(len(tracks)))
    if color not in ('z', 'label', None):
        raise ValueError('color must be "z", "label", or None')
    if color == 'label' and labels is None:
        raise ValueError('labels cannot be None if color == "label"')
    if color is None and cbar:
        raise ValueError('cbar must be False when color == None')

    dpi = matplotlib.rcParams['figure.dpi'] if dpi is None else dpi
    cmap = 'viridis' if cmap is None else cmap

    line_kws = {} if line_kws is None else line_kws.copy()
    line_kws.setdefault('linewidths', 1.5)
    line_kws.setdefault('alpha', 0.8)
    line_kws.setdefault('zorder', 1)

    scat_kws = {} if scat_kws is None else scat_kws.copy()
    scat_kws.setdefault('zorder', 2)
    scat_kws.setdefault('s', 0.5)
    scat_kws.setdefault('linewidths', 0.2)
    scat_kws.setdefault('c', 'black')

    # compute x and y limits
    if xlim is None:
        xlim = (tracks[:, :, 0].min(), tracks[:, :, 0].max())
        if image is not None:
            height, width = image.shape[:2]
            xlim = (min(0, xlim[0]), max(width, xlim[1]))
    if ylim is None:
        ylim = (tracks[:, :, 1].min(), tracks[:, :, 1].max())
        if image is not None:
            height, width = image.shape[:2]
            ylim = (min(0, ylim[0]), max(height, ylim[1]))

    # determine size of main axes
    width, height = np.fabs(xlim[0] - xlim[1]), np.fabs(ylim[0] - ylim[1])
    axsize = (np.array((width, height)) * scale).astype(np.int)
    # print('axsize: {}'.format(axsize))

    grid = FigureAxes(axsize, 20, dpi, cbar, cbar_width, 40, 40, 40, 10)
    grid.ax.set_aspect('equal')
    grid.ax.set_xlim(xlim)
    grid.ax.set_ylim(ylim[::-1])   # invert y axis

    # draw lines
    if color == 'label':
        linedata = tracks[:, :, :2]
        colordata = labels
        # TODO: Check that labels are strictly increasing and ints
        labels_unique = np.unique(labels)
        N = len(labels_unique)
        line_kws['cmap'] = plt.get_cmap(cmap, N)
    elif color == 'z':
        segments = trackviz.tools.tracks_to_segments(tracks)
        linedata = segments[:, :, :2]
        colordata = segments[:, 0, 2]
        line_kws['cmap'] = cmap
    else:
        linedata = tracks[:, :, :2]
        colordata = None

    lc = LineCollection(linedata, **line_kws)
    if colordata is not None:
        lc.set_array(colordata)
    grid.ax.add_collection(lc)

    # draw points
    if show_points:
        grid.ax.scatter(tracks[:, :, 0], tracks[:, :, 1], **scat_kws)

    # draw colorbar
    if cbar and color == 'label':
        # see https://gist.github.com/jakevdp/91077b0cae40f8f8244a
        cb = grid.ax.figure.colorbar(lc, cax=grid.cax, ticks=range(N))
        lc.set_clim(-0.5, N - 0.5)
    elif cbar and color == 'z':
        cb = grid.ax.figure.colorbar(lc, cax=grid.cax)

    return grid.fig, grid.ax


# TODO: Add ability to draw colorbar
# TODO: Better solution for setting aspect ratio of Axes that doesn't change range
# TODO: Add show_points parameter
def trajectory_3d(
    tracks,
    labels=None,
    color='z',
    xlim=None, ylim=None, zlim=None,
    cmap=None,
    line_kws=None,
):
    """
    3d plot of trajectories.

    Parameters
    ----------
    tracks : ndarray, shape (N, L, 3)
        Trajectory coordinates. There are N tracks, where each track has L (x, y, z) points.
    labels : ndarray, shape (N,), dtype int, optional
        Trajectory labels. Will be used to color trajectories.
    color : {'z', 'label', None}
        Value(s) used to color trajectories. If `color` == 'z', color each line segment of each trajectory according
        to its z value. If `color == 'label', color each trajectory based on its label. If `color` == None, do not
        color the trajectories.
    xlim, ylim, zlim : list, shape (2,) optional
        Data limits for the x, y, and z axis. If no limits are given, they will be computed from the data.
    cmap : matplotlib colormap name or object, optional
        Colormap used to map z coordinates to colors. Ignored if `labels` is not None.
    line_kws : dict, optional
        Keyword arguments passed to matplotlib LineCollection.

    Returns
    -------
    figure : matplotlib Figure
        Figure containing trajectory plot.
    axes : matplotlib Axes
        Axes on which the trajectories are plotted.
    """
    if not (tracks.ndim == 3 and tracks.shape[-1] == 3):
        raise ValueError('tracks must be a (N, L, 3) array')
    if tracks.shape[0] < 1:
        raise ValueError('tracks has zero length')
    if labels is not None and labels.shape != (len(tracks),):
        raise ValueError('labels must have same length as tracks ({},)'.format(len(tracks)))
    if color not in ('z', 'label', None):
        raise ValueError('color must be "z", "label", or None')
    if color == 'label' and labels is None:
        raise ValueError('labels cannot be None if color == "label"')

    X = tracks[:, :, 0]
    Y = tracks[:, :, 1]
    Z = tracks[:, :, 2]

    line_kws = {} if line_kws is None else line_kws.copy()
    cmap = 'viridis' if cmap is None else cmap

    # compute axis limits
    xlim = xlim if xlim else [np.floor(X.min()), np.ceil(X.max())]
    ylim = ylim if ylim else [np.floor(Y.min()), np.ceil(Y.max())][::-1]  # invert y axis
    zlim = zlim if zlim else [np.floor(Z.min()), np.ceil(Z.max())]

    fig = plt.figure()
    ax = fig.add_axes((0, 0, 1, 1), projection='3d')

    ax.set_xlim3d(xlim)
    ax.set_ylim3d(ylim)
    ax.set_zlim3d(zlim)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if color == 'label':
        linedata = tracks
        colordata = labels
        # TODO: Check that labels are strictly increasing and ints
        N = len(np.unique(labels))
        line_kws['cmap'] = plt.get_cmap(cmap, N)
    elif color == 'z':
        linedata = trackviz.tools.tracks_to_segments(tracks)
        colordata = linedata[:, 0, 2]
        line_kws['cmap'] = cmap
    else:
        linedata = tracks
        colordata = None

    lc = Line3DCollection(linedata, **line_kws)
    if colordata is not None:
        lc.set_array(colordata)
    ax.add_collection3d(lc)

    # set aspect
    ax.set_aspect('equal')  # will set aspect for x and y axes
    # _set_axes_equal(ax)

    return fig, ax


def _set_axes_equal(ax):
    """
    This was copied from stackoverflow here: https://stackoverflow.com/a/31364297

    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])