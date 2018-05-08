import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import BoundaryNorm


import trackviz.tools
from trackviz.tools import FigureAxes


def trajectory_2d(
    tracks,
    image=None,
    labels=None,
    color=None,
    xlim=None, ylim=None,
    cmap=None,
    cbar=False,
    cbar_width=30,
    show_points=False,
    line_kws=None,
    scat_kws=None,
    im_kws=None,
    dpi=None,
    scale=1.,
    margin=None
):
    """
    2d plot of trajectories.

    Parameters
    ----------
    tracks : pandas.DataFrame, columns (x, y, t, trackid)
        Trajectory data. Each row in `tracks` is a point in a trajectory, where each trajectory has a unique trackid.
    image : ndarray, shape (H, W) or (H, W, 3) or (H, W, 4), optional
        Image to plot in background.
    labels : pandas.DataFrame, columns (trackid, label)
        Trajectory labels used to color trajectories if `color` == `label`. There must be a label for each trajectory
        in `tracks`. Can contain integers or strings.
    color : {None, 't', 'label'}
        Determines how trajectories are colored. If `color` == 't', color each line segment of each trajectory according
        to its t value. If `color` == 'label', color each trajectory based on its label. If `color` == None, do not
        color the trajectories.
    xlim, ylim : array_like, shape (2,) optional
        Data limits for the x and y axis. If no limits are given, they will be computed from the data.
    cmap : matplotlib colormap name or object, optional
        Colormap used to map t coordinates to colors.
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
    im_kws : dict, optional
        Keyword arguments for ax.imshow.
    dpi : int or None, optional
        Figure dots per inch. If None, default to matplotlib.rcParams['figure.dpi'].
    scale : float, optional
        Output size scale.
    margin : dict, keys ('left', 'right', 'bottom', 'top') , optional
        Margin between plot elements and figure. Defaults are {'left': 40, 'bottom': 40, 'right': 40, 'top': 10}.

    Returns
    -------
    figure : matplotlib Figure
        Figure containing trajectory plot.
    axes : matplotlib Axes
        Axes on which the trajectories are plotted.
    """
    if not pd.Series(['x','y','t','trackid']).isin(tracks.columns).all():
        raise ValueError('tracks must have columns x, y, t, and trackid')
    if len(tracks) < 1:
        raise ValueError('tracks must have at least one row')
    if image is not None:
        if not (image.ndim == 2 or image.ndim == 3):
            raise ValueError('image must be (H, W), (H, W, 3) or (H, W, 4)')
        if image.ndim == 3 and not (image.shape[-1] == 3 or image.shape[-1] == 4):
            raise ValueError('image must be (H, W), (H, W, 3) or (H, W, 4)')
    if labels is not None and len(set(tracks['trackid']) - set(labels['trackid'])) != 0:
        raise ValueError('there must be a label for each trajectory')
    if color not in ('t', 'label', None):
        raise ValueError('color must be "t", "label", or None')
    if color == 'label' and labels is None:
        raise ValueError('labels cannot be None if color == "label"')
    if color is None and cbar:
        raise ValueError('cbar must be False when color == None')

    if labels is not None:
        labels = dict(zip(labels['trackid'], labels['label']))

    tracks = tracks.sort_values(['trackid', 't'])

    # set some default values
    dpi = matplotlib.rcParams['figure.dpi'] if dpi is None else dpi
    cmap = 'viridis' if cmap is None else cmap

    # set up keyword arguments
    line_kws = {} if line_kws is None else line_kws.copy()
    line_kws.setdefault('linewidths', 1.5)
    line_kws.setdefault('alpha', 0.8)
    line_kws.setdefault('zorder', 1)

    scat_kws = {} if scat_kws is None else scat_kws.copy()
    scat_kws.setdefault('zorder', 2)
    scat_kws.setdefault('s', 0.5)
    scat_kws.setdefault('linewidths', 0.2)
    scat_kws.setdefault('c', 'black')

    im_kws = {} if im_kws is None else im_kws.copy()

    margin = {} if margin is None else margin.copy()
    margin.setdefault('left', 0)
    margin.setdefault('bottom', 0)
    margin.setdefault('right', 25 if cbar else 0)
    margin.setdefault('top', 0)

    # compute x and y limits
    if xlim is None:
        xlim = (tracks['x'].min(), tracks['x'].max())
        if image is not None:
            height, width = image.shape[:2]
            xlim = (min(0, xlim[0]), max(width, xlim[1]))
    if ylim is None:
        ylim = (tracks['y'].min(), tracks['y'].max())
        if image is not None:
            height, width = image.shape[:2]
            ylim = (min(0, ylim[0]), max(height, ylim[1]))

    # determine size of main axes
    width, height = np.fabs(xlim[0] - xlim[1]), np.fabs(ylim[0] - ylim[1])
    axsize = (np.array((width, height)) * scale).astype(np.int)
    # print('axsize: {}'.format(axsize))

    grid = FigureAxes(axsize, 20, dpi, cbar, cbar_width,
                      margin['left'], margin['bottom'], margin['right'], margin['top'])
    grid.ax.set_aspect('equal')
    grid.ax.axis('off')
    grid.ax.set_xlim(xlim)
    grid.ax.set_ylim(ylim[::-1])   # invert y axis

    # plot image
    if image is not None:
        grid.ax.imshow(image, **im_kws)

    # set parameters for plotting trajectories
    if color == 'label':
        tracklines, trackids = trackviz.tools.tracks_to_lines(tracks, dims=2, return_trackid=True)

        # use indices of sorted labels to map labels to colors
        tracklabels = np.array([labels[tid] for tid in trackids])
        label_values, tracklabel_inds = np.unique(tracklabels, return_inverse=True)
        N = len(label_values)

        discrete_cmap = plt.get_cmap(cmap, N)
        norm = BoundaryNorm(np.linspace(-0.5, N-0.5, N+1), N)

        line_kws.update(segments=tracklines, cmap=discrete_cmap, norm=norm)
        colordata = tracklabel_inds
    elif color == 't':
        tracklines = trackviz.tools.tracks_to_lines(tracks, dims=3, return_trackid=False)
        segments = np.concatenate([trackviz.tools.line_to_segments(line) for line in tracklines], axis=0)

        line_kws.update(segments=segments[:, :, :2], cmap=cmap)
        colordata = segments[:, 0, 2]  # use t value of first point of segment to determine color of line
    else:
        tracklines = trackviz.tools.tracks_to_lines(tracks, dims=2, return_trackid=False)
        line_kws.update(segments=tracklines)

    # plot trajectories
    lc = LineCollection(**line_kws)
    if color == 'label' or color == 't':
        lc.set_array(colordata)  # set values that will be mapped to RGBA using cmap
    grid.ax.add_collection(lc)

    if cbar:
        if color == 'label':
            cb = grid.ax.figure.colorbar(lc, cax=grid.cax, ticks=range(N))
            cb.set_ticklabels(label_values)
        elif color == 't':
            cb = grid.ax.figure.colorbar(lc, cax=grid.cax)

    # draw points
    if show_points:
        grid.ax.scatter(tracks['x'], tracks['y'], **scat_kws)

    return grid.fig, grid.ax


# TODO: Add ability to draw colorbar
# TODO: Add show_points parameter
def trajectory_3d(
    tracks,
    labels=None,
    color=None,
    xlim=None, ylim=None, tlim=None,
    cmap=None,
    line_kws=None,
):
    """
    3d plot of trajectories.

    Parameters
    ----------
    tracks : pandas.DataFrame, columns (x, y, t, trackid)
        Trajectory data. Each row in `tracks` is a point in a trajectory, where each trajectory has a unique trackid.
    labels : pandas.DataFrame, columns (trackid, label)
        Trajectory labels used to color trajectories if `color` == `label`. There must be a label for each trajectory
        in `tracks`. Can contain integers or strings.
    color : {None, 't', 'label'}
        Value(s) used to color trajectories. If `color` == 't', color each line segment of each trajectory according
        to its t value. If `color == 'label', color each trajectory based on its label. If `color` == None, do not
        color the trajectories.
    xlim, ylim, tlim : list, shape (2,) optional
        Data limits for the x, y, and t axis. If no limits are given, they will be computed from the data.
    cmap : matplotlib colormap name or object, optional
        Colormap used to map t coordinates to colors. Ignored if `labels` is not None.
    line_kws : dict, optional
        Keyword arguments passed to matplotlib LineCollection.

    Returns
    -------
    figure : matplotlib Figure
        Figure containing trajectory plot.
    axes : matplotlib Axes
        Axes on which the trajectories are plotted.
    """
    if not pd.Series(['x','y','t','trackid']).isin(tracks.columns).all():
        raise ValueError('tracks must have columns x, y, t, and trackid')
    if len(tracks) < 1:
        raise ValueError('tracks must have at least one row')
    if labels is not None and len(set(tracks['trackid']) - set(labels['trackid'])) != 0:
        raise ValueError('there must be a label for each trajectory')
    if color not in ('t', 'label', None):
        raise ValueError('color must be "t", "label", or None')
    if color == 'label' and labels is None:
        raise ValueError('labels cannot be None if color == "label"')

    if labels is not None:
        labels = dict(zip(labels['trackid'], labels['label']))

    line_kws = {} if line_kws is None else line_kws.copy()
    cmap = 'viridis' if cmap is None else cmap

    # compute axis limits
    xlim = xlim if xlim else [np.floor(tracks['x'].min()), np.ceil(tracks['x'].max())]
    ylim = ylim if ylim else [np.floor(tracks['y'].min()), np.ceil(tracks['y'].max())][::-1]  # invert y axis
    zlim = tlim if tlim else [np.floor(tracks['t'].min()), np.ceil(tracks['t'].max())]

    fig = plt.figure()
    ax = fig.add_axes((0, 0, 1, 1), projection='3d')

    ax.set_xlim3d(xlim)
    ax.set_ylim3d(ylim)
    ax.set_zlim3d(zlim)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('t')

    # set values needed to plot trajectories
    tracklines, trackids = trackviz.tools.tracks_to_lines(tracks, dims=3, return_trackid=True)
    if color == 'label':
        tracklabels = np.array([labels[tid] for tid in trackids])
        label_values, tracklabel_inds = np.unique(tracklabels, return_inverse=True)
        N = len(label_values)

        discrete_cmap = plt.get_cmap(cmap, N)
        norm = BoundaryNorm(np.linspace(-0.5, N-0.5, N+1), N)

        line_kws.update(segments=tracklines, cmap=discrete_cmap, norm=norm)
        colordata = tracklabel_inds
    elif color == 't':
        segments = np.concatenate([trackviz.tools.line_to_segments(line) for line in tracklines], axis=0)

        line_kws.update(segments=segments, cmap=cmap)
        colordata = segments[:, 0, 2]  # use t value of first point of segment to determine color of line
    else:
        line_kws.update(segments=tracklines)

    lc = Line3DCollection(**line_kws)
    if color == 'label' or color == 't':
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