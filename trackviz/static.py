import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, LocatableAxes, Size
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
import seaborn as sns


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
    ax_cbar : matplotlib Axes
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
            ax_cbar = LocatableAxes(fig, divider.get_position())
            ax_cbar.set_axes_locator(divider.new_locator(nx=3, ny=1))
            fig.add_axes(ax_cbar)

        self.fig = fig
        self.ax = ax
        self.ax_cbar = ax_cbar if cbar else None


# TODO: Rather than tracks being a (N, L, 3) array where all the tracks must have the same
# TODO: length, use a pandas.DataFrame with columns (x, y, frame_num, track_id) or (x, y, z, track_id).
def trajectory_2d(
    tracks,
    image=None,
    labels=None,
    xlim=None, ylim=None,
    vmin=None, vmax=None,
    cmap=None,
    cbar=True,
    cbar_width=30,
    show_points=False,
    line_kws=None,
    scat_kws=None,
    dpi=100,
    scale=1.
):
    """
    2d plot of trajectories.

    If `labels` is not None, each trajectory will be colored according to its label. If `labels` is
    not None, `vmin`, `vmax`, cmap`, and `cbar` will be ignored. If no `labels` are specified, each
    trajectory point will be colored based on its z value.

    Parameters
    ----------
    tracks : ndarray, shape (N, L, 3)
        Trajectory coordinates. There are N tracks, where each track has L (x, y, z) points.
    image : ndarray, shape (H, W) or (H, W, 3) or (H, W, 4), optional
        Image to plot in background.
    labels : ndarray, shape (N,), optional
        Trajectory labels. Will be used to color trajectories.
    xlim, ylim : array_like, shape (2,) optional
        Data limits for the x and y axis. If no limits are given, they will be computed
        so that the trajectories, as well as the image (if it is used), fit into the limits.
    vmin, vmax : floats, optional
        Values to anchor the colormap. If None, they will be inferred from the data. Ignored
        if `labels` is not None.
    cmap : matplotlib colormap name or object, optional
        Colormap used to map z coordinates to colors. Ignored if `labels` is not None.
    cbar : bool, optional
        Whether to draw a colorbar. Ignored if `labels` is not None.
    cbar_width : int
        Width of colorbar (in pixels).
    show_points : bool, optional
        Whether to plot each trajectory as a point.
    line_kws : dict, optional
        Keyword arguments for matplotlib.collection.LineCollection.
    scat_kws : dict, optional
        Keyword arguments for ax.scatter.
    dpi : int, optional
        Dots per inch.
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
    if image is not None:
        if not (image.ndim == 2 or image.ndim == 3):
            raise ValueError('image must be (H, W), (H, W, 3) or (H, W, 4)')
        if image.ndim == 4 and not (image.shape[-1] == 3 or image.shape[-1] == 4):
            raise ValueError('image must be (H, W), (H, W, 3) or (H, W, 4)')
    if labels is not None and labels.shape != (len(tracks),):
        raise ValueError('labels must have shape ({},)'.format(len(tracks)))
    if labels is not None and cbar:
        raise ValueError('cbar must be False if labels are passed')

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

    # set color based on either labels or the z values
    if labels is not None:
        unique = np.unique(labels)
        # label_colors = dict(zip(unique, sns.husl_palette(len(unique))))
        label_colors = dict(zip(unique, sns.color_palette("Set2", len(unique))))
        colors = [label_colors[c] for c in labels]

        lc = LineCollection(tracks[:, :, :2], colors=colors, **line_kws)
    else:
        vmin = tracks[:, :, 2].min() if vmin is None else vmin
        vmax = tracks[:, :, 2].max() if vmax is None else vmax
        norm = Normalize(vmin, vmax)

        points = tracks[:, :, None, :]
        segments = np.concatenate((points[:, :-1], points[:, 1:]), axis=2)
        segments = segments.reshape(-1, 2, 3)

        lc = LineCollection(segments[:, :, :2], cmap=cmap, norm=norm, **line_kws)
        lc.set_array(segments[:, 0, 2])

        if cbar:
            cb = grid.ax.figure.colorbar(lc, cax=grid.ax_cbar)

    grid.ax.add_collection(lc)

    if show_points:
        grid.ax.scatter(tracks[:, :, 0], tracks[:, :, 1], **scat_kws)

    return grid.fig, grid.ax
