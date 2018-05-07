import numpy as np
import pandas as pd
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib.lines import Line2D

from trackviz.tools import FigureAxes


# TODO: Add trails parameter to which will not only show trajectories with a point in the current frame
class TrackAnimation2d:
    """
    Class for 2d animation of trajectories.

    Parameters
    ----------
    tracks : pandas.DataFrame, columns ('trackid', 'frame', 'x', 'y')
        Trajectory data.
    frames : ndarray, shape (L, H, W) or (L, H, W, 3) or (L, H, W, 4), optional
        Video frames to plot in background.
    labels : ndarray, shape (N,), dtype int, optional
        Trajectory labels. Will be used to color trajectories.
    color : {None, 'frame', 'label'}
        Value(s) used to color trajectories. If `color` == 'frame', color each line segment of each trajectory according
        to its frame value. If `color == 'label', color each trajectory based on its label. If `color` == None, do not
        color the trajectories. Default is None.
    xlim, ylim: list, shape (2,) optional
        Data limits for the x and y axis. If no limits are given, they will be computed from the data.
    cmap : matplotlib colormap name or object, optional
        Colormap used to map frame coordinates to colors. Ignored if `labels` is not None.
    cbar : bool, optional
        Whether to draw a colorbar. Must be False if `color` == None, because no colormapping is used.
    cbar_width : int
        Width of colorbar (in pixels).
    dpi : int or None, optional
        Figure dots per inch. If None, default to matplotlib.rcParams['figure.dpi'].
    scale : float, optional
        Output size scale.
    line_kws : dict, optional
        Keyword arguments passed to matplotlib LineCollection.
    frames_kws : dict, optional
        Keyword arguments passed to ax.imshow.

    Attributes
    -------
    figure : matplotlib Figure
        Figure containing trajectory plot.
    ax : matplotlib Axes
        Main Axes.
    cax : matplotlib Axes
        Colorbar Axes.
    """
    def __init__(
            self,
            tracks,
            frames=None,
            labels=None,
            color=None,
            xlim=None, ylim=None,
            cmap=None,
            cbar=None,
            cbar_width=30,
            dpi=None,
            scale=1.,
            line_kws=None,
            frames_kws=None
    ):
        if not (isinstance(tracks, pd.DataFrame) and all(x in tracks.columns for x in ('trackid', 'frame', 'x', 'y'))):
            raise ValueError('tracks must be pandas.DataFrame with columns "trackid", "frame", "x", and "y"')
        if frames is not None:
            if not (frames.ndim == 3 or frames.ndim == 4):
                raise ValueError('frames must be (L, H, W), (L, H, W, 3) or (L, H, W, 4)')
            if frames.ndim == 4 and not (frames.shape[-1] == 3 or frames.shape[-1] == 4):
                raise ValueError('frames must be (L, H, W), (L, H, W, 3) or (L, H, W, 4)')
        if color not in ('frame', 'label', None):
            raise ValueError('color must be "frame", "label", or None')
        if color == 'label' and labels is None:
            raise ValueError('labels cannot be None if color == "label"')
        if color is None and cbar:
            raise ValueError('cbar must be False when color == None')
        if color is not None:
            raise ValueError('only color=None is currently supported')

        tracks = tracks.sort_values(['trackid', 'frame'])
        tracks['pointid'] = tracks.groupby('trackid').cumcount()
        tracks['n_pts'] = tracks.groupby('trackid')['trackid'].transform('count')

        n_tracks = len(tracks['trackid'].unique())
        dpi = matplotlib.rcParams['figure.dpi'] if dpi is None else dpi

        line_kws = {} if line_kws is None else line_kws.copy()
        line_kws.update(animated=True)

        frames_kws = {} if frames_kws is None else frames_kws.copy()
        frames_kws.update(animated=True)
        if frames is not None and frames.ndim == 3:  # is grayscale
            frames_kws.update(cmap='gray')

        # compute x and y limits
        if xlim is None:
            xlim = (tracks['x'].min(), tracks['x'].max())
            if frames is not None:
                height, width = frames.shape[1:3]
                xlim = (min(0, xlim[0]), max(width, xlim[1]))
        if ylim is None:
            ylim = (tracks['y'].min(), tracks['y'].max())
            if frames is not None:
                height, width = frames.shape[1:3]
                ylim = (min(0, ylim[0]), max(height, ylim[1]))
            ylim = ylim[::-1]  # invert y axis

        # compute frame limits
        framelim = (tracks['frame'].min(), tracks['frame'].max())
        if frames is not None:
            framelim = (min(0, framelim[0]), max(len(frames), framelim[1]))

        # determine size of main axes
        width, height = np.fabs(xlim[0] - xlim[1]), np.fabs(ylim[0] - ylim[1])
        axsize = (np.array((width, height)) * scale).astype(np.int)

        # set up figure and axes
        grid = FigureAxes(axsize, 20, dpi, cbar, cbar_width, 40, 40, 40, 10, fig_func=Figure)
        FigureCanvasAgg(grid.fig)
        grid.ax.set_aspect('equal')
        grid.ax.set_xlim(xlim)
        grid.ax.set_ylim(ylim)

        lines = {trackid: Line2D([], [], **line_kws) for trackid in tracks['trackid'].unique()}

        im = grid.ax.imshow(np.zeros_like(frames[0]), **frames_kws) if frames is not None else None

        self.figure = grid.fig
        self.ax = grid.ax
        self.cax = grid.cax
        self.dpi = dpi

        self._framelim = framelim
        self._n_tracks = n_tracks
        self._tracks = tracks
        self._frames = frames

        self._lines = lines
        self._im = im

        self._track_id_old = list()

    def _draw_frame(self, frame_idx):
        # draw video frame
        if self._frames is not None:
            if frame_idx < len(self._frames):
                self._im.set_data(self._frames[frame_idx])
            elif frame_idx == len(self._frames):
                # self.ax.images.remove(self._im)
                self._im.remove()
                self._im = None

        # draw trajectories
        inds = np.flatnonzero(self._tracks['frame'] == frame_idx)
        for row in self._tracks.iloc[inds, :].itertuples(index=False):
            x, y = self._lines[row.trackid].get_data()
            x.append(row.x)
            y.append(row.y)
            self._lines[row.trackid].set_data(x, y)

            if row.pointid == 0:               # first trajectory point
                self.ax.add_line(self._lines[row.trackid])
            if row.pointid == (row.n_pts - 1): # last trajectory point
                self._track_id_old.append(row.trackid)

    def save(self, filename, fps=30, bitrate=-1, codec=None):
        writer = animation.FFMpegWriter(fps=fps, bitrate=bitrate, codec=codec)
        with writer.saving(self.figure, filename, self.dpi):
            for frame_idx in range(self._framelim[0], self._framelim[1]+1):
                print('frame {}'.format(frame_idx))
                self._draw_frame(frame_idx)
                writer.grab_frame()

                # remove old trajectories
                while len(self._track_id_old) > 0:
                    trackid = self._track_id_old.pop()
                    self._lines[trackid].remove()
                    self._lines[trackid] = None
