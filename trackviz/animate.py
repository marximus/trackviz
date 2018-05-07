import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3D

from trackviz.tools import FigureAxes


# TODO: Add option to color trajectories based on labels
# TODO: Add trails parameter which will not only show trajectories with a point in the current frame


class TrackAnimation:
    """Base class for creating 2d/3d trajectory animation. """
    def __init__(self, tracks):
        if not (isinstance(tracks, pd.DataFrame) and all(x in tracks.columns for x in ('trackid', 'frame', 'x', 'y'))):
            raise ValueError('tracks must be pandas.DataFrame with columns "trackid", "frame", "x", and "y"')
        if len(tracks) == 0:
            raise ValueError('tracks must be non-empty')

        # defined in base class
        self._dpi = 100
        self._track_id_old = list()

        # defined in subclass
        self.figure = None
        self._framelim = None
        self._lines = None

    def _draw_frame(self, frame_idx):
        """Override in subclass."""
        pass

    def save(self, filename, fps=30, bitrate=-1, codec=None):
        writer = animation.FFMpegWriter(fps=fps, bitrate=bitrate, codec=codec)
        with writer.saving(self.figure, filename, self._dpi):
            for frame_idx in range(self._framelim[0], self._framelim[1]+1):
                self._draw_frame(frame_idx)
                writer.grab_frame()

                # remove old trajectories
                while len(self._track_id_old) > 0:
                    trackid = self._track_id_old.pop()
                    self._lines[trackid].remove()
                    self._lines[trackid] = None


class TrackAnimation2d(TrackAnimation):
    """
    Class for 2d animation of trajectories.

    Parameters
    ----------
    tracks : pandas.DataFrame, columns ('trackid', 'frame', 'x', 'y')
        Trajectory data.
    frames : ndarray, shape (L, H, W) or (L, H, W, 3) or (L, H, W, 4), optional
        Video frames to plot in background.
    xlim, ylim: list, shape (2,) optional
        Data limits for the x and y axis. If no limits are given, they will be computed from the data.
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
    """
    def __init__(self, tracks, frames=None, scale=1., xlim=None, ylim=None, line_kws=None, frames_kws=None):
        super().__init__(tracks)
        if frames is not None:
            if not (frames.ndim == 3 or frames.ndim == 4):
                raise ValueError('frames must be (L, H, W), (L, H, W, 3) or (L, H, W, 4)')
            if frames.ndim == 4 and not (frames.shape[-1] == 3 or frames.shape[-1] == 4):
                raise ValueError('frames must be (L, H, W), (L, H, W, 3) or (L, H, W, 4)')

        tracks = tracks.sort_values(['trackid', 'frame'])
        tracks['pointid'] = tracks.groupby('trackid').cumcount()
        tracks['n_pts'] = tracks.groupby('trackid')['trackid'].transform('count')

        # set up keyword arguments
        line_kws = {} if line_kws is None else line_kws.copy()
        line_kws.update(animated=True)

        frames_kws = {} if frames_kws is None else frames_kws.copy()
        frames_kws.update(animated=True)
        if frames is not None and frames.ndim == 3:  # is grayscale
            frames_kws.update(cmap='gray')

        # compute x, y, and frame limits
        if xlim is None:
            xlim = (np.floor(tracks['x'].min()), np.ceil(tracks['x'].max()))
            if frames is not None:
                height, width = frames.shape[1:3]
                xlim = (min(0, xlim[0]), max(width, xlim[1]))
        if ylim is None:
            ylim = (np.floor(tracks['y'].min()), np.ceil(tracks['y'].max()))
            if frames is not None:
                height, width = frames.shape[1:3]
                ylim = (min(0, ylim[0]), max(height, ylim[1]))
            ylim = ylim[::-1]  # invert y axis

        framelim = (tracks['frame'].min(), tracks['frame'].max())
        if frames is not None:
            framelim = (min(0, framelim[0]), max(len(frames), framelim[1]))

        # determine size of output in pixels
        width, height = np.fabs(xlim[0] - xlim[1]), np.fabs(ylim[0] - ylim[1])
        axsize = (np.array((width, height)) * scale).astype(np.int)

        # set up figure and axes
        grid = FigureAxes(axsize, 20, self._dpi, False, 0, 0, 0, 0, 0, fig_func=Figure)
        fig, ax = grid.fig, grid.ax
        FigureCanvasAgg(fig)
        fig.set_frameon(False)
        ax.set_aspect('equal')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        lines = {trackid: Line2D([], [], **line_kws) for trackid in tracks['trackid'].unique()}
        im = ax.imshow(np.zeros_like(frames[0]), **frames_kws) if frames is not None else None

        self.figure = fig
        self.ax = ax

        self._tracks = tracks
        self._framelim = framelim
        self._frames = frames

        # artists on plot that will be updated
        self._lines = lines
        self._im = im

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


class TrackAnimation3d(TrackAnimation):
    """
    Class for 3d animation of trajectories.

    Parameters
    ----------
    tracks : pandas.DataFrame, columns ('trackid', 'frame', 'x', 'y')
        Trajectory data.
    xlim, ylim, zlim : list, shape (2,) optional
        Data limits for the x, y, and z axis. If no limits are given, they will be computed from the data.
    line_kws : dict, optional
        Keyword arguments passed to matplotlib LineCollection.

    Attributes
    -------
    figure : matplotlib Figure
        Figure containing trajectory plot.
    ax : matplotlib Axes
        Axes object in figure.
    """
    def __init__(self, tracks, xlim=None, ylim=None, zlim=None, line_kws=None):
        super().__init__(tracks)

        tracks = tracks.sort_values(['trackid', 'frame'])
        tracks['pointid'] = tracks.groupby('trackid').cumcount()
        tracks['n_pts'] = tracks.groupby('trackid')['trackid'].transform('count')

        # set up keyword arguments
        line_kws = {} if line_kws is None else line_kws.copy()
        line_kws.update(animated=True)

        # compute axis limits
        xlim = (np.floor(tracks['x'].min()), np.ceil(tracks['x'].max())) if xlim is None else xlim
        ylim = (np.floor(tracks['y'].min()), np.ceil(tracks['y'].max())) if ylim is None else ylim
        framelim = (tracks['frame'].min(), tracks['frame'].max())

        fig = Figure()
        FigureCanvasAgg(fig)
        ax = fig.add_axes([0, 0, 1, 1], projection='3d')

        ax.set_xlim3d(xlim)
        ax.set_ylim3d(ylim)
        ax.set_zlim3d(framelim)

        lines = {trackid: Line3D([], [], [], **line_kws) for trackid in tracks['trackid'].unique()}

        self.figure = fig
        self.ax = ax

        self._tracks = tracks
        self._framelim = framelim

        # artists on plot that will be updated
        self._lines = lines

    def _draw_frame(self, frame_idx):
        inds = np.flatnonzero(self._tracks['frame'] == frame_idx)
        for row in self._tracks.iloc[inds, :].itertuples(index=False):
            x, y, z = self._lines[row.trackid]._verts3d
            x.append(row.x)
            y.append(row.y)
            z.append(row.frame)
            self._lines[row.trackid]._verts3d = x, y, z

            if row.pointid == 0:               # first trajectory point
                self.ax.add_line(self._lines[row.trackid])
            if row.pointid == (row.n_pts - 1): # last trajectory point
                self._track_id_old.append(row.trackid)
