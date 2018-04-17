import numpy as np


def tracks_to_lines(tracks):
    ntracks, npts, _ = tracks['coords'].shape
    z = np.tile(np.arange(npts)-npts+1, (ntracks, 1)) + tracks['frame_num'][:, None]
    lines = np.append(tracks['coords'], z[:, :, None], axis=2)
    return lines


def tracks_to_segments(tracks):
    """
    Return individual line segments for each trajectory.

    Parameters
    ----------
    tracks : ndarray, shape (n_tracks, n_pts, 3)
        Trajectory data.

    Returns
    -------
    segments : ndarray, shape (n_tracks*(n_pts-1), 2, 3)
        Individual line segments.
    """
    points = tracks[:, :, None, :]
    segments = np.concatenate((points[:, :-1], points[:, 1:]), axis=2)
    segments = segments.reshape(-1, 2, 3)

    return segments
