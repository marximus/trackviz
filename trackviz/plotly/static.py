import pandas as pd
import plotly.graph_objs as go


def trajectory_3d(tracks, show_points=False, line_kws=None, marker_kws=None):
    """
    3d plot of trajectories.

    Parameters
    ----------
    tracks : pandas.DataFrame, columns (x, y, t, trackid)
        Trajectory data. Each row is a point in a trajectory, where each trajectory has a unique trackid.
    show_points : bool, optional
        Whether to plot trajectory points.
    line_kws : dict, optional
        Keyword arguments for plotly.graph_objs.Scatter.line. See https://plot.ly/python/reference/#scatter-line.
    marker_kws : dict, optional
        Keyword arguments for plotly.graph_objs.Scatter.marker. See https://plot.ly/python/reference/#scatter-marker.

    Returns
    -------
    figure : pyplot.graph_objs.Figure
        Figure containing trajectory plot.
    """
    if not pd.Series(['x','y','t','trackid']).isin(tracks.columns).all():
        raise ValueError('tracks must have columns x, y, t, and trackid')
    if tracks.shape[0] < 1:
        raise ValueError('tracks has zero length')

    mode = 'lines+markers' if show_points else 'lines'
    line_kws = {} if line_kws is None else line_kws.copy()
    marker_kws = {} if marker_kws is None else marker_kws.copy()

    data = []
    for trackid, group in tracks.groupby('trackid'):
        trace = go.Scatter3d(
            x=group['x'], y=group['y'], z=group['t'],
            mode=mode,
            line=line_kws, marker=marker_kws
        )
        data.append(trace)

    layout = dict(
        scene=dict(
            xaxis=dict(zeroline=True, title='x'),
            yaxis=dict(zeroline=True, title='y'),
            zaxis=dict(zeroline=True, title='t'),
        )
    )

    figure = go.Figure(data=data, layout=layout)

    return figure
