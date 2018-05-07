import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import trackviz.static


# load data
tracks = pd.read_csv('sample_data/ant_tracking_res.csv').rename(columns={'frame': 't'})

# create labels
trackid = tracks['trackid'].unique()
label = np.random.choice(['a', 'b', 'c'], size=len(trackid))
labels = pd.DataFrame(dict(trackid=trackid, label=label))

# plot
fig, ax = trackviz.static.trajectory_3d(
    tracks, labels=labels, color='label',
    line_kws=dict(linewidths=0.5)
)
fig.savefig('output/static_trajectory_3d_color_labels.png')
# plt.show()
