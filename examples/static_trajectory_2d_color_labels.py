import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import trackviz.static


# load data
tracks = pd.read_csv('sample_data/ant_tracking_res.csv').rename(columns={'frame': 't'})
image = imageio.get_reader('sample_data/ant_dataset.mp4').get_next_data()

# create labels
trackid = tracks['trackid'].unique()
label = np.random.choice(['a', 'b', 'c'], size=len(trackid))
labels = pd.DataFrame(dict(trackid=trackid, label=label))

# plot
fig, ax = trackviz.static.trajectory_2d(tracks, image=image, labels=labels, color='label', cbar=True)
fig.savefig('output/static_2d_color_labels.png')
# plt.show()
