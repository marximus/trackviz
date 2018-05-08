import imageio
import pandas as pd
import matplotlib.pyplot as plt

import trackviz.static


tracks = pd.read_csv('sample_data/ant_tracking_res.csv').rename(columns={'frame': 't'})
image = imageio.get_reader('sample_data/ant_dataset.mp4').get_next_data()

fig, ax = trackviz.static.trajectory_2d(tracks, image=image, line_kws=dict(linewidths=0.5))
fig.savefig('output/static_2d.png')
# plt.show()
