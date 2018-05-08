import imageio
import pandas as pd
import matplotlib.pyplot as plt

import trackviz.static


tracks = pd.read_csv('sample_data/ant_tracking_res.csv').rename(columns={'frame': 't'})
image = imageio.get_reader('sample_data/ant_dataset.mp4').get_next_data()

fig, ax = trackviz.static.trajectory_2d(tracks, color='t', cbar=True)
fig.savefig('output/static_2d_color_frame.png')
# plt.show()
