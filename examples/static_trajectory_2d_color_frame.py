import pandas as pd
import matplotlib.pyplot as plt

import trackviz.static


tracks = pd.read_csv('sample_data/ant_tracking_res.csv').rename(columns={'frame': 't'})
fig, ax = trackviz.static.trajectory_2d(tracks, color='t', cbar=True)
fig.savefig('output/static_trajectory_2d_color_frame.png')
# plt.show()
