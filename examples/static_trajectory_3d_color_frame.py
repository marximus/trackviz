import imageio
import pandas as pd
import matplotlib.pyplot as plt

import trackviz.static

tracks = pd.read_csv('sample_data/ant_tracking_res.csv').rename(columns={'frame': 't'})
fig, ax = trackviz.static.trajectory_3d(tracks, color='t', line_kws=dict(linewidths=0.5))
fig.savefig('output/static_3d_color_frame.png')
# plt.show()
