import pandas as pd
import matplotlib.pyplot as plt

import trackviz.static


tracks = pd.read_csv('sample_data/ant_tracking_res.csv').rename(columns={'frame': 't'})
fig, ax = trackviz.static.trajectory_2d(tracks, show_points=True)
fig.savefig('output/static_2d_points.png')
# plt.show()
