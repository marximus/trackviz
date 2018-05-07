import pandas as pd
import matplotlib.pyplot as plt
import imageio

import trackviz.static

tracks = pd.read_csv('sample_data/ant_tracking_res.csv').rename(columns={'frame': 't'})
video = imageio.mimread('sample_data/ant_dataset.mp4', memtest=False)
image = video[0]

fig, ax = trackviz.static.trajectory_2d(
    tracks,
    image=image,
    line_kws=dict(linewidths=0.8),
    im_kws=dict(interpolation='none')
)
fig.savefig('output/static_trajectory_2d_image.png')
# plt.show()
