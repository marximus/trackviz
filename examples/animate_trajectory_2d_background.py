import os
import numpy as np
import pandas as pd
import imageio
import trackviz.animate

os.makedirs('output', exist_ok=True)

tracks = pd.read_csv('sample_data/ant_tracking_res.csv')
frames = np.array(imageio.mimread('sample_data/ant_dataset.mp4', memtest=False))

trackanim = trackviz.animate.TrackAnimation2d(tracks, frames=frames)
trackanim.save('output/animate_trajectory_2d_background.mp4')
