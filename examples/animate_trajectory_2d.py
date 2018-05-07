import os
import pandas as pd
import trackviz.animate

tracks = pd.read_csv('sample_data/ant_tracking_res.csv')

os.makedirs('output', exist_ok=True)

trackanim = trackviz.animate.TrackAnimation2d(tracks)
trackanim.save('output/animate_trajectory_2d.mp4')