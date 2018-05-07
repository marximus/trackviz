import pandas as pd
import trackviz.animate


tracks = pd.read_csv('sample_data/ant_tracking_res.csv')
trackanim = trackviz.animate.TrackAnimation3d(tracks)
trackanim.save('output/animate_3d.mp4')
