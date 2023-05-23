import pickle as pkl
import pandas as pd
with open("episode_4973.pkl", "rb") as f:
    object = pkl.load(f)
    
df = pd.DataFrame(object)
df.to_csv('file.csv')