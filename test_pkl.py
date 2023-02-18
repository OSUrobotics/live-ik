import os
import pickle as pkl


path_to = os.path.abspath(os.path.dirname(__file__))
file_path = os.path.join(path_to, "actual_trajectories_2v2", "N_2v2_1.1_1.1_1.1_1.1.pkl")

with open(file_path, 'rb') as f:
    data = pkl.load(f)


path_to = os.path.abspath(os.path.dirname(__file__))
file_path = os.path.join(path_to, "Open_Loop_Data", "angles_N.pkl")

with open(file_path, 'rb') as f:
    data_old = pkl.load(f)

print("hi")