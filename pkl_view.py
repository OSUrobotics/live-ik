import pickle as pkl 
import os
# "/media/kyle/16ABA159083CA32B/kyle/live/2v2/1.1_1.1_1.1/N_2v2_1.1_1.1_1.1.pkl"
##path_to = os.path.abspath(os.path.dirname(__file__))
#file_path = os.path.join(path_to, "actual_trajectories_2v2/E_2v2_1.1_1.1_1.1_1.1.pkl")
with open("/media/kyle/16ABA159083CA32B/kyle/replay/2v2/1.1_1.1_1.1_1.1/N_2v2_1.1_1.1_1.1_1.1.pkl", 'rb') as f:
    data = pkl.load(f)

print("hi")