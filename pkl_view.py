import pickle as pkl 
import os
# "/media/kyle/16ABA159083CA32B/kyle/live/2v2/1.1_1.1_1.1/N_2v2_1.1_1.1_1.1.pkl"
path_to = os.path.abspath(os.path.dirname(__file__))
# "/media/kyle/16ABA159083CA32B/kyle/replay/3v3_50.0.25.0.25.0_40.0.35.0.25.0_1.1_540/N_3v3_50.0.25.0.25.0_40.0.35.0.25.0_1.1_540.pkl"
# "Open_Loop_Data/3v3_50.25.25_45.30.25_1.1_53/N_3v3_50.25.25_45.30.25_1.1_53.pkl"
file_path = os.path.join(path_to, "trial_state.pkl")#"Open_Loop_Data/2v2_50.50_50.50_1.1_63/NW_2v2_50.50_50.50_1.1_63.pkl")
with open(file_path, 'rb') as f:
    data = pkl.load(f)

    print("hi")