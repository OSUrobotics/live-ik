import csv
import pickle as pkl
import numpy as np
import pandas as pd




out_dict = {}




with open('joint_angs.csv', mode='r') as csv_file:
    csvreader = csv.reader(csv_file)
    for i, row in enumerate(csvreader):
        if i == 0:
            continue

        out_dict[i-1] = {"joint_1": float(row[2]), "joint_2": float(row[3]), "joint_3": float(row[0]), "joint_4": float(row[1])}

        
        
        

with open('rl_joints.pickle', 'wb') as handle:
    pkl.dump(out_dict, handle)


