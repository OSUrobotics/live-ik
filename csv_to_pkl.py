import csv
import pandas as pd

csv_in = pd.read_csv('quickstats_rerun4.csv')

hand = csv_in[csv_in["name"]=='3v3_25.40.35_40.35.25_1.1_53']

hand.to_pickle('3v3_25.40.35_40.35.25_1.1_53.pkl')
