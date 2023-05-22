import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from bvp_to_hr import read_data

def get_gsr_summary(df):
    summary = df[['Phase', 'GSR']].dropna().groupby("Phase").mean()
    summary['diff_from_baseline'] = np.concatenate([np.zeros(1), summary['GSR'].values[1:] - summary['GSR'].values[:-1]])
    summary = summary.rename(columns = {'GSR': 'mean_GSR'})
    return summary

def main(path_to_file):

    print('reading file...')
    df = read_data(path_to_file)

    print("processing file...")
    summary = get_gsr_summary(df)
    
    print("writing file...")
    filename, ext = path_to_file.split('.')
    summary.to_csv(f"{filename}_summary_gsr.{ext}")

    return summary


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Wrong usage; try $ python gsr.py <path_to_file>')
        exit()
    path_to_file = sys.argv[1]
    main(path_to_file)