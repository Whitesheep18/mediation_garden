import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def read_data(path_to_file, verbose=True) -> pd.DataFrame:
    df = pd.read_csv(path_to_file,
                    sep='\t', #stupid separator. What's wrong with comma?
                    names = ["Time", "ID", "blah", "BP", "GSR", "Phase"] # Let's give some nice column names
                    ).drop(columns=["blah"]) # We drop "Acquired Data" - I think it's spelled "Aquired" though...
    df['Time'] = pd.to_datetime(df["Time"])
    n = len(df)
    if verbose:
        print('Num samples:', n)
    df = df.set_index("Time")
    sampling_rate = df.groupby(pd.Grouper(freq="1s")).count().mode()['ID'].values[0]
    if verbose:
        print('sample rate:', sampling_rate)

    df['BP'] = df['BP'].rolling('50ms').mean().fillna(method='bfill')
    if verbose:
        print("Phases:", df['Phase'].unique())
    return df

def find_peaks(df, offset_scaler=6, smoothing_window='3s', verbose=True) -> pd.DataFrame: 
    """
    offset scaler and smoothing window depends heavily sample rate, base heart rate, tightness of device etc. 
        3s works well with 2 very different individuals, but it's best to ensure:
        sr 100Hz, healthy person, elastic band around middle finger if subject is small
    """

    def max_argmax_aggregate(x):
        if x.name == 'BP':
            return x.idxmax()
        else:
            return x.max()

    sample = df[['BP']]
    rolling_mean = sample.rolling(smoothing_window, center=True).mean()

    #proxy for amplitude - let's pray there's valid data in this range
    resting_range = sample.values[-2000:-1000].max() - sample.values[-2000:-1000].min()

     # we need to offset the running mean to avoid counting a pulse at the half pulse (only local absolute max counts)
    offset = resting_range / offset_scaler
    if verbose:
        print("Offset", offset)
    df['Onset'] = sample > rolling_mean + offset # find all samples higher than running mean + an offset

    df['Periods'] = np.cumsum(df['Onset'] != df['Onset'].shift(1))
    peaks = df.groupby('Periods').agg(max_argmax_aggregate)
    peaks = peaks.loc[peaks['Onset']]
    return peaks

def find_heart_rate(df, peaks, median_window='10s', smoothing_window='2s') -> pd.DataFrame:
    # get hr
    hr = pd.DataFrame({'HR': 60/peaks['BP'].diff().dt.total_seconds(), 'Time': peaks['BP']})
    hr = hr.set_index('Time')
    hr['HR_clean'] = hr['HR'].rolling(median_window, center=True).median().rolling(smoothing_window).mean()
    hr['Phase'] = np.nan
    baseline_mean = 0
    # append phase and get summary
    col = {'baseline': 'turquoise', 'stress':'red', 'experiment':'olive', 'control':'brown'}
    summary = {'Phase': [], 'mean_HR': [], 'diff_from_baseline': []}
    for phase, colour in col.items():
        if not phase in df['Phase'].unique():
            continue
        start_phase = df.loc[df['Phase']==phase].index.min()
        end_phase = df.loc[df['Phase']==phase].index.max()
        phase_mask = (hr.index > start_phase) & (hr.index < end_phase)
        mean_hr = hr['HR_clean'].loc[phase_mask].mean()
        if phase == 'baseline':
            baseline_mean = mean_hr
        hr['Phase'].loc[phase_mask] = phase
        summary['Phase'].append(phase)
        summary['mean_HR'].append(mean_hr)
        summary['diff_from_baseline'].append(mean_hr-baseline_mean)

    return hr, pd.DataFrame(summary)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Wrong usage; try $ python bvp_to_hr.py <path_to_file>')
        exit()

    print('reading file...')
    path_to_file = sys.argv[1]
    df = read_data(path_to_file)

    print('processing file...')
    peaks = find_peaks(df)
    hr, summary = find_heart_rate(df, peaks)

    filename, ext = path_to_file.split('.')
    print('writing files...')
    hr.to_csv(f"{filename}_hr.{ext}")
    summary.to_csv(f"{filename}_summary.{ext}")