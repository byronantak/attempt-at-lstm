import json

import matplotlib.pyplot as plt
import pandas as pd

from helpers.stats_calculator import StatsCalculator
from helpers.data_formatter import DataFormatter
from helpers.np_encoder import NpEncoder

csv_cols = ['temp', 'rel_humidity', 'precip', 'wind_dir', 'wind_spd', 'atmos_press']

plt.style.use('bmh')

df = pd.read_csv('./Train.csv')

for column in csv_cols:
    vals = df[column].str.split(',')
    df[column] = DataFormatter.convert_to_int_array(vals)
    # df[column] = pd.to_numeric(df[column], errors='coerce')

column_summaries = {}
for column in df:
    print(column.upper())
    if column in csv_cols:
        mean, std_dev_mean, min, max, nan_count_avg, min_nan_row, max_nan_row = StatsCalculator.get_descriptive_statistics_for_data(
            df[column])
        column_summaries[column] = {
            'mean': mean,
            'std_dev_mean': std_dev_mean,
            'nan_count_avg': nan_count_avg,
            'min': min,
            'max': max,
            'min_nan_row': min_nan_row,
            'max_nan_row': max_nan_row
        }
        json_str = json.dumps(column_summaries, cls=NpEncoder, indent=4)
        with open('results/descriptive-stats.json', 'w') as fp:
            fp.write(json_str)
    print()