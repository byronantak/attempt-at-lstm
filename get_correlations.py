import json

import matplotlib.pyplot as plt
import pandas as pd

from helpers.correlation_calculator import CorrelationCalculator
from helpers.data_formatter import DataFormatter
from helpers.np_encoder import NpEncoder

csv_cols = ['temp', 'rel_humidity', 'precip', 'wind_dir', 'wind_spd', 'atmos_press']
plt.style.use('bmh')
df = pd.read_csv('./Train.csv')

for column in csv_cols:
    vals = df[column].str.split(',')
    df[column] = DataFormatter.convert_to_int_array(vals)

correl_dict = CorrelationCalculator.calculate_correlations(csv_cols, df)
json_str = json.dumps(correl_dict, cls=NpEncoder, indent=4)
with open('results/correlations.json', 'w') as fp:
    fp.write(json_str)
