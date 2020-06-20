import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from helpers.data_formatter import DataFormatter

csv_cols = ['temp', 'rel_humidity', 'precip', 'wind_dir', 'wind_spd', 'atmos_press']
plt.style.use('bmh')
df = pd.read_csv('./Train.csv')

for column in csv_cols:
    vals = df[column].str.split(',')
    df[column] = DataFormatter.convert_to_int_array(vals)
    DataFormatter.fill_nans(df[column], -1)

meta_df = pd.read_csv('./airqo_metadata.csv')
train_data = DataFormatter.merge_meta_data_with_actual_data(df, meta_df)
train_data.drop('ID', axis=1, inplace=True)
train_data.drop('location', axis=1, inplace=True)

test_dataset, training_dataset = train_test_split(train_data, random_state=42, train_size=0.7)

test_targets = test_dataset['target']
train_targets = training_dataset['target']

test_dataset.drop('target', axis=1, inplace=True)
training_dataset.drop('target', axis=1, inplace=True)

test_dataset.to_pickle('./pickle/my_test_data.pkl')
training_dataset.to_pickle('./pickle/my_train_data.pkl')
test_targets.to_pickle('./pickle/my_test_targets.pkl')
training_dataset.to_pickle('./pickle/my_train_targets.pkl')
