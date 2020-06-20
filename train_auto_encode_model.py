import numpy as np
import pandas as pd

from helpers.data_formatter import DataFormatter

test_dataset = pd.read_pickle('./pickle/my_test_data.pkl')
training_dataset = pd.read_pickle('./pickle/my_train_data.pkl')
test_targets = pd.read_pickle('./pickle/my_test_targets.pkl')
training_targets = pd.read_pickle('./pickle/my_train_targets.pkl')

array_features = ['temp', 'precip', 'rel_humidity', 'wind_dir', 'wind_spd', 'atmos_press']
selected_features = ['temp', 'precip', 'rel_humidity', 'wind_dir', 'wind_spd', 'atmos_press', 'km2', 'popn',
                     'loc_altitude']
input_size = 726
train_matrix = np.stack(training_dataset[selected_features].values)
test_matrix = np.stack(test_dataset[selected_features].values)

BATCH_SIZE = 50

my_vals = []
for dimension in train_matrix:
    temp_array = DataFormatter.flatten_structure_to_one_dim_structure(dimension)
    my_vals.append(np.array(temp_array))
train_vectors = np.array(my_vals)

my_vals = []
for dimension in test_matrix:
    temp_array = DataFormatter.flatten_structure_to_one_dim_structure(dimension)
    my_vals.append(np.array(temp_array))
test_vectors = np.array(my_vals)
