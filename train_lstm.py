import numpy as np
import pandas as pd
import torch

from helpers.data_formatter import DataFormatter
from helpers.dataset import Dataset
from neural_net.other_lstm import OtherLstm

test_dataset = pd.read_pickle('./pickle/my_test_data.pkl')
training_dataset = pd.read_pickle('./pickle/my_train_data.pkl')
test_targets = pd.read_pickle('./pickle/my_test_targets.pkl')
training_targets = pd.read_pickle('./pickle/my_train_targets.pkl')

array_features = ['temp', 'precip', 'rel_humidity', 'wind_dir', 'wind_spd', 'atmos_press']
selected_features = ['temp', 'precip', 'rel_humidity', 'wind_dir', 'wind_spd', 'atmos_press', 'km2', 'popn',
                     'loc_altitude']
# input_size = 120 * (len(array_features) +1) + (len(selected_features) - len(array_features))
input_size = 726
train_matrix = np.stack(training_dataset[selected_features].values)
test_matrix = np.stack(test_dataset[selected_features].values)

BATCH_SIZE = 1

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

train_tensor = torch.from_numpy(train_vectors)
test_tensor = torch.from_numpy(test_vectors)

# train_dataset = Dataset(train_tensor, training_targets)
test_dataset = Dataset(test_tensor, test_targets)

model = OtherLstm(726, 50, batch_size=BATCH_SIZE, output_dim=1, num_layers=2)
loss_fn = torch.nn.MSELoss()

optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

NUM_EPOCS = 100
hist = np.zeros(NUM_EPOCS)

for t in range(NUM_EPOCS):
    model.zero_grad()
    current_epoc_loss = 0

    train_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)

    for i, (inputs, validation_labels) in enumerate(train_loader):
        model.train()
        inputs = inputs.view(50, input_size)
        outputs = model(inputs)
        loss = loss_fn(outputs, validation_labels)
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        model.eval()
        current_epoc_loss += loss.item()

    # Forward pass
    y_pred = model(train_loader)

    loss = loss_fn(y_pred, training_targets)
    if t % 5 == 0:
        print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
