import torch.nn.functional as F
from torch import nn


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # encoder
        self.enc1 = nn.Linear(in_features=726, out_features=626)
        self.enc2 = nn.Linear(in_features=626, out_features=526)
        self.enc3 = nn.Linear(in_features=526, out_features=426)
        self.enc4 = nn.Linear(in_features=426, out_features=326)
        self.enc5 = nn.Linear(in_features=326, out_features=32)

        # decoder
        self.dec1 = nn.Linear(in_features=32, out_features=326)
        self.dec2 = nn.Linear(in_features=326, out_features=426)
        self.dec3 = nn.Linear(in_features=426, out_features=526)
        self.dec4 = nn.Linear(in_features=526, out_features=626)
        self.dec5 = nn.Linear(in_features=626, out_features=726)

    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        x = F.relu(self.enc5(x))

        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        x = F.relu(self.dec5(x))
        return x
