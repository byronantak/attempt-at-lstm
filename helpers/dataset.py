from torch.utils import data


class Dataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, input, labels):
        """Initialization"""
        self.labels = labels.tolist()
        self.inputs = input

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.inputs)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        ID = self.inputs[index]
        # Load data and get label
        X = ID
        y = self.labels[index]
        return X, y
