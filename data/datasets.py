import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

import numpy as np


class TweeterDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        text = self.data[idx]
        label = self.labels[idx]

        sample = {"text": text, "label": label}

        if self.transform:
            sample = self.transform(sample)

        return sample


def load_datasets(csv_file):
    tweeter_frame = pd.read_csv(csv_file)
    tweeter_frame = tweeter_frame[tweeter_frame["sarcastic"] == 1]
    # tweeter_frame = tweeter_frame.drop([1062])
    x0 = tweeter_frame.loc[:, "tweet"].values
    x1 = tweeter_frame.loc[:, "rephrase"].values
    data_x = np.append(x0, x1)
    data_y = np.append(np.ones(len(x1)), np.zeros(len(x0))).astype('int')
    x_train, x_test, y_train, y_test = train_test_split(
        data_x, data_y, test_size=0.2, random_state=0
    )
    train_dataset = TweeterDataset(x_train, y_train)
    test_dataset = TweeterDataset(x_test, y_test)

    return train_dataset, test_dataset
