import pandas as pd
import numpy as np


class LoadDataset:
    def __init__(self, base_path):
        self.training_features, self.training_classes = (
            load_dataset(f"{base_path}/../resources/data_training.csv")
        )

        self.test_features, self.test_classes = (
            load_dataset(f"{base_path}/../resources/data_test.csv")
        )

        self.training_features, _, _ = normalize_dataset(self.training_features)
        self.test_features, _, _ = normalize_dataset(self.test_features)


def load_dataset(file):
    input_data = pd.read_csv(file, header=None)

    classes = np.array(input_data.get(1))
    input_data.drop([0, 1], axis=1, inplace=True)
    input_data.fillna(input_data.median(), inplace=True)
    features = np.array(input_data).astype(float)

    classes = np.where(classes == 'M', 0, 1).astype(int)

    return features, classes


def normalize_dataset(features):
    _mean = np.mean(features, axis=0)
    _std = np.std(features)

    normalized = (features - _mean) / _std

    return normalized, _mean, _std


def convert_to_one_hot(labels):
    y_one_hot = np.zeros((labels.size, labels.max() + 1))
    y_one_hot[np.arange(labels.size), labels] = 1

    return y_one_hot
