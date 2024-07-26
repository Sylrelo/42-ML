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

        self.training_features, _data_for_std = normalize_dataset(self.training_features)
        self.test_features, _ = normalize_dataset(self.test_features, std_data=_data_for_std)

        self.std_values = _data_for_std


def load_dataset(file):
    input_data = pd.read_csv(file, header=None)

    classes = np.array(input_data.get(1))
    input_data.drop([0, 1], axis=1, inplace=True)
    input_data.fillna(input_data.median(), inplace=True)
    features = np.array(input_data).astype(float)

    classes = np.where(classes == 'M', 0, 1).astype(int)

    return features, classes


def normalize_dataset(features, std_data=None):
    _std_values = []
    _len = features.shape[1]

    for i in range(_len):
        _column = features[:, i]

        _mean = float(np.mean(_column)) if std_data is None else std_data[i]["mean"]
        _std = float(np.std(_column)) if std_data is None else std_data[i]["std"]

        _std_values.append({
            "mean": _mean,
            "std": _std,
        })

        features[:, i] = (_column - _mean) / _std

    return features, _std_values


def convert_to_one_hot(labels):
    y_one_hot = np.zeros((labels.size, labels.max() + 1))
    y_one_hot[np.arange(labels.size), labels] = 1

    return y_one_hot

def split_dataset(base_path, cut_percent=0.8):
    input_data = pd.read_csv(base_path + "/../resources/data.csv", header=None)

    train_dataset = input_data.sample(frac=cut_percent)
    test_dataset = input_data.drop(train_dataset.index)

    train_dataset.to_csv(
        f"{base_path}/../resources/data_training.csv",
        header=False,
        index=False
    )

    test_dataset.to_csv(
        f"{base_path}/../resources/data_test.csv",
        header=False,
        index=False
    )
