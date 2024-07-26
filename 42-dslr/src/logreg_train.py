import sys
from math import ceil

import numpy as np
import pandas as pd
import logreg_utils as lu
from utils import open_arg_csv, prepare_x
import matplotlib.pyplot as plt


def compute_cost(theta, bias, X, y):
    m = len(y)
    h = lu.sigmoid(np.dot(X, theta) + bias)
    epsilon = 1e-5
    cost = -(1 / m) * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
    return cost


class House:
    def __init__(self, features_count, learning_rate):
        self.weights = np.random.randn(features_count) * 0.1
        # self.weights = np.zeros(features_count)
        self.bias = 0
        self.cost = []
        self.cost_test = []

        self._learning_rate = learning_rate

    def gradient_descent(self, x, y, m):
        sig = lu.sigmoid(x.dot(self.weights) + self.bias)

        self.weights = self.weights - self._learning_rate * m * np.dot(x.T, (sig - y))
        self.bias = self.bias - self._learning_rate * m * np.sum(sig - y)

    def compute_cost(self, x_train, y_train, x_test, y_test):
        self.cost.append(
            compute_cost(self.weights, self.bias, x_train, y_train)
        )

        self.cost_test.append(
            compute_cost(self.weights, self.bias, x_test, y_test)
        )


def predict(x, houses):
    _n_samples = len(x)
    _n_classes = len(houses)

    scores = np.zeros((_n_samples, _n_classes))

    house_names = list(houses.keys())

    for i, (house_name, house) in enumerate(houses.items()):
        scores[:, i] = lu.sigmoid(np.dot(x, house.weights) + house.bias)

    predictions_indices = np.argmax(scores, axis=1)
    predictions = [house_names[idx] for idx in predictions_indices]

    return predictions


def get_accuracy(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    total = y_true.shape[0]
    return correct / total


class Dataset:
    def __init__(self, x, y):
        _x_len = len(x)
        _split_offset = ceil(_x_len * 0.70)

        self.train_len = _split_offset

        print(f"Total samples : {_x_len}")
        print(f"- Kept for training : {_split_offset}")
        print(f"- Kept for testing  : {_x_len - _split_offset}")

        _permutation = np.random.permutation(_x_len)
        _x = x[_permutation]
        _y = y[_permutation]

        self.train_x = np.array(_x[:_split_offset])
        self.train_y = np.array(_y[:_split_offset])

        self.test_x = np.array(_x[_split_offset:])
        self.test_y = np.array(_y[_split_offset:])

    def shuffle_train(self):
        _permutation = np.random.permutation(self.train_len)
        self.train_x = self.train_x[_permutation]
        self.train_y = self.train_y[_permutation]


def main():
    dataset_file = open_arg_csv()
    # np.random.seed(42)

    # 1                  : Stochastic
    # sample_count       : Batch
    # > 1 < sample_count : Mini-Batch
    batch_size = 32
    inv_sample = 1.0 / batch_size

    _learning_rate = 0.01
    _iterations = 1500

    _tmp_house = dict.fromkeys(set(d for d in dataset_file.iloc[:, 1]))
    houses = {k: _tmp_house[k] for k in sorted(_tmp_house.keys(), reverse=False)}

    excluded_features = [
        "Arithmancy",
        "Care of Magical Creatures",
        # "Defense Against the Dark Arts",
        "Potions",
        # "History of Magic",
        # "Muggle Studies",
        # "Transfiguration",
        # "Herbology",
        # "Astronomy"
    ]
    dataset_file = dataset_file.drop(excluded_features, axis=1)
    val_x = prepare_x(dataset_file)

    exported_values = {
        "normalization": [],
        "weights": _tmp_house.copy(),
        "biases": _tmp_house.copy(),
        "excluded_features": excluded_features,
    }

    # Standardisation des valeurs
    x_shape = val_x.shape
    for i in range(x_shape[1]):
        column_data = val_x[:, i].astype(float)

        exported_values["normalization"].append({
            "mean": float(column_data.mean()),
            "std": float(column_data.std())
        })
        val_x[:, i] = (column_data - column_data.mean()) / column_data.std()

    val_y = dataset_file["Hogwarts House"]
    samples_count, features_count = val_x.shape
    dataset = Dataset(val_x, val_y)

    # Initialisation des données pour le train
    for house in houses:
        houses[house] = House(
            features_count=features_count,
            learning_rate=_learning_rate
        )

    # Courbes
    test_accuracy_over_epochs = []
    train_accuracy_over_epochs = []

    for epoch in range(_iterations):
        if epoch % 2 == 0:
            dataset.shuffle_train()

        x = dataset.train_x[:batch_size]
        y = dataset.train_y[:batch_size]

        for house_name in houses:
            _house = houses[house_name]
            _house.gradient_descent(
                x=x,
                y=np.where(house_name == y, 1, 0),
                m=inv_sample
            )

            _house.compute_cost(
                x_train=dataset.train_x,
                y_train=np.where(house_name == dataset.train_y, 1, 0),
                x_test=dataset.test_x,
                y_test=np.where(house_name == dataset.test_y, 1, 0)
            )

        # if epoch == 0 or epoch % 100 == 0 or epoch >= _iterations - 1:
        test_prediction = predict(dataset.test_x, houses)
        test_accuracy = get_accuracy(dataset.test_y, test_prediction)

        train_prediction = predict(dataset.train_x, houses)
        train_accuracy = get_accuracy(dataset.train_y, train_prediction)

        test_accuracy_over_epochs.append(test_accuracy)
        train_accuracy_over_epochs.append(train_accuracy)
        print(
            f"Accuracy    : Epoch {epoch:6}/{_iterations} - Train : {train_accuracy * 100:.3f}% - Test : {test_accuracy * 100:.3f}%")

    fig, ax = plt.subplots(len(houses) // 2, len(houses) // 2, figsize=(10, 10))
    fig.tight_layout()

    for i, house in enumerate(houses):
        _x = i // 2
        _y = i % 2
        ax[_x, _y].plot(houses[house].cost, label="Train")
        ax[_x, _y].plot(houses[house].cost_test, label="Test")
        ax[_x, _y].set_title('Cost over epochs')
        ax[_x, _y].legend()
        # ax[_x, _y].plot(loss_per_house[house])

    plt.show()

    plt.plot(train_accuracy_over_epochs, label="Train Accuracy")
    plt.plot(test_accuracy_over_epochs, label="Test Accuracy")
    plt.legend()
    plt.show()

    for house_name in houses:
        _house = houses[house_name]
        print(f"Saving weights for {house_name}")
        exported_values["weights"][house_name] = [w for w in _house.weights]
        exported_values["biases"][house_name] = float(_house.bias)

    with open('model.json', 'w') as file:
        import json
        json.dump(exported_values, file, indent=4)

    print("Train done !")


if __name__ == "__main__":
    main()
