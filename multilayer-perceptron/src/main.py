import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from activation_functions import sigmoid, sigmoid_derivative, relu, relu_derivative, softmax, ACTIVATION_FN, \
    ACTIVATION_DERIVATIVE_FN
from typing import List
from dataset_handle import load_dataset, normalize_dataset, convert_to_one_hot, LoadDataset
from utils import binary_cross_entropy, compute_accuracy
from initialization_functions import INITIALIZATION_FN


################################
# RAPPELS
################################
# Poids (weights) détermine l'influence sur le résultat de sortie
# Biais : Ajustement en entrée pour aider la prédiction
#
# Backpropagation (rétro-propagation) : Ajuste les poids et biais
#    pour réduire l'erreur en fonction de la prédiction du réseau
#    et des valeurs réelles
#
#

class Layer:
    def __init__(self, size, activation=None, initializer=None):
        self.size = size

        self._initializer = initializer
        self._activation = activation

        self.initializer_fn = None
        self.activation_fn = None
        self.derivative_activation_fn = None

        if initializer is None:
            self.initializer_fn = INITIALIZATION_FN['random']
        elif initializer is not None and initializer in INITIALIZATION_FN:
            self.initializer_fn = INITIALIZATION_FN[initializer]
        else:
            print(f"Initialization function {initializer} not found.")
            exit(1)

        if activation is not None and activation in ACTIVATION_FN:
            self.activation_fn = ACTIVATION_FN[activation]
            self.derivative_activation_fn = ACTIVATION_DERIVATIVE_FN[activation]
        elif activation is not None:
            print(f"Activation function {activation} not found.")
            exit(1)

    def __str__(self):
        return "Layer size : {0} - Activation {1} - Initialization {2}".format(self.size, self._activation,
                                                                               self._initializer)
    def to_dict(self):
        return {
            "size": self.size,
            "activation": self._activation,
            "initializer": self._initializer,
        }

class mlp:
    def __init__(self, X, y, x_validation, y_validation):
        self._layers: List[Layer] = []
        self._weights = []
        self._biases = []
        self._activations = []

        self.learning_rate: float = 0.025
        self.epochs: int = 15000

        self.x: [] = X if X is not None else []
        self.y: [] = y if y is not None else []
        self.x_test: [] = x_validation
        self.y_test: [] = y_validation

        self.batch_size = 32

        # self.add_layer(np.shape(X)[1])
        # self.add_layer(size=32, activation="sigmoid", initializer="xavierUniform")
        # self.add_layer(size=32, activation="sigmoid", initializer="xavierUniform")
        # self.add_layer(size=32, activation="sigmoid", initializer="xavierUniform")
        # self.add_layer(size=2, activation="softmax")

        # self.init_weights_and_biases()

    # Initialisation des poids et biais pour les couches cachées
    def init_weights_and_biases(self):
        for i in range(len(self._layers) - 1):
            layer = self._layers[i]

            self._weights.append(
                layer.initializer_fn(self._layers[i].size, self._layers[i + 1].size)
            )
            self._biases.append(
                np.zeros((1, self._layers[i + 1].size))
            )

    def add_layer(self, size, activation=None, initializer=None):
        self._layers.append(Layer(size, activation, initializer))

    def train(self):
        overtime_loss_train = []
        overtime_loss_test = []
        overtime_precision_train = []
        overtime_precision_test = []

        _m = len(self.y)

        for epoch in range(self.epochs):

            _permutation = np.random.permutation(_m)
            _minibatch_count = _m // self.batch_size

            self.x = self.x[_permutation]
            self.y = self.y[_permutation]

            for batch in range(_minibatch_count + 1):
                _offset_start = self.batch_size * batch
                _offset_end = min(_offset_start + self.batch_size, _m)

                _activations = self.forward_propagation(self.x[_offset_start:_offset_end])
                self.back_propagation(y_true=self.y[_offset_start:_offset_end], activations=_activations)

            # exit(1)
            if epoch % (self.epochs // 100) == 0 or epoch >= self.epochs:
                _train_activations = self.forward_propagation(self.x)
                _train_loss = binary_cross_entropy(self.y, _train_activations[-1])

                _test_activations = self.forward_propagation(self.x_test)
                _test_loss = binary_cross_entropy(self.y_test, _test_activations[-1])

                _test_accuracy = compute_accuracy(_test_activations[-1], self.y_test)
                _train_accuracy = compute_accuracy(_train_activations[-1], self.y)

                overtime_loss_train.append(_train_loss)
                overtime_loss_test.append(_test_loss)
                overtime_precision_train.append(_train_accuracy)
                overtime_precision_test.append(_test_accuracy)

                print(
                    f"\033[37;1mEpoch {epoch:6} / {self.epochs}\033[0m "
                    f"- Loss [{_train_loss:<5.3f}  {_test_loss:>5.3f}] "
                    f"- Accuracy [{_train_accuracy:<5.3f}  {_test_accuracy:>5.3f}]"
                )

        # plt.plot(overtime_loss_train, label="Training Loss")
        # plt.plot(overtime_loss_test, label="Test loss")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        ax1.plot(overtime_loss_train, label='Train dataset')
        ax1.plot(overtime_loss_test, label='Test dataset')
        ax1.set_title('Loss over epochs')
        ax1.legend()

        ax2.plot(overtime_precision_train, label='Train dataset')
        ax2.plot(overtime_precision_test, label='Test dataset')
        ax2.set_title('Accuracy over epochs')
        ax2.legend()

        plt.show()

        print(len(overtime_precision_test))

    def predict(self, x):
        y_pred = self.forward_propagation(x)
        return y_pred[-1]

    #############################################################################
    # PASSE-AVANT (FORWARD PROPAGATION)
    #############################################################################
    # Calcul de la somme pondérée des poids de la couche actuelle
    #   et des sorties de la couche précédente.
    #############################################################################
    def forward_propagation(self, x):
        _activations = [x]

        for layer_id in range(1, len(self._layers)):
            layer = self._layers[layer_id]
            weights = self._weights[layer_id - 1]
            biases = self._biases[layer_id - 1]

            z = np.dot(_activations[-1], weights) + biases
            a = layer.activation_fn(z)

            _activations.append(a)

        return _activations

    ########################################
    # RÉTRO-PROPAGATION (BACK PROPAGATION)
    ########################################
    # Ajustement des poids pour minimiser l'erreur entre les prédictions et les résultats réel.
    def back_propagation(self, y_true, activations):
        m = 1 / np.shape(y_true)[0]

        deltas = [np.zeros_like(a) for a in self._layers]
        deltas[-1] = activations[-1] - y_true

        for l in range(len(self._layers) - 2, -1, -1):
            layer = self._layers[l]

            derivative_value = layer.derivative_activation_fn(activations[l]) \
                if layer.derivative_activation_fn is not None else 1.0

            deltas[l] = np.dot(deltas[l + 1], self._weights[l].T) * derivative_value

            gw = m * np.dot(activations[l].T, deltas[l + 1])
            gb = m * np.sum(deltas[l + 1], axis=0, keepdims=True)

            self._weights[l] = self._weights[l] - (self.learning_rate * gw)
            self._biases[l] = self._biases[l] - (self.learning_rate * gb)

    def export_topology(self, file_path):
        exported_data = {
            "layers": [layer.to_dict() for layer in self._layers],
            "weights": [w.tolist() for w in self._weights],
            "biases": [b.tolist() for b in self._biases]
        }
        with open(file_path, 'w') as file:
            import json
            json.dump(exported_data, file, indent=4)

    def import_topology(self, file_path):
        self._layers = []
        self._weights = []
        self._biases = []

        with open(file_path, "r") as file:
            import json
            _file_str = file.read()
            _data = json.loads(_file_str)

            for layer in _data['layers']:
                self.add_layer(layer['size'], layer['activation'], layer['initializer'])

            for weight in _data['weights']:
                self._weights.append(weight)

            for bias in _data['biases']:
                self._biases.append(bias)

def main():
    import os
    current_file_path = os.path.dirname(os.path.realpath(__file__))
    _TOPOLOGY_PATH = os.path.join(current_file_path, "topology.json")


    parser = argparse.ArgumentParser(description="")

    parser.add_argument('action', type=str, choices=['train', 'predict', 'split'])

    parser.add_argument('--lr', type=float, default=0.025)
    parser.add_argument('--epochs', type=int, default=15000)
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--layers_size', type=str, default=None)
    parser.add_argument('--layers_activation', type=str, default=None)
    parser.add_argument('--layers_init', type=str, default=None)

    args = parser.parse_args()

    if args.action == 'train':
        dataset = LoadDataset(base_path=current_file_path)

        _mlp = mlp(
            X=dataset.training_features,
            y=convert_to_one_hot(dataset.training_classes),

            x_validation=dataset.test_features,
            y_validation=convert_to_one_hot(dataset.test_classes),
        )

        _mlp.epochs = args.epochs
        _mlp.learning_rate = args.lr
        _mlp.batch_size = args.batch_size

        if (args.layers_activation is not None or args.layers_init is not None) and args.layers_size is None:
            print("Invalid configuration. You need to specify layers size.")
            exit(1)

        _mlp.add_layer(np.shape(dataset.training_features)[1])

        if args.layers_size is not None:
            _sizes = args.layers_size.split(',')
            _activations = args.layers_activation.split(',') if args.layers_activation is not None else []
            _initializers = args.layers_init.split(',') if args.layers_init is not None else []

            for l in range(len(_sizes)):
                _current_activation = "sigmoid" if l >= len(_activations) else _activations[l]
                _current_initializer = "xavierUniform" if l >= len(_initializers) else _initializers[l]
                _mlp.add_layer(size=int(_sizes[l]), activation=_current_activation, initializer=_current_initializer)

        else:
            _mlp.add_layer(size=32, activation="sigmoid", initializer="xavierUniform")
            _mlp.add_layer(size=32, activation="sigmoid", initializer="xavierUniform")
            _mlp.add_layer(size=32, activation="sigmoid", initializer="xavierUniform")

        _mlp.add_layer(size=2, activation="softmax")

        _mlp.init_weights_and_biases()
        _mlp.train()
        _mlp.export_topology(_TOPOLOGY_PATH)

    elif args.action == 'predict':
        dataset = LoadDataset(base_path=current_file_path)

        _mlp = mlp(
            # X=dataset.training_features,
            # y=dataset.training_classes,

            x_validation=dataset.test_features,
            y_validation=convert_to_one_hot(dataset.test_classes),
        )

        _mlp.import_topology(_TOPOLOGY_PATH)


    elif args.action == 'split':
        None

    #
    # exit(1)
    # training_features, training_classes = (
    #     load_dataset(f"{current_file_path}/../resources/data_training.csv")
    # )
    #
    # test_features, test_classes = (
    #     load_dataset(f"{current_file_path}/../resources/data_test.csv")
    # )
    #
    # training_features, _, _ = normalize_dataset(training_features)
    # test_features, _, _ = normalize_dataset(test_features)
    #
    # yeet = mlp(
    #     X=training_features,
    #     y=convert_to_one_hot(training_classes),
    #     x_validation=test_features,
    #     y_validation=convert_to_one_hot(test_classes)
    # )
    #
    # yeet.epochs = 100
    #
    # # yeet.train()
    #
    # # yeet.export_topology(file_path=current_file_path + "/../topology.json")
    #
    # # yeet.import_topology(file_path=current_file_path + "/../topology.json")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user")
        exit(0)
    except Exception as e:
        print("Error", e)
        exit(1)
