import argparse

import numpy as np
import matplotlib.pyplot as plt

from activation_functions import ACTIVATION_FN, ACTIVATION_DERIVATIVE_FN
from typing import List
from dataset_handle import load_dataset, normalize_dataset
from utils import binary_cross_entropy, compute_accuracy
from initialization_functions import INITIALIZATION_FN

# Notes bonus :
# - Accuracy affichée pendant le training
# - Graphe accuracy train/test
# - Changement du mode (Batch, Mini-GD, Stochastic)
# - Early stopping
# - Ajout de deux méthodes d'initialisation en plus (xavier_uniform_initialization, lecun_uniform_initialization)
# - Ajout de deux méthodes d'activations (tanh, reLU, tanh)

################################
# RAPPELS
################################
#
# GRADIENT DESCENT :
#    Méthode d'optimisation pour ajuster les poids afin de
#    minimiser l'erreur entre les valeurs prédites et réelle
#
# FEEDFORWARD :
#    Calcul de la somme pondérée des poids de la couche actuelle
#    et des sorties de la couche précédente.
#
# BACKPROPAGATION (rétro-propagation) :
#    Parcour des couches du réseau de neurones à partir de la fin
#    pour calculer les gradients afin de mettre à jour les poids
#    du modèle
#
# HYPERPARAMÈTRES :
#    Paramètres qui ne sont pas appris et adapter pendant le training
#    mais qui impactent les performances du modèles.
#      - Learning rate           : Taille des pas lors de la mise à jour des poids
#      - Epochs                  : Nombre d'itération de l'algorithme
#      - Taille du batch         : Nombre de données utilisée simultanément par epoch
#      - Nombre de couche/taille : Structure du réseau de neurones
#      - Fonction d'activation   : Fonction non linéaire utilisée pour transformer la sortie d'un neurone,
#                                  affecte la convergence/performance du modèle

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
    def __init__(self, X=None, y=None, x_validation=None, y_validation=None):
        self._layers: List[Layer] = []
        self._weights = []
        self._biases = []
        self._activations = []

        self.learning_rate: float = 0.01
        self.epochs: int = 3000

        self.x: [] = X if X is not None else []
        self.y: [] = y if y is not None else []
        self.x_test: [] = x_validation if x_validation is not None else []
        self.y_test: [] = y_validation if y_validation is not None else []

        self.batch_size = 32
        self.early_stopping = False

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


        _patience_counter = 0
        _best_loss = float('inf')

        for epoch in range(self.epochs):
            _permutation = np.random.permutation(_m)
            self.x = self.x[_permutation]
            self.y = self.y[_permutation]

            _activations = self.forward_propagation(self.x[self.batch_size:])
            self.back_propagation(y_true=self.y[self.batch_size:], activations=_activations)

            _test_activations = self.forward_propagation(self.x_test)
            _test_loss = binary_cross_entropy(self.y_test, _test_activations[-1])

            # print(_best_loss - _test_loss)

            if _best_loss - _test_loss > 0.0000001:
                _best_loss = _test_loss
                _patience_counter = 0
            else:
                _patience_counter += 1

            if _patience_counter >= 30 and self.early_stopping == True:
                print("EARLY STOPPING")
                break

            # exit(1)
            if epoch % 2 == 0 or epoch >= self.epochs:
                _train_activations = self.forward_propagation(self.x)
                _train_loss = binary_cross_entropy(self.y, _train_activations[-1])

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
        deltas = [ activations[-1] - y_true ]

        for w in reversed(range(len(self._weights))):
            layer_id = w + 1
            layer = self._layers[layer_id]

            delta = deltas[-1]
            curr_activation = activations[w]

            dw = m * np.dot(curr_activation.T, delta)
            db = m * np.sum(delta, axis=0, keepdims=True)

            derivative_value = layer.derivative_activation_fn(curr_activation) \
                if layer.derivative_activation_fn is not None else 1.0

            deltas.append(np.dot(delta, self._weights[w].T) * derivative_value)

            self._weights[w] -= (self.learning_rate * dw)
            self._biases[w] -= (self.learning_rate * db)

    def export_topology(self, file_path, std_values):
        exported_data = {
            "layers": [layer.to_dict() for layer in self._layers],
            "weights": [w.tolist() for w in self._weights],
            "biases": [b.tolist() for b in self._biases],
            "std_values": std_values
        }
        with open(file_path, 'w') as file:
            import json
            json.dump(exported_data, file, indent=4)

    def import_topology_and_dataset_for_predict(self, base_path, topology_path):
        self._layers = []
        self._weights = []
        self._biases = []

        _std_values = []
        with open(topology_path, "r") as file:
            import json
            _file_str = file.read()
            _data = json.loads(_file_str)

            for layer in _data['layers']:
                self.add_layer(layer['size'], layer['activation'], layer['initializer'])

            for weight in _data['weights']:
                self._weights.append(weight)

            for bias in _data['biases']:
                self._biases.append(bias)

            for std_value in _data['std_values']:
                _std_values.append(std_value)

        self.x_test, self.y_test = (
            load_dataset(f"{base_path}/../resources/data_test.csv")
        )

        self.x_test, _ = normalize_dataset(self.x_test, std_data=_std_values)
