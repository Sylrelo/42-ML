import argparse

import numpy as np

from dataset_handle import LoadDataset, convert_to_one_hot, split_dataset
from mlp import mlp
from utils import compute_accuracy, binary_cross_entropy


def main():
    import os
    current_file_path = os.path.dirname(os.path.realpath(__file__))
    _TOPOLOGY_PATH = os.path.join(current_file_path, "topology.json")

    parser = argparse.ArgumentParser(description="")

    parser.add_argument('action', type=str, choices=['train', 'predict', 'split'])

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=7000)
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--layers_size', type=str, default=None)
    parser.add_argument('--layers_activation', type=str, default=None)
    parser.add_argument('--layers_init', type=str, default=None)

    parser.add_argument('--cut',  type=float, default=0.8)

    parser.add_argument("--early_stopping", type=bool, default=False)

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
        _mlp.early_stopping = args.early_stopping

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
            _mlp.add_layer(size=32, activation="sigmoid", initializer="heUniform")
            _mlp.add_layer(size=32, activation="tanh", initializer="heUniform")
            _mlp.add_layer(size=32, activation="tanh", initializer="heUniform")

        _mlp.add_layer(size=2, activation="softmax")

        _mlp.init_weights_and_biases()
        _mlp.train()
        _mlp.export_topology(file_path=_TOPOLOGY_PATH, std_values=dataset.std_values)

    elif args.action == 'predict':
        _mlp = mlp()
        _mlp.import_topology_and_dataset_for_predict(
            base_path=current_file_path,
            topology_path=_TOPOLOGY_PATH,
        )
        predicted = _mlp.predict(_mlp.x_test)

        y_onehot = convert_to_one_hot(_mlp.y_test)

        loss = binary_cross_entropy(y_onehot, predicted)
        accuracy = compute_accuracy(predicted, y_onehot)

        print(f"Loss: {loss}, Accuracy: {accuracy}")

    elif args.action == 'split':
        split_dataset(base_path=current_file_path, cut_percent=args.cut)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user")
        exit(0)
    except Exception as e:
        print("Error", e)
        exit(1)
