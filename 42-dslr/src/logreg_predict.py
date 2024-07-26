import sys
import pandas as pd
import logreg_utils as lu
from utils import prepare_x

def main():
    if len(sys.argv) != 3:
        print("Usage: logreg_predit.py [data file (.csv)] [model file (.json)]")
        sys.exit(0)

    # Parse model
    with open('model.json', 'r') as file:
        import json
        model = json.load(file)

    dataset_file = pd.read_csv(sys.argv[1])
    dataset_file = dataset_file.drop(model["excluded_features"], axis=1)

    val_x = prepare_x(dataset_file)

    # Standardise les données à prédire avec les même valeurs que le train
    x_shape = val_x.shape
    for i in range(x_shape[1]):
        column_data = val_x[:, i].astype(float)
        _mean = model["normalization"][i]["mean"]
        _std = model["normalization"][i]["std"]
        val_x[:, i] = (column_data - _mean) / _std

    # Prédiction
    predicted_house = []
    for x_row in val_x:
        pred = [0, sys.float_info.min]

        for house in model["weights"]:
            _weights = model["weights"][house]
            _bias = model["biases"][house]

            y_pre = lu.sigmoid(x_row.dot(_weights) + _bias)
            if y_pre > pred[1]:
                pred = [house, y_pre]
        predicted_house.append(pred[0])

    # Enregistrement de la prédiction
    pd.DataFrame(
            predicted_house,
            columns=["Hogwarts House"]
        ).to_csv("./houses.csv", index_label="Index")


if __name__ == "__main__":
    main()
