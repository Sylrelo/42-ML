#!/bin/bash
python3 ./src/logreg_train.py datasets/dataset_train.csv
python3 ./src/logreg_predict.py datasets/dataset_test.csv ./weights.csv
python3 ./evaluate.py
