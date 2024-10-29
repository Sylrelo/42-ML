from concurrent.futures import ProcessPoolExecutor
from time import sleep
from typing import List, Tuple

import joblib
from matplotlib import pyplot as plt
import mne
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit, cross_val_score, train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import RobustScaler, MinMaxScaler, MaxAbsScaler

from custom_csp import CustomCSP
from data_processing import filter_data, get_events, load_and_process, prepare_data
from utils import load_eegbci_data

mne.set_log_level('WARNING')

RANGE_SUBJECT = range(10, 14)


def _train(X: np.ndarray, y: np.ndarray, calculate_xval=False) -> Pipeline:
    _best_pipeline = None
    _best_score = None

    cv = ShuffleSplit(4, test_size=0.2, random_state=42)

    transformer = CustomCSP(n_components=6)

    # lda = LinearDiscriminantAnalysis(solver='svd')
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    pipeline_rfc = Pipeline([('transformer', transformer), ('scaler', None), ('classifier', rfc)])

    # param_grid = {
    #     'classifier__max_depth': [75, 100, 120],
    #     'classifier__max_features': [3, 5, 8],
    #     'classifier__min_samples_leaf': [10, 12, 14],
    #     'classifier__min_samples_split': [8, 10, 12],
    #     'classifier__n_estimators': [90, 100, 110],
    #     'transformer__n_components': [6, 10],
    #     'scaler': [StandardScaler(), RobustScaler(), None],
    # }
    param_grid = {
        'classifier__max_depth': [100],
        'classifier__max_features': [3],
        'classifier__min_samples_leaf': [10],
        'classifier__min_samples_split': [8],
        'classifier__n_estimators': [75, 100, 110],
        'transformer__n_components': [6, 10],
        'scaler': [StandardScaler(), RobustScaler(), None],
    }
    grid_search_rfc = GridSearchCV(
        estimator=pipeline_rfc,
        param_grid=param_grid,
        cv=2,
        n_jobs=6,
        verbose=2,
        scoring='accuracy',
        error_score='raise',
        return_train_score=True,
    )

    grid_search_rfc.fit(X, y)

    transformer = CustomCSP(n_components=6)
    print(f"Best score (Cross-Validation): {grid_search_rfc.best_score_}")
    print(grid_search_rfc.best_params_)

    # lda = LinearDiscriminantAnalysis(solver='svd', )
    # transformer = CustomCSP(n_components=6)
    # pipeline_lda = Pipeline([('transformer', transformer), ('scaler', None), ('classifier', lda)])
    # param_grid_lda = {
    #     'classifier__solver': ['svd', 'lsqr', 'eigen'],
    #     'classifier__shrinkage': ['auto', None],
    #     'transformer__n_components': [2, 4, 6],
    #     'scaler': [StandardScaler(), RobustScaler(), None],
    # }
    # grid_search_lda = GridSearchCV(
    #     estimator=pipeline_lda,
    #     param_grid=param_grid_lda,
    #     cv=2,
    #     n_jobs=10,
    #     verbose=2,
    #     scoring='accuracy',
    #     error_score=np.nan,
    #     return_train_score=True,
    # )
    #
    # grid_search_lda.fit(X, y)
    #
    # print(grid_search_rfc.best_params_)
    # print(grid_search_lda.best_params_)
    sleep(2)
    #
    # if grid_search_lda.best_score_ > grid_search_rfc.best_score_:
    #     print("Using LDA")
    #     return grid_search_lda.best_estimator_
    # else:
    #     print("Using RFC")
    return grid_search_rfc.best_estimator_

    # _pipeline = grid_search_rfc.best_estimator_.fit(X, y)

    # return _pipeline
    # return _pipeline

    # n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # # Number of features to consider at every split
    # max_features = ['auto', 'sqrt']
    # # Maximum number of levels in tree
    # max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    # max_depth.append(None)
    # # Minimum number of samples required to split a node
    # min_samples_split = [2, 5, 10]
    # # Minimum number of samples required at each leaf node
    # min_samples_leaf = [1, 2, 4]
    # # Method of selecting samples for training each tree
    # bootstrap = [True, False]  # Create the random grid
    # random_grid = {'n_estimators': n_estimators,
    #                'max_features': max_features,
    #                'max_depth': max_depth,
    #                'min_samples_split': min_samples_split,
    #                'min_samples_leaf': min_samples_leaf,
    #                'bootstrap': bootstrap}
    #
    # rf_random = RandomizedSearchCV(
    #     estimator=rfc,
    #     param_distributions=random_grid,
    #     n_iter=100,
    #     cv=3,
    #     verbose=2,
    #     random_state=42,
    #     n_jobs=-1
    # )
    # rf_random.fit(X, y)

    # rf_random.best_estimator_

    # if calculate_xval is True:
    #     # score_lda = cross_val_score(pipeline_lda, X=X, y=y, cv=cv, verbose=False)
    #     # score_logreg = cross_val_score(pipeline_logreg, X=X, y=y, cv=cv, verbose=False)
    #     score_rfc = cross_val_score(pipeline_rfc, X=X, y=y, cv=cv, verbose=False)
    #
    #     # print("score_lda ", score_lda.mean())
    #     # print("score_logreg ", score_logreg.mean())
    #     print("score_rfc ", score_rfc.mean())
    #
    # _pipeline = pipeline_rfc.fit(X, y)
    #
    # return _pipeline


def _get_train_data_some_subjects_aled_nom_fonction(experiment=None) -> Tuple[np.ndarray, np.ndarray]:
    if experiment is None:
        _runs = 1

    _SUBJECTS = range(1, 110)
    # _SUBJECTS = [1, 23, 12, 19, 34, 55, 64, 98, 101, 44]
    all_epochs = []
    all_labels = []


    # try:
    #     with open(f"../all_model_xy.xy", "rb") as f:
    #         data = joblib.load(f)
    #
    #         return data[0], data[1]
    # except Exception as e:
    #     print(e)


    for subject in _SUBJECTS:
        # if subject % 2 == 0:
        #     continue
        # eegbci_raw = load_eegbci_data(subject=subject, runs=_runs)
        # filtered_raw = filter_data(eegbci_raw)
        # _, labels, epochs = get_events(filtered_raw)

        X, y = load_and_process(subject=subject, experiment=experiment)
        print(f"Subject {subject} loaded. ", np.shape(X), np.shape(y))

        all_epochs.append(X)
        all_labels.append(y)

    X = np.concatenate(all_epochs, axis=0)
    y = np.concatenate(all_labels, axis=0)

    # with open(f"../all_model_xy.xy", "wb") as f:
    #     joblib.dump((X, y), f)

    return X, y


if __name__ == '__main__':
    expiriment = 4

    (X, y) = _get_train_data_some_subjects_aled_nom_fonction(experiment=0)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.4, random_state=42)
    pipeline = _train(train_X, train_y, calculate_xval=True)

    predicted_y = pipeline.predict(test_X)
    score = np.mean(predicted_y == test_y)

    total = 0
    total_subjects = 0
    for subject in range(1, 110):
        (X, y) = load_and_process(subject, experiment=3)
        predicted_y = pipeline.predict(X)
        score = np.mean(predicted_y == y)
        print(f"Sub {subject}: {score}")

        total += score
        total_subjects += 1

        print(f"Total score: {total / total_subjects}")

    print(f"Total score: {total / total_subjects}")
