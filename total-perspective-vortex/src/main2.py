from time import sleep
from typing import  Tuple

import mne
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit,train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from custom_csp import CustomCSP
from data_processing import load_and_process
from csp_transformer import CSPTransformer
from wavelet_transformer import WaveletTransformer

mne.set_log_level('WARNING')

RANGE_SUBJECT = range(10, 14)

def _train(X: np.ndarray, y: np.ndarray) -> Pipeline:
    _best_pipeline = None
    _best_score = None

    cv = ShuffleSplit(2, test_size=0.2, random_state=42)

    rfc = RandomForestClassifier(n_estimators=100, random_state=42)

    csp_transformer = CSPTransformer(n_components=6)
    wavelet_transformer = WaveletTransformer()
    pipeline_rfc = Pipeline([
        ('wavelet', wavelet_transformer),
        ('csp', csp_transformer),
        ('scaler', None),
        ('classifier', rfc)
    ])

    param_grid = {
        'classifier__max_depth': [75, 100],
        'classifier__max_features': [3],
        'classifier__min_samples_leaf': [10, 14],
        'classifier__min_samples_split': [8, 12],
        'classifier__n_estimators': [60, 75, 100],
        'csp__n_components': [6, 10],
        'scaler': [StandardScaler(), RobustScaler(), None],
    }

    param_grid = {
        'classifier__max_depth': [75],
        'classifier__max_features': [3],
        'classifier__min_samples_leaf': [14],
        'classifier__min_samples_split': [8],
        'classifier__n_estimators': [60],
        'csp__n_components': [10],
        'scaler': [StandardScaler()],
    }

    grid_search_rfc = GridSearchCV(
        estimator=pipeline_rfc,
        param_grid=param_grid,
        cv=cv,
        n_jobs=6,
        verbose=2,
        scoring='accuracy',
        error_score='raise',
        return_train_score=True,
    )

    grid_search_rfc.fit(X, y)

    print(f"  Cross-validation Score: {grid_search_rfc.best_score_}")
    print(grid_search_rfc.best_params_)

    sleep(1)
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


def _get_train_data_some_subjects_aled_nom_fonction(experiment=None, run=None) -> Tuple[np.ndarray, np.ndarray]:
    if experiment is None:
        _runs = 1

    _SUBJECTS = range(1, 110)
    all_epochs = []
    all_labels = []

    for subject in _SUBJECTS:
        X, y = load_and_process(subject=subject, experiment=experiment, run=run)
        print(f"Subject {subject} loaded. ", np.shape(X), np.shape(y))
        all_epochs.append(X)
        all_labels.append(y)

    X = np.concatenate(all_epochs, axis=0)
    y = np.concatenate(all_labels, axis=0)

    return X, y


def train_only_one_run(run=None):
    (X, y) = _get_train_data_some_subjects_aled_nom_fonction(run=run)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.4, random_state=42)
    pipeline = _train(train_X, train_y)

    predicted_y = pipeline.predict(test_X)
    score = np.mean(predicted_y == test_y)

    print(f"  Score on test dataset: {score}")

    _total_subhect = 0
    _total_score = 0
    for subject in range(1, 110):
        (X, y) = load_and_process(subject, run=run)
        _, test_X, _, test_y = train_test_split(X, y, test_size=0.4, random_state=42)

        predicted_y = pipeline.predict(test_X)
        score = np.mean(predicted_y == test_y)
        _total_score += score
        _total_subhect += 1

    print(f"TOT : {_total_score / _total_subhect}")

if __name__ == '__main__':
    # EXPERIMENT_RANGE = range(1, 7)
    EXPERIMENT_RANGE = range(1, 7)
    _pipelines = {}

    # train_only_one_run(3)
    # exit(1)

    for experiment in EXPERIMENT_RANGE:
        print(f"Training for experiment {experiment}")
        (X, y) = _get_train_data_some_subjects_aled_nom_fonction(experiment=experiment)
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.4, random_state=42)
        pipeline = _train(train_X, train_y)

        predicted_y = pipeline.predict(test_X)
        score = np.mean(predicted_y == test_y)

        _pipelines[experiment] = pipeline

        print(f"  Score on test dataset: {score}")

    _total_experiment_score = 0
    _total_experiments = 0
    total_score = 0
    _per_subject_score = {}

    for experiment in EXPERIMENT_RANGE:
        _experiment_subject_score = 0
        _total_subjects = 0

        for subject in range(1, 110):
            (X, y) = load_and_process(subject, experiment=experiment)
            _, test_X, _, test_y = train_test_split(X, y, test_size=0.4, random_state=42)

            predicted_y = _pipelines[experiment].predict(test_X)
            score = np.mean(predicted_y == test_y)

            _experiment_subject_score += score
            _total_subjects += 1

            if subject not in _per_subject_score:
                _per_subject_score[subject] = 0
            _per_subject_score[subject] += score

        _total_experiments += 1
        _total_experiment_score += _experiment_subject_score / _total_subjects

        print(f"Experiment {experiment} accuracy: {(_experiment_subject_score / _total_subjects) * 100}%")

    for subid in _per_subject_score:
        _per_subject_score[subid] /= _total_experiments
        print(f"Subject {subid} total accuracy: {_per_subject_score[subid] * 100}%")

    print(f"Total accuracy: {(_total_experiment_score / _total_experiments) * 100}%")
    #
    # total = 0
    # total_subjects = 0
    #
    #
    #     print(f"Total score: {total / total_subjects}")
    #
    # print(f"Total score: {total / total_subjects}")
