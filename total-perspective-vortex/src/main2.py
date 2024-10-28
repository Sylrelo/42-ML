from typing import List, Tuple
from matplotlib import pyplot as plt
import mne
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit, cross_val_score, train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from custom_csp import CustomCSP
from data_processing import filter_data, get_events, load_and_process, prepare_data
from utils import load_eegbci_data

mne.set_log_level('WARNING')

RANGE_SUBJECT = range(10, 14)


def _train(X: np.ndarray, y: np.ndarray, calculate_xval=False) -> Pipeline:
    _best_pipeline = None
    _best_score = None

    cv = ShuffleSplit(10, test_size=0.2, random_state=42)

    # transformer = mne.decoding.CSP(n_components=6)
    transformer = CustomCSP(n_components=6)

    # print(np.shape(X), np.shape(y))
    # exit(1)
    newX = transformer.fit_transform(X, y)

    lda = LinearDiscriminantAnalysis(solver='svd')
    log_reg = LogisticRegression(penalty='l1', solver='liblinear')
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)

    pipeline_lda = make_pipeline(transformer, lda)
    pipeline_logreg = make_pipeline(transformer, log_reg)
    pipeline_rfc = make_pipeline(transformer, rfc)

    param_grid = {
        'max_depth': [80, 90, 100, 110],
        'max_features': [2, 3],
        'min_samples_leaf': [2, 3, 4, 5],
        'min_samples_split': [2, 4, 8, 10],
        'n_estimators': [100, 200, 300, 1000]
    }

    grid_search = GridSearchCV(
        estimator=rfc,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=2,
        error_score='raise'
    )

    grid_search.fit(newX, y)

    print(grid_search.best_score_)
    # print(grid_search.best_params_)
    #
    # _pipeline = grid_search.best_estimator_.fit(newX, y)

    new_rfc = RandomForestClassifier(
        n_estimators=grid_search.best_params_['n_estimators'],
        max_depth=grid_search.best_params_['max_depth'],
        max_features=grid_search.best_params_['max_features'],
        min_samples_leaf=grid_search.best_params_['min_samples_leaf'],
        min_samples_split=grid_search.best_params_['min_samples_split'],
        random_state=42
    )

    new_rfc_pipeline = make_pipeline(transformer, new_rfc)

    new_rfc_pipeline.fit(X, y)
    return new_rfc_pipeline
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

    if calculate_xval is True:
        # score_lda = cross_val_score(pipeline_lda, X=X, y=y, cv=cv, verbose=False)
        # score_logreg = cross_val_score(pipeline_logreg, X=X, y=y, cv=cv, verbose=False)
        score_rfc = cross_val_score(pipeline_rfc, X=X, y=y, cv=cv, verbose=False)

        # print("score_lda ", score_lda.mean())
        # print("score_logreg ", score_logreg.mean())
        print("score_rfc ", score_rfc.mean())

    _pipeline = pipeline_rfc.fit(X, y)

    return _pipeline


def _get_train_data_some_subjects_aled_nom_fonction(runs=None) -> Tuple[np.ndarray, np.ndarray]:
    _runs = runs
    if runs is None:
        _runs = [3, 7, 11]

    _SUBJECTS = range(1, 109)
    # _SUBJECTS = [1, 23, 12, 19, 34, 55, 64, 98, 101, 44]
    all_epochs = []
    all_labels = []

    for subject in _SUBJECTS:
        # eegbci_raw = load_eegbci_data(subject=subject, runs=_runs)
        # filtered_raw = filter_data(eegbci_raw)
        # _, labels, epochs = get_events(filtered_raw)

        X, y = load_and_process(subject=subject, experiment=_runs)
        print(f"Subject {subject} loaded. ", np.shape(X), np.shape(y))

        all_epochs.append(X)
        all_labels.append(y)

    X = np.concatenate(all_epochs, axis=0)
    y = np.concatenate(all_labels, axis=0)

    return X, y


if __name__ == '__main__':
    # for subject in RANGE_SUBJECT:
    #     (X, y) = load_and_process(subject)
    #     train_X, text_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
    #     pipeline = _train(train_X, train_y)

    #     predicted_y = pipeline.predict(text_X)
    #     score = np.mean(predicted_y == test_y)
    #     print(f"====== End {subject}: {score}%")

    (X, y) = _get_train_data_some_subjects_aled_nom_fonction()
    # csp = mne.decoding.CSP()

    # print(np.shape(X), np.shape(y))
    # print(len(X), len(y))
    # X = csp.fit_transform(X, y)

    train_X, text_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline = _train(train_X, train_y, calculate_xval=True)

    predicted_y = pipeline.predict(text_X)
    score = np.mean(predicted_y == test_y)
    print(f"Train score : {score}")
    # for subject in RANGE_SUBJECT:

    # exit(1)
    # pipeline.predict()
    total = 0
    total_subjects = 0
    for subject in range(1, 110):
        (X, y) = load_and_process(subject)
        # _, test_X, _, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
        # #
        predicted_y = pipeline.predict(X)
        score = np.mean(predicted_y == y)
        print(f"Sub {subject}: {score}")

        total += score
        total_subjects += 1

        print(f"Total score: {total / total_subjects}")
    #

    print(f"Total score: {total / total_subjects}")
