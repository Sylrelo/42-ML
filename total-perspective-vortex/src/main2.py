import argparse
import random
import sys
from time import sleep
from typing import  Tuple

from matplotlib import pyplot as plt
import mne
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit, cross_val_score,train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from custom_csp import CustomCSP
from data_processing import load_and_process
from csp_transformer import CSPTransformer
import global_data
from utils import model_cache_get, model_cache_save
from wavelet_transformer import WaveletTransformer

mne.set_log_level('WARNING')

RANGE_SUBJECT = range(10, 14)

def _train(X: np.ndarray, y: np.ndarray) -> Pipeline:
    _best_pipeline = None
    _best_score = None

    cv = ShuffleSplit(2, test_size=0.2, random_state=global_data.RANDOM_STATE)
    rfc = RandomForestClassifier(
        n_estimators=75, 
        random_state=global_data.RANDOM_STATE,
        max_depth=75,
        max_features=3,
        min_samples_leaf=10,
        min_samples_split=8,
        )

    csp_transformer = CSPTransformer(n_components=6)
    wavelet_transformer = WaveletTransformer()
    csp = CustomCSP()
    
    pipeline_rfc = Pipeline([
        ('wavelet', wavelet_transformer),
        ('csp', csp_transformer),
        ('scaler', None),
        ('classifier', rfc)
    ])
    
    if global_data.DISABLE_WAVELET is True:
        pipeline_rfc = Pipeline([
            ('csp', csp),
            ('scaler', None),
            ('classifier', rfc)
        ])    
        
    if global_data.DISABLE_HYPERTUNNING is True:
        
        _crossval_score = cross_val_score(pipeline_rfc, cv=cv, X=X, y=y)
        print(f"   Cross-Validation score: {_crossval_score.mean()}")
        _pipeline = pipeline_rfc.fit(X, y)
        
        return _pipeline

    param_grid = {
        'classifier__max_depth': [75, 100],
        'classifier__max_features': [3],
        'classifier__min_samples_leaf': [10, 14],
        'classifier__min_samples_split': [8, 12],
        'classifier__n_estimators': [60, 75, 100],
        'csp__n_components': [6, 10],
        'scaler': [StandardScaler(), RobustScaler(), None],
    }

    grid_search_rfc = GridSearchCV(
        estimator=pipeline_rfc,
        param_grid=param_grid,
        cv=cv,
        n_jobs=4,
        verbose=2,
        scoring='accuracy',
        error_score=np.nan,
        return_train_score=True,
    )

    grid_search_rfc.fit(X, y)

    print(f"  Cross-validation Score: {grid_search_rfc.best_score_}")
    print(grid_search_rfc.best_params_)

    sleep(1)
    return grid_search_rfc.best_estimator_

def _get_train_data_some_subjects_aled_nom_fonction(subject=None, experiment=None, run=None) -> Tuple[np.ndarray, np.ndarray]:
    if experiment is None:
        _runs = 1

    _SUBJECTS = range(1, 110)
    if subject is not None:
        _SUBJECTS = [subject]

    all_epochs = []
    all_labels = []

    for subjectid in _SUBJECTS:
        X, y = load_and_process(subject=subjectid, experiment=experiment, run=run)
        print(f"Subject {subjectid} loaded. ", np.shape(X), np.shape(y))
        all_epochs.append(X)
        all_labels.append(y)

    X = np.concatenate(all_epochs, axis=0)
    y = np.concatenate(all_labels, axis=0)

    return X, y


# def train_only_one_run(run=None):
#     (X, y) = _get_train_data_some_subjects_aled_nom_fonction(run=run)
#     train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.4, random_state=42)
#     pipeline = _train(train_X, train_y)

#     predicted_y = pipeline.predict(test_X)
#     score = np.mean(predicted_y == test_y)

#     print(f"  Score on test dataset: {score}")

#     _total_subhect = 0
#     _total_score = 0
#     for subject in range(1, 110):
#         (X, y) = load_and_process(subject, run=run)
#         _, test_X, _, test_y = train_test_split(X, y, test_size=0.4, random_state=42)

#         predicted_y = pipeline.predict(test_X)
#         score = np.mean(predicted_y == test_y)
#         _total_score += score
#         _total_subhect += 1

#     print(f"TOT : {_total_score / _total_subhect}")


def _train_experiment_for_all_subjects():
    pass


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    
    args.add_argument("--train", action="store_true", help="Train" , default=False)
    args.add_argument("--predict", action="store_true", help="Predict", default=False)
    
    args.add_argument("--task", type=int, choices=range(3, 15), help="Task to run (incompatible with --experiment)")
    args.add_argument("--experiment", type=int, choices=[1, 2, 3, 4, 5, 6], help="Experiment to run (incompatible with --task)")
    
    args.add_argument("--subject", type=int, choices=range(1, 110), help="Subject to use")
    
    # args.add_argument("--force-train", action="store_true", default=False, help="Force the model to re-train everything (ignore cached model)")
    args.add_argument("--force-processing", action="store_true", default=False, help="Force the data processing pipeline process the data (ignore matrices cache)")
   
    args.add_argument("--disable-tunning", action="store_true", default=False, help="Disable hyper-parameters tunning (use default values)")
    args.add_argument("--disable-wavelet", action="store_true", default=False, help="Disable 'Discrete Wavelet Transform' pipeline step")
    
    args.add_argument("--show-analytics", action="store_true", default=False, help="Show analytic graphs")
    
    args.add_argument("--rnd", type=int, default=42, help="Random state")
    
    args.add_argument("--directory", type=str, help="Default directory for data")
    
    parsargs = args.parse_args()
    
    if parsargs.task and parsargs.experiment:
        print("\033[91m--experiment and --run are mutually exclusive.\x1b[0m\n")
        args.print_usage()
        exit(1)
    
    if parsargs.directory is not None:
        global_data.BASE_DIRECTORY = parsargs.directory
        global_data.DATA_DIRECTORY = f"{global_data.BASE_DIRECTORY}/_data"
        global_data.EEGBCI_DIRECTORY = f"{global_data.BASE_DIRECTORY}/eegbci"
    
    if parsargs.disable_tunning is True:
        global_data.DISABLE_HYPERTUNNING = True
        
    if parsargs.rnd != -1:
        global_data.RANDOM_STATE = parsargs.rnd
    elif parsargs.rnd == -1:
        global_data.RANDOM_STATE = random.randint(1, sys.maxsize)
        
    print(f"Random state: {global_data.RANDOM_STATE}")
    
    if parsargs.show_analytics is True:
        global_data.FORCE_DATA_PROCESSING = True
        
        print("=== ANALYTICS ===")
        _subject = parsargs.subject or 1
        _task = parsargs.task or 4
        _experiment = parsargs.experiment or 1
        
        _, _ = load_and_process(
            subject=_subject, 
            experiment=_experiment, 
            run=_task
        )
        
        plt.show()
        exit(0)
    
    if parsargs.train is True:
        print("=== TRAIN ===")
        _experiments_to_train = range(1, 7)
        
        if parsargs.experiment is not None:
            _experiments_to_train = [parsargs.experiment]
        elif parsargs.task is not None:
            _experiments_to_train = []
        else:
            print("Training all experiments.")
        
        if parsargs.subject is None:
            print("Training all subjects.")
            
        if len(_experiments_to_train) > 0:
            for experiment in _experiments_to_train:
                print(f"Training Experiment {experiment}...")
                x, y = _get_train_data_some_subjects_aled_nom_fonction(subject=parsargs.subject, experiment=experiment)
                train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=global_data.RANDOM_STATE)
                pipeline = _train(train_X, train_y)
                predicted_y = pipeline.predict(test_X)
                score = np.mean(predicted_y == test_y)
                print(f"   Test dataset: {score}")
                model_cache_save(pipeline=pipeline, subject=parsargs.subject, experiment=experiment)
        
        elif parsargs.task is not None:
            print(f"Training Task {parsargs.task}")
            x, y = _get_train_data_some_subjects_aled_nom_fonction(subject=parsargs.subject, experiment=None, run=parsargs.task)
            train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=global_data.RANDOM_STATE)
            pipeline = _train(train_X, train_y)
            predicted_y = pipeline.predict(test_X)
            score = np.mean(predicted_y == test_y)
            model_cache_save(pipeline=pipeline, subject=parsargs.subject, task=parsargs.task)
            print(f"   Test dataset: {score}")
        else:
            print("Invalid settings.")
            
    if parsargs.predict is True:
        print("=== PREDICT ===")
        _experiments_to_predict = range(1, 7)
        
        if parsargs.experiment is not None:
            _experiments_to_predict = [parsargs.experiment]
            print(f"Predicting experiment {parsargs.experiment}")
        elif parsargs.task is not None:
            _experiments_to_predict = []
            print(f"Predicting task {parsargs.task}")
        else:
            print("Predicting all experiments.")
        
        _subjects_to_predict = range(1, 110)
        if parsargs.subject is not None:
            _subjects_to_predict = [parsargs.subject]
            
        _subjects_scores = []

        if len(_experiments_to_predict) > 0:
            _per_experiments_scores = {}
            
            for subject in _subjects_to_predict:
                _experiment_scores = []
                
                for experiment in _experiments_to_predict:
                    _model = model_cache_get(subject=parsargs.subject, experiment=experiment)
                    if _model is None:
                        continue
                    x, y = _get_train_data_some_subjects_aled_nom_fonction(subject=subject, experiment=experiment)
                    predicted_y = _model.predict(x)
                    score = np.mean(predicted_y == y)
                    # print(f"Accuracy: {score}")
                    if experiment not in _per_experiments_scores:
                        _per_experiments_scores[experiment] = []
                    _per_experiments_scores[experiment].append(score)
                    
                    _experiment_scores.append(score)
                _subjects_scores.append(np.mean(_experiment_scores))
            for experiment in _per_experiments_scores:
                print(f"Accuracy for experiment {experiment}: {np.mean(_per_experiments_scores[experiment])}%")
            
        elif parsargs.task is not None:
            for subject in _subjects_to_predict:
                _model = model_cache_get(subject=parsargs.subject, experiment=experiment)
                if _model is None:
                    exit(1)
                x, y = _get_train_data_some_subjects_aled_nom_fonction(subject=subject, experiment=None, run=parsargs.task)
                predicted_y = _model.predict(x)
                score = np.mean(predicted_y == y)
            _subjects_scores.append(score)
            
        else:
            print("Invalid settings.")
            exit(1)
        
        print(f"Total accuracy : {np.mean(_subjects_scores)}")
            
    
    if parsargs.task is False and parsargs.experiment is False and parsargs.subject is False:
        print("Hello")
        pass
    
    exit(1)
    
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
