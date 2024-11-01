import argparse
from os import mkdir, path
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
    cv = ShuffleSplit(2, test_size=0.2, random_state=global_data.RANDOM_STATE)
    
    # Construit un "arbre" (forest, duh) formé à partir d'échantillons aléatoire du jeu de donnée
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
        n_jobs=5,
        verbose=2,
        scoring='accuracy',
        error_score=np.nan,
        return_train_score=True,
    )

    grid_search_rfc.fit(X, y)

    print(f"  Cross-validation Accuracy: {grid_search_rfc.best_score_}")
    print(grid_search_rfc.best_params_)

    sleep(1)
    return grid_search_rfc.best_estimator_

def _get_data_for_training(subject=None, experiment=None, run=None) -> Tuple[np.ndarray, np.ndarray]:
    if experiment is None:
        _runs = 1

    _SUBJECTS = range(1, 110)
    if subject is not None:
        _SUBJECTS = [subject]

    all_epochs = []
    all_labels = []
    all_raw = []

    for subjectid in _SUBJECTS:
        X, y, raw = load_and_process(subject=subjectid, experiment=experiment, run=run)
        print(f"Subject {subjectid} loaded. ", np.shape(X), np.shape(y))
        all_epochs.append(X)
        all_labels.append(y)
        
        if subject is not None:
            all_raw.append(raw)

    X = np.concatenate(all_epochs, axis=0)
    y = np.concatenate(all_labels, axis=0)

    return X, y, all_raw[0] if len(all_raw) > 0 else None

def _realtime_predict(raw, model):
    # Time (s): 0       1       2       3       4       5       6       7       8
    #       |-------|-------|-------|-------|-------|-------|-------|-------|
    # Window 1: [-------1.0s-------]
    # Step 1: Advance by 1.5s
    # Window 2:             [-------1.0s-------]
    # Step 2: Advance by 1.5s
    # Window 3:                     [-------1.0s-------]


    window_size = 2.0
    step_size = 1.0
    sfreq = raw.info['sfreq']                    # Sampling frequency in Hz
    n_samples_window = int(window_size * sfreq)  # Number of samples in the window
    n_samples_step = int(step_size * sfreq)      # Number of samples to advance the window
    buffer = np.empty((0, raw.info['nchan']))
    total_samples = raw.n_times
    start_idx = 0

    while start_idx + n_samples_step <= total_samples:
        end_idx = start_idx + n_samples_step
        data, _ = raw[:, start_idx:end_idx]
        buffer = np.vstack((buffer, data.T))
        
        if buffer.shape[0] >= n_samples_window:
            window_data = buffer[-n_samples_window:] 
            epoch = mne.EpochsArray(window_data[np.newaxis, ...].transpose(0, 2, 1), raw.info)
            
            X = epoch.get_data(copy=True)
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X).max() * 100
            print(f"[{((start_idx + n_samples_step) / total_samples * 100):.3f}] Prediction: {prediction}, Probability: {probability:.2f}%")

        start_idx += n_samples_step
        sleep(step_size)


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
    
    args.add_argument("--realtime", action="store_true", default=False, help="Enable realtime stream processing as asked in the subject (slow, use with --experiment and --subject)")
    
    args.add_argument("--show-analytics", action="store_true", default=False, help="Show analytic graphs")
    
    args.add_argument("--test-size", type=float, default=0.2, help="Split value")
    
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
    
    if path.isdir(global_data.BASE_DIRECTORY) == False:
        mkdir(global_data.BASE_DIRECTORY)
    if path.isdir(global_data.DATA_DIRECTORY) == False:
        mkdir(global_data.DATA_DIRECTORY)
    if path.isdir(global_data.EEGBCI_DIRECTORY) == False:
        mkdir(global_data.EEGBCI_DIRECTORY)
    
    if parsargs.disable_tunning is True:
        global_data.DISABLE_HYPERTUNNING = True
        
    if parsargs.force_processing is True:
        global_data.FORCE_DATA_PROCESSING = True
        
    if parsargs.rnd != -1:
        global_data.RANDOM_STATE = parsargs.rnd
    elif parsargs.rnd == -1:
        global_data.RANDOM_STATE = random.randint(1, sys.maxsize)
        
    if parsargs.disable_wavelet is True:
        global_data.DISABLE_WAVELET = True
        
    
    if parsargs.show_analytics is True:
        global_data.FORCE_DATA_PROCESSING = True
        global_data.SHOW_ANALYTIC_GRAPHS = True
        
        print("=== ANALYTICS ===")
        _subject = parsargs.subject or 1
        _task = parsargs.task or 4
        _experiment = parsargs.experiment or 1
        
        print(_subject, _task, _experiment)
        
        _, _, _ = load_and_process(
            subject=_subject, 
            experiment=_experiment, 
            run=_task
        )
        
        plt.show()
        exit(0)
    
    _test_size = max(0.2, min(0.8, parsargs.test_size or 0.2))
    
    print(f"Random state: {global_data.RANDOM_STATE}")
    print(f"Test Size: {_test_size}")
    print(f"Base directory: {global_data.BASE_DIRECTORY}")
    sleep(1)
        
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
                x, y, _ = _get_data_for_training(subject=parsargs.subject, experiment=experiment)
                train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=_test_size, random_state=global_data.RANDOM_STATE)
                pipeline = _train(train_X, train_y)
                predicted_y = pipeline.predict(test_X)
                score = np.mean(predicted_y == test_y)
                print(f"   Test dataset Accuracy: {score}")
                model_cache_save(pipeline=pipeline, subject=parsargs.subject, experiment=experiment)
        
        elif parsargs.task is not None:
            print(f"Training Task {parsargs.task}")
            x, y, _ = _get_data_for_training(subject=parsargs.subject, experiment=None, run=parsargs.task)
            train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=_test_size, random_state=global_data.RANDOM_STATE)
            pipeline = _train(train_X, train_y)
            predicted_y = pipeline.predict(test_X)
            score = np.mean(predicted_y == test_y)
            model_cache_save(pipeline=pipeline, subject=parsargs.subject, task=parsargs.task)
            print(f"   Test dataset Accuracy: {score}")
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
                    x, y, raw = _get_data_for_training(subject=subject, experiment=experiment)
                    _, test_x, _, test_y = train_test_split(x, y, test_size=_test_size, random_state=global_data.RANDOM_STATE)
                    
                    predicted_y = _model.predict(test_x)
                    score = np.mean(predicted_y == test_y)
                    
                    if parsargs.realtime is True and parsargs.experiment is not None and parsargs.subject is not None:
                        _realtime_predict(raw=raw, model=_model)
                        
                    if experiment not in _per_experiments_scores:
                        _per_experiments_scores[experiment] = []
                    _per_experiments_scores[experiment].append(score)
                    
                    _experiment_scores.append(score)
                _subjects_scores.append(np.mean(_experiment_scores))
            for experiment in _per_experiments_scores:
                print(f"Accuracy for experiment {experiment}: {np.mean(_per_experiments_scores[experiment])}%")
            
        elif parsargs.task is not None:
            for subject in _subjects_to_predict:
                _model = model_cache_get(subject=parsargs.subject, experiment=None, task=parsargs.task)
                if _model is None:
                    exit(1)
                x, y, raw = _get_data_for_training(subject=subject, experiment=None, run=parsargs.task)
                _, test_x, _, test_y = train_test_split(x, y, test_size=_test_size, random_state=global_data.RANDOM_STATE)
                
                predicted_y = _model.predict(test_x)
                score = np.mean(predicted_y == test_y)
                
                if parsargs.realtime is True and parsargs.task is not None and parsargs.subject is not None:
                    _realtime_predict(raw=raw, model=_model)
            _subjects_scores.append(score)
            
        else:
            print("Invalid settings.")
            exit(1)
        
        print(f"Total accuracy : {np.mean(_subjects_scores)}")