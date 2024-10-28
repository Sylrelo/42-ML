import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from multiprocessing import freeze_support

import numpy as np
sys.path.append(os.path.abspath(os.path.dirname("__file__")))
sys.path.append(os.path.join(os.path.dirname("__file__"), "src"))

import os
import threading
import time
import matplotlib.pyplot as plt  

import joblib
import mne
from mne.io.edf.edf import RawEDF
from preprocess import filter_dataset, get_picks, get_epochs
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import (GridSearchCV, KFold, cross_val_score, train_test_split)

RUNS = {
    "BaselineEyesOpen": [1],
    "BaselineEyesClosed": [2],
    "OpenCloseFist": [3, 7, 11],
    "ImagineOpenCloseFist": [4, 8, 12],
    "OpenCloseBothFistsAndFeet": [5, 9, 13],
    "ImagineOpenCloseBothFistsAndFeet": [6, 10, 14],
    "OpenAndImagineOpenCloseBothFistsAndFeet": [5, 9, 13, 6, 10, 14],
    "OpenAndImagineOpenCloseFist": [3, 7, 11, 4, 8, 12],
}
EXPERIMENTS = {
    0: RUNS["BaselineEyesOpen"],
    1: RUNS["BaselineEyesClosed"],
    2: RUNS["OpenCloseFist"],
    3: RUNS["ImagineOpenCloseFist"],
    4: RUNS["OpenCloseBothFistsAndFeet"],
    5: RUNS["ImagineOpenCloseBothFistsAndFeet"],
    6: RUNS["OpenAndImagineOpenCloseBothFistsAndFeet"],
    7: RUNS["OpenAndImagineOpenCloseFist"],
}

############################################################
VERBOSE_LEVEL = 30
mne.set_log_level('WARNING')

def load_dataset(subject=1, runs=None) -> RawEDF:
    if runs is None:
        runs = [3, 7, 11]

    _data = mne.datasets.eegbci.load_data(
        subject=subject,
        runs=runs,
        verbose=VERBOSE_LEVEL,
        # path="../eegbi_data",
        path="/home/slopez/sgoinfre/eegbci",
        update_path=False
    )
    raw_files: [RawEDF] = [mne.io.read_raw_edf(f, preload=True, verbose=VERBOSE_LEVEL) for f in _data]
    _raw: RawEDF = mne.concatenate_raws(raw_files)
    mne.datasets.eegbci.standardize(_raw)
    return _raw


############################################################
# T0 Rest
# T1 Motion (real or imagined)
# T2 motion (real or imagined)

# raw_baseline = load_dataset(subject=1, runs=[1])
# mne.events_from_annotations(raw_baseline)
# raw_baseline.plot(
#     n_channels=64,
#     scalings='auto',
#     title='RAW Baseline Data',
#     show=False,
#     block=True,
# )
# baseline_cpy: RawEDF = raw_baseline.copy()
# baseline_filtered: RawEDF = baseline_cpy.filter(l_freq=8, h_freq=41, picks="eeg", fir_design='firwin')

# raw = load_dataset(
#     subject=1,
#     runs=RUNS["OpenCloseFist"]
# )

# Visualisation de tous les channels, sans filtrage
# raw.plot(
#     n_channels=64,
#     duration=10,
#     scalings='auto',
#     title='RAW Data',
#     show=False,
#     block=True,
#     verbose=VERBOSE_LEVEL
# )

# print(f"Sample rate: {raw.info['sfreq']} Hz, Shape: {raw._data.shape}", )

# Filtrage des bandes de fréqueunces 8 - 41
# data_cpy: RawEDF = raw.copy()
# data_filtered = filter_dataset(raw)
# events, event_ids, picks = get_picks(data_filtered)
# epochs = get_epochs(data_filtered, events, event_ids, picks)

##

def prepare_dataset(raw: RawEDF, baseline: RawEDF = None):
    data_cpy: RawEDF = raw.copy()
    data_filtered = filter_dataset(raw, baseline)
    events, event_ids, picks = get_picks(data_filtered)
    epochs = get_epochs(data_filtered, events, event_ids, picks)
    
    # epochs.apply_baseline()
    
    # data_filtered.plot(
    #     n_channels=64,
    #     scalings='auto',
    #     title='Filtered Data',
    #     show=False,
    #     block=True,
    #     picks=picks
    # )

    # plt.show()
    
    train_data = epochs.copy().crop(tmin=-1.0, tmax=2.0)
    train_data = epochs.get_data(copy=True)

    return train_data, epochs.events[:, -1]


def train_model(train_x, train_y, test_x, test_y, subject=None, experiment=None):
    crossval_strategy = KFold(n_splits=5)
    # csp = mne.decoding.CSP(n_components=4, log=True)
    # pipeline = make_pipeline(csp, LinearDiscriminantAnalysis(solver="lsqr"), verbose=False)

    try:
        f = open(f"../_best/best_config_{subject}E{experiment}.td", "rb")
        data = joblib.load(f)
        csp = mne.decoding.CSP(n_components=data[0], log=False)
        _pipeline = make_pipeline(csp, LinearDiscriminantAnalysis(solver="lsqr", tol=0.0001, shrinkage="auto"), verbose=False)
        _pipeline.fit(train_x, train_y)
        return _pipeline
    except Exception as ex:
        print(f"Error reading preference: {ex}")

    best_score = 0
    best_pipeline = None

    try:
        for ncomp in [7, 10]:
            for tol in [1]:
                csp = mne.decoding.CSP(n_components=ncomp, log=False)
                _pipeline = make_pipeline(csp, LinearDiscriminantAnalysis(solver="lsqr", tol=0.0001, shrinkage="auto"), verbose=False)
                _score = cross_val_score(_pipeline, X=train_x, y=train_y, cv=crossval_strategy, verbose=False)
                print(f"Score {_score.mean()}. Old score: {best_score} - CSP {ncomp}")
                if _score.mean() > best_score:
                    best_score = _score.mean()
                    best_pipeline = _pipeline
                    with open(f"../_best/best_config_{subject}E{experiment}.td", "wb") as f:
                        joblib.dump((ncomp, tol), f)
    except Exception as ex:
        print(f"Aled: {ex}")


    # print(f"Score: {score.mean()}")
    # print("Searching for best Hyperparameters...")
    # grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', verbose=1)
    # grid_search.fit(X=train_x.copy(), y=train_y)
    
    # grid_search.best_estimator_
    
    # print("Best parameters found:", grid_search.best_params_)
    # print("Best cross-validation score:", grid_search.best_score_)
    #
    #
    # return grid_search.best_estimator_
    # grid_search.best_estimator_.pred

    best_pipeline.fit(train_x, train_y)
    return best_pipeline



    # print(f"Accuracy: {pipeline.score(train_x, train_y)}")

    # y_predicted = pipeline.predict(test_x)
    # score = np.mean(y_predicted == test_y)
    #
    # print(f"Score: {score}")

    # print(y_predicted)
    # print(test_y)

    return pipeline

    # print(f"{len(score)} { len(train_y)}")

    # fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
    # PredictionErrorDisplay.from_predictions(
    #     y,
    #     y_pred=score,
    #     kind="actual_vs_predicted",
    #     subsample=100,
    #     ax=axs[0],
    #     random_state=0,
    # )
    # axs[0].set_title("Actual vs. Predicted values")
    # PredictionErrorDisplay.from_predictions(
    #     y,
    #     y_pred=score,
    #     kind="residual_vs_predicted",
    #     subsample=100,
    #     ax=axs[1],
    #     random_state=0,
    # )
    # axs[1].set_title("Residuals vs. Predicted Values")
    # fig.suptitle("Plotting cross-validated predictions")
    # plt.tight_layout()
    # plt.show()


# raw = load_dataset(
#     subject=1,
#     runs=RUNS["OpenCloseFist"]
# )
# x, y = prepare_dataset(raw)
#
# train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
#
# print(f"{len(train_x)} {len(train_y)}")

# train_model(train_x, train_y, test_x, test_y)


# 1..110
# SUBJECTS_RANGE = range(1, 110)
# SUBJECTS_RANGE = range(1, 4)

# 0..5
# EXPERIMENTS_RANGE = range(2, 5)
# EXPERIMENTS_RANGE = range(2, 4)

MODELS_DIRECTORY = "../_data/"


def save_test_dataset(test_x, test_y, subject=1, experiment=0):
    print(f"Saving S{subject}E{experiment}.td")
    with open(f"{MODELS_DIRECTORY}/testdata_S{subject}E{experiment}.td", "wb") as f:
        joblib.dump((test_x, test_y), f)

def save_model_to_file(model, subject=1, experiment=0):
    print(f"Saving S{subject}E{experiment}.model")
    with open(f"{MODELS_DIRECTORY}/model_S{subject}E{experiment}.model", "wb") as f:
        joblib.dump(model, f)

def open_model_file(subject=1, experiment=0) -> Pipeline:
    with open(f"{MODELS_DIRECTORY}/model_S{subject}E{experiment}.model", "rb") as f:
        data = joblib.load(f)
        return data
    
def open_test_dataset(subject=1, experiment=0):
    with open(f"{MODELS_DIRECTORY}/testdata_S{subject}E{experiment}.td", "rb") as f:
        data = joblib.load(f)
        return (data[0], data[1])


if not os.path.exists(MODELS_DIRECTORY):
    os.makedirs(MODELS_DIRECTORY)

def start_experiments(subject=1, experiment=0):
    print(f"START THREAD FOR SUBJECT {subject} EXPERIMENT {experiment}")


def start_predict(subjects_range: range, experiments_range: range):
    total_score = 0
    total_exp = 0

    for subject in subjects_range:
        subject_tasks_total_score = 0
        subject_total_tasks = 0
        
        for experiment in experiments_range:
            try:
                (test_x, test_y) = open_test_dataset(subject, experiment)
                pipeline = open_model_file(subject, experiment)
                y_predicted = pipeline.predict(test_x)
                score = np.mean(y_predicted == test_y)
                # print(f"S{subject}E{experiment} : {score}")
                
                subject_tasks_total_score += score
                subject_total_tasks += 1
                
                total_score += score
                total_exp += 1
                
            except FileNotFoundError as err:
                print(f"File not found for S{subject}E{experiment} ({err.filename})")
            except Exception as err:
                print(f"An unknown error occured. {err}")

        if subject_total_tasks > 0:
            print(f"Accuracy for subject {subject} : {subject_tasks_total_score / subject_total_tasks}")
    print(f"Total Accuracy {total_score / total_exp}")


def _train(subject: int, experiments_range: range):
    for experiment in experiments_range:
        # baseline_eo =  load_dataset(subject=subject, runs=EXPERIMENTS[0])
        baseline_eo = None

        raw = load_dataset(subject=subject, runs=EXPERIMENTS[experiment])
        x, y = prepare_dataset(raw, baseline_eo)
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
        save_test_dataset(test_x, test_y, subject, experiment)

        model = train_model(train_x, train_y, test_x, test_y, subject, experiment)
        save_model_to_file(model, subject, experiment)
        print(f"Subject {subject} Experiment {experiment} done training. =========")
    print(f"Subject {subject} done training. =========")

current_threads = []

def start_training(subjects_range: range, experiments_range: range):

    with ProcessPoolExecutor(max_workers=5) as executor:

        for subject in subjects_range:
            print(f"Queuing subject {subject}")

            _futures = executor.submit(_train, subject, experiments_range)
            current_threads.append(_futures)
        #     _thread = threading.Thread(target=_train, args=(subject, experiments_range))
        #     _thread.start()
        #
        #     if len(current_threads) >= 7:
        #         for thread in as_completed(current_threads):
        #             _ = thread
        #         current_threads.clear()
        # #
        # for thread in current_threads:
        #     thread.join()
        for thread in as_completed(current_threads):
            _ = thread
    #     print(f"Subject {subject}")
    #     baseline_eo = load_dataset(subject=subject, runs=EXPERIMENTS[0])
    #     baseline_eo_filtered = filter_dataset(baseline_eo)
    #
    #     # events, event_ids, picks = get_picks(baseline_eo_filtered)
    #     # epochs = get_epochs(baseline_eo_filtered, events, event_ids, picks)
    #     # baseline_ec = load_dataset(subject=subject, runs=EXPERIMENTS[0])
    #
    #     for experiment in experiments_range:
    #         print(f"Experiment {experiment}")
    #         raw = load_dataset(subject=subject, runs=EXPERIMENTS[experiment])
    #         x, y = prepare_dataset(raw, baseline_eo)
    #         train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
    #         save_test_dataset(test_x, test_y, subject, experiment)
    #
    #         model = train_model(train_x, train_y, test_x, test_y)
    #         save_model_to_file(model, subject, experiment)


if __name__ == '__main__':
    freeze_support()
    range_subject = range(23, 29)
    start_training(range_subject, range(2, 8))
    start_predict(range_subject, range(2, 8))

# Récupération du nom des évènements
# events, event_id = mne.events_from_annotations(data_filtered)
# rename_events(event_id)
#
# '''
#     Récupération des channels EEG
# '''
# picks = mne.pick_types(
#     data_filtered.info,
#     meg=False,
#     eeg=True,
#     stim=False,
#     eog=False,
#     exclude="bads",
# )
# # picks = picks[::2] # A VOIR ? SELECTION UNIQUEMENT DES CHANNEL ODD ?

# data_filtered.plot(
#     n_channels=64,
#     scalings='auto',
#     title='Filtered Data',
#     show=False,
#     block=True,
#     picks=picks
# )


'''
    Affichage du graphique "Power Spectral Analysis"
    Mesure de la puissance d'un signal contre la fréquence
    
    It represents the proportion of the total signal power contributed by each frequency component of a voltage signal.
'''
# data_filtered.compute_psd().plot(picks=picks, exclude="bads", amplitude=False)


# Affichage des évènements filtrés par epochs (frequency and time-frequency)
'''

'''
# epochs = mne.Epochs(
#     data_filtered,
#     event_id=event_id,
#     events=events,
#     tmin=-1,
#     tmax=5,
#     proj=True,
#     picks=picks,
#     baseline=None,
#     preload=True,
# )
# epochs.plot(
#     events=events,
#     event_id=event_ids,
# )


# epochs.compute_psd().plot(average=True, picks=picks, exclude="bads", amplitude=False)
# #
#
# plt.show()
# mne.viz.plot_events(events, event_id=event_id,  sfreq=raw.info["sfreq"], first_samp=raw.first_samp)

# plt.show()
