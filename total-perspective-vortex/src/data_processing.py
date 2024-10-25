from typing import List, Tuple
import mne
from matplotlib import pyplot as plt
from mne.io.edf.edf import RawEDF
from mne.datasets import eegbci
from mne.channels import make_standard_montage
import numpy as np

from utils import load_eegbci_data

def prepare_data(raw: RawEDF) -> RawEDF :
    
    raw_copy = raw.copy();  
    eegbci.standardize(raw_copy)
    
    montage = make_standard_montage("standard_1020")
    raw_copy.set_montage(montage, on_missing='ignore')
    
    
    #montage = raw_copy.get_montage()
    # p = montage.plot()
    # p = mne.viz.plot_raw(raw_copy, scalings={"eeg": 75e-6})
    #
    # raw_copy.plot_psd()
    # raw_copy.plot_psd(average=True)
    #
    # plt.show()
    # exit(1)
    return raw_copy


def filter_data(raw: RawEDF) -> RawEDF :
    
    for annot in raw.annotations:
        if annot['onset'] > raw.times[-1]:
            annot['onset'] = raw.times[-1]

    data_filtered = raw.copy()
    
    # data_filtered.compute_psd().plot()
    # data_filtered.notch_filter(60, fir_design='firwin')
    # data_filtered.compute_psd().plot()
    
    data_filtered: RawEDF = data_filtered.filter(
        l_freq=8,
        h_freq=30,
        picks="eeg",
        fir_design='firwin',
        skip_by_annotation="edge",
        verbose="ERROR"
    )
    
    # p = mne.viz.plot_raw(data_filtered, scalings={"eeg": 75e-6})
    # plt.show()
    # exit(1)
    return data_filtered

def get_events(filtered_raw: RawEDF, tmin=-1, tmax=4.0) -> Tuple[any, any, any]:
    events, event_ids = mne.events_from_annotations(filtered_raw)

    mapping = {1: 'Rest', 2: 'ExecuteLeftOrBothFists', 3: 'ExecuteRightOrBothFeet'}
    annot_from_events = mne.annotations_from_events(
        events=events,
        event_desc=mapping,
        sfreq=filtered_raw.info['sfreq'],
        orig_time=filtered_raw.info['meas_date'],
    )

    filtered_raw.set_annotations(annot_from_events)

    # rename_events(event_ids)


    #p = mne.viz.plot_raw(filtered_raw, scalings={"eeg": 75e-6})

    # Power Spectral Analysis (PSD) - https://youtu.be/Gka11q5VfFI
    #filtered_raw.plot_psd()
    #filtered_raw.plot_psd(average=True)

    picks = mne.pick_types(
        filtered_raw.info,
        meg=False,
        eeg=True,
        stim=False,
        eog=False,
        exclude="bads",
    )

    epochs = mne.Epochs(
        filtered_raw,
        events,
        event_ids,
        tmin,
        tmax,
        proj=True,
        picks=picks,
        baseline=None,
        preload=True,
    )

    # epochs_before_baseline = epochs.copy()
    #
    # rest_start, rest_end = 0, 3
    # rest_data = filtered_raw.copy().crop(tmin=rest_start, tmax=rest_end).get_data()
    # rest_baseline = rest_data.mean(axis=1, keepdims=True)


    labels = epochs.events[:, -1]
    # epochs = epochs[["ExecuteLeftOrBothFists", "ExecuteRightOrBothFeet"]]
    # epochs = epochs[["ExecuteLeftOrBothFists", "ExecuteRightOrBothFeet"]]

    # epochs._data -= rest_baseline
    epochs = epochs[["T1", "T2"]]
    #
    # plt.figure(figsize=(10, 5))
    # plt.plot(epochs.times , epochs_before_baseline.get_data()[5, 2, :], label="Before Baseline Correction", color="blue")
    # plt.plot(epochs.times , epochs.get_data()[5, 2, :], label="After Baseline Correction", color="red", linestyle="--")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Amplitude (µV)")
    # plt.title("EEG Signal Before and After Baseline Correction")
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    # exit(1)
    # epochs.apply_baseline(baseline=(None, 0))
    ica = mne.preprocessing.ICA(n_components=20, random_state=42, max_iter=800)
    ica.fit(epochs)
    # ica.plot_components()

    eog_indices, eog_scores = ica.find_bads_eog(epochs, ch_name='Fp1', threshold=3.0)
    # eog_indices, eog_scores = ica.find_bads_eog(epochs, threshold=3.0)  # or other threshold value
    ica.exclude = eog_indices  #Mark components for exclusion
    ica.apply(epochs)

    # print(eog_scores, eog_indices)
    #ica.plot_overlay(filtered_raw, exclude=[0], picks="eeg")
    # print(epochs)
    # print(labels)
    # print(epochs)
    # print(picks)

    # X = epochs.get_data()
    # y = epochs.events[:, -1] - 1

    # Visualizes the occurrence and types of events in the EEG data.
    # mne.viz.plot_events(
    #     events,
    #     sfreq=filtered_raw.info['sfreq'],
    #     first_samp=filtered_raw.first_samp,
    #     event_id=event_ids
    # )

    # epochs.plot(n_channels=32, scalings=dict(eeg=250e-6))
    #
    # plt.show()

    return picks, labels, epochs
    
def rename_events(evt: dict):
    evt['Rest'] = evt['T0']
    evt['ExecuteLeftOrBothFists'] = evt['T1']
    evt['ExecuteRightOrBothFeet'] = evt['T2']

    del evt['T0']
    del evt['T1']
    del evt['T2']
    

def load_and_process(subject=None, experiment=None) -> Tuple[np.ndarray, np.ndarray]:
    raw_edf = load_eegbci_data(subject, experiment)
    prepared_data = prepare_data(raw_edf)
    filtered_data = filter_data(prepared_data)
    (picks, labels, epochs) = get_events(filtered_data)
    

    X = epochs.get_data(copy=True)
    y = epochs.events[:, -1] - 1
    
    # print(picks, labels, epochs)
    return np.array(X), np.array(y)
    # return X, y