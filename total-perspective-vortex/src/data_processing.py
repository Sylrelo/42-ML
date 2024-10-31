from typing import List, Tuple

import joblib
import mne
from matplotlib import pyplot as plt
from mne.io.edf.edf import RawEDF
from mne.datasets import eegbci
from mne.channels import make_standard_montage
import numpy as np
from mne.preprocessing import ICA
from toolz.itertoolz import no_pad

import global_data
from utils import load_eegbci_data

def prepare_data(raw: RawEDF) -> RawEDF:
    raw_copy = raw.copy()
    eegbci.standardize(raw_copy)

    montage = make_standard_montage("standard_1020")
    raw_copy.set_montage(montage, on_missing='ignore')

    if global_data.SHOW_ANALYTIC_GRAPHS:
        p = montage.plot()
        plt.show()
        
        fig = raw_copy.plot(title='Données EEG brute', scalings={"eeg": 75e-6})
        fig.axes[0].set_xlabel('Temps (s)')
        fig.axes[0].set_ylabel('Amplitude (µV)')
        plt.show()
        
        # Densité Spectrale de Puissance (PSD)
        # Identification de l'intensité des fréquences par channel
        fig = raw_copy.plot_psd()
        fig.axes[0].set_title('Densité Spectrale de Puissance (PSD)')
        fig.axes[0].set_xlabel('Fréquence (Hz)')
        fig.axes[0].set_ylabel('Densité Spectrale (µV²/Hz)')
        plt.show()
        
        fig = raw_copy.plot_psd(average=True)
        fig.axes[0].set_title('Densité Spectrale de Puissance Moyenne')
        fig.axes[0].set_xlabel('Fréquence (Hz)')
        fig.axes[0].set_ylabel('Densité Spectrale (µV²/Hz)')
        plt.show()

    return raw_copy


# ICA (Independent Component Analysis)
# - Néttoyer les données pour préserver l'activité motrice uniquement
#       (retire/diminue les signaux liés aux mouvements occulaire, par exemple)
# - Amélioration de la qualité du signal
#
def _apply_ICA_filtering(data_filtered: RawEDF):
    _data_before = None
    if global_data.SHOW_ANALYTIC_GRAPHS is True:
        _data_before = data_filtered.get_data()

    ica = ICA(n_components=15, random_state=global_data.RANDOM_STATE, method='fastica', verbose=None, )
    ica.fit(data_filtered)

    eog_channels = [
        'Fp1', 'Fp2',           # Mouvements verticaux
        'F7', 'F8',             # Mouvements horizontaux
        'Fpz', 'AF3', 'AF4',    # Clignements
        'T7', 'T8',             # Mouvement machoire/tempes
        'FC5', 'FC6',           # Cou
    ]

    eog_indices, eog_scores = _detect_eog_artifacts(data_filtered, ica, eog_channels=eog_channels)
    ica.exclude = eog_indices
    ica.apply(data_filtered)

    if global_data.SHOW_ANALYTIC_GRAPHS is True:
        ica.plot_scores(eog_scores)
        plt.show()

        _data_after = ica.get_sources(data_filtered).get_data()

        ###########################################
        plt.figure(figsize=(15, 10))
        n_channels = min(_data_before.shape[0], _data_after.shape[0])
        n_samples = min(_data_before.shape[1], _data_after.shape[1])
        time = data_filtered.times[:n_samples]
        # time = data_filtered.times[:n_channels]  # Temps pour les tracés

        # Tracer le signal avant ICA
        plt.subplot(3, 1, 1)
        plt.plot(time, _data_before[:, :n_samples].T, color='grey', alpha=0.5)
        plt.title('Signal EEG avant filtrage ICA')
        plt.xlabel('Temps (s)')
        plt.ylabel('Amplitude (uV)')

        # Tracer le signal après ICA
        plt.subplot(3, 1, 2)
        plt.plot(time, _data_after[:, :n_samples].T, color='blue', alpha=0.5)
        plt.title('Signal EEG après filtrage ICA')
        plt.xlabel('Temps (s)')
        plt.ylabel('Amplitude (uV)')

        plt.subplot(3, 1, 3)
        for channel in range(n_channels):
            difference = _data_before[channel, :n_samples] - _data_after[channel, :n_samples]
            plt.plot(time, difference, alpha=0.5, label=f'Canal {channel + 1}')

        plt.title('Différence entre Signal avant et après ICA')
        plt.xlabel('Temps (s)')
        plt.ylabel('Différence (uV)')
        plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
        plt.legend(loc='upper right', fontsize='small', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

        plt.tight_layout()
        plt.show()

    ###########################################

    # plt.show()


def _detect_eog_artifacts(data_filtered: RawEDF, ica, eog_channels=None, threshold=3.0):
    if eog_channels is None:
        eog_channels = ['Fp1', 'Fp2', 'F7', 'F8']

    all_indices = []
    all_scores = []

    for ch_name in eog_channels:
        if ch_name in data_filtered.ch_names:
            indices, scores = ica.find_bads_eog(
                data_filtered,
                ch_name=ch_name,
                threshold=threshold,
            )
            all_indices.extend(indices)
            all_scores.append(scores)

    eog_indices = list(dict.fromkeys(all_indices))

    return eog_indices, all_scores


def filter_data(raw: RawEDF) -> RawEDF:
    for annot in raw.annotations:
        if annot['onset'] > raw.times[-1]:
            annot['onset'] = raw.times[-1]

    data_filtered = raw.copy()

    # if global_data.SHOW_ANALYTIC_GRAPHS:
    #     data_filtered.compute_psd().plot()
    #     data_filtered.notch_filter(60, fir_design='firwin')
    #     data_filtered.compute_psd().plot()

    data_filtered: RawEDF = data_filtered.filter(
        l_freq=8,
        h_freq=36,
        picks="eeg",
        fir_design='firwin',
        skip_by_annotation="edge",
        verbose="ERROR"
    )

    data_filtered.notch_filter(freqs=60, fir_design='firwin')
    print("Filter OK.", end=" ", flush=True)
    
    _apply_ICA_filtering(data_filtered)
    print("ICA OK.", end=" ", flush=True)
    
    data_filtered = data_filtered.copy().resample(90, npad="auto")

    if global_data.SHOW_ANALYTIC_GRAPHS is True:
        fig = data_filtered.compute_psd().plot()
        fig.axes[0].set_title('Densité Spectrale de Puissance (PSD) filtrée')
        fig.axes[0].set_xlabel('Fréquence (Hz)')
        fig.axes[0].set_ylabel('Densité Spectrale (µV²/Hz)')
        plt.show()
        
        fig = data_filtered.plot(title='Données EEG filtrée', scalings={"eeg": 75e-6})
        fig.axes[0].set_xlabel('Temps (s)')
        fig.axes[0].set_ylabel('Amplitude (µV)')
        plt.show()
        
        
    return data_filtered


def get_events(filtered_raw: RawEDF, tmin=-0.4, tmax=2.5) -> Tuple[any, any, any]:
    events, event_ids = mne.events_from_annotations(filtered_raw)

    mapping = {1: 'Rest', 2: 'ExecuteLeftOrBothFists', 3: 'ExecuteRightOrBothFeet'}
    annot_from_events = mne.annotations_from_events(
        events=events,
        event_desc=mapping,
        sfreq=filtered_raw.info['sfreq'],
        orig_time=filtered_raw.info['meas_date'],
    )

    filtered_raw.set_annotations(annot_from_events)

    # Power Spectral Analysis (PSD) - https://youtu.be/Gka11q5VfFI

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
        baseline=(None, 0),
        preload=True,
    )
    epochs.drop_bad()

    labels = epochs.events[:, -1]

    epochs = epochs[["T1", "T2"]]
    
 

    # print(eog_scores, eog_indices)
    #ica.plot_overlay(filtered_raw, exclude=[0], picks="eeg")
    # print(epochs)
    # print(labels)
    # print(epochs)
    # print(picks)

    # X = epochs.get_data()
    # y = epochs.events[:, -1] - 1

    if global_data.SHOW_ANALYTIC_GRAPHS is True:
        # Visualizes the occurrence and types of events in the EEG data.
        fig = mne.viz.plot_events(
            events,
            sfreq=filtered_raw.info['sfreq'],
            first_samp=filtered_raw.first_samp,
            event_id=event_ids
        )
        fig.axes[0].title = "Évenements"
        plt.show()
        
        epochs.plot(n_channels=32, scalings=dict(eeg=250e-6))
        plt.show()

    return picks, labels, epochs


def rename_events(evt: dict):
    evt['Rest'] = evt['T0']
    evt['ExecuteLeftOrBothFists'] = evt['T1']
    evt['ExecuteRightOrBothFeet'] = evt['T2']

    del evt['T0']
    del evt['T1']
    del evt['T2']


def load_and_process(subject=None, experiment=None, run=None) -> Tuple[any, any, any]:
    if global_data.FORCE_DATA_PROCESSING is False:
        try:
            with open(f"{global_data.DATA_DIRECTORY}/s{subject}e{experiment}r{run}.xy", "rb") as f:
                data = joblib.load(f)

                return data[0].astype(np.float64), data[1].astype(np.float64), data[2]
        except Exception as e:
            pass

    print(f"Loading subject {subject} Experiment {experiment}...", end=" ", flush=True)
    raw_edf = load_eegbci_data(subject, experiment, run)
    print("Load OK.", end=" ", flush=True)
    prepared_data = prepare_data(raw_edf)
    print("Prepare OK.", end=" ", flush=True)
    filtered_data = filter_data(prepared_data)
    print("Filtering OK.", end=" ", flush=True)
    (picks, labels, epochs) = get_events(filtered_data)

    X = epochs.get_data(copy=True)
    y = epochs.events[:, -1] - 1

    with open(f"{global_data.DATA_DIRECTORY}/s{subject}e{experiment}r{run}.xy", "wb") as f:
        joblib.dump((X.astype(np.float64), y.astype(np.float64), filtered_data), f)

    return X, y, filtered_data
