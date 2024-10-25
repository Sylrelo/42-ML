from typing import Any

import mne
import numpy as np
from mne import Epochs
from mne.io.edf.edf import RawEDF

def rename_events(evt: dict):
    evt['Rest'] = evt['T0']
    evt['Left Fist'] = evt['T1']
    evt['Right Fist'] = evt['T2']

    del evt['T0']
    del evt['T1']
    del evt['T2']

def filter_dataset(data_raw: RawEDF, baseline: RawEDF = None):
    """
        TEMPORAL FILTERING
        Sélectionne uniquement les fréquences qui nous intéresse.

        8 - 12      : Alpha (Rest / Relaxed / Motor Functions)
        12 - 30     : Beta  (Thinking / Focus / Aware / Motor Functions)
        30 - 100    : Gamma (Concentration / Problem Solving / Alertness)

        [ https://www.sciencedirect.com/science/article/pii/S0736584521000223 ]
        Movement intentions mostly detected within 8 Hz - 22  Hz
    """

    for annot in data_raw.annotations:
        if annot['onset'] > data_raw.times[-1]:
            annot['onset'] = data_raw.times[-1]

    if baseline is not None:
        baseline_data = baseline.get_data()
        baseline_start, baseline_end = 0, 10  # in seconds
        baseline_mean = baseline_data[:, int(baseline_start * baseline.info['sfreq']):int(baseline_end * baseline.info['sfreq'])].mean(axis=1)
    
        target_data = data_raw.get_data()
        target_corrected = target_data - baseline_mean[:, None]
        data_raw._data = target_corrected

    # data_raw = data_raw.copy()
    # data_raw.notch_filter(np.arange(50, 251, 50))
    # data_raw.set_eeg_reference('average')

    data_filtered: RawEDF = data_raw.filter(
        l_freq=7,
        h_freq=32,
        picks="eeg",
        fir_design='firwin',
        skip_by_annotation="edge",
        verbose="ERROR"
    )

    """
        Montage : Placement des électrodes
        standard_1020 : International 10-20 System
    """
    montage = mne.channels.make_standard_montage("standard_1020")
    data_filtered.set_montage(montage)
    data_filtered.set_eeg_reference(projection=True)

    # picks = mne.pick_types(
    #     data_filtered.info,
    #     meg=False,
    #     eeg=True,
    #     stim=False,
    #     eog=False,
    #     exclude="bads",
    # )
    # data_filtered.plot(
    #     n_channels=64,
    #     scalings='auto',
    #     title='Filtered Data',
    #     show=False,
    #     block=True,
    #     picks=picks
    # )
    
    # plt.show()

    return data_filtered

def get_picks(data_filtered: RawEDF) -> (Any, Any, []):
    events, event_ids = mne.events_from_annotations(data_filtered)
    rename_events(event_ids)

    '''
        Récupération des channels EEG
    '''
    picks = mne.pick_types(
        data_filtered.info,
        meg=False,
        eeg=True,
        stim=False,
        eog=False,
        exclude="bads",
    )

    return events, event_ids, picks

def get_epochs(data_filtered: RawEDF, events, event_ids, picks: []) -> Epochs:
    epochs = mne.Epochs(
        data_filtered,
        event_id=event_ids,
        events=events,
        tmin=-1,
        tmax=4,
        proj=True,
        picks=picks,
        baseline=None,
        preload=True,
    )

    return epochs