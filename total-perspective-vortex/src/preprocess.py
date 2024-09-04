from typing import Any

import mne
from mne import Epochs
from mne.io.edf.edf import RawEDF

def rename_events(evt: dict):
    evt['Rest'] = evt['T0']
    evt['Left Fist'] = evt['T1']
    evt['Right Fist'] = evt['T2']

    del evt['T0']
    del evt['T1']
    del evt['T2']


def filter_dataset(data_raw: RawEDF):
    """
        TEMPORAL FILTERING
        Sélection d'uniquement les fréqueunces qui nous intéresse.

        8 - 12      : Alpha (Rest / Relaxed / Motor Functions)
        12 - 30     : Beta  (Thinking / Focus / Aware / Motor Functions)
        30 - 100    : Gamma (Concentration / Problem Solving / Alertness)

        [ https://www.sciencedirect.com/science/article/pii/S0736584521000223 ]
        Movement intentions mostly detected within 8 Hz - 22  Hz
    """
    data_filtered: RawEDF = data_raw.filter(
        l_freq=8,
        h_freq=41,
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
        tmax=5,
        proj=True,
        picks=picks,
        baseline=None,
        preload=True,
    )

    return epochs