from typing import List, Tuple
import mne
from mne.io.edf.edf import RawEDF
from mne.datasets import eegbci
from mne.channels import make_standard_montage

from utils import load_eegbci_data

def prepare_data(raw: RawEDF) -> RawEDF :
    
    raw_copy = raw.copy();  
    eegbci.standardize(raw_copy)
    
    montage = make_standard_montage("biosemi64")
    raw_copy.set_montage(montage, on_missing='ignore')
    
    
    #montage = raw_copy.get_montage()
    #p = montage.plot()
    #p = mne.viz.plot_raw(raw_copy, scalings={"eeg": 75e-6})
    
    # raw.plot_psd()
    # raw.plot_psd(average=True)

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
        l_freq=7,
        h_freq=32,
        picks="eeg",
        fir_design='firwin',
        skip_by_annotation="edge",
        verbose="ERROR"
    )
    
    # p = mne.viz.plot_raw(data_filtered, scalings={"eeg": 75e-6})
    
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
    
    labels = epochs.events[:, -1]
    # epochs = epochs[["ExecuteLeftOrBothFists", "ExecuteRightOrBothFeet"]]
    # epochs = epochs[["ExecuteLeftOrBothFists", "ExecuteRightOrBothFeet"]]
    
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
    
    return picks, labels, epochs
    
def rename_events(evt: dict):
    evt['Rest'] = evt['T0']
    evt['ExecuteLeftOrBothFists'] = evt['T1']
    evt['ExecuteRightOrBothFeet'] = evt['T2']

    del evt['T0']
    del evt['T1']
    del evt['T2']
    

def load_and_process(subject=None, experiment=None):
    raw_edf = load_eegbci_data(subject, experiment)
    prepared_data = prepare_data(raw_edf)
    filtered_data = filter_data(prepared_data)
    (picks, labels, epochs) = get_events(filtered_data)
    

    print(picks, labels, epochs)