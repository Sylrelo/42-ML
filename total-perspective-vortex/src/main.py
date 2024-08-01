import mne
import matplotlib.pyplot as plt
from mne.io.edf.edf import RawEDF

RUNS = {
    "BaselineEyesOpen": [1],
    "BaselineEyesClosed": [2],
    "OpenCloseFist": [3, 7, 11],
    "ImagineOpenCloseFist": [4, 8, 12],
    "OpenCloseBothFistsAndFeet": [5, 9, 13],
    "ImagineOpenCloseBothFistsAndFeet": [6, 10, 14],
}

############################################################
VERBOSE_LEVEL = 30

def load_dataset(subject=1, runs=None) -> RawEDF:
    if runs is None:
        runs = [3, 7, 11]

    _data = mne.datasets.eegbci.load_data(
        subject=subject,
        runs=runs,
        verbose=VERBOSE_LEVEL
    )

    raw_files: [RawEDF] = [mne.io.read_raw_edf(f, preload=True, verbose=VERBOSE_LEVEL) for f in _data]
    _raw: RawEDF = mne.concatenate_raws(raw_files)

    mne.datasets.eegbci.standardize(_raw)
    return _raw


def rename_events(evt: dict):
    evt['Rest'] = evt['T0']
    evt['Left Fist'] = evt['T1']
    evt['Right Fist'] = evt['T2']

    del evt['T0']
    del evt['T1']
    del evt['T2']

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

raw = load_dataset(
    subject=1,
    runs=RUNS["OpenCloseFist"]
)

# Visualisation de tous les channels, sans filtrage
raw.plot(
    n_channels=64,
    duration=10,
    scalings='auto',
    title='RAW Data',
    show=False,
    block=True,
    verbose=VERBOSE_LEVEL
)

print(f"Sample rate: {raw.info['sfreq']} Hz, Shape: {raw._data.shape}", )

# Filtrage des bandes de fréqueunces 8 - 41
data_cpy: RawEDF = raw.copy()

'''
    TEMPORAL FILTERING
    Sélection d'uniquement les fréqueunces qui nous intéresse.

    8 - 12      : Alpha (Rest / Relaxed / Motor Functions)
    12 - 30     : Beta  (Thinking / Focus / Aware / Motor Functions)
    30 - 100    : Gamma (Concentration / Problem Solving / Alertness)
    
    [ https://www.sciencedirect.com/science/article/pii/S0736584521000223 ]
    Movement intentions mostly detected within 8 Hz - 22  Hz
'''
data_filtered: RawEDF = data_cpy.filter(l_freq=8, h_freq=41, picks="eeg", fir_design='firwin')

'''
    Montage : Placement des électrodes
    standard_1020 : International 10-20 System
'''
montage = mne.channels.make_standard_montage("standard_1020")
data_filtered.set_montage(montage)
data_filtered.set_eeg_reference(projection=True)


# Récupération du nom des évènements
events, event_id = mne.events_from_annotations(data_filtered)
rename_events(event_id)

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
# picks = picks[::2] # A VOIR ? SELECTION UNIQUEMENT DES CHANNEL ODD ?

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
data_filtered.compute_psd().plot(picks=picks, exclude="bads", amplitude=False)


# Affichage des évènements filtrés par epochs (frequency and time-frequency)
'''

'''
epochs = mne.Epochs(
    data_filtered,
    event_id=event_id,
    events=events,
    tmin=-1,
    tmax=5,
    proj=True,
    picks=picks,
    baseline=None,
    preload=True,
)
epochs.plot(
    events=events,
    event_id=event_id,
)


epochs.compute_psd().plot(average=True, picks=picks, exclude="bads", amplitude=False)
#
mne.viz.plot_events(events, event_id=event_id)

plt.show()
