import mne
import matplotlib.pyplot as plt

RUNS = {
    "BaselineEyesOpen": [1],
    "BaselineEyesClosed": [2],
    "OpenCloseFist": [3, 7, 11],
    "ImagineOpenCloseFist": [4, 8, 12],
    "OpenCloseBothFistsAndFeet": [5, 9, 13],
    "ImagineOpenCloseBothFistsAndFeet": [6, 10, 14],
}

############################################################

def load_dataset(subject=1, runs=[3, 7, 11]):

    data = mne.datasets.eegbci.load_data(
        subject=subject,
        runs=runs
    )

    raw_files = [mne.io.read_raw_edf(f, preload=True) for f in data]
    raw = mne.concatenate_raws(raw_files)

    mne.datasets.eegbci.standardize(raw) 
    return raw

############################################################

raw = load_dataset(
    subject=1,
    runs=RUNS["OpenCloseFist"]
)
# RAW DATASET PLOT
raw.plot(
    n_channels=64, 
    scalings='auto', 
    title='RAW Data', 
    show=True,
    block=True,
)

montage = mne.channels.make_standard_montage("standard_1020")
raw.set_montage(montage)
raw.set_eeg_reference(projection=True)

psd = raw.compute_psd()#.plot(average=True, spatial_colors=False);

axes = plt.subplot() 
fig = psd.plot(axes=axes, show=False)
plt.show()

# 
# raw.annotations.rename(dict(T1="hands", T2="feet"))

# raw.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")

# picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")

# tmin, tmax = -1.0, 4.0

# # Read epochs (train will be done only between 1 and 2s)
# # Testing will be done with a running classifier
# epochs = mne.Epochs(
#     raw,
#     event_id=["hands", "feet"],
#     tmin=tmin,
#     tmax=tmax,
#     proj=True,
#     picks=picks,
#     baseline=None,
#     preload=True,
# )
# epochs_train = epochs.copy().crop(tmin=1.0, tmax=2.0)
# labels = epochs.events[:, -1] - 2

# raw.plot_psd(fmin=1, fmax=40, tmin=0, tmax=60, n_fft=2048)

# montage = mne.channels.make_standard_montage('standard_1020')
# raw.set_montage(montage)
# raw.filter(1., 40., fir_design='firwin')

# print(mne.channels.get_builtin_montages())

# montage = mne.channels.make_standard_montage('standard_1020')
# raw.set_montage(montage)

# # Apply band-pass filter (e.g., 1-40 Hz)
# raw.filter(1., 40., fir_design='firwin')
# raw.plot(n_channels=10, scalings='auto', title='Raw EEG Data',
#          show=True, block=True)