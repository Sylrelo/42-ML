from typing import List
import mne
from mne.io.edf.edf import RawEDF

from global_data import EXPERIMENT
import global_data

VERBOSE_LEVEL = 30


def load_eegbci_data(subject=1, experiment=None, run=None) -> RawEDF:
    if experiment is None:
        experiment = 1

    runs = EXPERIMENT[experiment]
    if run is not None:
        runs = [run]

    _data = mne.datasets.eegbci.load_data(
        subject=subject,
        runs=runs,
        verbose=VERBOSE_LEVEL,
        path=global_data.EEGBCI_DIRECTORY,
        update_path=False
    )
    raw_files: List[RawEDF] = [mne.io.read_raw_edf(f, preload=True, verbose=VERBOSE_LEVEL) for f in _data]
    _raw: RawEDF = mne.concatenate_raws(raw_files)
    mne.datasets.eegbci.standardize(_raw)

    return _raw

