from typing import List
import mne
from mne.io.edf.edf import RawEDF

from enums import RUN_FISTS


VERBOSE_LEVEL = 30


def load_eegbci_data(subject=1, runs=None) -> RawEDF:
    if runs is None:
        runs = RUN_FISTS

    _data = mne.datasets.eegbci.load_data(
        subject=subject,
        runs=runs,
        verbose=VERBOSE_LEVEL,
        # path="../eegbi_data",
        path="/home/slopez/sgoinfre/eegbci",
        update_path=False
    )
    raw_files: List[RawEDF] = [mne.io.read_raw_edf(f, preload=True, verbose=VERBOSE_LEVEL) for f in _data]
    _raw: RawEDF = mne.concatenate_raws(raw_files)
    mne.datasets.eegbci.standardize(_raw)
    return _raw

