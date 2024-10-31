from typing import List
import joblib
import mne
from mne.io.edf.edf import RawEDF
from sklearn.pipeline import Pipeline

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
        verbose='CRITICAL',
        path=global_data.EEGBCI_DIRECTORY,
        update_path=False
    )
    raw_files: List[RawEDF] = [mne.io.read_raw_edf(f, preload=True, verbose='CRITICAL') for f in _data]
    _raw: RawEDF = mne.concatenate_raws(raw_files)
    mne.datasets.eegbci.standardize(_raw)

    return _raw

def _model_cache_file(subject=None, experiment=None, task=None):
    return f"{global_data.DATA_DIRECTORY}/model_s{subject}e{experiment}t{task}.model"
    
def model_cache_get(subject=None, experiment=None, task=None) -> Pipeline | None:
    filepath = _model_cache_file(subject, experiment, task)
    
    try:
        with open(filepath, "rb") as f:
            data = joblib.load(f)
            return data
    except Exception as e:
        print(f"Cannot load model for Subject [{subject}] - Experiment [{experiment}] - Task [{task}]: {e}")
        return None
    
    return None

def model_cache_save(pipeline=None, subject=None, experiment=None, task=None):
    filepath = _model_cache_file(subject, experiment, task)
    try:
        with open(filepath, "wb") as f:
            joblib.dump(pipeline, f)
    except Exception as e:
        print("Cannot save model.")
