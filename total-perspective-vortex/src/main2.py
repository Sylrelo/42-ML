from typing import List, Tuple
from matplotlib import pyplot as plt
import mne
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from custom_csp import CustomCSP
from data_processing import filter_data, get_events, load_and_process, prepare_data
from utils import load_eegbci_data

mne.set_log_level('WARNING')


RANGE_SUBJECT = range(10, 14)


def _train(X: np.ndarray, y: np.ndarray, calculate_xval=False) -> Pipeline:
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    
    # transformer = mne.decoding.CSP(n_components=6)
    transformer = CustomCSP(n_components=6)

    lda = LinearDiscriminantAnalysis(solver='svd')
    log_reg = LogisticRegression(penalty='l1', solver='liblinear')
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    
    pipeline_lda = make_pipeline(transformer, lda)
    pipeline_logreg = make_pipeline(transformer, log_reg)
    pipeline_rfc = make_pipeline(transformer, rfc)
    
    # if calculate_xval is True:
    #     score_lda = cross_val_score(pipeline_lda, X=X, y=y, cv=cv, verbose=False)
    #     score_logreg = cross_val_score(pipeline_logreg, X=X, y=y, cv=cv, verbose=False)
    #     score_rfc = cross_val_score(pipeline_rfc, X=X, y=y, cv=cv, verbose=False)
        
    #     print("score_lda ", score_lda.mean())    
    #     print("score_logreg ", score_logreg.mean())
    #     print("score_rfc ", score_rfc.mean())  
             
    _pipeline = pipeline_rfc.fit(X, y)
    
    return _pipeline
          
def _get_train_data_some_subjects_aled_nom_fonction(runs=[3, 7, 11]) -> Tuple[np.ndarray, np.ndarray]:
    _SUBJECTS = [1, 10, 20, 30, 40, 50, 60, 70]
    _runs = runs
    all_epochs = []
    all_labels = []
    
    for subject in _SUBJECTS:
        # eegbci_raw = load_eegbci_data(subject=subject, runs=_runs)
        # filtered_raw = filter_data(eegbci_raw)
        # _, labels, epochs = get_events(filtered_raw)
        
        # print(np.shape(epochs))
        X, y = load_and_process(subject=subject, experiment=_runs)
        all_epochs.append(X)
        all_labels.append(y)
    
    X = np.concatenate(all_epochs, axis=0)
    y = np.concatenate(all_labels, axis=0)
                
    return X, y

if __name__ == '__main__':
    # for subject in RANGE_SUBJECT:
    #     (X, y) = load_and_process(subject)
    #     train_X, text_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
    #     pipeline = _train(train_X, train_y)

    #     predicted_y = pipeline.predict(text_X)
    #     score = np.mean(predicted_y == test_y)
    #     print(f"====== End {subject}: {score}%")
        
    (X, y) = _get_train_data_some_subjects_aled_nom_fonction()
    # csp = mne.decoding.CSP()
    
    # print(np.shape(X), np.shape(y))
    # print(len(X), len(y))
    # X = csp.fit_transform(X, y)
    
    train_X, text_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline = _train(train_X, train_y, calculate_xval=True)
    
    # for subject in RANGE_SUBJECT:

    # exit(1)
    # pipeline.predict()
    total = 0
    total_subjects = 0
    for subject in range(1, 110):
        (X, y) = load_and_process(subject)
    #     _, text_X, _, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
    # #
        predicted_y = pipeline.predict(X)
        score = np.mean(predicted_y == y)
        print(f"Sub {subject}: {score}")

        total += score
        total_subjects += 1
        
        print(f"Total score: {total / total_subjects}")
    #

    print(f"Total score: {total / total_subjects}")