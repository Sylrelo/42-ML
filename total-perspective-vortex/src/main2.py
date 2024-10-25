from typing import List
from matplotlib import pyplot as plt
import mne
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from data_processing import filter_data, get_events, load_and_process, prepare_data
from utils import load_eegbci_data

mne.set_log_level('WARNING')


RANGE_SUBJECT = range(1, 110)


def _train(X: List[any], y: List[any]) -> Pipeline:
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    
    transformer = mne.decoding.CSP(n_components=8)
    
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    # log_reg = LogisticRegression(penalty='l1', solver='liblinear')
    # rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    
    pipeline_lda = make_pipeline(transformer, lda)
    # pipeline_logreg = make_pipeline(transformer, log_reg)
    # pipeline_rfc = make_pipeline(transformer, rfc)
    
    score_lda = cross_val_score(pipeline_lda, X=X, y=y, cv=cv, verbose=False)
    # score_logreg = cross_val_score(pipeline_logreg, X=X, y=y, cv=cv, verbose=False)
    # score_rfc = cross_val_score(pipeline_rfc, X=X, y=y, cv=cv, verbose=False)
             
    print("  ", score_lda.mean())      
    pipeline = pipeline_lda.fit(X, y)      
    # print("  ", score_logreg.mean())            
    # print("  ", score_rfc.mean())
    
    return pipeline
                
if __name__ == '__main__':
    
    (X, y) = load_and_process(2)
    
    train_X, text_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline = _train(train_X, train_y)
    
    # pipeline.predict()
    total = 0
    total_subjects = 0
    for subject in RANGE_SUBJECT:
        (X, y) = load_and_process(subject)
        _, text_X, _, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
        
        predicted_y = pipeline.predict(text_X)
        score = np.mean(predicted_y == test_y)
        
        total += score
        total_subjects += 1
        
        print(f"Sub {subject}: {score}")
    
    print(f"Total score: {total / total_subjects}")
    #     print(f"Subject {subject}")
    #     (X, y) = load_and_process(subject)
    #     _train(X, y)