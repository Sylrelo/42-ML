from matplotlib import pyplot as plt
from data_processing import filter_data, get_events, load_and_process, prepare_data
from utils import load_eegbci_data


RANGE_SUBJECT = range(23, 29)

if __name__ == '__main__':
    print("hello")

    for subject in RANGE_SUBJECT:
        print(f"Subject {subject}")
        
        load_and_process(subject)
        # raw_edf = load_eegbci_data(subject)
        
        # data = prepare_data(raw_edf)
        # filetered_data = filter_data(data)
        # get_events(filetered_data)
        
        plt.show()
        exit(1)