
RUN_FISTS = [3, 7, 11]
RUN_IMAGINE_FISTS = [4, 8, 12]

RUN_FISTS_AND_FEET = [5, 9, 13]
RUN_IMAGINE_FISTS_AND_FEET = [6, 10, 14]

RUN_BASELINE_EYE_OPEN = [1]
RUN_BASELINE_EYE_CLOSED = [2]


EXPERIMENT = {
    0: RUN_FISTS + RUN_IMAGINE_FISTS + RUN_FISTS_AND_FEET + RUN_IMAGINE_FISTS_AND_FEET,

    1: RUN_FISTS,
    2: RUN_IMAGINE_FISTS,

    3: RUN_FISTS_AND_FEET,
    4: RUN_IMAGINE_FISTS_AND_FEET,

    5: RUN_FISTS + RUN_IMAGINE_FISTS,
    6: RUN_FISTS_AND_FEET + RUN_IMAGINE_FISTS_AND_FEET,

    # 7: RUN_BASELINE_EYE_OPEN,
    # 8: RUN_BASELINE_EYE_CLOSED,
}

RUNS = {
    "BaselineEyesOpen": [1],
    "BaselineEyesClosed": [2],
    
    "OpenCloseFist": [3, 7, 11],
    "OpenCloseBothFistsAndFeet": [5, 9, 13],
    
    "ImagineOpenCloseFist": [4, 8, 12],
    "ImagineOpenCloseBothFistsAndFeet": [6, 10, 14],
    
    "OpenAndImagineOpenCloseBothFistsAndFeet": [5, 9, 13, 6, 10, 14],
    "OpenAndImagineOpenCloseFist": [3, 7, 11, 4, 8, 12],
}

EXPERIMENTS = {
    0: RUNS["BaselineEyesOpen"],
    1: RUNS["BaselineEyesClosed"],
    
    2: RUNS["ImagineOpenCloseFist"],
    3: RUNS["ImagineOpenCloseBothFistsAndFeet"],
    
    4: RUNS["OpenCloseFist"],
    5: RUNS["OpenCloseBothFistsAndFeet"],
    
    6: RUNS["OpenAndImagineOpenCloseFist"],
    7: RUNS["OpenAndImagineOpenCloseBothFistsAndFeet"],
}



BASE_DIRECTORY = "/media/slopez/Aled"
DATA_DIRECTORY = f"{BASE_DIRECTORY}/_data"
EEGBCI_DIRECTORY = f"{BASE_DIRECTORY}/eegbci"