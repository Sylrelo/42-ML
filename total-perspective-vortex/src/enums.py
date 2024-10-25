
RUN_FISTS = [3, 7, 11]
RUN_IMAGINE_FISTS = [4, 8, 12]

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
