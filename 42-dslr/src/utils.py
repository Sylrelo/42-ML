import sys
import pandas as pd
import numpy as np
from pathlib import Path
from maths_utils import remove_nan


def open_arg_csv(file) -> pd.DataFrame:
    file_as_path = Path(file)

    if not file_as_path.exists() or not file_as_path.is_file():
        print("File does not exists.")
        sys.exit(1)

    pcsv = pd.read_csv(file)

    return pcsv


def prepare_x(dataset: pd.DataFrame) -> np.ndarray:
    val_x = dataset.iloc[:, 6:]
    val_x = val_x.reindex(sorted(val_x), axis=1)
    val_x.fillna(val_x.mean(), inplace=True)
    val_x = np.array(val_x)

    return val_x


def get_courses_data(csvdata: pd.DataFrame):
    columns_names = csvdata.columns[6:].array;
    data = []
    data_np_raw = np.array(csvdata)

    for entry in enumerate(columns_names):
        data.append(remove_nan(data_np_raw[:, entry[0] + 6]))

    return columns_names, data


def get_color_per_house(house: str):
    if house == "Slytherin":
        return "#366447"
    if house == "Gryffindor":
        return "#a6332e"
    if house == "Hufflepuff":
        return "#efbc2f"
    if house == "Ravenclaw":
        return "#3c4e91"

    return "#000000"
