import sys
import maths_utils
from utils import open_arg_csv
from utils import get_courses_data


def main():
    assert len(sys.argv) == 2, "usage: python describe.py <dataset/path>"
    pcsv = open_arg_csv(sys.argv[1])
    columns_names, data = get_courses_data(pcsv)
    result = []

    result.append(["", *columns_names])
    result.append(["Count", *[data[ind].size for ind, _ in enumerate(columns_names)]])
    result.append(["Mean", *[maths_utils.mean(data[ind]) for ind, _ in enumerate(columns_names)]])
    result.append(["Std", *[maths_utils.std(data[ind]) for ind, _ in enumerate(columns_names)]])
    result.append(["Min", *[maths_utils.min(data[ind]) for ind, _ in enumerate(columns_names)]])
    result.append(["25%", *[maths_utils.percentile(data[ind], 25) for ind, _ in enumerate(columns_names)]])
    result.append(["50%", *[maths_utils.percentile(data[ind], 50) for ind, _ in enumerate(columns_names)]])
    result.append(["75%", *[maths_utils.percentile(data[ind], 75) for ind, _ in enumerate(columns_names)]])
    result.append(["Max", *[maths_utils.max(data[ind]) for ind, _ in enumerate(columns_names)]])
    result.append(["Skewness", *[maths_utils.skewness(data[ind]) for ind, _ in enumerate(columns_names)]])
    result.append(["Kurtosis", *[maths_utils.kurtosis(data[ind]) for ind, _ in enumerate(columns_names)]])

    columns_name_len = [10, *[max(len(v), 15) for v in columns_names]]

    for val in result:
        print("  ".join([format_str(colval).rjust(columns_name_len[ind]) for ind, colval in enumerate(val)]))
    
def format_str(input: str):
    is_float = str(input).replace(".", "").replace("-", "").isnumeric()
    if is_float:
        return "{:0.6f}".format(float(input))
    else:
        return str(input).upper()

if __name__ == "__main__":
    main()
