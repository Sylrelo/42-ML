import numpy as np
import math


def sum(ndarray: np.ndarray):
    total_sum = 0
    [total_sum := total_sum + x for x in ndarray]
    return total_sum


def mean(ndarray: np.ndarray):
    return sum(ndarray) / len(ndarray)


def std(ndarray: np.ndarray):
    mean_val = mean(ndarray)
    sum_deviation = [(x - mean_val) ** 2 for x in ndarray]
    return math.sqrt(sum(sum_deviation) / len(ndarray))


def percentile(ndarray: np.ndarray, centille: float):
    ndarray.sort()

    rank = (len(ndarray) - 1) * (centille / 100)

    fid = np.floor(rank)
    cid = np.ceil(rank)

    if fid == cid:
        return ndarray[int(rank)]

    d0 = ndarray[int(fid)] * (cid - rank)
    d1 = ndarray[int(cid)] * (rank - fid)

    return d0 + d1


def remove_nan(ndarray: np.ndarray):
    return np.array([value for value in ndarray if not math.isnan(value)])


def min(ndarray: np.ndarray):
    curr_min = float('inf')

    for value in ndarray:
        curr_min = value if value < curr_min else curr_min

    return curr_min


def max(ndarray: np.ndarray):
    curr_max = -float('inf')

    for value in ndarray:
        curr_max = value if value > curr_max else curr_max

    return curr_max
