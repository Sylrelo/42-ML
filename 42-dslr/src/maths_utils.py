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


def skewness(ndarray: np.ndarray) -> float:
    """Mesure the skewness of the given numeric values.

    Interpret result:
        - {-0.5, 0.5} --- low or approximately symmetric
        - {-1, -0.5} U {0.5, 1} --- moderately skewed
        - {-inf, -1} U {1, +inf} --- highly skewed

    Args:
        ndarray (np.ndarray): array of numeric values to study

    Returns:
        float: skewness
    """
    n = len(ndarray)
    _mean = mean(ndarray)
    _std = std(ndarray)

    a = n / ((n - 1) * (n - 2))
    b = np.sum(((ndarray - _mean) / _std) ** 3)

    return a * b


def kurtosis(ndarray: np.ndarray) -> float:
    """Mesure the kurtosis of the given numeric values.

    Args:
        ndarray (np.ndarray): array of numeric values to study

    Returns:
        float: kurtosis
    """
    n = len(ndarray)
    _mean = mean(ndarray)
    _std = std(ndarray)

    kurt = (1 / n) * np.sum(((ndarray - _mean) / _std) ** 4) - 3
    return kurt
