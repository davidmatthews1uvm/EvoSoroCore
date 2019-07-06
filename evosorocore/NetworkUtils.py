import numpy as np


def identity(x):
    return x


def sigmoid(x):
    return 2.0 / (1.0 + np.exp(-x)) - 1.0


def positive_sigmoid(x):
    return (1 + sigmoid(x)) * 0.5


def rescaled_positive_sigmoid(x, x_min=0, x_max=1):
    return (x_max - x_min) * positive_sigmoid(x) + x_min


def inverted_sigmoid(x):
    return sigmoid(x) ** -1


def neg_abs(x):
    return -np.abs(x)


def neg_square(x):
    return -np.square(x)


def sqrt_abs(x):
    return np.sqrt(np.abs(x))


def neg_sqrt_abs(x):
    return -sqrt_abs(x)


def mean_abs(x):
    return np.mean(np.abs(x))


def std_abs(x):
    return np.std(np.abs(x))


def count_positive(x):
    return np.sum(np.greater(x, 0))


def count_negative(x):
    return np.sum(np.less(x, 0))


def normalize(x):
    x -= np.min(x)
    with np.errstate(divide='ignore', invalid='ignore'):
        x /= np.max(x)
    x = np.nan_to_num(x)
    x *= 2
    x -= 1
    return x


def normalize_scaled(x, new_min=0, new_max=1):
    x = np.nan_to_num(x)
    x -= np.min(x)
    with np.errstate(divide='ignore', invalid='ignore'):
        x /= np.max(x)
    x = np.nan_to_num(x)
    x = np.where(abs(x) > 1, 0, x)  # force to range 0, 1. (nan_to_num gives np.inf large values)
    new_spread = (new_max - new_min)
    new_mean = new_min + new_spread / 2
    x *= new_spread
    x += new_mean - new_spread / 2
    return x

def vox_xyz_from_id(idx, size):
    z = idx / (size[0]*size[1])
    y = (idx - z*size[0]*size[1]) / size[0]
    x = idx - z*size[0]*size[1] - y*size[0]
    return x, y, z

