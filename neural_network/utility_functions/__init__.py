import numpy as np
# from numba import jit


# @jit(nopython=True, parallel=True)
def humps(x: np.ndarray) -> np.ndarray:
    return (1 / ((x - .3) ** 2 + .01)) + (1 / ((x - .9) ** 2 + .04)) - 6


# @jit(nopython=True, parallel=True, fastmath=True)
def fx(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (x ** 2 + y ** 2) * humps(x)


def progress_bar(epoch, max_epoch, max_bar=80):
    portions_precent = epoch / float(max_epoch)

    bar_procced = int(portions_precent * max_bar)

    progress_str = f"{epoch}/{max_epoch} ["
    i = 0
    for _ in range(bar_procced):
        progress_str += "="
        i += 1

    if epoch < max_epoch:
        progress_str += ">"
    else:
        progress_str += "="

    for _ in range(i, max_bar):
        progress_str += " "

    progress_str += "]"
    return progress_str
