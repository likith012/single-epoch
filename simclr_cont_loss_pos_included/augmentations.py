from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter
import numpy as np
import copy


def masking(x):
    segments = 50 + int(np.random.rand()*(200 - 50))
    ret = copy.deepcopy(x)
    for i,k in enumerate(x):
        points = np.random.randint(0,3000-segments)
        ret[i,points:points+segments] = 0
    return ret


def noise_channel(ts, mode, degree):
    """
    Add noise to ts

    mode: high, low, both
    degree: degree of noise, compared with range of ts

    Input:
        ts: (n_length)
    Output:
        out_ts: (n_length)

    """
    len_ts = len(ts)
    num_range = np.ptp(ts) + 1e-4  # add a small number for flat signal

    ### high frequency noise
    if mode == "high":
        noise = degree * num_range * (2 * np.random.rand(len_ts) - 1)
        out_ts = ts + noise

    ### low frequency noise
    elif mode == "low":
        noise = degree * num_range * (2 * np.random.rand(len_ts // 100) - 1)
        x_old = np.linspace(0, 1, num=len_ts // 100, endpoint=True)
        x_new = np.linspace(0, 1, num=len_ts, endpoint=True)
        f = interp1d(x_old, noise, kind="linear")
        noise = f(x_new)
        out_ts = ts + noise

    ### both high frequency noise and low frequency noise
    elif mode == "both":
        noise1 = degree * num_range * (2 * np.random.rand(len_ts) - 1)
        noise2 = degree * num_range * (2 * np.random.rand(len_ts // 100) - 1)
        x_old = np.linspace(0, 1, num=len_ts // 100, endpoint=True)
        x_new = np.linspace(0, 1, num=len_ts, endpoint=True)
        f = interp1d(x_old, noise2, kind="linear")
        noise2 = f(x_new)
        out_ts = ts + noise1 + noise2

    else:
        out_ts = ts

    return out_ts


def add_noise(x, ratio):
    """
    Add noise to multiple ts
    Input:
        x: (n_channel, n_length)
    Output:
        x: (n_channel, n_length)
    """
    for i in range(x.shape[0]):

        mode = np.random.choice(["high", "low", "both", "no"])
        x[i, :] = noise_channel(x[i, :], mode=mode, degree=0.05)

    return x

def crop(x):
    n_length = x.shape[1]
    l = np.random.randint(1, n_length - 1)
    x[:, :l], x[:, l:] = x[:, -l:], x[:, :-l]

    return x


def augment(x):
    t = np.random.rand()
    if t > 0.75:
        x = add_noise(x, ratio=0.5)
    elif t > 0.5:
        x = masking(x)
    elif t > 0.25:
        x = crop(x)
    else:
        x = x[[1, 0], :]  # channel flipping
    return x