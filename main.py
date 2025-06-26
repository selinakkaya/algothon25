import numpy as np

def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    n_inst, n_days = prcSoFar.shape
    pos = np.zeros(n_inst)

    short_window = 5
    long_window = 20

    if n_days < long_window:
        return pos  # Not enough data yet

    for i in range(n_inst):
        prices = prcSoFar[i, :]
        short_ma = np.mean(prices[-short_window:])
        long_ma = np.mean(prices[-long_window:])

        if short_ma > long_ma:
            pos[i] = 100  # Long position
        elif short_ma < long_ma:
            pos[i] = -100  # Short position
        else:
            pos[i] = 0  # Flat

    return pos
