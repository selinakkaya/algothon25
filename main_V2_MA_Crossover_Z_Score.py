import numpy as np

# This is a modified version of the Moving Average Crossover strategy with added volatility adjustment.

# Comparison to v1:

# [✓] Still uses MA crossover: 5-day short MA vs 20-day long MA to detect trend
# [✗] Replaces fixed ±100 position sizing from v1
# [✓] Still runs independently per instrument (loop over all 50 assets)
# [✓] Still exits early if < 20 days of price data
# [+] New: Adds volatility adjustment using 20-day rolling standard deviation
# [+] New: Computes z-score = (short MA - long MA) / std
# [+] New: Scales position size proportionally to z-score, capped at ±100
# [-] Performance currently worse than v1 (Sharpe ↓, Score ↓); possibly too reactive or overfitting

def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    n_inst, n_days = prcSoFar.shape
    pos = np.zeros(n_inst)

    short_window = 5
    long_window = 20
    std_window = 20
    max_position = 100

    if n_days < long_window:
        return pos  # Not enough data yet

    for i in range(n_inst):
        prices = prcSoFar[i, :]

        short_ma = np.mean(prices[-short_window:])
        long_ma = np.mean(prices[-long_window:])
        std = np.std(prices[-std_window:])

        if std == 0:  # Avoid divide-by-zero
            continue

        z_score = (short_ma - long_ma) / std

        # Scale position proportionally to z-score
        scaled_pos = np.clip(z_score * 50, -max_position, max_position)
        pos[i] = int(scaled_pos)

    return pos

