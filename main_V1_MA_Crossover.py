import numpy as np

# This is a simple Moving Average Crossover strategy.

# Currently Implemented Features:
    # Trend Signal:	Uses a 5-day short MA vs 20-day long MA to detect upward/downward momentum
    # Per-Instrument Logic:	Strategy runs independently per instrument using its price history
    # Position Sizing: Uses fixed position size of ±100 units depending on crossover
    # Early Exit: Returns all-zero if there's insufficient data (i.e., < 20 days)

# To Improve/Add:
    # Volatility Filter: Only take trades if volatility is above a threshold (e.g., StdDev of returns)	Helps reduce noise in low-activity markets
    # Risk Management: Add stop-loss / max drawdown control
    # Position Scaling:	Scale position size based on confidence (e.g., distance between MAs or z-score)

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
