import numpy as np

# Mean Reversion Strategy with Trend Filter — Version 3

# Current Performance (Day 750):
    # mean(PL): -7.2
    # return: -0.00141
    # StdDev(PL): 37.48
    # annSharpe(PL): -3.01
    # totDvolume: 1,013,128
    # Score: -10.90

# Strategy Description:
    # This is a Mean Reversion strategy using short/long MA divergence (Z-score) with a trend filter.

# Currently Implemented Features:
    # Mean Reversion Signal: Enters trades when short MA significantly deviates from long MA,
    #                        measured by Z-score ((short - long) / std deviation)
    # Trend Filter: Only activates mean reversion logic when market shows <1% trend strength
    # Position Persistence: Holds positions across days unless Z-score flips or weakens
    # Exit Signal: Closes position if Z-score changes direction or magnitude drops below exit threshold
    # Per-Instrument Logic: Applies independently to each instrument using its price history
    # Position Sizing: Position is scaled linearly with Z-score (bounded to ±100 units)
    # Early Exit: Skips trading if insufficient data (< 20 days) or unstable MA/std values

# To Improve/Add:
    # Volatility Targeting: Downweight position size when std deviation is high to manage risk
    # Holding Duration Limit: Exit after N days even if Z-score persists, to avoid position stagnation
    # Cooldown Period: Prevent immediate re-entry after an exit to reduce overtrading
    # Adaptive Thresholds: Dynamically adjust entry/exit thresholds based on asset volatility regime
    # Risk Management: Add stop-loss, value at risk, or max drawdown controls

# Persistent position and z-score
current_pos = np.zeros(50)  # 50 instruments
previous_z = np.zeros(50)

def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    global current_pos, previous_z

    n_inst, n_days = prcSoFar.shape
    pos = np.copy(current_pos)

    short_window = 5
    long_window = 20
    std_window = 20
    max_position = 100

    entry_z = 1.0
    exit_z = 0.2
    trend_thresh = 0.01  # 1% difference between short and long MA

    if n_days < long_window:
        return np.zeros(n_inst)

    for i in range(n_inst):
        prices = prcSoFar[i, :]
        short_ma = np.mean(prices[-short_window:])
        long_ma = np.mean(prices[-long_window:])
        std = np.std(prices[-std_window:])

        if std == 0 or long_ma == 0:
            continue  # Avoid division by zero

        z = (short_ma - long_ma) / std
        trend_strength = abs(short_ma - long_ma) / long_ma

        # If trend is weak → look for mean reversion signals
        if trend_strength < trend_thresh:
            # Exit if z flips direction or gets close to zero
            if np.sign(z) != np.sign(previous_z[i]) or abs(z) < exit_z:
                pos[i] = 0
            # Entry or continuation if strong enough z-score
            elif abs(z) >= entry_z:
                pos[i] = int(np.clip(-z * 50, -max_position, max_position))
        else:
            pos[i] = 0  # Avoid trading in strong trends

        previous_z[i] = z  # Update z-score for next day

    current_pos = pos
    return pos
