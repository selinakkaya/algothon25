
import numpy as np

##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

# Momentum: It buys assets with strong recent returns and sells underperformers.
# Directionally biased: There’s no volatility or trend filter, it always takes positions.
# Cumulative positions: Each day adds more to currentPos, stacking positions infinitely. This means: if you bought something yesterday and it goes up again, you buy even more. This creates overexposure quickly.
# No volatility filter	You trade every day, even when price movements are noise, not signal.
# No mean reversion / trend filter	You assume price will continue in the same direction after one up day, which isn't reliable without confirmation.
# Commission costs	Every day you rebalance (even slightly), you pay commissions — this eats into profits fast.
# No cap or stop-loss/risk control:If one asset keeps going the wrong way, your position just keeps growing and so do your losses.

# 50 instruments (Stocks/Assets)
nInst = 50
# Tracks current positions for each instrument
currentPos = np.zeros(nInst)

# prcSoFar is a 2D NumPy array of shape [nInst, nTime]
def getMyPosition(prcSoFar):
    # global currentPos so changes persist between calls
    global currentPos

    # Edge case: If there aren’t at least two time points, you can’t calculate returns
    (nins, nt) = prcSoFar.shape
    if (nt < 2):
        return np.zeros(nins)
    
    # Calculates log returns for each instrument between the last two days (how much each stock moved between the last two time points).
    lastRet = np.log(prcSoFar[:, -1] / prcSoFar[:, -2])

    # Normalizes the return vector using Euclidean norm (like a unit vector). This gives relative performance direction across instruments.
    lNorm = np.sqrt(lastRet.dot(lastRet))
    lastRet /= lNorm

    # You assign positions proportional to recent returns. Each position is scaled by 5000 and adjusted for price, so you're equalising dollar exposure.
    # This is momentum-style trading, buy what’s going up, short what’s going down.
    rpos = np.array([int(x) for x in 5000 * lastRet / prcSoFar[:, -1]])

    # Positions are cumulative over time. You keep adding to winning positions each round which could cause high turnover or oversizing.
    currentPos = np.array([int(x) for x in currentPos+rpos])
    return currentPos
