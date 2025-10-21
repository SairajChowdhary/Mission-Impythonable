"""Monte Carlo based portfolio optimizer and sentiment-driven adjustment"""
import numpy as np
import pandas as pd

def compute_daily_returns(price_df):
    return price_df.pct_change().dropna()

def monte_carlo_portfolios(returns, num_portfolios=5000, risk_free_rate=0.0, seed=42):
    np.random.seed(seed)
    n = returns.shape[1]
    mean_returns = returns.mean()
    cov = returns.cov()
    results = []
    for _ in range(num_portfolios):
        weights = np.random.random(n)
        weights /= weights.sum()
        port_return = np.dot(weights, mean_returns) * 252
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov * 252, weights)))
        sharpe = (port_return - risk_free_rate) / (port_vol + 1e-9)
        results.append({
            'weights': weights,
            'return': port_return,
            'vol': port_vol,
            'sharpe': sharpe
        })
    return results

def pick_best_by_sharpe(results):
    df = pd.DataFrame([{'return': r['return'], 'vol': r['vol'], 'sharpe': r['sharpe']} for r in results])
    idx = df['sharpe'].idxmax()
    best = results[idx]
    return best, df

def adjust_weights_by_sentiment(weights, sentiment_label, multiplier=0.7):
    """Adjust weights towards safer assets when sentiment negative, towards riskier when positive.
    This is a demo strategy. It assumes asset ordering: [STOCK_A, STOCK_B, BOND_C, COMMOD_D]
    where BOND_C is the 'safe' asset."""
    w = weights.copy()
    # identify safe asset index (2)
    safe_idx = 2
    if sentiment_label == 'negative':
        # move a fraction of allocation into bonds
        move = (1 - multiplier) * w.sum()
        # shift from non-safe assets proportionally
        non_safe = [i for i in range(len(w)) if i != safe_idx]
        reduction = (w[non_safe] / w[non_safe].sum()) * (move)
        for i, idx in enumerate(non_safe):
            w[idx] = max(0.0, w[idx] - reduction[i])
        w[safe_idx] += move
    elif sentiment_label == 'positive':
        # reduce bond exposure and distribute to risky assets
        move = (1 - multiplier) * w[safe_idx]
        w[safe_idx] = max(0.0, w[safe_idx] - move)
        non_safe = [i for i in range(len(w)) if i != safe_idx]
        addition = (w[non_safe] / w[non_safe].sum()) * move
        for i, idx in enumerate(non_safe):
            w[idx] += addition[i]
    # normalize
    w = w / w.sum()
    return w
