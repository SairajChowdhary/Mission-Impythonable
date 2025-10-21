"""Demo runner for the project (CLI-style)"""
import argparse
import pandas as pd
from src.utils import load_credit, load_prices, train_test_split_credit
from src.credit_model import train_credit_model, save_model
from src.sentiment import get_sentiment_score, interpret_sentiment
from src.portfolio_opt import compute_daily_returns, monte_carlo_portfolios, pick_best_by_sharpe, adjust_weights_by_sentiment

def demo_pipeline():
    print('Loading credit data...')
    credit = load_credit()
    X_train, X_val, y_train, y_val = train_test_split_credit(credit)
    print('Training credit model...')
    pipe = train_credit_model(X_train, y_train, X_val, y_val)
    save_model(pipe, 'credit_model.joblib')
    print('Saved credit model to credit_model.joblib')

    print('\nLoading price data...')
    prices = load_prices()
    rets = compute_daily_returns(prices)
    print('Running Monte Carlo portfolio generation...')
    results = monte_carlo_portfolios(rets, num_portfolios=3000)
    best, _ = pick_best_by_sharpe(results)
    print('Best portfolio (sharpe):', best['sharpe'])
    print('Weights:', best['weights'])

    print('\nSentiment demo:')
    text = 'Central bank signals raise worries about growth and volatility.'
    score = get_sentiment_score(text)
    label = interpret_sentiment(score)
    print('Text:', text)
    print('Score:', score, 'Label:', label)
    adj = adjust_weights_by_sentiment(best['weights'], label)
    print('Adjusted weights based on sentiment:', adj)

if __name__ == '__main__':
    import sys
    if '--demo' in sys.argv:
        demo_pipeline()
    else:
        print('Run with --demo to execute the demo pipeline.')
