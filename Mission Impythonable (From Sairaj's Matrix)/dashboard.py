"""A simple Streamlit dashboard to demo components"""
import streamlit as st
import pandas as pd
import numpy as np
from src.utils import load_credit, load_prices
from src.credit_model import train_credit_model, build_pipeline
from src.sentiment import get_sentiment_score, interpret_sentiment
from src.portfolio_opt import compute_daily_returns, monte_carlo_portfolios, pick_best_by_sharpe, adjust_weights_by_sentiment

st.title('Finance + Risk Demo Dashboard')

st.sidebar.header('Controls')
demo = st.sidebar.button('Run Demo Pipeline')

if demo:
    st.header('1) Credit Model (train on sample)')
    credit = load_credit()
    st.dataframe(credit)
    X = credit.drop(columns=['id','defaulted'])
    y = credit['defaulted']
    # quick split (small dataset)
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.3, random_state=42)
    pipe = train_credit_model(X_train, y_train, X_val, y_val)
    st.success('Trained a RandomForest credit classifier (see logs)')

    st.header('2) Sentiment: demo texts')
    sample_texts = [
        'The market rallied strongly today after positive earnings reports.',
        'Economic indicators show a slowdown; investors are worried.',
        'Mixed signals from central bank lead to uncertainty.'
    ]
    for t in sample_texts:
        s = get_sentiment_score(t)
        st.write(t)
        st.write('Score:', s, 'Label:', interpret_sentiment(s))

    st.header('3) Portfolio Optimization demo')
    prices = load_prices()
    st.line_chart(prices)
    rets = compute_daily_returns(prices)
    results = monte_carlo_portfolios(rets, num_portfolios=2000)
    best, df = pick_best_by_sharpe(results)
    st.write('Best Sharpe portfolio (weights):', best['weights'].round(3))
    st.write('Return:', best['return'], 'Vol:', best['vol'], 'Sharpe:', best['sharpe'])

    st.header('4) Adjust portfolio by sentiment')
    sscore = get_sentiment_score('The market rallied strongly today after positive earnings reports.')
    label = interpret_sentiment(sscore)
    st.write('Sentiment label used:', label)
    adj = adjust_weights_by_sentiment(best['weights'], label)
    st.write('Adjusted weights:', adj.round(3))
