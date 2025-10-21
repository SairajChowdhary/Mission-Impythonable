"""Sentiment utilities using NLTK VADER"""
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def get_sentiment_score(text, analyzer=None):
    if analyzer is None:
        analyzer = SentimentIntensityAnalyzer()
    res = analyzer.polarity_scores(str(text))
    # return compound score between -1 and 1
    return res['compound']

def interpret_sentiment(score, pos_thresh=0.05, neg_thresh=-0.05):
    if score >= pos_thresh:
        return 'positive'
    elif score <= neg_thresh:
        return 'negative'
    else:
        return 'neutral'
