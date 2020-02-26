import pandas as pd
import swifter
from cleaner import simple_clean
from encoders import encoder_Doc2vec, encoder_tfidf, mean_word2vec, encoder_sbert
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly
import plotly.graph_objects as go
from reducer_dim import reduce_dim

def load():
    df = pd.read_pickle("data/sick.pkl")
    return(df)

def clean(df):
    df = simple_clean(df, "sentence_a", "clean_a")
    df = simple_clean(df, "sentence_b", "clean_b")

def encode(df, params):
    corpus = np.concatenate([[ ' '.join(s) for s in df.clean_b.values], [' '.join(s) for s in df.clean_a.values]], axis=None)
    for param in params:
        print("column", param[1])
        algo = param[0]
        algo.train(corpus)
        df[param[1]+"_a"] = df.clean_a.swifter.apply(lambda x: algo.encode(' '.join(x)))
        df[param[1]+"_b"] = df.clean_b.swifter.apply(lambda x: algo.encode(' '.join(x)))

def find_cosine_similarity(df, params):
    for param in params:
        column = param[1]
        df[column+"_cos_score"] = df.swifter.apply(lambda x: cosine_similarity([x[column+"_a"]], [x[column+"_b"]])[0][0], axis=1)

def plot(df, params):
    for param in params:
        column = param[1]
        fig = go.Figure(data=[go.Scatter(x=df.score.values, y=df[column+"_cos_score"].values, name=column, mode="markers")])
        plotly.offline.plot(fig, filename='html/sick_'+column+'.html')
    

df = load()
clean(df)
params = [(encoder_sbert(), "sbert"), (encoder_Doc2vec(100, 100), "doc2vec" ),
 (mean_word2vec(100, 5, 3, 0), "w2v_cbow"), (mean_word2vec(100, 5, 3, 1), "w2v_skip"),
(encoder_tfidf(), "tfidf")]

encode(df, params)
find_cosine_similarity(df, params)
plot(df, params)





