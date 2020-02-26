###This file represents all encoders used in this projet,
## An encoder uses 2 functions, encode: transform e string into encoded vector
## Train: train the model


## For the moment: tfidf, Doc2vec, skip-thoughts, Quick-thoughts, Word2Vec moyenne, Sbert

from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np 
import re
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from gensim.models import Word2Vec
from gensim.test.utils import common_texts

class encoder_tfidf:
    def train(self, corpus):
        print("[SYSTEME] train tfidf")
        vocabulary = np.unique(word_tokenize(' '.join(corpus)))
        self.pipe = Pipeline([('count', CountVectorizer(vocabulary=vocabulary)), ('tfid',TfidfTransformer())]).fit(corpus)

    def encode(self, text):
        return(self.pipe.transform([text]).toarray()[0])

class encoder_Doc2vec:
    def __init__(self, vec_size, max_epochs):
        self.model = Doc2Vec(vector_size=vec_size, batch_words=50)
        self.max_epochs = max_epochs

    def train(self, corpus):
        print("[SYSTEME] train d2v")
        tagged_data = [TaggedDocument(words=word_tokenize(str(_d).lower()), tags=[str(i)]) for i, _d in enumerate(corpus)]
        self.model.build_vocab(tagged_data)
        for _ in range(self.max_epochs):
            self.model.train(tagged_data,
                  total_examples=self.model.corpus_count,
                  epochs=self.model.epochs)
            self.model.alpha -= 0.0002
            self.model.min_alpha = self.model.alpha
        
    def encode(self, text):
        return(self.model.infer_vector([text]))

class mean_word2vec:
    def __init__(self, output_size, window, workers, sg):
        self.output_size = output_size
        self.window = window
        self.workers = workers
        self.sg =sg

    def train(self, corpus):
        print("[SYSTEME] train w2v")
        corpus = [word_tokenize(sentence) for sentence in corpus]
        self.model = Word2Vec(corpus, size=self.output_size, window=self.window, min_count=1, workers=self.workers, sg=self.sg)

    def encode(self, text):
        vectors = []
        for word in word_tokenize(text):
            try:
                vectors.append(self.model.wv[word])
            except: 
                ()
        return(np.mean(vectors, axis=0))
        
#class quick_thoughts:

class encoder_sbert:
    def train(self, corpus):
        print("[SYSTEME] train sbert")
        self.model = SentenceTransformer('bert-base-nli-mean-tokens')
        
    def encode(self, text):
        return(self.model.encode([text])[0])
        

 


