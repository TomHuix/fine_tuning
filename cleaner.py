import swifter
import pandas as pd
import string
import functools
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
import timeit
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize 
import string
from sklearn.feature_extraction.text import CountVectorizer
import functools
import operator
import plotly.express as px
foldl = lambda func, acc, xs: functools.reduce(func, xs, acc)
import matplotlib.pyplot as plt
import timeit
import spacy
import json
import pickle
import plotly
   
nlp = spacy.load('en_core_web_lg')
foldl = lambda func, acc, xs: functools.reduce(func, xs, acc)
stopwords = set(stopwords.words('english'))

def lowercasing(df, output_column):
    df[output_column] = df[output_column].swifter.apply(lambda x : x.lower())
    
def replace_values(df, output_column):
    df[output_column] = df[output_column].swifter.apply(lambda x : foldl(lambda a,b: re.sub(b[0], b[1], a), x, [("( .*@.* )", " xxemail "), 
        (r"(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})", "xxphone"),
        ("([1-9] |1[0-9]| 2[0-9]|3[0-1])(.|-)([1-9] |1[0-2])(.|-|)20[0-9][0-9]", "xxdate"),
        ("[0-9]", "")]))

def clean_header(df, output_column):
    df[output_column] = df[output_column].swifter.apply(lambda x : foldl(lambda a,b: re.sub(b, "", a), x, ["From: .*\n", "Subject: .*\n",
               "Organization: .*\n", "Lines: .*\n", "To: .*\n", "NNTP-Posting-Host: .*\n", "Nntp-Posting-Host: .*\n",
              "Article-I.D.: .*\n", "Reply-To: .*\n"]))

def replace_syn(corpus, dic):
    result = []
    for text in corpus:
        current_text = []
        tokens = word_tokenize(text)
        for token in tokens:
            if token not in stopwords:
                syn  = dic.get(token, -1)
                if syn == -1: #synonym is not in the dictionnary
                    try:
                        for syn in wordnet.synsets(token)[0].lemma_names():
                            dic[syn] = token
                        dic[token] = token
                        current_text.append(word)
                    except: #token does not exist on wordnet
                        ()
                else:
                    current_text.append(syn)
        result.append(current_text)
    pickle.dump(dic, open("dict_word_cleaner.pkl","wb"))
    return(result)
    
def clean_syn(df, output_column):
    try:
        dic = pickle.load(open( "dict_word_cleaner.pkl", "rb" ))
    except:
        dic = {}
    df[output_column] = replace_syn(df.column.values, dic)


def replace(word, word_type):
    if word_type == "CARDINAL":
        return("xxnumber")
    if word_type == 'DATE':
        return("xxdate")
    elif word_type == "QUANTITY":
        return("xxquantity")
    elif word_type == "TIME":
        return("xxtime")
    elif word_type == "PERCENT":
        return("xxpercent")
    elif word_type == "MONEY":
        return("xxmoney")
    elif word_type == "PERSON":
        return("xxperson")
    elif word_type == "ORG":
        return("xxorg")
    else:
        return(word)
            
def clean_words(df, output_column):
    df[output_column] = df[output_column].swifter.apply(lambda x : ' '.join([str(replace(w, w.ent_type_)) for w in nlp(x)]))
    


def remove_punctuation(df, output_column):
    df[output_column] = df[output_column].swifter.apply(lambda x: foldl(lambda a,b: a.replace(b, ""), x, string.punctuation))
        
def remove_sw(df, output_column):
    df[output_column] = df[output_column].swifter.apply(lambda x : [w for w in word_tokenize(x) if w not in stopwords])
    
def remove_number(df, output_column):
    df[output_column] = df[output_column].swifter.apply(lambda x : re.sub('[0-9]', '', x))

def simple_clean(df, column_to_clean, output_column):
    df[output_column] = df[column_to_clean]
    #remove_header(df, output_column)
    lowercasing(df, output_column)
    remove_punctuation(df, output_column)
    remove_number(df, output_column)
    remove_sw(df, output_column)
    return(df)    

def medium_clean(df, column_to_clean, output_column):
    df[output_column] = df[column_to_clean]
    #remove_header(df, output_column)
    clean_header(df, column_to_clean)
    lowercasing(df, output_column)
    replace_values(df, output_column)
    remove_punctuation(df, output_column)
    clean_syn(df, output_column)
    return(df)


def complet_clean(df, column_to_clean, output_column):
    df[output_column] = df[column_to_clean]
    #remove_header(df, output_column)
    clean_words(df, output_column)
    lowercasing(df, output_column)
    replace_values(df, output_column)
    remove_punctuation(df, output_column)
    clean_syn(df, output_column)     
    return(df)
 

