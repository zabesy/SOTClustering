import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import scipy.stats
from PIL import Image
import time
import math
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import ipywidgets as widgets
import plotly.graph_objects as go
from ipywidgets import Layout
from sklearn.metrics.cluster import v_measure_score

from sklearn.manifold import TSNE

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

import matplotlib.pyplot as plt
from nltk.cluster import KMeansClusterer
import nltk
import sklearn
from sklearn import cluster
from sklearn import metrics
import gensim.downloader as api

from scipy.stats import mode
from sklearn.metrics import accuracy_score
import tempfile
from os.path import exists

# Ignore the "FutureWarning" on TSNE.
import warnings

warnings.filterwarnings('ignore')
warnings.filterwarnings(
    action='ignore',
    category=FutureWarning,
    module=r'.*_t_sne'
)


def download_googles_w2v_model(file_name):
    loaded_w2v = api.load('word2vec-google-news-300')
    return loaded_w2v


def load_and_or_save_googles_w2v_model(file_name):
    # Loads and saves model depending on if it already exist.
    if not exists(file_name):
        loaded_w2v = download_googles_w2v_model(file_name)
        # Saves it in current directory
        loaded_w2v.save(file_name)
    return KeyedVectors.load(file_name)


def remove_empty_rows(DF):
    empty_sentences = []
    for index, sentence in enumerate(DF['tokenized_txt']):
        if len(sentence.strip("[]")) == 0:
            empty_sentences.append(index)
    return DF.drop(DF.index[empty_sentences], inplace=True)


def collect_sentences_from_tokenized_txt(sentences, tokenized_txt):
    # Ship those sentences that are empty
    for i, sentence in enumerate(tokenized_txt):
        sentences.append(sentence)
    return sentences


def try_googles_w2v_model(w2v):
    for index, word in enumerate(w2v.index_to_key):
        if index == 10:
            break
        print("word #{}/{} is {}".format(index, len(w2v.index_to_key), word))

    pairs = [
        ('car', 'minivan'),  # a minivan is a kind of car
        ('car', 'bicycle'),  # still a wheeled vehicle
        ('car', 'airplane'),  # ok, no wheels, but still a vehicle
        ('car', 'cereal'),  # ... and so on
        ('car', 'communism'),
    ]
    for w1, w2 in pairs:
        print('%r\t%r\t%.2f' % (w1, w2, w2v.similarity(w1, w2)))

    print(w2v.most_similar(positive=['car', 'minivan'], topn=5))
    print(w2v.doesnt_match(['fire', 'water', 'land', 'sea', 'air', 'car']))


def sentence_vectorizer(sentence, model):
    words = []
    num_words = 0
    for index, w in enumerate(sentence):
        try:
            if num_words == 0:
                words = model[w]
            else:
                words = np.add(words, model[w])
            num_words += 1
        except:
            pass
    return np.asarray(words) / num_words


def apply_kmeans(num_clusters, sentences_vectorized):
    kmeans = cluster.KMeans(n_clusters=num_clusters).fit(sentences_vectorized)
    labels = kmeans.predict(sentences_vectorized)
    centroids = kmeans.cluster_centers_
    return kmeans, labels, centroids
