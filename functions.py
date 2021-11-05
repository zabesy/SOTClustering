import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import scipy.stats
import webencodings
from PIL import Image
import time
import math
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import ipywidgets as widgets
import plotly.graph_objects as go
from ipywidgets import Layout
from sklearn.metrics.cluster import v_measure_score
import unidecode
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD


from statistics import stdev
from sklearn.decomposition import TruncatedSVD

import umap.umap_ as umap
from wordcloud import STOPWORDS, WordCloud, ImageColorGenerator
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.test.utils import get_tmpfile, datapath

import matplotlib.pyplot as plt
from nltk.cluster import KMeansClusterer
import nltk
from nltk.corpus import stopwords
from nltk.corpus import words
import sklearn
from sklearn import cluster
from sklearn import metrics
import gensim.downloader as api
from gensim.scripts.glove2word2vec import glove2word2vec

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


def load_and_or_save_w2v_model(file_name="glove.840B.300d.txt.word2vec", w2v_type="glove", input_file="glove.840B.300d.txt"):
    assert w2v_type in ["glove", "googles"], "w2v type parameter must be either 'glove' or 'googles'."
    if w2v_type == "glove":
        if exists(file_name):
            return KeyedVectors.load_word2vec_format(file_name, binary=False)
        elif exists(input_file):
            print(f"Generating {file_name}")
            _ = glove2word2vec(input_file, file_name)
            return KeyedVectors.load_word2vec_format(file_name, binary=False)
        else:
            raise AttributeError(f"Neither {file_name} nor {input_file} exists.")
    else:
        # Loads and saves model depending on if it already exist.
        if not exists(file_name):
            loaded_w2v = api.load('word2vec-google-news-300')
            # Saves it in current directory
            loaded_w2v.save(file_name)
        return KeyedVectors.load(file_name)


def collect_sentences_from_tokenized(sentences, tokenized_sentences):
    for i, sentence in enumerate(tokenized_sentences):
        sentences.append(sentence)
    return sentences


def try_w2v_model(w2v):
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
    #Check if list if empty. Basically none of the words in the sentence are found in the model.
    if num_words < 1:
        return None
    return np.asarray(words) / num_words


def apply_kmeans(num_clusters, sentences_vectorized, tolerance=1e-4):
    kmeans = cluster.KMeans(n_clusters=num_clusters, algorithm="full", tol=tolerance).fit(sentences_vectorized)
    # print(kmeans.n_iter_)
    labels = kmeans.predict(sentences_vectorized)
    centroids = kmeans.cluster_centers_
    return kmeans, labels, centroids


# Read and load text from path, normal utf-8 encoded.
def read_and_load_text(file_name):
    file = open(file_name, 'r', encoding='utf-8')
    the_text = file.read()
    file.close()
    return the_text


def english_decontractions(text):
    text = re.sub(r"n\'t\s+", " not ", text)
    text = re.sub(r"\'re\s+", " are ", text)
    text = re.sub(r"\'s\s+", " is ", text)
    text = re.sub(r"\'d\s+", " would ", text)
    text = re.sub(r"\'ll\s+", " will ", text)
    text = re.sub(r"\'t\s+", " not ", text)
    text = re.sub(r"\'ve\s+", " have ", text)
    text = re.sub(r"\'m\s+", " am ", text)
    return text


def check_whitespace_misspells(text, english_words):
    whole_text = []
    for sentence in text.split("."):
        # sentence = "j ust hell o"
        #Should result in "just hello"
        changed = True
        temp_words_1 = sentence.split()
        temp_words_2 = []
        while changed:
            changed = False
            i = 0
            while i < len(temp_words_1):
                if i <= len(temp_words_1) - 2:
                    if (temp_words_1[i] + temp_words_1[i+1]) in english_words:
                        temp_words_2.append(temp_words_1[i] + temp_words_1[i+1])
                        i += 1
                        changed = True
                    else:
                        temp_words_2.append(temp_words_1[i])
                i += 1
            if changed:
                temp_words_1 = temp_words_2[:]
            temp_words_2 = []

        whole_text.append(" ".join(temp_words_1))

    return ".".join(whole_text)


def find_unusual_words(text, english_words):
    #Using googles w2v model
    model_vocab = load_and_or_save_w2v_model("w2v-gensim-model").index_to_key
    model_vocab = load_and_or_save_w2v_model("").index_to_key
    words_orig_text_list = [word for word in (" ".join(text.split("."))).split()]
    words_orig_text_set = set(words_orig_text_list)
    unusual_words = words_orig_text_set - english_words
    unusual_words = sorted(unusual_words)
    unusual_words_not_in_model = set(unusual_words).difference(model_vocab)

    #Write to file the first <int> (or more) most used wrongly spelled words with count.
    num_word_count = dict.fromkeys(unusual_words_not_in_model, 0)
    for index in words_orig_text_list:
        if index not in num_word_count:
            continue
        num_word_count[index] += 1
    sorted_dict = sorted(num_word_count.items(), key=lambda x: x[1], reverse=True)
    write_words_to_file(sorted_dict[:20])

    # print("The unusual words that doesn't exist in the model = {}".format(set(unusual_words).difference(model_vocab)))

def custom_convert_wrongly_spelled_words(text):
    #US SPELLING
    text = re.sub(r'\shonour', ' honor ', text)
    text = re.sub(r'\sfavours', 'favors ', text)
    text = re.sub(r'\shonourable', ' honorable ', text)
    text = re.sub(r'\sneighbourhood', ' neighborhood ', text)
    text = re.sub(r'\scolour', ' color ', text)

    text = re.sub(r'\sappertains', ' pertain ', text)
    text = re.sub(r'\svolitions', ' volition ', text)
    text = re.sub(r'\shighmindedness', ' high mindedness ', text)
    text = re.sub(r'\saffectiones', ' affections ', text)
    text = re.sub(r'\sbetaking', ' be taking ', text)
    text = re.sub(r'\sanalysed', ' analyzed ', text)
    text = re.sub(r'\sgeneralisation', ' generalization ', text)
    text = re.sub(r'\sfavour', ' favor ', text)
    text = re.sub(r'\srecognised', ' recognized ', text)
    text = re.sub(r'\slanguagegame', ' language game ', text)
    text = re.sub(r'\sbservation', ' observation ', text)
    text = re.sub(r'\sbstract', ' abstract ', text)
    text = re.sub(r'\shetween', ' between ', text)
    text = re.sub(r'\snthe\s', 'n the ', text)
    text = re.sub(r'\sntaking\s', 'n taking ', text)

    return text



def remove_unnecessary_phrases_chars(text, stop_words, english_words):
    text = re.sub(r'This\s*page\s*intentionally\s*left\s*blank', ' ', text)
    # Remove numbers conjoined with a dot or comma.
    text = re.sub(r'\d\,\d', ' ', text)
    text = re.sub(r'\d\.\d', ' ', text)
    # Remove headers/titles
    text = re.sub(r'[A-Z]{2,}', ' ', text)
    # To lowercase.
    text = text.lower()
    # Remove atmostrophes.
    text = unidecode.unidecode(text)
    # Remove roman numericals, usually followed by a dot. Need to be done several times because some roman numericals can be xvii.xii
    #And so only the resulting sub will be .xii, but if ran several times, it will result in an empty string.
    int_changed = 1
    while int_changed > 0:
        text, int_changed = re.subn(r'\s((i*v*x*\.)|(i*x*v*\.)|(x*v*i*\.)|(x*i*v*\.)|(v*i*\.)|(l*x*v*i*\.)|(x*l*i*\.)|(f*x*v*i*\.)|(x*i*f*\.))', ' ', text)

    # Convert ?! to dots.
    text = re.sub(r'[\x21\x3f]', '.', text)
    # Replace all non-ascii characters except [ASCII 32-126 and newline] with space.
    text = re.sub(r'[^\x20\x21\x2e\x3f\x61-\x7a]', ' ', text)
    # Remove stopwords.
    text = " ".join([word for word in text.split() if word not in stop_words])

    # Find unusual words (will also write to file the first 20 (or more) most used wrongly spelled words with count.)
    # find_unusual_words(text, english_words)

    # Convert custom wrongly spelled words.
    text = custom_convert_wrongly_spelled_words(text)

    return text


def customized_cleaning(text, stop_words, english_words):
    #Check for unusual words for each text
    text = remove_unnecessary_phrases_chars(text, stop_words, english_words)
    #Fix if misspells with whitespace occurs.
    text = check_whitespace_misspells(text, english_words)

    return text

def write_words_to_file(words):
    with open('words_not_in_w2v.txt','a') as f:
        for word in words:
            f.write("{}\n".format(word))

def text_cleaning(dictionary_list_of_strings):
    stop_words = stopwords.words('english') + custom_stop_words()
    english_words = set(word.lower() for word in nltk.corpus.words.words())
    for school, list_books in dictionary_list_of_strings.items():
        for index, text in enumerate(list_books):
            # print("Number of words before cleaning in text = {}".format(len(text.split())))
            #Unravel decontractions.
            text = english_decontractions(text)
            #Customized text cleaning.
            text = customized_cleaning(text, stop_words, english_words)
            #Remove extra whitespace
            text = " ".join(text.split())
            #Reassign the text to the dict
            dictionary_list_of_strings[school][index] = text
            # print("Number of words after cleaning in text = {}".format(len(text.split())))
            # print("First 500 words: {}".format(text.split()[:500]))

    return dictionary_list_of_strings

def custom_stop_words():
    return ["another",
            "else",
            "made",
            "without",
            "shall",
            "still",
            "made",
            "also",
            "may",
            "sein",
            "b",
            "da",
            "p",
            "c",
            "e",
            "n",
            "therefore",
            "cannot",
            "yet",
            "upon",
            "mean",
            "one",
            "might",
            "g",
            "x",
            "e",
            "f",
            "l",
            "ie",
            "however",
            "cf",
            "also",
            "would",
            "even",
            "way",
            "since",
            "thus",
            "two",
            "say",
            "said",
            "thing",
            "things",
            "must",
            "us",
            "something",
            "first",
            "every",
            "part",
            "much",
            "well",
            "make",
            "like",
            "could",
            "given",
            "though",
            "certain",
            "com",
            "whether",
            "let",
            "either",
            "e",
            "merely",
            "iv",
            "instead",
            "xiv",
            "xi",
            "i.e.",
            "i.e"]

def generate_dataframe(sot_text_dict_list_cleaned):
    books_titles = ["ethics",
                    "improvment of the understanding",
                    "theodicy: essays on the goodness of god, the freedom of man, and the origin of evil",
                    "discourse on the method",
                    "meditations on first philosophy",
                    "malebranche: the search after truth",
                    "the analysis of mind",
                    "the problems of philosophy",
                    "philosophical studies",
                    "philosophical investigations",
                    "tractatus logico philosophicus",
                    "philosophical papers volume 1",
                    "philosophical papers volume 2",
                    "quintessence",
                    "the logic of scientific discovery",
                    "naming and necessity",
                    "philosophical troubles",
                    "second treatise on government",
                    "an essay concerning human understanding 1",
                    "an essay concerning human understanding 2",
                    "a treatise of human nature",
                    "dialogues concerning natural religion",
                    "three dialogues between gylas and philonous",
                    "a treatise concerning the principles of human knowledge",
                    "phenomenology of perception",
                    "the crisis of european sciences and transcendental",
                    "phenomenology",
                    "being and time",
                    "off the beaten track",
                    "the wealth of nations",
                    "on the principles of political economy and taxation",
                    "the general theory of employment, interest and money",
                    "the birth of the clinic",
                    "history of madness",
                    "the order of things",
                    "writing and difference",
                    "difference and repetition",
                    "anti-oedipus: capitalism and schizophrenia",
                    "plato: complete works",
                    "the complete works of aristotle vol 1",
                    "the complete works of aristotle vol 2",
                    "critique of practical reason",
                    "critique of judgment",
                    "critique of pure reason",
                    "the system of ethics",
                    "science of logic",
                    "phenomenology of spirit",
                    "elements of the philosophy of right",
                    "das kapital",
                    "the communist manifesto",
                    "essential works"]

    books_authors = ["Baruch Spinoza",
                     "Baruch Spinoza",
                     "Gottfried Wilhelm Leibniz",
                     "René Descartes",
                     "René Descartes",
                     "Nicolas Malebranche",
                     "Bertrand Russell",
                     "Bertrand Russell",
                     "George Edward Moore",
                     "Ludwig Wittgenstein",
                     "Ludwig Wittgenstein",
                     "David Kellogg Lewis",
                     "David Kellogg Lewis",
                     "Willard Van Orman Quine",
                     "Karl Popper",
                     "Saul Kripke",
                     "Saul Kripke",
                     "John Locke",
                     "John Locke",
                     "John Locke",
                     "David Hume",
                     "David Hume",
                     "George Berkeley",
                     "George Berkeley",
                     "Maurice Merleau-Ponty",
                     "Edmund Husserl",
                     "Edmund Husserl",
                     "Martin Heidegger",
                     "Martin Heidegger",
                     "Adam Smith",
                     "David Ricardo",
                     "John Maynard Keynes",
                     "Michel Foucault",
                     "Michel Foucault",
                     "Michel Foucault",
                     "Jacques Derrida",
                     "Gilles Deleuze",
                     "Gilles Deleuze",
                     "Plato",
                     "Aristotle",
                     "Aristotle",
                     "Immanuel Kant",
                     "Immanuel Kant",
                     "Immanuel Kant",
                     "Johann Gottlieb Fichte",
                     "Georg Wilhelm Friedrich Hegel",
                     "Georg Wilhelm Friedrich Hegel",
                     "Georg Wilhelm Friedrich Hegel",
                     "Karl Marx",
                     "Karl Marx",
                     "Vladimir Lenin"]

    books_sot = ["rationalism",
                 "rationalism",
                 "rationalism",
                 "rationalism",
                 "rationalism",
                 "rationalism",
                 "analytic",
                 "analytic",
                 "analytic",
                 "analytic",
                 "analytic",
                 "analytic",
                 "analytic",
                 "analytic",
                 "analytic",
                 "analytic",
                 "analytic",
                 "empiricism",
                 "empiricism",
                 "empiricism",
                 "empiricism",
                 "empiricism",
                 "empiricism",
                 "empiricism",
                 "phenomenology",
                 "phenomenology",
                 "phenomenology",
                 "phenomenology",
                 "phenomenology",
                 "capitalism",
                 "capitalism",
                 "capitalism",
                 "continental",
                 "continental",
                 "continental",
                 "continental",
                 "continental",
                 "continental",
                 "plato",
                 "aristotle",
                 "aristotle",
                 "german_idealism",
                 "german_idealism",
                 "german_idealism",
                 "german_idealism",
                 "german_idealism",
                 "german_idealism",
                 "german_idealism",
                 "communism",
                 "communism",
                 "communism"]

    dataframe_sot = pd.DataFrame(columns=['school', 'author', 'title', 'tokenized_sentence'])

    count = 0
    for school, list_text in sot_text_dict_list_cleaned.items():
        for text in list_text:
            temp_list = []
            for sentence in text.split("."):
                sentence_dict = {'school': books_sot[count],
                                 'author': books_authors[count],
                                 'title': books_titles[count],
                                 'tokenized_sentence': sentence.split()}

                temp_list.append(sentence_dict)
            dataframe_sot = dataframe_sot.append(temp_list, ignore_index=True)
            count += 1

    return dataframe_sot


def scatter_plot(centroids, labels, values, type='tsne', n_components=2, n_jobs=-1):
    if type == 'tsne':
        model = TSNE(n_components=n_components, n_jobs=n_jobs)
    elif type == 'umap':
        model = umap.UMAP(n_components=2, n_jobs=n_jobs)
    else:
        model = PCA(n_components=2)

    start = time.time()

    np.set_printoptions(suppress=True)

    unique_labels_list = list(set(labels))
    Y = model.fit_transform(values)

    tsne_time = time.time() - start
    print(f"TIME: {tsne_time}")

    scatter = plt.scatter(Y[:, 0], Y[:, 1], c=labels, s=5, cmap='tab20')
    plt.legend(handles=scatter.legend_elements()[0], labels=unique_labels_list, prop={"size": 5})

    # If we want to have centroids join the party aswell.
    CENTER=model.fit_transform(centroids)
    plt.scatter(CENTER[:,0], CENTER[:,1], c='black', s=10)
    for j in range(len(centroids)):
        plt.annotate(unique_labels_list[j], xy=(CENTER[j][0], CENTER[j][1]), xytext=(0,0), textcoords='offset points')

    plt.show()
    pass

def cluster_n_assigned(DF, name, labels, schools):
    amount_SOT_got_clustered_dict = dict()
    for i, school in enumerate(DF["school"]):
        if amount_SOT_got_clustered_dict.get(school) is None:
            amount_SOT_got_clustered_dict[school] = dict.fromkeys([i for i in range(len(schools))], 0)
        amount_SOT_got_clustered_dict[school][labels[i]] += 1

    print(amount_SOT_got_clustered_dict)

    cluster_n_assigned_to_SOT = pd.DataFrame(amount_SOT_got_clustered_dict)
    cluster_n_assigned_to_SOT.plot(kind="bar", stacked=True)
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    plt.title(f"{name} Total cluster assignment of School of Thought")
    plt.legend(prop={"size": 5})
    plt.show()

    count = 0
    for school, list_tuples in amount_SOT_got_clustered_dict.items():
        plt.subplot(1, 2, (count%2)+1)
        plt.bar(range(len(list_tuples)), [value[1] for value in list_tuples.items()], align='center')
        plt.xticks(range(len(list_tuples)), [value[0] for value in list_tuples.items()])
        plt.xlabel("Cluster")
        plt.ylabel("Count")
        plt.title(f"{school}")
        if (count+1) % 2 == 0:
            plt.show()
        count += 1

def average_v_measure(orig_labels, num_clusters, DF, min_df_range ,n , max_df_range, bigram_range=(1,2), trigram_range=(1,3)):
    exceptions_dict = []
    methods = ["tf_idf", "tf_idf_bigram", "tf_idf_trigram", "bigram", "trigram"]
    v_measure_dict = {val: [] for val in methods}
    iteration = 0
    for min_df in np.arange(min_df_range[0],min_df_range[1], min_df_range[2]):
        for max_df in np.arange(max_df_range[0],max_df_range[1],max_df_range[2]):
            iteration+=1
            try:
                tf_idf = TfidfVectorizer(min_df=min_df, max_df=max_df,
                                         stop_words='english').fit_transform(DF["string_sentence"])
                tf_idf_bigram = TfidfVectorizer(min_df=min_df, max_df=max_df,
                                                stop_words='english', ngram_range=bigram_range).fit_transform(
                    DF["string_sentence"])
                tf_idf_trigram = TfidfVectorizer(min_df=min_df, max_df=max_df,
                                                 stop_words='english', ngram_range=trigram_range).fit_transform(
                    DF["string_sentence"])
                bigram = CountVectorizer(min_df=min_df, max_df=max_df, stop_words='english',
                                         ngram_range=bigram_range).fit_transform(DF["string_sentence"])
                trigram = CountVectorizer(min_df=min_df, max_df=max_df, stop_words='english',
                                          ngram_range=trigram_range).fit_transform(DF["string_sentence"])

                tf_idf = TruncatedSVD(n_components=100).fit_transform(tf_idf)
                tf_idf_bigram = TruncatedSVD(n_components=100).fit_transform(tf_idf_bigram)
                tf_idf_trigram = TruncatedSVD(n_components=100).fit_transform(tf_idf_trigram)
                bigram = TruncatedSVD(n_components=100).fit_transform(bigram)
                trigram = TruncatedSVD(n_components=100).fit_transform(trigram)

                tf_idf_kmeans, tf_idf_labels, tf_idf_centroids = apply_kmeans(num_clusters, tf_idf)
                tf_idf_bigram_kmeans, tf_idf_bigram_labels, tf_idf_bigram_centroids = apply_kmeans(num_clusters,
                                                                                                   tf_idf_bigram)
                tf_idf_trigram_kmeans, tf_idf_trigram_labels, tf_idf_trigram_centroids = apply_kmeans(num_clusters,
                                                                                                      tf_idf_trigram)
                bigram_kmeans, bigram_labels, bigram_centroids = apply_kmeans(num_clusters, bigram)
                trigram_kmeans, trigram_labels, trigram_centroids = apply_kmeans(num_clusters, trigram)

                tf_idf_v_measure = v_measure_score(orig_labels, tf_idf_labels)
                tf_idf_bigram_v_measure = v_measure_score(orig_labels, tf_idf_bigram_labels)
                tf_idf_trigram_v_measure = v_measure_score(orig_labels, tf_idf_trigram_labels)
                bigram_v_measure = v_measure_score(orig_labels, bigram_labels)
                trigram_v_measure = v_measure_score(orig_labels, trigram_labels)

                tf_idf_v_measure_dict = {"v_measure": tf_idf_v_measure, "min_df": min_df,
                                         "max_df": max_df}
                tf_idf_bigram_v_measure_dict = {"v_measure": tf_idf_bigram_v_measure, "min_df": min_df,
                                                "max_df": max_df}
                tf_idf_trigram_v_measure_dict = {"v_measure": tf_idf_trigram_v_measure, "min_df": min_df,
                                                 "max_df": max_df}
                bigram_v_measure_dict = {"v_measure": bigram_v_measure, "min_df": min_df / min_df_range,
                                         "max_df": max_df}
                trigram_v_measure_dict = {"v_measure": trigram_v_measure, "min_df": min_df,
                                          "max_df": max_df}

                v_measure_dict["tf_idf"].append(tf_idf_v_measure_dict)
                v_measure_dict["tf_idf_bigram"].append(tf_idf_bigram_v_measure_dict)
                v_measure_dict["tf_idf_trigram"].append(tf_idf_trigram_v_measure_dict)
                v_measure_dict["bigram"].append(bigram_v_measure_dict)
                v_measure_dict["trigram"].append(trigram_v_measure_dict)
            except:
                exception = {"min_df": min_df, "max_df": max_df}
                exceptions_dict.append(exception)
                pass
            print("Iteration: " + str(iteration) + " / " + str(n), end='\r')
    return v_measure_dict, exceptions_dict



