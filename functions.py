import numpy as np
import re

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import time
import umap.umap_ as umap
import unidecode
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn import cluster
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.cluster import v_measure_score
from wordcloud import STOPWORDS, WordCloud, ImageColorGenerator
from statistics import stdev
import seaborn as sns

# Ignore the "FutureWarning" on TSNE.
import warnings

warnings.filterwarnings('ignore')
warnings.filterwarnings(
    action='ignore',
    category=FutureWarning,
    module=r'.*_t_sne'
)


# def load_and_or_save_w2v_model(file_name="glove.trained.txt", w2v_type="glove", input_file="glove.840B.300d.txt"):
#     assert w2v_type in ["glove", "googles"], "w2v type parameter must be either 'glove' or 'googles'."
#     if w2v_type == "glove":
#         if exists(file_name):
#             return KeyedVectors.load_word2vec_format(file_name, binary=False)
#         elif exists(input_file):
#             print(f"Generating {file_name}")
#             _ = glove2word2vec(input_file, file_name)
#             return KeyedVectors.load_word2vec_format(file_name, binary=False)
#         else:
#             raise AttributeError(f"Neither {file_name} nor {input_file} exists.")
#     else:
#         # Loads and saves model depending on if it already exist.
#         if not exists(file_name):
#             loaded_w2v = api.load('word2vec-google-news-300')
#             # Saves it in current directory
#             loaded_w2v.save(file_name)
#         return KeyedVectors.load(file_name)


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
    # Check if list if empty, if none of the words in the sentence are found in the model.
    if num_words < 1:
        return None
    return np.asarray(words) / num_words


def apply_kmeans(num_clusters, sentences_vectorized, tolerance=1e-4):
    kmeans = cluster.KMeans(n_clusters=num_clusters, algorithm="full", tol=tolerance, n_init=50).fit(sentences_vectorized)
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
        # Should result in "just hello"
        changed = True
        temp_words_1 = sentence.split()
        temp_words_2 = []
        while changed:
            changed = False
            i = 0
            while i < len(temp_words_1):
                if i <= len(temp_words_1) - 2:
                    if (temp_words_1[i] + temp_words_1[i + 1]) in english_words:
                        temp_words_2.append(temp_words_1[i] + temp_words_1[i + 1])
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


# def find_unusual_words(text, english_words):
#     #Using googles w2v model
#     model_vocab = load_and_or_save_w2v_model("w2v-gensim-model").index_to_key
#     model_vocab = load_and_or_save_w2v_model("").index_to_key
#     words_orig_text_list = [word for word in (" ".join(text.split("."))).split()]
#     words_orig_text_set = set(words_orig_text_list)
#     unusual_words = words_orig_text_set - english_words
#     unusual_words = sorted(unusual_words)
#     unusual_words_not_in_model = set(unusual_words).difference(model_vocab)
#
#     #Write to file the first <int> (or more) most used wrongly spelled words with count.
#     num_word_count = dict.fromkeys(unusual_words_not_in_model, 0)
#     for index in words_orig_text_list:
#         if index not in num_word_count:
#             continue
#         num_word_count[index] += 1
#     sorted_dict = sorted(num_word_count.items(), key=lambda x: x[1], reverse=True)
#     write_words_to_file(sorted_dict[:20])
#
#     # print("The unusual words that doesn't exist in the model = {}".format(set(unusual_words).difference(model_vocab)))

def custom_convert_wrongly_spelled_words(text):
    text = re.sub(r'\bappertains', ' pertain ', text)
    text = re.sub(r'\bvolitions', ' volition ', text)
    text = re.sub(r'\bhighmindedness', ' high mindedness ', text)
    text = re.sub(r'\baffectiones', ' affections ', text)
    text = re.sub(r'\bbetaking', ' be taking ', text)
    text = re.sub(r'\banalysed', ' analyzed ', text)
    text = re.sub(r'\bgeneralisation', ' generalization ', text)
    text = re.sub(r'\bfavour', ' favor ', text)
    text = re.sub(r'\brecognised', ' recognized ', text)
    text = re.sub(r'\blanguagegame', ' language game ', text)
    text = re.sub(r'\bbservation', ' observation ', text)
    text = re.sub(r'\bbstract', ' abstract ', text)
    text = re.sub(r'\bhetween', ' between ', text)
    text = re.sub(r'\bnthe', 'n the ', text)
    text = re.sub(r'\bntaking', 'n taking ', text)
    text = re.sub(r'\bathing', ' a thing ', text)
    text = re.sub(r'\btobe', ' to be ', text)
    text = re.sub(r'\binsofar', ' in so far ', text)

    return text


def remove_unnecessary_phrases_chars(text, stop_words, english_words):
    text = re.sub(r"This\s*page\s*intentionally\s*left\s*blank", " ", text)
    text = re.sub(r"Translator’s\s*Preface", ' ', text)
    # Remove numbers conjoined with a dot or comma.
    text = re.sub(r'\d\,\d', ' ', text)
    text = re.sub(r'\d\.\d', ' ', text)
    # Remove headers/titles
    text = re.sub(r'[A-Z]{2,}', ' ', text)
    # To lowercase.
    text = text.lower()
    # Remove atmostrophes.
    text = unidecode.unidecode(text)
    # Need to be done before roman numericals.
    text.replace('i.e.', ' ').replace('i.e', ' ')
    # Remove roman numericals.
    text, int_changed = re.subn(r'\b[ivx]+\.', ' ', text)

    # Convert ?! to dots.
    text = re.sub(r'[\x21\x3f]', '.', text)
    # Replace all non-ascii characters except [ASCII 32-126 and newline] with space.
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'[^\x20\x21\x2e\x3f\x61-\x7a]', ' ', text)

    # Find unusual words (will also write to file the first 20 (or more) most used wrongly spelled words with count.)
    # find_unusual_words(text, english_words)

    # Convert custom wrongly spelled words.
    text = custom_convert_wrongly_spelled_words(text)

    # Fix if misspells with whitespace occurs.
    text = check_whitespace_misspells(text, english_words)

    # Remove 1 character words
    text, num_removals = re.subn(r"\b[a-z]{1,1}\b", " ", text)

    # Remove stop_words
    text = ".".join([" ".join([word for word in sentence.split() if word not in stop_words]) for sentence in text.split(".")])

    return text

def customized_cleaning(text, stop_words, english_words):
    # Check for unusual words for each text
    text = remove_unnecessary_phrases_chars(text, stop_words, english_words)

    return text

def write_words_to_file(words):
    with open('words_not_in_w2v.txt', 'a') as f:
        for word in words:
            f.write("{}\n".format(word))


def text_cleaning(dictionary_list_of_strings):
    stop_words = stopwords.words('english') + custom_stop_words()
    english_words = set(word.lower() for word in list(nltk.corpus.brown.words()))
    for school, list_books in dictionary_list_of_strings.items():
        for index, text in enumerate(list_books):
            # print("Number of words before cleaning in text = {}".format(len(text.split())))
            # Unravel decontractions.
            text = english_decontractions(text)
            # Customized text cleaning.
            text = customized_cleaning(text, stop_words, english_words)
            # Remove extra whitespace
            text = " ".join(text.split())
            # Reassign the text to the dict
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
            "da",
            "therefore",
            "cannot",
            "yet",
            "upon",
            "mean",
            "one",
            "might",
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
            "merely",
            "instead",
            "yes",
            ""]


def scatter_plot(labels, values, type='tsne', centroids=None, n_components=2, n_jobs=-1):
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

    # If we want to have centroids join aswell.
    if centroids != None:
        CENTER = model.fit_transform(centroids)
        plt.scatter(CENTER[:, 0], CENTER[:, 1], c='black', s=10)
        for j in range(len(centroids)):
            plt.annotate(unique_labels_list[j], xy=(CENTER[j][0], CENTER[j][1]), xytext=(0, 0),
                         textcoords='offset points')

    plt.show()


def cluster_n_assigned(DF, name, labels, schools):
    amount_SOT_got_clustered_dict = dict()
    for i, school in enumerate(DF["school"]):
        if amount_SOT_got_clustered_dict.get(school) is None:
            amount_SOT_got_clustered_dict[school] = dict.fromkeys([i for i in range(len(schools))], 0)
        amount_SOT_got_clustered_dict[school][labels[i]] += 1

    # print(amount_SOT_got_clustered_dict)

    cluster_n_assigned_to_SOT = pd.DataFrame(amount_SOT_got_clustered_dict)
    cluster_n_assigned_to_SOT.plot(kind="bar", stacked=True)
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    plt.title(f"{name.upper()} total cluster assignment of School of Thought")
    plt.legend(prop={"size": 6})
    plt.show()

    j = 0
    for school, list_tuples in amount_SOT_got_clustered_dict.items():
        plt.subplot(1, 2, (j % 2) + 1)
        plt.bar(range(len(list_tuples)), [value[1] for value in list_tuples.items()], align='center')
        plt.xticks(range(len(list_tuples)), [value[0] for value in list_tuples.items()])
        plt.xlabel("Cluster")
        plt.ylabel("Count")
        plt.title(f"{school.capitalize()}")
        plt.tight_layout()
        if (j + 1) % 2 == 0:
            plt.show()
        j += 1


def vectorize_sentences(w2v, DF):
    indexes_to_remove_from_df = set()
    sentences_vectorized = []
    for index, sentence in enumerate(DF["tokenized_sentence"]):
        temp_variable = sentence_vectorizer(sentence, w2v)
        # Check if return variable is empty
        if temp_variable is None:
            indexes_to_remove_from_df.add(index)
            continue
        sentences_vectorized.append(temp_variable)

    DF.drop(DF.index[list(indexes_to_remove_from_df)], inplace=True)
    DF.reset_index(drop=True, inplace=True)
    return DF, sentences_vectorized


def best_v_measure_w2v(FULL_DF, train=False, train_ratio=0.2, iterations=1, model=1, window=5, epochs=10, vector_size=100):
    v_measure_list = []
    start = time.time()
    for i in range(iterations):

        if train:
            indexes = [i for i in list(FULL_DF.sample(frac=train_ratio).index)]
            DF_TRAIN = FULL_DF.iloc[indexes]
            DF_TRAIN.reset_index(drop=True, inplace=True)
            DF_TEST = FULL_DF.iloc[~FULL_DF.index.isin(indexes)]
            DF_TEST.reset_index(drop=True, inplace=True)
        else:
            DF_TEST = FULL_DF
            DF_TRAIN = FULL_DF

        w2v = Word2Vec(
            DF_TRAIN["tokenized_sentence"].tolist(),
            vector_size=vector_size,
            sg=model,  # 1 for skipgram, otherwise CBOW
            window=window,
            epochs=epochs)

        DF_TEST, w2v_sentences = vectorize_sentences(DF=DF_TEST, w2v=w2v.wv)

        unique_schools = DF_TRAIN["school"].unique().tolist()
        original_labels = [unique_schools.index(school) for school in DF_TEST["school"]]
        num_clusters = len(unique_schools)

        w2v_kmeans, w2v_labels, w2v_centroids = apply_kmeans(num_clusters, w2v_sentences)

        w2v_v_measure = v_measure_score(original_labels, w2v_labels)

        v_measure_list.append(w2v_v_measure)
        print(f"{i + 1} / {iterations} run", end='\r')

    end = time.time()
    print(f"\rAverage Time per Iteration: {(end - start) / iterations}")
    return v_measure_list


def plot_similar_words(model, words, title, top_words):
    similar_words = {word: [word_tuple[0] for word_tuple in model.most_similar([word], topn=top_words)] for word in words}
    words = sum([[k] + v for k, v in similar_words.items()], [])
    words_vec = model[words]

    tsne = TSNE(n_components=2)
    Y = tsne.fit_transform(words_vec)

    plt.figure(figsize=(10, 6))
    plt.scatter(Y[:, 0], Y[:, 1], c='red', edgecolors='r')
    plt.title(title)
    for label, x, y in zip(words, Y[:, 0], Y[:, 1]):
        plt.annotate(label, xy=(x + 1, y + 1), xytext=(0, 0), textcoords='offset points')

def plot_v_measure(v_measure_dict, title):
    df = pd.DataFrame(v_measure_dict.values(), index=["skipgram_2".upper(),"skipgram_3".upper(),"skipgram_5".upper(),"skipgram_7".upper(),"cbow_2".upper(),"cbow_3".upper(),"cbow_5".upper(),"cbow_7".upper()])
    df.T.boxplot(vert=False)
    plt.subplots_adjust(left=0.25)
    plt.title(title)
    plt.xlabel("V/Vmax")
    plt.show()


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
                 "marxism",
                 "marxism",
                 "marxism"]

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
