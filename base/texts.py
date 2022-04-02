import json
import pandas as pd
import networkx as nx
import numpy as np
import statsmodels.stats.api as sms
import nltk

from pprint import pprint
from scipy.spatial.distance import squareform, pdist
from tqdm.auto import tqdm, trange
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def load_papers_df(path="data/papers.json"):
    with open(path) as f:
        papers = json.load(f)

    return pd.DataFrame(papers)


def load_sst_df(encodings_path="data/sst_encodings.npy", sentences_path="data/sst_sentences.json"):
    encodings = np.load(encodings_path)

    with open(sentences_path) as f:
        texts = json.load(f)
        
    data = pd.DataFrame({"text": texts, "label": 1})
    data.loc[100:, "label"] = 0

    idxs = np.arange(encodings.shape[0])
    np.random.shuffle(idxs)

    return data.loc[idxs], encodings[idxs]


def encodings2distances(encodings):
    distances = squareform(pdist(encodings, metric="cosine"))
    np.fill_diagonal(distances, 1)
    return distances


def distances2fixed_edges_graph(distances, max_edges, add_single_nodes=False):
    G = nx.Graph()
    max_dist = np.sort(distances.flatten())[max_edges * 2 - 1]
    x, y = np.where(distances <= max_dist)

    if add_single_nodes:
        for i in range(distances.shape[0]):
            G.add_node(i)

    for i, j in zip(x, y):
        G.add_edge(i, j)

    return G


def distances2fixed_degree_graph(distances, max_degree):
    G = nx.Graph()
    neighbors = distances.argsort(axis=1)[:,:max_degree]

    for i in range(distances.shape[0]):
        for j in neighbors[i]:
            G.add_edge(i, j)

    return G


def print_paper(paper_row):
    pprint({
        "Title": paper_row["title"],
        "Session": paper_row["session"],
        "Abstract": paper_row["abstract"],
    })


def kernighan_lin_clustering(G):
    partition = nx.algorithms.community.kernighan_lin_bisection(G)
    res_labels = np.zeros(len(G.nodes))
    res_labels[list(partition[0])] = 1
    return res_labels


def clustering_accuracy(prediction_func, y_true, *args, num_repetitions=100, verbose=True, **kwargs):
    results = []

    loop = trange(num_repetitions) if verbose else range(num_repetitions)
    for _ in loop:
        res_labels = prediction_func(*args, **kwargs)

        acc = max(accuracy_score(y_true, res_labels), accuracy_score(y_true, 1 - res_labels))
        results.append(acc)
    
    left, right = sms.DescrStatsW(results).tconfint_mean()
    return (left + right) / 2, (right - left) / 2


def preprocess_paper_texts(titles, abstracts):
    texts = titles.apply(lambda x: x + " ") + abstracts

    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')

    stop_words = set(stopwords.words('english') + ['ha', 'wa', 'say', 'said'])
    lemmatizer = WordNetLemmatizer()

    def preprocess(text):
        text = list(filter(str.isalpha, word_tokenize(text.lower())))
        text = list(lemmatizer.lemmatize(word) for word in text)
        text = list(word for word in text if word not in stop_words)
        return ' '.join(text)

    return texts.apply(preprocess)


def draw_wordcloud(texts, max_words=1000, width=1000, height=500):
    wordcloud = WordCloud(background_color='white', max_words=max_words,
                          width=width, height=height)
    
    joint_texts = ' '.join(list(texts))
    wordcloud.generate(joint_texts)
    return wordcloud.to_image()


def lda_topics(texts, n_topics=2, n_words=10):
    np.random.seed(42)

    count_vect = CountVectorizer()
    counts = count_vect.fit_transform(texts)

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42).fit(counts)
    argmaxes = lda.components_.argsort()[:,-n_words:]

    topics = []
    for cluster in range(n_topics):
        topics.append([])
        for w in range(n_words):
            topics[-1].append(count_vect.get_feature_names()[argmaxes[cluster, -w - 1]])

    return topics
