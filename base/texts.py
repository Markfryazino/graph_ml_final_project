import json
import pandas as pd
import networkx as nx
import numpy as np
import statsmodels.stats.api as sms

from pprint import pprint
from scipy.spatial.distance import squareform, pdist
from tqdm.auto import tqdm, trange
from sklearn.metrics import accuracy_score


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
