import json
import pandas as pd
import numpy as np
import networkx as nx
from tqdm.auto import tqdm, trange


def read_source(filename="../data/ht09_contact_list.dat"):
    data = pd.read_csv(filename, sep="\t", header=None)
    data = data.rename(columns={0: "timestep", 1: "person1", 2: "person2"})
    data["time"] = data["timestep"].apply(lambda x: pd.to_datetime(1246255200 + 7200 + x, unit="s"))
    return data


def encode_persons(data):
    persons = pd.concat([data["person1"], data["person2"]]).unique()
    mapping = {el: i for i, el in enumerate(persons)}

    data2 = data.copy()
    data2[["person1", "person2"]] = data[["person1", "person2"]].replace(mapping)
    return data2, persons, mapping


def build_graph_from_pandas(data, weighted=False):
    agg = data.groupby(["person1", "person2"])["timestep"].count().reset_index()
    agg = agg.rename(columns={"timestep": "sum"})
    G = nx.Graph()
    
    for i, row in agg.iterrows():
        if weighted:
            G.add_edge(row["person1"], row["person2"], weight=row['sum'])
        else:
            G.add_edge(row["person1"], row["person2"])

    return G


def extract_relevant_data(data, start, end):
    start_dt = pd.to_datetime(start) if type(start) == str else start
    end_dt = pd.to_datetime(end) if type(end) == str else end
    return data[(data["time"] >= start) & (data["time"] <= end)]


def plot_graph(G, plot_single_nodes=False, **kwargs):
    options = {
        "node_size": 50,
        "node_color": "white",
        "edgecolors": "black",
        "linewidths": 1,
        "width": 2,
    }
    options.update(kwargs)

    nodes_to_plot = G.nodes
    if not plot_single_nodes:
        nodes_to_plot = [key for key, val in dict(nx.degree(G)).items() if val > 0]
    
    pos = nx.spring_layout(G)
    nx.draw(G, pos, **options, nodelist=nodes_to_plot)


def measure_centralities(G):
    return {
        "closeness": nx.closeness_centrality(G),
        "betweenness": nx.betweenness_centrality(G),
        "katz": nx.katz_centrality_numpy(G),
        "pagerank": nx.pagerank(G),
        "degree": nx.degree_centrality(G)
    }


def get_max_connected_component(G):
    max_size = 0
    for component in nx.connected_components(G):
        if len(component) > max_size:
            max_size = len(component)
    
    return max_size


def graph_features(G):
    degrees = np.array(list(dict(nx.degree(G)).values()))

    features = {
        "n_nodes": len(G.nodes),
        "n_edges": len(G.edges),
        "mean_degree": degrees.mean(),
        "max_degree": degrees.max(),
        "connected_components": nx.number_connected_components(G),
        "max_component_size": get_max_connected_component(G),
        "clustering_coef": nx.average_clustering(G),
        "diameter": None,
        "average_distance": None,
    }

    if features["connected_components"] == 1:
        features["diameter"] = nx.diameter(G)
        features["average_distance"] = nx.average_shortest_path_length(G)

    return features
