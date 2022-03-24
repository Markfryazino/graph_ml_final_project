import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer


def prepare_texts(papers_json):
    data = pd.DataFrame(papers_json)
    data["other_abstract"] = data["abstract"].apply(lambda x: "\n" + x.replace("No abstract available.", ""))
    data["text"] = data["title"] + data["other_abstract"]
    return data["text"]


if __name__ == '__main__':
    with open("data/papers.json") as f:
        papers = json.load(f)
    
    texts = prepare_texts(papers)

    model = SentenceTransformer('all-mpnet-base-v2')
    encodings = model.encode(data["text"])
    np.save("data/encodings.npy", encodings)
