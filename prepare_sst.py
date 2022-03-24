import numpy as np
import json

from sentence_transformers import SentenceTransformer
from datasets import load_dataset


if __name__ == "__main__":
    dataset = load_dataset("sst", "default")

    sentences = dataset["train"]["sentence"]
    label = dataset["train"]["label"]

    positives_idx = np.random.choice(np.where(np.array(label) > 0.5)[0], replace=False, size=100)
    negatives_idx = np.random.choice(np.where(np.array(label) < 0.5)[0], replace=False, size=100)

    positive_sentences = dataset["train"].select(positives_idx)["sentence"]
    negative_sentences = dataset["train"].select(negatives_idx)["sentence"]

    model = SentenceTransformer('all-mpnet-base-v2')
    positive_encodings = model.encode(positive_sentences)
    negative_encodings = model.encode(negative_sentences)
    encodings = np.vstack([positive_encodings, negative_encodings])

    np.save("data/sst_encodings.npy", encodings)

    with open("data/sst_sentences.json", "w") as f:
        json.dump(positive_sentences + negative_sentences, f)
