import os
import pickle

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine


def create_keyword_embeddings(kw_path, nlp):
    """Read keywords from file and create embeddings for them using spaCy.

    Args:
        kw_path (str): Path to keywords .csv file.
        nlp (spacy.Language): NLP model from spaCy, used to get embeddings.

    Returns:
        dict: Maps keywords to their embeddings (numpy vector).
    """
    keywords_df = pd.read_csv(kw_path)
    embeddings = {}
    for _, row in keywords_df.iterrows():
        keyword = row['keyword']
        embedding = np.mean([t.vector for t in nlp(keyword)], axis=0)
        embeddings[keyword] = embedding
    return embeddings


def save_keyword_embeddings(kw_embed, path):
    with open(path, 'wb') as f:
        pickle.dump(kw_embed, f)


def load_keyword_embeddings(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_keyword_embeddings(kw_embed_path, kw_path, nlp):
    if os.path.isfile(kw_embed_path):
        print(f'Loading keyword embeddings from file: {kw_embed_path}')
        return load_keyword_embeddings(kw_embed_path)
    else:
        print(f'Creating keyword embeddings from keywords: {kw_path}')
        keyword_embeddings = create_keyword_embeddings(kw_path, nlp)
        print(f'Saving keyword embeddings to file: {kw_embed_path}')
        save_keyword_embeddings(keyword_embeddings, kw_embed_path)
        return keyword_embeddings


def cos_dist(a, b):
    return cosine(a, b)


def cos_sim(a, b):
    return 1 - cosine(a, b)


def get_top_k_closest_keywords(embedding, keyword_embeddings, k=5):
    scores = []
    for kw, kw_embedding in keyword_embeddings.items():
        scores.append((kw, cos_sim(embedding, kw_embedding)))
    return sorted(scores, reverse=True, key=lambda tup: tup[1])[:k]


def bold(s):
    return '\033[1m{}\033[0m'.format(s)
