import os
import pickle

import numpy as np
import pandas as pd


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

