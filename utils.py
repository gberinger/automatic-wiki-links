import os
import pickle
import re

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from nltk.corpus import stopwords


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


def get_keyword_embeddings(kw_embed_path, kw_path, nlp, verbose=False):
    if os.path.isfile(kw_embed_path):
        if verbose:
            print(f'Loading keyword embeddings from file: {kw_embed_path}')
        return load_keyword_embeddings(kw_embed_path)
    else:
        if verbose:
            print(f'Creating keyword embeddings from keywords: {kw_path}')
        keyword_embeddings = create_keyword_embeddings(kw_path, nlp)
        if verbose:
            print(f'Saving keyword embeddings to file: {kw_embed_path}')
        save_keyword_embeddings(keyword_embeddings, kw_embed_path)
        return keyword_embeddings


def cos_dist(a, b):
    return cosine(a, b)


def cos_sim(a, b):
    return 1 - cosine(a, b)


def get_top_k_closest_keywords(embedding, keyword_embeddings, k=3):
    scores = []
    for kw, kw_embedding in keyword_embeddings.items():
        scores.append((kw, cos_dist(embedding, kw_embedding)))
    return sorted(scores, key=lambda tup: tup[1])[:k]


def bold(s):
    return '\033[1m{}\033[0m'.format(s)


def preprocess_text(path):
    with open(path) as f:
        sentence = re.sub(r'[^\w\s*]', ' ', f.read().strip().lower())
        words = [w for w in sentence.split() if w not in stopwords.words('english')]
        return words, sentence


def get_keyword_pos(words):
    pos = [i for i, w in enumerate(words) if w.startswith('*')][0]  # Find position of keyword in text
    words[pos] = words[pos][1:-1]  # Remove '*' on both sides
    return pos


def get_context_embedding(words, kw_idx, ctx, nlp, method='avg'):
    ctx_embed = None
    if method == 'avg':
        word_w_context = words[max(kw_idx - ctx, 0):min(kw_idx + 1 + ctx, len(words))]
        ctx_embed = np.mean([nlp(w).vector for w in word_w_context], axis=0)
    else:
        raise Exception('Context embedding method "{}" does not exist!'.format(method))
    return ctx_embed


def update_keyword_embeddings_with_context(kw_embeds, kw, ctx_embed, method='alpha_beta', **kwargs):
    if method == 'alpha_beta':
        alpha = kwargs['alpha']
        kw_embeds[kw] += alpha * (ctx_embed - kw_embeds[kw])
        if kwargs['beta'] and kwargs['beta'] > 0:
            topk = get_top_k_closest_keywords(ctx_embed, kw_embeds)
            for other_kw, dist in topk:
                if other_kw != kw:
                    beta = kwargs['beta'] or dist
                    kw_embeds[other_kw] -= beta * (ctx_embed - kw_embeds[other_kw])
    else:
        raise Exception('Update method "{}" does not exist!'.format(method))
