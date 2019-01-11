import argparse
import re

import numpy as np
import spacy
import pandas as pd
from scipy.spatial.distance import cosine


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vocabulary", help="Vocabulary (model) for spaCy to use", default="en_core_web_lg")
    parser.add_argument("-k", "--keywords", help="File with keywords", default="keywords.csv")
    parser.add_argument("-t", "--text", help="Text file containing a refernce to one or more keywords",
                        default="texts/tree_1.txt")
    parser.add_argument("-c", "--context", help="Number of words in left/right context", type=int, default=3)
    return parser.parse_args()


def get_keyword_embeddings(keywords, nlp):
    keywords_df = pd.read_csv(keywords)
    embeddings = {}
    for _, row in keywords_df.iterrows():
        keyword = row['keyword']
        embedding = np.mean([t.vector for t in nlp(keyword)], axis=0)
        embeddings[keyword] = embedding
    return embeddings


def get_top_k_closest_keywords(embedding, keyword_embeddings, k=5):
    scores = []
    for kw, kw_embedding in keyword_embeddings.items():
        scores.append((kw, 1 - cosine(embedding, kw_embedding)))
    return sorted(scores, reverse=True, key=lambda tup: tup[1])[:k]


def main():
    config = get_config()
    print('Config: {}'.format(config))
    nlp = spacy.load(config.vocabulary)
    keyword_embeddings = get_keyword_embeddings(config.keywords, nlp)
    with open(config.text) as f:
        sentence = re.sub(r'[^\w\s*]', ' ', f.read().strip().lower())

    print(f'Input text: {sentence}')
    words = sentence.split()
    c = config.context
    for i, word in enumerate(words):
        if not word.startswith('*'):
            # We only care about marked words i.e. ones with '*' on both sides of the word
            continue
        word = words[i] = word[1:-1]  # Remove '*' on both sides
        word_w_context = [w for w in words[max(i - c, 0):min(i + 1 + c, len(words))]]
        context_keyword = np.mean([nlp(w).vector for w in word_w_context], axis=0)
        top_k = get_top_k_closest_keywords(context_keyword, keyword_embeddings)
        print("--------------------------")
        print("Index: {}".format(i))
        print("Word: \033[1m{}\033[0m".format(word))
        print("Context: {}".format(" ".join(word_w_context)))
        print("Closest keywords:")
        for keyword, sim in top_k:
            print("\t{} ({:.6f})".format(keyword, sim))


if __name__ == "__main__":
    main()
