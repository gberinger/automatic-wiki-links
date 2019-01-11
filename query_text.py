import argparse
import os
import re

import numpy as np
import spacy
from scipy.spatial.distance import cosine

import utils


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vocabulary", help="Vocabulary (model) for spaCy to use", default="en_core_web_lg")
    parser.add_argument("-k", "--keywords", help="File with keywords", default="keywords.csv")
    parser.add_argument("-e", "--keyword_embeddings", default="keyword_embeddings.pickle",
                        help="File with keyword embeddings. If path does not exist, keyword embeddings are created "
                             "from vocabulary model and saved to this path.")
    parser.add_argument("-t", "--text", help="Text file containing a refernce to one or more keywords",
                        default="texts/tree_1.txt")
    parser.add_argument("-c", "--context", help="Number of words in left/right context", type=int, default=3)
    return parser.parse_args()


def get_top_k_closest_keywords(embedding, keyword_embeddings, k=5):
    scores = []
    for kw, kw_embedding in keyword_embeddings.items():
        scores.append((kw, 1 - cosine(embedding, kw_embedding)))
    return sorted(scores, reverse=True, key=lambda tup: tup[1])[:k]


def main():
    config = get_config()
    print('Config: {}'.format(config))
    nlp = spacy.load(config.vocabulary)
    keyword_embeddings = utils.get_keyword_embeddings(config.keyword_embeddings, config.keywords, nlp)
    with open(config.text) as f:
        sentence = re.sub(r'[^\w\s*]', ' ', f.read().strip().lower())

    print(f'\n\033[1mInput text\033[0m: {sentence}')
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
