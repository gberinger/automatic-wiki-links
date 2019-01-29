import argparse
import re

import numpy as np
import spacy
from scipy.spatial.distance import cosine

import utils


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vocabulary", help="Vocabulary (model) for spaCy to use", default="en_vectors_web_lg")
    parser.add_argument("-k", "--keywords", help="File with keywords", default="keywords.csv")
    parser.add_argument("-e", "--kw_embeds", default="kw_embeds.pickle",
                        help="File with keyword embeddings. If path does not exist, keyword embeddings are created "
                             "from vocabulary model and saved to this path.")
    parser.add_argument("-t", "--text", help="Text file containing a refernce to one or more keywords",
                        default="texts/test/tree_forest_1.txt")
    parser.add_argument("-c", "--context", help="Number of words in left/right context", type=int, default=3)
    return parser.parse_args()


def main():
    config = get_config()
    print('Config: {}'.format(config))
    nlp = spacy.load(config.vocabulary)
    kw_embeds = utils.get_keyword_embeddings(config.kw_embeds, config.keywords, nlp)
    words, sentence = utils.preprocess_text(config.text)
    print('\n{}: {}'.format(utils.bold('Input text'), sentence))
    c = config.context
    for i, word in enumerate(words):
        if not word.startswith('*'):
            # We only care about marked words i.e. ones with '*' on both sides of the word
            continue
        word = words[i] = word[1:-1]  # Remove '*' on both sides
        word_w_context = [w for w in words[max(i - c, 0):min(i + 1 + c, len(words))]]
        context_embedding = np.mean([nlp(w).vector for w in word_w_context], axis=0)
        top_k = utils.get_top_k_closest_keywords(context_embedding, kw_embeds)
        print("--------------------------")
        print("Index: {}".format(i))
        print("Word: {}".format(utils.bold(word)))
        print("Context: {}".format(" ".join(word_w_context)))
        print("Closest keywords:")
        for keyword, sim in top_k:
            print("\t{} ({:.6f})".format(keyword, sim))


if __name__ == "__main__":
    main()
