import argparse
from collections import defaultdict
import re

import numpy as np
import pandas as pd
import spacy

import utils


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vocabulary", help="Vocabulary (model) for spaCy to use", default="en_core_web_lg")
    parser.add_argument("-k", "--keywords", help="CSV file with keywords", default="keywords.csv")
    parser.add_argument("-e", "--keyword_embeddings", default="keyword_embeddings.pickle",
                        help="File with keyword embeddings. If path does not exist, keyword embeddings are created "
                             "from vocabulary model and saved to this path.")
    parser.add_argument("-t", "--test", help="CSV file with test texts", default="test.csv")
    parser.add_argument("-c", "--context", help="Number of words in left/right context", type=int, default=3)
    return parser.parse_args()


def main():
    config = get_config()
    print('Config: {}'.format(config))
    nlp = spacy.load(config.vocabulary)
    keyword_embeddings = utils.get_keyword_embeddings(config.keyword_embeddings, config.keywords, nlp)
    test_texts_df = pd.read_csv(config.test)
    c = config.context

    cos_dists = defaultdict(list)
    top1_scores = defaultdict(list)
    top5_scores = defaultdict(list)
    cos_dists_all = []
    top1_scores_all = []
    top5_scores_all = []
    for _, sample in test_texts_df.iterrows():
        path = sample['path']
        keyword = sample['keyword']
        with open(path) as f:
            sentence = re.sub(r'[^\w\s*]', ' ', f.read().strip().lower())
            words = sentence.split()
        
        i = [i for i, w in enumerate(words) if w.startswith('*')][0]  # Find position of keyword in text
        words[i] = words[i][1:-1]  # Remove '*' on both sides
        word_w_context = [w for w in words[max(i - c, 0):min(i + 1 + c, len(words))]]
        context_embedding = np.mean([nlp(w).vector for w in word_w_context], axis=0)
        cos_dist = utils.cos_dist(context_embedding, keyword_embeddings[keyword])
        cos_dists[keyword].append(cos_dist)
        cos_dists_all.append(cos_dist)
        
        top5_kws = [kw for kw, _ in utils.get_top_k_closest_keywords(context_embedding, keyword_embeddings)]
        is_top1 = int(keyword == top5_kws[0])
        is_top5 = int(keyword in top5_kws)
        top1_scores[keyword].append(is_top1)
        top5_scores[keyword].append(is_top5)
        top1_scores_all.append(is_top1)
        top5_scores_all.append(is_top5)
        # print('{}, {}, {}, {}, {}'.format(keyword, top5_kws, is_top1, is_top5, cos_dist))
    
    print('\n---------------------------------------------------------------')
    print('| {} | avg cos dist | top-1 acc | top-5 acc |'.format(
        utils.bold('keyword'.ljust(20))))
    print('|-------------------------------------------------------------|')
    for kw, dists in cos_dists.items():
        print('| {} |   {:.6f}   |   {:.2f}    |   {:.2f}    |'.format(
            utils.bold(kw.ljust(20)), np.mean(dists), np.mean(top1_scores[kw]), np.mean(top5_scores[kw]))
        )
    print('---------------------------------------------------------------')
    print('\nAvg cosine distance: {:.6f}'.format(np.mean(cos_dists_all)))
    print('Top-1 acc:           {:.2f}'.format(np.mean(top1_scores_all)))
    print('Top-5 acc:           {:.2f}\n'.format(np.mean(top5_scores_all)))


if __name__ == "__main__":
    main()