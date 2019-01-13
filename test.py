import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import spacy

import utils


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vocabulary", help="Vocabulary (model) for spaCy to use", default="en_core_web_lg")
    parser.add_argument("-k", "--keywords", help="CSV file with keywords", default="keywords.csv")
    parser.add_argument("-e", "--kw_embeds", default="kw_embeds.pickle",
                        help="File with keyword embeddings. If path does not exist, keyword embeddings are created "
                             "from vocabulary model and saved to this path.")
    parser.add_argument("-t", "--test", help="CSV file with test texts", default="test.csv")
    parser.add_argument("-c", "--context", help="Number of words in left/right context", type=int, default=3)
    return parser.parse_args()


def main():
    config = get_config()
    print('Config: {}'.format(config))
    nlp = spacy.load(config.vocabulary)
    kw_embeds = utils.get_keyword_embeddings(config.kw_embeds, config.keywords, nlp)
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
        kw = sample['keyword']
        words = utils.preprocess_text(path)
        
        i = [i for i, w in enumerate(words) if w.startswith('*')][0]  # Find position of keyword in text
        words[i] = words[i][1:-1]  # Remove '*' on both sides
        word_w_context = [w for w in words[max(i - c, 0):min(i + 1 + c, len(words))]]
        ctx_embed = np.mean([nlp(w).vector for w in word_w_context], axis=0)
        cos_dist = utils.cos_dist(ctx_embed, kw_embeds[kw])
        cos_dists[kw].append(cos_dist)
        cos_dists_all.append(cos_dist)
        
        top5_kws = [kw for kw, _ in utils.get_top_k_closest_keywords(ctx_embed, kw_embeds)]
        is_top1 = int(kw == top5_kws[0])
        is_top5 = int(kw in top5_kws)
        top1_scores[kw].append(is_top1)
        top5_scores[kw].append(is_top5)
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