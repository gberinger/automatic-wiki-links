import argparse
from collections import defaultdict, namedtuple

import numpy as np
import pandas as pd
import spacy

import utils


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vocabulary", help="Vocabulary (model) for spaCy to use", default="en_vectors_web_lg")
    parser.add_argument("-k", "--keywords", help="CSV file with keywords", default="keywords.csv")
    parser.add_argument("-e", "--kw_embeds_opt", default="kw_embeds_opt.pickle",
                        help="File with keyword embeddings. If path does not exist, keyword embeddings are created "
                             "from vocabulary model and saved to this path.")
    parser.add_argument("-t", "--test", help="CSV file with test texts", default="test.csv")
    parser.add_argument("-c", "--context", help="Number of words in left/right context", type=int, default=3)
    return parser.parse_args()


ResultsTuple = namedtuple('ResultsTuple', ['cos_dist', 'top1_acc', 'top2_acc', 'top3_acc', 'cos_dists',
                                           'top1_scores', 'top2_scores', 'top3_scores'])


def test(nlp, config):
    kw_embeds = utils.get_keyword_embeddings(config.kw_embeds_opt, config.keywords, nlp)
    test_texts_df = pd.read_csv(config.test)
    c = config.context

    cos_dists = defaultdict(list)
    top1_scores = defaultdict(list)
    top2_scores = defaultdict(list)
    top3_scores = defaultdict(list)
    cos_dists_all = []
    top1_scores_all = []
    top2_scores_all = []
    top3_scores_all = []
    for _, sample in test_texts_df.iterrows():
        path = sample['path']
        kw = sample['keyword']
        words, _ = utils.preprocess_text(path)

        kw_pos = utils.get_keyword_pos(words)
        ctx_embed = utils.get_context_embedding(words, kw_pos, c, nlp)
        cos_dist = utils.cos_dist(ctx_embed, kw_embeds[kw])
        cos_dists[kw].append(cos_dist)
        cos_dists_all.append(cos_dist)

        top2_kws = [kw for kw, _ in utils.get_top_k_closest_keywords(ctx_embed, kw_embeds, k=2)]
        top3_kws = [kw for kw, _ in utils.get_top_k_closest_keywords(ctx_embed, kw_embeds, k=3)]
        is_top1 = int(kw == top3_kws[0])
        is_top2 = int(kw in top2_kws)
        is_top3 = int(kw in top3_kws)
        top1_scores[kw].append(is_top1)
        top2_scores[kw].append(is_top2)
        top3_scores[kw].append(is_top3)
        top1_scores_all.append(is_top1)
        top2_scores_all.append(is_top2)
        top3_scores_all.append(is_top3)
    cos_dist_avg = np.mean(cos_dists_all)
    top1_acc = np.mean(top1_scores_all)
    top2_acc = np.mean(top2_scores_all)
    top3_acc = np.mean(top3_scores_all)
    return ResultsTuple(cos_dist_avg, top1_acc, top2_acc, top3_acc, cos_dists, top1_scores, top2_scores, top3_scores)


def main():
    config = get_config()
    print('Config: {}'.format(config))
    nlp = spacy.load(config.vocabulary)

    results = test(nlp, config)

    kw_col_size = 24
    print('\n-------------------------------------------------------------------')
    print('| {} | avg cos dist | top-1 acc |  top-2 acc | top-3 acc |'.format(
        utils.bold('keyword'.ljust(kw_col_size))))
    print('|-----------------------------------------------------------------|')
    for kw, dists in results.cos_dists.items():
        print('| {} |   {:.6f}   |   {:.2f}    |   {:.2f}    |   {:.2f}    |'.format(
            utils.bold(kw.ljust(kw_col_size)), np.mean(dists), np.mean(results.top1_scores[kw]),
            np.mean(results.top2_scores[kw]), np.mean(results.top3_scores[kw]))
        )
    print('-------------------------------------------------------------------')
    print('\nAvg cosine distance: {:.6f}'.format(results.cos_dist))
    print('Top-1 acc:           {:.2f}'.format(results.top1_acc))
    print('Top-2 acc:           {:.2f}'.format(results.top2_acc))
    print('Top-3 acc:           {:.2f}\n'.format(results.top3_acc))


if __name__ == "__main__":
    main()
