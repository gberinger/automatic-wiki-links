import argparse
from collections import defaultdict, namedtuple

import numpy as np
import pandas as pd
import spacy

import utils


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vocabulary", help="Vocabulary (model) for spaCy to use", default="en_core_web_lg")
    parser.add_argument("-k", "--keywords", help="CSV file with keywords", default="keywords.csv")
    parser.add_argument("-e", "--kw_embeds_opt", default="kw_embeds_opt.pickle",
                        help="File with keyword embeddings. If path does not exist, keyword embeddings are created "
                             "from vocabulary model and saved to this path.")
    parser.add_argument("-t", "--test", help="CSV file with test texts", default="test.csv")
    parser.add_argument("-c", "--context", help="Number of words in left/right context", type=int, default=3)
    return parser.parse_args()


ResultsTuple = namedtuple('ResultsTuple', ['cos_dist', 'top1_acc', 'top5_acc', 'cos_dists',
                                           'top1_scores', 'top5_scores'])


def test(nlp, config):
    kw_embeds = utils.get_keyword_embeddings(config.kw_embeds_opt, config.keywords, nlp)
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

        kw_pos = utils.get_keyword_pos(words)
        ctx_embed = utils.get_context_embedding(words, kw_pos, c, nlp)
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
    cos_dist_avg = np.mean(cos_dists_all)
    top1_acc = np.mean(top1_scores_all)
    top5_acc = np.mean(top5_scores_all)
    return ResultsTuple(cos_dist_avg, top1_acc, top5_acc, cos_dists, top1_scores, top5_scores)


def main():
    config = get_config()
    print('Config: {}'.format(config))
    nlp = spacy.load(config.vocabulary)

    results = test(nlp, config)

    kw_col_size = 24
    print('\n-------------------------------------------------------------------')
    print('| {} | avg cos dist | top-1 acc | top-5 acc |'.format(
        utils.bold('keyword'.ljust(kw_col_size))))
    print('|-----------------------------------------------------------------|')
    for kw, dists in results.cos_dists.items():
        print('| {} |   {:.6f}   |   {:.2f}    |   {:.2f}    |'.format(
            utils.bold(kw.ljust(kw_col_size)), np.mean(dists), np.mean(results.top1_scores[kw]),
            np.mean(results.top5_scores[kw]))
        )
    print('-------------------------------------------------------------------')
    print('\nAvg cosine distance: {:.6f}'.format(results.cos_dist))
    print('Top-1 acc:           {:.2f}'.format(results.top1_acc))
    print('Top-5 acc:           {:.2f}\n'.format(results.top5_acc))


if __name__ == "__main__":
    main()