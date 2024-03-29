import argparse

import numpy as np
import pandas as pd
import spacy

import utils


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vocabulary", help="Vocabulary (model) for spaCy to use", default="en_vectors_web_lg")
    parser.add_argument("-k", "--keywords", help="CSV file with keywords", default="keywords.csv")
    parser.add_argument("-e", "--kw_embeds", default="kw_embeds.pickle",
                        help="File with input keyword embeddings. If path does not exist, keyword embeddings are created "
                             "from vocabulary model and saved to this path.")
    parser.add_argument("-o", "--kw_embeds_opt", default="kw_embeds_opt.pickle",
                        help="File with updated keyword embeddings.")
    parser.add_argument("-cm", "--ctx_embed_method", default="avg",
                        help="Method for getting the context embedding. Default: avg")
    parser.add_argument("-um", "--update_method", default="alpha_beta",
                        help="Method for updating the keyword embeddings. Default: alpha_beta")
    parser.add_argument("-t", "--train", help="CSV file with train texts", default="train.csv")
    parser.add_argument("-c", "--context", help="Number of words in left/right context", type=int, default=3)
    parser.add_argument("-a", "--alpha", type=float, default=0.15,
                        help="Strength of a single update. If not provided, cosinus distance between the keyword "
                             "and context embeddings will be used as alpha (default).")
    parser.add_argument("-b", "--beta", type=float, default=0.0,
                        help="Strength of a single update on WRONG keywords, disabled by default.")
    parser.add_argument("-ep", "--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    return parser


def train(nlp, config):
    kw_embeds = utils.get_keyword_embeddings(config.kw_embeds, config.keywords, nlp)
    train_texts_df = pd.read_csv(config.train)
    c = config.context

    for epoch in range(1, config.epochs + 1):
        print('Epoch {}/{}'.format(epoch, config.epochs))
        df = train_texts_df.sample(frac=1) if config.shuffle else train_texts_df
        for _, sample in df.iterrows():
            path = sample['path']
            kw = sample['keyword']
            words, _ = utils.preprocess_text(path)
            kw_pos = utils.get_keyword_pos(words)
            ctx_embed = utils.get_context_embedding(words, kw_pos, c, nlp, config.ctx_embed_method)
            cos_dist_prev = utils.cos_dist(ctx_embed, kw_embeds[kw])
            utils.update_keyword_embeddings_with_context(kw_embeds, kw, ctx_embed, config.update_method,
                                                        alpha=config.alpha, beta=config.beta)
            cos_dist_new = utils.cos_dist(ctx_embed, kw_embeds[kw])
            if config.verbose:
                print('{}:\n\tCos dist: {:.6f} -> {:.6f}'.format(kw, cos_dist_prev, cos_dist_new))

    utils.save_keyword_embeddings(kw_embeds, config.kw_embeds_opt)


def main():
    parser = get_parser()
    config = parser.parse_args()
    print('Config: {}'.format(config))
    nlp = spacy.load(config.vocabulary)
    train(nlp, config)


if __name__ == "__main__":
    main()