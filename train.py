import argparse

import numpy as np
import pandas as pd
import spacy

import utils


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vocabulary", help="Vocabulary (model) for spaCy to use", default="en_core_web_lg")
    parser.add_argument("-k", "--keywords", help="CSV file with keywords", default="keywords.csv")
    parser.add_argument("-e", "--kw_embeds", default="kw_embeds.pickle",
                        help="File with input keyword embeddings. If path does not exist, keyword embeddings are created "
                             "from vocabulary model and saved to this path.")
    parser.add_argument("-o", "--kw_embeds_out", default="kw_embeds_opt.pickle",
                        help="File with updated keyword embeddings.")
    parser.add_argument("-t", "--train", help="CSV file with train texts", default="train.csv")
    parser.add_argument("-c", "--context", help="Number of words in left/right context", type=int, default=3)
    parser.add_argument("-a", "--alpha", help="Strength of a single update.", type=float, default=0.2)
    return parser.parse_args()


def main():
    config = get_config()
    print('Config: {}'.format(config))
    nlp = spacy.load(config.vocabulary)
    kw_embeds = utils.get_keyword_embeddings(config.kw_embeds, config.keywords, nlp)
    train_texts_df = pd.read_csv(config.train)
    c = config.context

    for _, sample in train_texts_df.iterrows():
        path = sample['path']
        kw = sample['keyword']
        kw_embed = kw_embeds[kw]
        words = utils.preprocess_text(path)
        
        i = [i for i, w in enumerate(words) if w.startswith('*')][0]  # Find position of keyword in text
        words[i] = words[i][1:-1]  # Remove '*' on both sides
        word_w_context = [w for w in words[max(i - c, 0):min(i + 1 + c, len(words))]]
        ctx_embed = np.mean([nlp(w).vector for w in word_w_context], axis=0)
        cos_dist = utils.cos_dist(ctx_embed, kw_embed)

        kw_embed += config.alpha * (ctx_embed - kw_embed)
        cos_dist_new = utils.cos_dist(ctx_embed, kw_embed)
        print('{}:\n\tCos dist: {:.6f} -> {:.6f}'.format(kw, cos_dist, cos_dist_new))
        kw_embeds[kw] = kw_embed
    
    utils.save_keyword_embeddings(kw_embeds, config.kw_embeds_out)


if __name__ == "__main__":
    main()