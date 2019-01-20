import argparse
import json
import os

import numpy as np
import pandas as pd
import spacy

import utils
import train
import test


def get_config():
    parser = train.get_parser()
    parser.add_argument("-te", "--test", help="CSV file with test texts", default="test.csv")
    parser.add_argument("outdir", help="Output directory")
    return parser.parse_args()


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main():
    config = get_config()
    print('Config: {}'.format(config))
    makedirs(config.outdir)
    with open(os.path.join(config.outdir, 'args.txt'), 'w') as f:
        json.dump(vars(config), f, indent=4, sort_keys=True)
    results = []

    nlp = spacy.load(config.vocabulary)
    contexts = range(1, 2)
    for c in contexts:
        config.context = c
        print('\n##################################################################')
        print('context = {}'.format(c))
        print('\nTraining:')
        print('------------')
        train.train(nlp, config)
        print('\nTesting:')
        print('-----------')
        cos_dist_avg, top1_acc, top5_acc, _, _, _ = test.test(nlp, config)
        results.append((c, cos_dist_avg, top1_acc, top5_acc))
        print('Avg cosine distance: {:.6f}'.format(cos_dist_avg))
        print('Top-1 acc:           {:.2f}'.format(top1_acc))
        print('Top-5 acc:           {:.2f}'.format(top5_acc))
    results_df = pd.DataFrame(results, columns=['context','cos_dist','top1_acc','top5_acc'])
    results_df.to_csv(os.path.join(config.outdir, 'results.csv'), index=False)

if __name__ == "__main__":
    main()