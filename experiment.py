import argparse

import numpy as np
import pandas as pd
import spacy

import utils
import train
import test


def get_config():
    parser, _ = train.get_config()
    parser.add_argument("-te", "--test", help="CSV file with test texts", default="test.csv")
    return parser.parse_args()


def main():
    config = get_config()
    print('Config: {}'.format(config))

    nlp = spacy.load(config.vocabulary)
    contexts = range(1, 4)
    for c in contexts:
        config.context = c
        print('##################################################################')
        print('context = {}'.format(c))
        print('\nTraining:')
        print('------------')
        train.train(nlp, config)
        print('\nTesting:')
        print('-----------')
        cos_dist_avg, top1_acc, top5_acc, _, _, _ = test.test(nlp, config)
        print('Avg cosine distance: {:.6f}'.format(cos_dist_avg))
        print('Top-1 acc:           {:.2f}'.format(top1_acc))
        print('Top-5 acc:           {:.2f}\n'.format(top5_acc))

if __name__ == "__main__":
    main()