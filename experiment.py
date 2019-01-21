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
    parser.add_argument("experiment", help="Experiment name. Options: context, epochs",
                        choices=["context", "epochs"])
    parser.add_argument("outdir", help="Output directory")
    config = parser.parse_args()
    makedirs(config.outdir)
    return config


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


class Experiment(object):
    def __init__(self, config):
        self.config = config
        self.nlp = spacy.load(config.vocabulary)
        self.is_done = False

    def on_experiment_begin(self):
        pass

    def on_experiment_end(self):
        pass

    def on_iter_begin(self):
        pass

    def on_iter_end(self, results):
        pass

    def train(self):
        print('\nTraining:')
        print('------------')
        train.train(self.nlp, self.config)

    def test(self):
        print('\nTesting:')
        print('-----------')
        results = test.test(self.nlp, self.config)
        print('Avg cosine distance: {:.6f}'.format(results.cos_dist))
        print('Top-1 acc:           {:.2f}'.format(results.top1_acc))
        print('Top-5 acc:           {:.2f}'.format(results.top5_acc))
        return results

    def run(self):
        self.on_experiment_begin()
        while not self.is_done:
            print('\n##################################################################')
            self.on_iter_begin()
            self.train()
            results = self.test()
            self.on_iter_end(results)
        self.on_experiment_end()


class ContextExperiment(Experiment):
    def __init__(self, config, contexts=range(1, 11)):
        super().__init__(config)
        self.contexts = contexts
        self.results = []
        self.idx = 0

    def on_iter_begin(self):
        self.config.context = self.contexts[self.idx]
        print('Context = {}'.format(self.config.context))

    def on_iter_end(self, results):
        self.results.append((self.config.context, results.cos_dist, results.top1_acc, results.top5_acc))
        self.idx += 1
        if self.idx >= len(self.contexts):
            self.is_done = True

    def on_experiment_end(self):
        results_df = pd.DataFrame(self.results, columns=['context','cos_dist','top1_acc','top5_acc'])
        results_df.to_csv(os.path.join(self.config.outdir, 'results.csv'), index=False)


class EpochExperiment(Experiment):
    def __init__(self, config, max_epoch=15):
        super().__init__(config)
        self.max_epoch = max_epoch
        self.results = []
        self.current_epoch = 1
        self.config.epochs = 1
        self.kw_embeds_path = self.config.kw_embeds

    def on_iter_begin(self):
        print('Current epoch = {}'.format(self.current_epoch))

    def on_iter_end(self, results):
        # Update the keyword embeddings with each epoch
        self.config.kw_embeds = self.config.kw_embeds_opt
        self.results.append((self.current_epoch, results.cos_dist, results.top1_acc, results.top5_acc))
        self.current_epoch += 1
        if self.current_epoch > self.max_epoch:
            self.is_done = True

    def on_experiment_end(self):
        results_df = pd.DataFrame(self.results, columns=['epoch','cos_dist','top1_acc','top5_acc'])
        results_df.to_csv(os.path.join(self.config.outdir, 'results_{}.csv'.format(self.config.context)), index=False)
        self.config.kw_embeds = self.kw_embeds_path


def main():
    config = get_config()
    print('Config: {}'.format(config))
    with open(os.path.join(config.outdir, 'args.txt'), 'w') as f:
        json.dump(vars(config), f, indent=4, sort_keys=True)

    if config.experiment == 'context':
        ContextExperiment(config, contexts=range(1, 11)).run()
    elif config.experiment == 'epochs':
        for c in range(1, 11):
            config.context = c
            print('****************************************************')
            print('\nRunning experiment for context={}'.format(c))
            print('****************************************************')
            EpochExperiment(config, max_epoch=10).run()
    else:
        print('Wrong experiment name!')

if __name__ == "__main__":
    main()