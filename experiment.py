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
    parser.add_argument("experiment", help="Experiment name",
                        choices=["context", "context_epochs", "epochs", "alpha", "beta"])
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
        print('Top-3 acc:           {:.2f}'.format(results.top3_acc))
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


class EpochExperiment(Experiment):
    def __init__(self, config, max_epoch=15, out_csv='results.csv'):
        super().__init__(config)
        self.max_epoch = max_epoch
        self.results = []
        self.current_epoch = 1
        self.config.epochs = 1
        self.kw_embeds_path = self.config.kw_embeds
        self.out_csv = out_csv

    def on_experiment_begin(self):
        # Check results for untrained embeddings
        kw_embeds_opt = self.config.kw_embeds_opt
        self.config.kw_embeds_opt = self.config.kw_embeds
        results = self.test()
        self.results.append((0, results.cos_dist, results.top1_acc, results.top3_acc))
        self.config.kw_embeds_opt = kw_embeds_opt

    def on_iter_begin(self):
        print('Current epoch = {}'.format(self.current_epoch))

    def on_iter_end(self, results):
        # Update the keyword embeddings with each epoch
        self.config.kw_embeds = self.config.kw_embeds_opt
        self.results.append((self.current_epoch, results.cos_dist, results.top1_acc, results.top3_acc))
        self.current_epoch += 1
        if self.current_epoch > self.max_epoch:
            self.is_done = True

    def on_experiment_end(self):
        results_df = pd.DataFrame(self.results, columns=['epoch','cos_dist','top1_acc','top3_acc'])
        results_df.to_csv(os.path.join(self.config.outdir, self.out_csv), index=False)
        self.config.kw_embeds = self.kw_embeds_path


class ValueExperiment(Experiment):
    def __init__(self, config, value_name, values, out_csv='results.csv'):
        super().__init__(config)
        self.value_name = value_name
        self.values = values
        self.value_current = getattr(self.config, self.value_name)
        self.results = []
        self.idx = 0
        self.out_csv = out_csv

    def on_iter_begin(self):
        self.value_current = self.values[self.idx]
        setattr(self.config, self.value_name, self.value_current)
        print('{} = {}'.format(self.value_name, self.value_current))

    def on_iter_end(self, results):
        self.results.append((self.value_current, results.cos_dist, results.top1_acc, results.top3_acc))
        self.idx += 1
        if self.idx >= len(self.values):
            self.is_done = True

    def on_experiment_end(self):
        results_df = pd.DataFrame(self.results, columns=[self.value_name,'cos_dist','top1_acc','top3_acc'])
        results_df.to_csv(os.path.join(self.config.outdir, self.out_csv), index=False)


def main():
    config = get_config()
    print('Config: {}'.format(config))
    with open(os.path.join(config.outdir, 'args.txt'), 'w') as f:
        json.dump(vars(config), f, indent=4, sort_keys=True)

    exp = config.experiment
    if exp == 'context':
        ValueExperiment(config, 'context', values=range(1, 11)).run()
    elif exp == 'context_epochs':
        for c in range(1, 11):
            config.context = c
            print('****************************************************')
            print('\nRunning experiment for context={}'.format(c))
            print('****************************************************')
            EpochExperiment(config, max_epoch=10, out_csv='results_{}.csv'.format(c)).run()
    elif exp == 'epochs':
        EpochExperiment(config, max_epoch=config.epochs).run()
    elif exp == 'alpha':
        ValueExperiment(config, 'alpha', values=np.linspace(0.05, 0.5, 10)).run()
    elif exp == 'beta':
        ValueExperiment(config, 'beta', values=np.linspace(0, 0.1, 11)).run()
    else:
        print('Wrong experiment name!')

if __name__ == "__main__":
    main()