import json
import pandas as pd
import random
import numpy as np
import logging
import argparse
import os
from os.path import join, dirname, basename, exists
from glob import glob
import random

import sys
sys.path.append('.')

from scorer.main import evaluate
# from format_checker.subtask_1a import check_format

random.seed(0)
ROOT_DIR = dirname(dirname(__file__))

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


def load_vclaims(dir):
    vclaims_fp = glob(f'{dir}/*.json')
    vclaims_fp.sort()
    vclaims = {}
    for vclaim_fp in vclaims_fp:
        with open(vclaim_fp) as f:
            vclaim = json.load(f)
        # print(vclaim)
        # print(vclaim['vclaim_id'])
        vclaims[vclaim['vclaim_id']] = vclaim
    return vclaims

def run_random_baseline(data_fpath, vclaims, results_fpath):
    print(data_fpath)
    gold_df = pd.read_csv(data_fpath, sep='\t', names=['iclaim_id', '0', 'vclaim_id', 'relevance'])
    print(vclaims)
    vclaims_list = list(vclaims.keys())
    outs = {}
    with open(results_fpath, "w") as results_file:
        for i, line in gold_df.iterrows():
            if line.iclaim_id in outs:
                outs[line.iclaim_id] += 1
            else:
                outs[line.iclaim_id] = 1
            results_file.write(f'{line.iclaim_id}\tQ0\t{random.choice(vclaims_list)}\t{outs[line.iclaim_id]}\t1\trandom\n')


def run_baselines(train_fpath, test_fpath, vclaim_dpath, lang, subtask):
    if not exists('baselines/data'):
        os.mkdir('baselines/data')
    vclaims = load_vclaims(vclaim_dpath)

    random_baseline_fpath = join(ROOT_DIR, f'baselines/data/subtask_{subtask}_random_baseline_{lang}_{basename(test_fpath)}')
    run_random_baseline(test_fpath, vclaims, random_baseline_fpath)
    maps, mrr, precisions = evaluate(test_fpath, random_baseline_fpath)
    logging.info(f"Random Baseline for Subtask-{subtask}--{lang}")
    logging.info(f'All MAP scores on threshold from [1, 3, 5, 10, 20, 50, 1000]. {maps}')
    logging.info(f'MRR score {mrr}')
    logging.info(f'All P scores on threshold from [1, 3, 5, 10, 20, 50, 1000]. {precisions}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file-path", "-t", required=True, type=str,
                        help="The absolute path to the training data")
    parser.add_argument("--dev-file-path", "-d", required=True, type=str,
                        help="The absolute path to the dev data")
    parser.add_argument("--vclaims-dir-path", "-v", required=True, type=str,
                        help="The absolute path to the directory with the verified claim documents")
    parser.add_argument("--subtask", "-m", required=True, 
                        choices=['2a', '2b'],
                        help="The subtask you want to check the format of.")
    parser.add_argument("--lang", "-l", required=True, type=str,
                        choices=['arabic', 'english'],
                        help="The language of the subtask")
    args = parser.parse_args()
    run_baselines(args.train_file_path, args.dev_file_path, args.vclaims_dir_path, args.lang, args.subtask)
