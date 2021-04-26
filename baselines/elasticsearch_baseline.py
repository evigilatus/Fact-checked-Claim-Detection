import pdb
import json
import pandas as pd
import random
import numpy as np
import logging
import argparse
import os
from os.path import join, dirname, basename, exists
from tqdm import tqdm
from glob import glob
import random
from elasticsearch import Elasticsearch

import sys
sys.path.append('.')

from scorer.main import evaluate

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
        vclaims[vclaim['vclaim_id']] = vclaim
    return vclaims

def create_connection(conn_string='localhost:9200'):
    logging.debug("Starting ElasticSearch client")
    try:
        es = Elasticsearch([conn_string], sniff_on_start=True)
    except:
        raise ConnectionError(f"Couldn't connect to Elastic Search instance at: {conn_string} \
                                Check if you've started it or if it listens on the port listed above.")
    logging.debug("Elasticsearch connected")
    return es

def clear_index(es, index):
    cleared = True
    try:
        es.indices.delete(index=index)
    except:
        cleared = False
    return cleared

def build_index(es, vclaims, index):
    vclaims_count = len(vclaims)
    logging.info(f"Builing index of {vclaims_count} vclaims")
    clear_index(es, index)
    for vclaim_id in tqdm(vclaims):
        vclaim = vclaims[vclaim_id]
        id = vclaim['vclaim_id']
        if not es.exists(index=index, id=id):
            es.create(index=index, id=id, body=vclaim)

def get_score(es, iclaim, index, search_keys, size=1000):
    query = {"query": {"multi_match": {"query": iclaim, "fields": search_keys}}}
    try:
        response = es.search(index=index, body=query, size=size)
    except:
        logging.error(f"No elasticsearch results for {iclaim}")
        raise

    results = response['hits']['hits']
    score = []
    for result in results:
        score.append((result['_id'], result['_score']))
    return score

def get_scores(es, iclaims, vclaims, index, search_keys, size):
    iclaims_count, vclaims_count = len(iclaims), len(vclaims)
    scores = {}

    logging.info(f"Geting RM5 scores for {iclaims_count} iclaims and {vclaims_count} vclaims")
    for iclaim_id, iclaim in tqdm(iclaims):
        score = get_score(es, iclaim, index, search_keys=search_keys, size=size)
        scores[iclaim_id] = score
    return scores

def format_scores(scores):
    output_string = ''
    for iclaim_id in scores:
        for i, (vclaim_id, score) in enumerate(scores[iclaim_id]):
            output_string += f"{iclaim_id}\tQ0\t{vclaim_id}\t{i+1}\t{score}\telasic\n"
    return output_string

def run_baselines(args):
    if not exists('baselines/data'):
        os.mkdir('baselines/data')
    vclaims = load_vclaims(args.vclaims_dir_path)
    all_iclaims = pd.read_csv(args.iclaims_file_path, sep='\t', names=['iclaim_id', 'iclaim'])
    
    wanted_iclaim_ids = pd.read_csv(args.dev_file_path, sep='\t', names=['iclaim_id', '0', 'vclaim_id', 'relevance'])
    wanted_iclaim_ids = wanted_iclaim_ids.iclaim_id.tolist()

    iclaims = []
    for iclaim_id in wanted_iclaim_ids:
        iclaim = all_iclaims.iclaim[all_iclaims.iclaim_id == iclaim_id].iloc[0]
        iclaims.append((iclaim_id, iclaim))
    
    index = f"{args.subtask}-{args.lang}"

    es = create_connection(args.conn)
    build_index(es, vclaims, index)
    print("here")
    scores = get_scores(es, iclaims, vclaims, index, search_keys=args.keys, size=args.size)
    print("her")
    ngram_baseline_fpath = join(ROOT_DIR, f'baselines/data/subtask_{args.subtask}_elastic_baseline_{args.lang}_{basename(args.dev_file_path)}')
    formatted_scores = format_scores(scores)
    with open(ngram_baseline_fpath, 'w') as f:
        f.write(formatted_scores)
    maps, mrr, precisions = evaluate(args.dev_file_path, ngram_baseline_fpath)
    logging.info(f"Elasticsearch Baseline for Subtask-{args.subtask}--{args.lang}")
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
    parser.add_argument("--iclaims-file-path", "-i", required=True,
                        help="TSV file with iclaims. Format: iclaim_id iclaim_content")
    parser.add_argument("--keys", "-k", default=['vclaim', 'title'],
                        help="Keys to search in the document")
    parser.add_argument("--size", "-s", default=10000,
                        help="Maximum results extracted for a query")
    parser.add_argument("--conn", "-c", default="localhost:9200",
                        help="HTTP/S URI to a instance of ElasticSearch")
    parser.add_argument("--subtask", "-m", required=True, 
                        choices=['2a', '2b'],
                        help="The subtask you want to check the format of.")
    parser.add_argument("--lang", "-l", required=True, type=str,
                        choices=['arabic', 'english'],
                        help="The language of the subtask")

    args = parser.parse_args()
    run_baselines(args)
