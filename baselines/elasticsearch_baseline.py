import logging
import os
import random
import sys
from os.path import dirname, exists

from elasticsearch import Elasticsearch
from tqdm import tqdm

from baselines.util.baselines_util import create_args_parser, print_evaluation
from baselines.util.preprocessing_util import load_vclaims, load_iclaims

sys.path.append('.')

random.seed(0)
ROOT_DIR = dirname(dirname(__file__))

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


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




def run_baselines(args):
    if not exists('baselines/data'):
        os.mkdir('baselines/data')
    vclaims, vclaims_list = load_vclaims(args.vclaims_dir_path)
    iclaims, all_iclaims = load_iclaims(args)
    
    index = f"{args.subtask}-{args.lang}"

    es = create_connection(args.conn)
    build_index(es, vclaims, index)
    scores = get_scores(es, iclaims, vclaims, index, search_keys=args.keys, size=args.size)
    print_evaluation(args, scores, ROOT_DIR)


if __name__ == '__main__':
    parser = create_args_parser()
    args = parser.parse_args()
    run_baselines(args)
