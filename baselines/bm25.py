import argparse
import json
import logging
import os
import random
import sys
from glob import glob
from os.path import join, dirname, basename, exists

import pandas as pd

sys.path.append('.')

from baselines.util.baselines_util import format_scores, print_evaluation
from gensim import corpora
from gensim.summarization import bm25
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

nltk.download('stopwords')

en_stop = set(stopwords.words('english'))
p_stemmer = PorterStemmer()

random.seed(0)
ROOT_DIR = dirname(dirname(__file__))


def load_vclaims(dir):
    vclaims_fp = glob(f'{dir}/*.json')
    vclaims_fp.sort()
    vclaims = {}
    vclaims_list = []
    for vclaim_fp in vclaims_fp:
        with open(vclaim_fp) as f:
            vclaim = json.load(f)
        vclaims[vclaim['vclaim_id']] = vclaim
        vclaims_list.append(vclaim)
    return vclaims, vclaims_list


def preprocess(doc):
    texts = doc.split()
    lower = [text.lower() for text in texts]
    stopped_tokens = [a for a in lower if not a in en_stop]
    return [p_stemmer.stem(j) for j in stopped_tokens]


def get_bm25(docs):
    texts = [preprocess(doc) for doc in docs]  # you can do preprocessing as removing stopwords
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    bm25_obj = bm25.BM25(corpus)
    return [bm25_obj, dictionary]


def get_score(iclaim, bm25_obj, dictionary, vclaims_list, index, search_keys, size=10000):
    claim_tokens = preprocess(iclaim)
    query_doc = dictionary.doc2bow(claim_tokens)
    scores = bm25_obj.get_scores(query_doc)
    best_docs = sorted(range(len(scores)), key=lambda i: scores[i])[-size:]
    score = []
    for best_doc in best_docs:
        score.append((vclaims_list[best_doc]['vclaim_id'], scores[best_doc]))
    return score


def get_scores(iclaims, bm25_obj, dictionary, vclaims_list, index, search_keys, size):
    iclaims_count, vclaims_count = len(iclaims), len(vclaims_list)
    scores = {}

    logging.info(f"Geting RM5 scores for {iclaims_count} iclaims and {vclaims_count} vclaims")
    for iclaim_id, iclaim in iclaims:
        score = get_score(iclaim, bm25_obj, dictionary, vclaims_list, index, search_keys=search_keys, size=size)
        scores[iclaim_id] = score
    return scores


def run_baselines(args):
    if not exists('baselines/data'):
        os.mkdir('baselines/data')
    vclaims, vclaims_list = load_vclaims(args.vclaims_dir_path)
    all_iclaims = pd.read_csv(args.iclaims_file_path, sep='\t', names=['iclaim_id', 'iclaim'])
    wanted_iclaim_ids = pd.read_csv(args.dev_file_path, sep='\t', names=['iclaim_id', '0', 'vclaim_id', 'relevance'])
    wanted_iclaim_ids = wanted_iclaim_ids.iclaim_id.tolist()

    iclaims = []
    for iclaim_id in wanted_iclaim_ids:
        iclaim = all_iclaims.iclaim[all_iclaims.iclaim_id == iclaim_id].iloc[0]
        iclaims.append((iclaim_id, iclaim))

    index = f"{args.subtask}-{args.lang}"

    # options are title, vclaim, text
    bm25_obj, dictionary = get_bm25([vclaim['text'] for vclaim in vclaims_list])
    print(dictionary)
    scores = get_scores(iclaims, bm25_obj, dictionary, vclaims_list, index, search_keys=args.keys, size=args.size)
    ngram_baseline_fpath = join(ROOT_DIR,
                                f'baselines/data/subtask_{args.subtask}_bm25_{args.lang}_{basename(args.dev_file_path)}')
    formatted_scores = format_scores(scores)
    print_evaluation(args, scores, ROOT_DIR)


# python baselines/bm25.py --train-file-path=baselines/v1/train.tsv --dev-file-path=baselines/v1/train.tsv --vclaims-dir-path=baselines/politifact-vclaims --iclaims-file-path=baselines/v1/iclaims.queries --subtask=2b --lang=english
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
    parser.add_argument("--size", "-s", default=19250,
                        help="Maximum results extracted for a query")
    parser.add_argument("--subtask", "-m", required=True,
                        choices=['2a', '2b'],
                        help="The subtask you want to check the format of.")
    parser.add_argument("--lang", "-l", required=True, type=str,
                        choices=['arabic', 'english'],
                        help="The language of the subtask")

    args = parser.parse_args()
    run_baselines(args)
