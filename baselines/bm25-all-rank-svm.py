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

from scorer.main import evaluate
from gensim import corpora
from gensim.summarization import bm25
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


import numpy as np

import joblib

en_stop = set(stopwords.words('english'))

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
    return texts

def get_bm25(docs):
    texts = [preprocess(doc) for doc in docs]  # you can do preprocessing as removing stopwords
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    bm25_obj = bm25.BM25(corpus)
    return [bm25_obj, dictionary]

def get_score(iclaim,
              bm25_title,
              dictionary_title,
              bm25_vclaim,
              dictionary_vclaim,
              bm25_text,
              dictionary_text,
              vclaims_list,
              tfidf_title,
              vectorizer,
              size=10000
              ):
    iclaims = preprocess(iclaim)
    query_doc_title = dictionary_title.doc2bow(iclaims)
    scores_title = bm25_title.get_scores(query_doc_title)
    query_doc_vclaim = dictionary_vclaim.doc2bow(iclaims)
    scores_vclaim = bm25_vclaim.get_scores(query_doc_vclaim)
    query_doc_text = dictionary_text.doc2bow(iclaims)
    scores_text = bm25_text.get_scores(query_doc_text)
    tfidf_iclaim = vectorizer.transform([iclaim])
    tfidf_scores = linear_kernel(tfidf_iclaim, tfidf_title).flatten()
    loaded_rank_svm = joblib.load(join(ROOT_DIR, "baselines/data/rank_svm-6.joblib"))
    scores = [(scores_title[i] + scores_vclaim[i] + scores_text[i]) / 3 for i in range(len(scores_title))]
    X = [
        scores_title,
        scores_vclaim,
        scores_text,
        tfidf_scores,
    ]
    X = np.transpose(X)
    scores = loaded_rank_svm.decision_function(X)
    best_docs = sorted(range(len(scores)), key=lambda i: scores[i])[-size:]
    score = []
    for best_doc in best_docs:
        score.append((vclaims_list[best_doc]['vclaim_id'], scores[best_doc]))
    return score

def get_scores(iclaims, vclaims_list, index, search_keys, size):
    iclaims_count, vclaims_count = len(iclaims), len(vclaims_list)
    titles = [vclaim['title'] for vclaim in vclaims_list]
    bm25_title, dictionary_title = get_bm25(titles)
    bm25_vclaim, dictionary_vclaim = get_bm25([vclaim['vclaim'] for vclaim in vclaims_list])
    bm25_text, dictionary_text = get_bm25([vclaim['text'] for vclaim in vclaims_list])
    vectorizer = TfidfVectorizer(tokenizer=preprocess)
    tfidf_title = vectorizer.fit_transform(titles)
    scores = {}

    logging.info(f"Geting RM5 scores for {iclaims_count} iclaims and {vclaims_count} vclaims")
    for iclaim_id, iclaim in iclaims:
        score = get_score(
            iclaim,
            bm25_title,
            dictionary_title,
            bm25_vclaim,
            dictionary_vclaim,
            bm25_text,
            dictionary_text,
            vclaims_list,
            tfidf_title,
            vectorizer,
            size=size)
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
    vclaims, vclaims_list = load_vclaims(args.vclaims_dir_path)
    all_iclaims = pd.read_csv(args.iclaims_file_path, sep='\t', names=['iclaim_id', 'iclaim'])
    wanted_iclaim_ids = pd.read_csv(args.train_file_path, sep='\t', names=['iclaim_id', '0', 'vclaim_id', 'relevance'])
    wanted_iclaim_ids = wanted_iclaim_ids.iclaim_id.tolist()

    iclaims = []
    for iclaim_id in wanted_iclaim_ids:
        iclaim = all_iclaims.iclaim[all_iclaims.iclaim_id == iclaim_id].iloc[0]
        iclaims.append((iclaim_id, iclaim))
    
    index = f"{args.subtask}-{args.lang}"

    scores = get_scores(iclaims, vclaims_list, index, search_keys=args.keys, size=args.size)

    ngram_baseline_fpath = join(ROOT_DIR, f'baselines/data/subtask_{args.subtask}_bm25-all-rank-svm_{args.lang}_{basename(args.dev_file_path)}')
    formatted_scores = format_scores(scores)
    with open(ngram_baseline_fpath, 'w') as f:
        f.write(formatted_scores)
    maps, mrr, precisions = evaluate(args.dev_file_path, ngram_baseline_fpath)
    logging.info(f"Elasticsearch Baseline for Subtask-{args.subtask}--{args.lang}")
    logging.info(f'All MAP scores on threshold from [1, 3, 5, 10, 20, 50, 1000]. {maps}')
    logging.info(f'MRR score {mrr}')
    logging.info(f'All P scores on threshold from [1, 3, 5, 10, 20, 50, 1000]. {precisions}')

# python baselines/bm25-all-rank-svm.py --train-file-path=baselines/v1/train.tsv --dev-file-path=baselines/v1/train.tsv --vclaims-dir-path=baselines/politifact-vclaims --iclaims-file-path=baselines/v1/iclaims.queries --subtask=2b --lang=english

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
