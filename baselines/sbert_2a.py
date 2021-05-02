import argparse
import json
import logging
import os
import random
import sys
import re
import string
from glob import glob
from os.path import join, dirname, basename, exists

from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import sent_tokenize
import pandas as pd

sys.path.append('.')

from scorer.main import evaluate

random.seed(0)
ROOT_DIR = dirname(dirname(__file__))
sbert = SentenceTransformer('paraphrase-distilroberta-base-v1')

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

def delete_link(text):
    return re.sub(r'(http|pic\.twitter\.com)\S+', '', text) 

def separate_words(text):
    return re.sub(' +', ' ', re.sub('([A-Z][a-z]+)', r' \1', re.sub('(#|@)([A-Z]+)', r'\2', text)))

def remove_last_punctuation(text):
    punctuation = '!"#$%&\'()*+, -./:;<=>?@[\]^_`{|}~”'
    index = len(text) - 1
    while text[index] in punctuation:
        index = index - 1
    return text[:index + 1]

def preprocess_iclaims(iclaim):
    return separate_words(delete_link(iclaim))
    
def preprocess_vclaims(vclaim):
    return remove_last_punctuation(separate_words(vclaim['title'] + " " + vclaim['subtitle'] + " " +  vclaim['vclaim'])) + ' ' +  vclaim['date']

def get_score(iclaim_encodding, vclaims_list, vclaim_encodings, index, search_keys, size=10000):
    score = {}

    for i, vclaim in enumerate(vclaim_encodings):
        results = util.semantic_search(iclaim_encodding, vclaim,  top_k=1)
        result_sum = 0
        for j, result in enumerate(results):
            result_sum += result[j]['score']
        average = result_sum / len(results)
        vclaim_id = vclaims_list[i]['vclaim_id']
        score[vclaim_id] = average
    score = sorted(list(score.items()), key=lambda x: x[1], reverse=True)

    return score


def get_scores(iclaims, vclaims_list, index, search_keys, size):
    iclaims_count, vclaims_count = len(iclaims), len(vclaims_list)
    scores = {}

    # Compute the encodings for all iclaims
    iclaims_encodings = [sbert.encode(preprocess_iclaims(iclaim[1])) for iclaim in iclaims]
    logging.info("All iclaims encoded successfully.")
    

    # Compute the encodings for all vclaims in all texts
    texts = [preprocess_vclaims(vclaim)  for vclaim in vclaims_list]
    vclaim_encodings = [sbert.encode(sent_tokenize(text)) for text in texts]
    logging.info("All vclaims encoded successfully.")
    logging.info(f"Geting RM5 scores for {iclaims_count} iclaims and {vclaims_count} vclaims")
    counter = 0

    for iclaim_id, iclaim in iclaims:
        score = get_score(iclaims_encodings[counter], vclaims_list, vclaim_encodings, index, search_keys=search_keys, size=size)
        counter += 1
        scores[iclaim_id] = score
    return scores


def format_scores(scores):
    output_string = ''
    i = 1
    for iclaim_id in scores:
        i = i + 1
        for i, (vclaim_id, score) in enumerate(scores[iclaim_id]):
            output_string += f"{iclaim_id}\tQ0\t{vclaim_id}\t{i + 1}\t{score}\tsbert\n"
    return output_string


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
    scores = get_scores(iclaims, vclaims_list, index, search_keys=args.keys, size=args.size)
    ngram_baseline_fpath = join(ROOT_DIR,
                                f'baselines/data/subtask_{args.subtask}_sbert_{args.lang}_{basename(args.dev_file_path)}')
    formatted_scores = format_scores(scores)
    with open(ngram_baseline_fpath, 'w') as f:
        f.write(formatted_scores)
    maps, mrr, precisions = evaluate(args.dev_file_path, ngram_baseline_fpath)
    logging.info(f"S-BERT Baseline for Subtask-{args.subtask}--{args.lang}")
    logging.info(f'All MAP scores on threshold from [1, 3, 5, 10, 20, 50, 1000]. {maps}')
    logging.info(f'MRR score {mrr}')
    logging.info(f'All P scores on threshold from [1, 3, 5, 10, 20, 50, 1000]. {precisions}')


# python baselines/bm25.py --train-file-path=baselines/v1/train.tsv --dev-file-path=baselines/v1/train.tsv
# --vclaims-dir-path=baselines/politifact-vclaims --iclaims-file-path=baselines/v1/iclaims.queries --subtask=2b
# --lang=english
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
    parser.add_argument("--subtask", "-m", required=True,
                        choices=['2a', '2b'],
                        help="The subtask you want to check the format of.")
    parser.add_argument("--lang", "-l", required=True, type=str,
                        choices=['arabic', 'english'],
                        help="The language of the subtask")

    args = parser.parse_args()
    run_baselines(args)
