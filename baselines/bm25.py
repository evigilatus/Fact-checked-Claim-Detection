import logging
import logging
import os
import random
import sys
from os.path import dirname, exists

from baselines.random_baseline import load_vclaims
from baselines.util.preprocessing_util import load_iclaims

sys.path.append('.')

from baselines.util.baselines_util import print_evaluation, create_args_parser
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
    iclaims, all_claims = load_iclaims(args)

    index = f"{args.subtask}-{args.lang}"

    bm25_obj, dictionary = get_bm25([vclaim['text'] for vclaim in vclaims_list])
    print(dictionary)
    scores = get_scores(iclaims, bm25_obj, dictionary, vclaims_list, index, search_keys=args.keys, size=args.size)
    print_evaluation(args, scores, ROOT_DIR)


if __name__ == '__main__':
    parser = create_args_parser()
    args = parser.parse_args()
    run_baselines(args)
