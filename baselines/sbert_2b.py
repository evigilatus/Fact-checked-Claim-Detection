import json
import json
import logging
import os
import random
import sys
from os.path import dirname, exists

import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

sys.path.append('.')

from baselines.util.baselines_util import create_args_parser, print_evaluation, get_labels
from baselines.util.classifier import predict, load_classifier
from baselines.util.preprocessing_util import preprocess_iclaim, preprocess_vclaim, \
    parse_datasets, load_vclaims, load_iclaims

random.seed(0)
ROOT_DIR = dirname(dirname(__file__))
DATA_DIR = "data/subtask-2b--english/politifact-vclaims/"

sbert = SentenceTransformer('paraphrase-distilroberta-base-v1')
num_sentences = 4


def get_encodings(args, all_iclaims, tclaims, iclaims, vclaims_list, dclaims):
    if args.train_embeddings_path:
        # Load train data encodings from path
        train_encodings = np.load(args.train_embeddings_path, allow_pickle=True)
        logging.info("All train embeddings loaded successfully.")
    else:
        # Compute the encodings for the train data
        train_encodings = []
        tclaim_ids = tclaims.iclaim_id.tolist()
        for iclaim_id in tclaim_ids:
            iclaim = all_iclaims.iclaim[all_iclaims.iclaim_id == iclaim_id].iloc[0]
            train_encodings.append(sbert.encode(preprocess_iclaim(iclaim)))

        if args.store_embeddings:
            np.save('embeddings/tclaims_embeddings.npy', np.array(train_encodings))
        logging.info("All train claims encoded successfully.")

    if args.iclaims_embeddings_path:
        # Load encodings from path
        iclaims_encodings = np.load(args.iclaims_embeddings_path, allow_pickle=True)
        logging.info("All iclaims embeddings loaded successfully.")
    else:
        # Compute the encodings for all iclaims
        iclaims_encodings = [sbert.encode(preprocess_iclaim(iclaim[1])) for iclaim in iclaims]
        if args.store_embeddings:
            np.save('embeddings/iclaims_embeddings.npy', np.array(iclaims_encodings))
        logging.info("All iclaims encoded successfully.")

    if args.vclaims_embeddings_path:
        # Load encodings from path
        vclaim_encodings = np.load(args.vclaims_embeddings_path, allow_pickle=True)
        logging.info("All vclaims embeddings loaded successfully.")
    else:
        # Compute the encodings for all vclaims in all texts
        texts = [preprocess_vclaim(vclaim) for vclaim in vclaims_list]
        vclaim_encodings = [sbert.encode(sent_tokenize(text)) for text in texts]
        if args.store_embeddings:
            np.save('embeddings/vclaims_embeddings.npy', np.array(vclaim_encodings))
        logging.info("All vclaims encoded successfully.")

    if args.dev_embeddings_path:
        # Load encodings from path
        dclaim_encodings = np.load(args.dev_embeddings_path, allow_pickle=True)
        logging.info("All dclaims embeddings loaded successfully.")
    else:
        # Compute the encodings for all vclaims in all texts
        dclaim_encodings = []
        dclaim_ids = dclaims.iclaim_id.tolist()
        for iclaim_id in dclaim_ids:
            iclaim = all_iclaims.iclaim[all_iclaims.iclaim_id == iclaim_id].iloc[0]
            dclaim_encodings.append(sbert.encode(preprocess_iclaim(iclaim)))

        if args.store_embeddings:
            np.save('embeddings/dclaims_embeddings.npy', np.array(dclaim_encodings))
        logging.info("All dclaims encoded successfully.")

    return train_encodings, iclaims_encodings, vclaim_encodings, dclaim_encodings


def run_baselines(args):
    if not exists('baselines/data'):
        os.mkdir('baselines/data')
    vclaims, vclaims_list = load_vclaims(args.vclaims_dir_path)
    iclaims, all_iclaims = load_iclaims(args)
    dev_dataset, train_dataset = parse_datasets(args)
    train_encodings, iclaims_encodings, vclaim_encodings, dev_encodings = get_encodings(args, all_iclaims,
                                                                                        train_dataset, iclaims,
                                                                                        vclaims_list, dev_dataset)
    # Classify S-BERT scores
    train_labels = get_labels(train_dataset.vclaim_id, vclaims)
    classifier = load_classifier(args, train_labels, train_encodings, vclaim_encodings)
    predictions = predict(classifier, dev_encodings, vclaim_encodings, iclaims, vclaims_list)
    print_evaluation(args, predictions, ROOT_DIR)


# python baselines/bm25.py --train-file-path=baselines/v1/train.tsv --dev-file-path=baselines/v1/dev.tsv
# --vclaims-dir-path=baselines/politifact-vclaims --iclaims-file-path=baselines/v1/iclaims.queries --subtask=2b
# --lang=english
if __name__ == '__main__':
    parser = create_args_parser()
    args = parser.parse_args()
    run_baselines(args)

