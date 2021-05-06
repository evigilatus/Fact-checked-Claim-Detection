import argparse
import json
import logging
import os
import random
import sys
from glob import glob
from os.path import join, dirname, basename, exists
from sklearn.metrics.pairwise import cosine_similarity


import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tqdm import tqdm

sys.path.append('.')

from scorer.main import evaluate

random.seed(0)
ROOT_DIR = dirname(dirname(__file__))
sbert = SentenceTransformer('paraphrase-distilroberta-base-v1')
num_sentences = 4


def load_dataset(ids_fp, pairs_fp):
    INPUT_COLUMNS = ["sentence_id", "sentence"]
    dataset = pd.read_csv(ids_fp,
                            sep='\t',
                            header=None,
                            names=INPUT_COLUMNS,
                            index_col=False)
    dataset['vclaim_ids'] = [[] for _ in range(len(dataset))]
    with open(pairs_fp) as f:
        for line in f.readlines():
            sentence_id, vclaim_id = line.strip().split('\t')
            sentence_id, vclaim_id = int(sentence_id), int(vclaim_id)
            row = dataset[dataset.sentence_id == sentence_id].iloc[0]
            row.vclaim_ids.append(vclaim_id)
    return dataset

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


def get_score(iclaim_encodding, vclaims_list, vclaim_encodings):
    score = {}

    for i, vclaim in enumerate(vclaim_encodings):
        results = util.semantic_search(iclaim_encodding, vclaim, top_k=5)
        result_sum = 0
        for j, result in enumerate(results):
            result_sum += result[j]['score']
        average = result_sum / len(results)
        vclaim_id = vclaims_list[i]['vclaim_id']
        score[vclaim_id] = average
    score = sorted(list(score.items()), key=lambda x: x[1], reverse=True)

    return score


def get_encodings(args, iclaims, vclaims_list, tclaims):
    iclaims_count, vclaims_count = len(iclaims), len(vclaims_list)
    scores = {}

    if args.train_embeddings_path:
        # Load train data encodings from path
        train_encodings = np.load(args.train_embeddings_path, allow_pickle=True)
        logging.info("All train embeddings loaded successfully.")
    else:
        # Compute the encodings for the train data
        train_encodings = [sbert.encode(tclaim) for tclaim in tclaims]
        if args.store_embeddings:
            np.save('embeddings/tclaims_embeddings.npy', np.array(train_encodings))
        logging.info("All train claims encoded successfully.")

    if args.iclaims_embeddings_path:
        # Load encodings from path
        iclaims_encodings = np.load(args.iclaims_embeddings_path, allow_pickle=True)
        logging.info("All iclaims embeddings loaded successfully.")
    else:
        # Compute the encodings for all iclaims
        iclaims_encodings = [sbert.encode(iclaim) for iclaim in iclaims]
        if args.store_embeddings:
            np.save('embeddings/iclaims_embeddings.npy', np.array(iclaims_encodings))
        logging.info("All iclaims encoded successfully.")

    if args.vclaims_embeddings_path:
        # Load encodings from path
        vclaim_encodings = np.load(args.vclaims_embeddings_path, allow_pickle=True)
        logging.info("All vclaims embeddings loaded successfully.")
    else:
        # Compute the encodings for all vclaims in all texts
        texts = [vclaim['text'] for vclaim in vclaims_list]
        vclaim_encodings = [sbert.encode(sent_tokenize(text)) for text in texts]
        if args.store_embeddings:
            np.save('embeddings/vclaims_embeddings.npy', np.array(vclaim_encodings))
        logging.info("All vclaims encoded successfully.")

    return train_encodings, iclaims_encodings, vclaim_encodings


def get_scores(iclaims, vclaims_list, iclaims_encodings, vclaim_encodings):
    iclaims_count, vclaims_count = len(iclaims), len(vclaims_list)
    scores = {}

    logging.info(f"Geting RM5 scores for {iclaims_count} iclaims and {vclaims_count} vclaims")
    counter = 0

    for iclaim_id, iclaim in iclaims:
        score = get_score(iclaims_encodings[counter][1], vclaims_list, vclaim_encodings)
        counter += 1
        scores[iclaim_id] = score
    return scores


def format_scores(scores):
    output_string = ''
    for iclaim_id in scores:
        for i, (vclaim_id, score) in enumerate(scores[iclaim_id]):
            output_string += f"{iclaim_id}\tQ0\t{vclaim_id}\t{i + 1}\t{score}\tsbert\n"
    return output_string


def create_classifier(train_encodings):
    # Define the model
    model = Sequential()
    model.add(Dense(20, input_dim=num_sentences, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit the keras model on the dataset
    model.fit(train_encodings,
              epochs=15,
              batch_size=2048)

    return model


def get_sbert_body_scores(input_embeddings, vclaim_embeddings, num_sentences):
    sbert_vclaims_text_scores = np.zeros((len(input_embeddings), num_sentences, len(vclaim_embeddings)))
    for vclaim_id, sbert_embeddings in enumerate(tqdm(vclaim_embeddings)):
        if not len(sbert_embeddings):
            continue
        vclaim_text_score = cosine_similarity(input_embeddings, sbert_embeddings)
        vclaim_text_score = np.sort(vclaim_text_score)
        n = min(num_sentences, len(sbert_embeddings))
        sbert_vclaims_text_scores[:, :n, vclaim_id] = vclaim_text_score[:, -n:]

    return sbert_vclaims_text_scores.transpose((0, 2, 1))

def predict(model, iclaim_embeddings, vclaim_embeddings):
    test_scores = get_sbert_body_scores(iclaim_embeddings, vclaim_embeddings, num_sentences)
    return model.predict(test_scores.reshape((-1, num_sentences)))

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

    tclaims = pd.read_csv(args.dev_file_path, sep='\t', names=['iclaim_id', '0', 'vclaim_id', 'relevance'])

    train_encodings, iclaims_encodings, vclaim_encodings = get_encodings(args, iclaims, vclaims_list, tclaims)

    # Classify S-BERT scores
    classifier = create_classifier(train_encodings)
    predictions = predict(classifier, iclaims_encodings, vclaim_encodings)
    # options are title, vclaim, text
    # scores = get_scores(iclaims, vclaims_list, iclaims_encodings, vclaim_encodings)
    ngram_baseline_fpath = join(ROOT_DIR,
                                f'baselines/data/subtask_{args.subtask}_sbert_{args.lang}_{basename(args.dev_file_path)}')
    formatted_scores = format_scores(predictions)
    with open(ngram_baseline_fpath, 'w') as f:
        f.write(formatted_scores)
    maps, mrr, precisions = evaluate(args.dev_file_path, ngram_baseline_fpath)
    logging.info(f"S-BERT Baseline for Subtask-{args.subtask}--{args.lang}")
    logging.info(f'All MAP scores on threshold from [1, 3, 5, 10, 20, 50, 1000]. {maps}')
    logging.info(f'MRR score {mrr}')
    logging.info(f'All P scores on threshold from [1, 3, 5, 10, 20, 50, 1000]. {precisions}')


# python baselines/bm25.py --train-file-path=baselines/v1/train.tsv --dev-file-path=baselines/v1/dev.tsv
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
    parser.add_argument("--size", "-s", default=19250,
                        help="Maximum results extracted for a query")
    parser.add_argument("--subtask", "-m", required=True,
                        choices=['2a', '2b'],
                        help="The subtask you want to check the format of.")
    parser.add_argument("--lang", "-l", required=True, type=str,
                        choices=['arabic', 'english'],
                        help="The language of the subtask")
    parser.add_argument("--iclaims-embeddings-path", "-ie", required=False, type=str,
                        help="The absolute path to embeddings to be used for iclaims")
    parser.add_argument("--vclaims-embeddings-path", "-ve", required=False, type=str,
                        help="The absolute path to embeddings to be used for vclaims")
    parser.add_argument("--train-embeddings-path", "-te", required=False, type=str,
                        help="The absolute path to embeddings to be used for test")
    parser.add_argument("--store-embeddings", "-se", required=False, type=bool, default=False,
                        help="The absolute path to embeddings to be used for vclaims")

    args = parser.parse_args()
    run_baselines(args)
