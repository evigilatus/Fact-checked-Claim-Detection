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
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, model_from_json
from tqdm import tqdm

from util.preprocessing_util import separate_words, remove_new_lines

sys.path.append('.')

from scorer.main import evaluate

random.seed(0)
ROOT_DIR = dirname(dirname(__file__))

sbert = SentenceTransformer('paraphrase-distilroberta-base-v1')
num_sentences = 5


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
    index = 0
    for vclaim_fp in vclaims_fp:
        with open(vclaim_fp) as f:
            vclaim = json.load(f)
        vclaim['file_id'] = index
        vclaims[vclaim['vclaim_id']] = vclaim
        vclaims_list.append(vclaim)
        index = index + 1
    return vclaims, vclaims_list


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


def format_scores(scores):
    output_string = ''
    for iclaim_id in scores:
        for i, (vclaim_id, score) in enumerate(scores[iclaim_id]):
            output_string += f"{iclaim_id}\tQ0\t{vclaim_id}\t{i + 1}\t{score}\tsbert\n"
    return output_string


def get_sbert_body_scores(input_embeddings, vclaim_embeddings, num_sentences):
    sbert_vclaims_text_scores = np.zeros((len(input_embeddings), num_sentences, len(vclaim_embeddings)))

    for vclaim_id, sbert_embeddings in enumerate(tqdm(vclaim_embeddings)):
        if not len(sbert_embeddings):
            continue

        vclaim_text_score = cosine_similarity(input_embeddings,
                                              sbert_embeddings)  # util.semantic_search(input_embedding, sbert_embeddings, top_k = num_sentences)
        scores = []

        vclaim_text_score = np.sort(vclaim_text_score)
        n = min(num_sentences, len(sbert_embeddings))
        sbert_vclaims_text_scores[:, :n, vclaim_id] = vclaim_text_score[:, -n:]

    return sbert_vclaims_text_scores.transpose((0, 2, 1))


def preprocess_iclaim(iclaim):
    return separate_words(iclaim)


def preprocess_vclaim(vclaim):
    return separate_words(remove_new_lines(vclaim['title'] + " " + vclaim['text']))


def create_classifier(train_labels, train_embeddings, vclaim_embeddings):
    # Define the model
    model = Sequential()
    model.add(Dense(20, input_dim=num_sentences, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Compute class weights
    total = len(train_labels.reshape((-1, 1)))
    pos = train_labels.reshape((-1)).sum()
    neg = total - pos

    # Scaling by total/2 helps keep the loss to a similar magnitude.
    # The sum of the weights of all examples stays the same.
    weight_for_0 = (1 / neg) * (total) / 2.0
    weight_for_1 = (1 / pos) * (total) / 2.0

    class_weight = {0: weight_for_0, 1: weight_for_1}

    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))

    # Obtain the embeddings and scores to train the MLP (top-4 sentences per each article)
    train_embeddings = get_sbert_body_scores(train_embeddings, vclaim_embeddings, num_sentences)

    logging.info(f"Train embeddings shape-{train_embeddings.reshape((-1, num_sentences))}")
    logging.info(f"Train labels shape-{train_labels.reshape((-1, 1))}")

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    model.fit(train_embeddings.reshape((-1, num_sentences)),
              train_labels.reshape((-1, 1)),
              epochs=50,
              batch_size=2048,
              class_weight=class_weight,
              callbacks=[callback])

    return model


def predict(model, iclaim_embeddings, vclaim_embeddings, iclaims, vclaims_list):
    test_scores = get_sbert_body_scores(iclaim_embeddings, vclaim_embeddings, num_sentences)
    predictions = model.predict(test_scores.reshape((-1, num_sentences)))

    return map_predictions(predictions, iclaims, vclaims_list)


def map_predictions(predictions, iclaims, vclaims_list):
    splitted_predictions = predictions.reshape(len(iclaims), len(vclaims_list))
    all_scores = {}
    for i, iclaim in enumerate(iclaims):
        iclaim_score = {}
        results = splitted_predictions[i]
        for j, result in enumerate(results):
            vclaim_id = vclaims_list[j]['vclaim_id']
            iclaim_score[vclaim_id] = result
        all_scores[iclaim[0]] = sorted(list(iclaim_score.items()), key=lambda x: x[1], reverse=True)

    return all_scores


def get_labels(vclaim_ids, verified_claims):
    labels = np.zeros((len(vclaim_ids), len(verified_claims)))

    for i, vclaim_id in enumerate(vclaim_ids):
        labels[i][verified_claims[vclaim_id]['file_id']] = 1
    return labels


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

    dev_dataset = pd.read_csv(args.dev_file_path, sep='\t', names=['iclaim_id', '0', 'vclaim_id', 'relevance'])
    train_dataset = pd.read_csv(args.train_file_path, sep='\t', names=['iclaim_id', '0', 'vclaim_id', 'relevance'])

    train_encodings, iclaims_encodings, vclaim_encodings, dev_encodings = get_encodings(args, all_iclaims,
                                                                                        train_dataset, iclaims,
                                                                                        vclaims_list, dev_dataset)

    # Classify S-BERT scores
    train_labels = get_labels(train_dataset.vclaim_id, vclaims)

    if args.model_path:
        json_file = open(args.model_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        classifier = model_from_json(loaded_model_json)
        classifier.load_weights(args.weights_path)
        logging.info(f"Loaded model from {args.weights_path}")
    else:
        classifier = create_classifier(train_labels, train_encodings, vclaim_encodings)
        if args.store_model:
            model_json = classifier.to_json()
            # TODO: Fix this so that it works without previously created model directory
            with open("model/classifier.json", "w") as json_file:
                json_file.write(model_json)
            # Serialize weights to HDF5
            classifier.save_weights("model/classifier.h5")
            logging.info("Saved model to disk")

    predictions = predict(classifier, dev_encodings, vclaim_encodings, iclaims, vclaims_list)

    # options are title, vclaim, text
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

# translated 2a
# python baselines/classifier.py --train-file-path=data/subtask-2a--english/qrels-train.tsv --dev-file-path=data/subtask-2a--english/qrels-dev.tsv --vclaims-dir-path=data/subtask-2a--english/vclaims --iclaims-file-path=data/subtask-2a--english/tweets-train-dev.tsv --subtask=2a --lang=english --iclaims-embeddings-path=embeddings/2a/iclaims_embeddings.npy --vclaims-embeddings-path=embeddings/2a/vclaims_embeddings.npy --dev-embeddings-path=embeddings/2a/dclaims_embeddings.npy --train-embeddings-path=embeddings/2a/tclaims_embeddings.npy --model-path=model/2a/translated_similarity/classifier.json --weights-path=model/2a/translated_similarity/classifier.h5
# cosine 2a
# python baselines/classifier.py --train-file-path=data/subtask-2a--english/qrels-train.tsv --dev-file-path=data/subtask-2a--english/qrels-dev.tsv --vclaims-dir-path=data/subtask-2a--english/vclaims --iclaims-file-path=data/subtask-2a--english/tweets-train-dev.tsv --subtask=2a --lang=english --iclaims-embeddings-path=embeddings/2a/iclaims_embeddings.npy --vclaims-embeddings-path=embeddings/2a/vclaims_embeddings.npy --dev-embeddings-path=embeddings/2a/dclaims_embeddings.npy --train-embeddings-path=embeddings/2a/tclaims_embeddings.npy --model-path=model/2a/cosine_similarity/classifier.json --weights-path=model/2a/cosine_similarity/classifier.h5
# train 2a
# python baselines/classifier.py --train-file-path=data/subtask-2a--english/qrels-train.tsv --dev-file-path=data/subtask-2a--english/qrels-dev.tsv --vclaims-dir-path=data/subtask-2a--english/vclaims --iclaims-file-path=data/subtask-2a--english/tweets-train-dev.tsv --subtask=2a --lang=english --iclaims-embeddings-path=embeddings/2a/iclaims_embeddings.npy --vclaims-embeddings-path=embeddings/2a/vclaims_embeddings.npy --dev-embeddings-path=embeddings/2a/dclaims_embeddings.npy --train-embeddings-path=embeddings/2a/tclaims_embeddings.npy

# train 2b
# python baselines/classifier.py --train-file-path=data/subtask-2b--english/v1/train.tsv --dev-file-path=data/subtask-2b--english/v1/dev.tsv --vclaims-dir-path=data/subtask-2b--english/politifact-vclaims --iclaims-file-path=data/subtask-2b--english/v1/iclaims.queries --subtask=2b --lang=english --iclaims-embeddings-path=embeddings/2b/iclaims_embeddings.npy --vclaims-embeddings-path=embeddings/2b/vclaims_embeddings.npy --dev-embeddings-path=embeddings/2b/dclaims_embeddings.npy --train-embeddings-path=embeddings/2b/tclaims_embeddings.npy

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
                        help="The absolute path to embeddings to be used for train")
    parser.add_argument("--dev-embeddings-path", "-de", required=False, type=str,
                        help="The absolute path to embeddings to be used for dev")
    parser.add_argument("--store-embeddings", "-se", required=False, type=bool, default=False,
                        help="If  true, store all computed embeddings")
    parser.add_argument("--store-model", "-sm", required=False, type=bool, default=False,
                        help="If true, store the model weights. Please create a model directory first.")
    parser.add_argument("--model-path", "-mp", required=False, type=str,
                        help="The absolute path to the model")
    parser.add_argument("--weights-path", "-wp", required=False, type=str,
                        help="The absolute path to the model weights")

    args = parser.parse_args()
    run_baselines(args)
