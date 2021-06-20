import argparse
import logging
from os.path import join, basename

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from scorer.main import evaluate


# S-BERT Experiment
def get_sbert_body_scores(input_embeddings, vclaim_embeddings, num_sentences):
    sbert_vclaims_text_scores = np.zeros((len(input_embeddings), num_sentences, len(vclaim_embeddings)))

    for vclaim_id, sbert_embeddings in enumerate(tqdm(vclaim_embeddings)):
        if not len(sbert_embeddings):
            continue

        vclaim_text_score = cosine_similarity(input_embeddings,
                                              sbert_embeddings)

        vclaim_text_score = np.sort(vclaim_text_score)
        n = min(num_sentences, len(sbert_embeddings))
        sbert_vclaims_text_scores[:, :n, vclaim_id] = vclaim_text_score[:, -n:]

    print(sbert_vclaims_text_scores.shape)

    return sbert_vclaims_text_scores.transpose((0, 2, 1))


def get_labels(vclaim_ids, verified_claims):
    labels = np.zeros((len(vclaim_ids), len(verified_claims)))

    for i, vclaim_id in enumerate(vclaim_ids):
        #verified_claims[vclaim_id]['file_id']
        labels[i][verified_claims[vclaim_id]['file_id']] = 1
    return labels

# General


def format_scores(scores):
    output_string = ''
    for iclaim_id in scores:
        for i, (vclaim_id, score) in enumerate(scores[iclaim_id]):
            output_string += f"{iclaim_id}\tQ0\t{vclaim_id}\t{i + 1}\t{score}\tsbert\n"
    return output_string


def create_args_parser():
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

    return parser


def print_evaluation(args, predictions, ROOT_DIR):
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