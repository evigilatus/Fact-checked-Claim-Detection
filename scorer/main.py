import pdb
import logging
import argparse
import os
import pandas as pd
from trectools import TrecRun, TrecQrel, TrecEval
from os.path import join, dirname, abspath

import sys
sys.path.append('.')
from format_checker.main import check_format
from scorer.utils import print_thresholded_metric, print_single_metric
"""
Scoring of Task 2 with the metrics Average Precision, R-Precision, P@N, RR@N. 
"""

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


MAIN_THRESHOLDS = [1, 3, 5, 10, 20, 50, 1000]

def evaluate(gold_fpath, pred_fpath, thresholds=None):
    """
    Evaluates the predicted line rankings w.r.t. a gold file.
    Metrics are: Average Precision, R-Pr, Reciprocal Rank, Precision@N
    :param gold_fpath: the original annotated gold file, where the last 4th column contains the labels.
    :param pred_fpath: a file with line_number at each line, where the list is ordered by check-worthiness.
    :param thresholds: thresholds used for Reciprocal Rank@N and Precision@N.
    If not specified - 1, 3, 5, 10, 20, 50, len(ranked_lines).
    """
    gold_labels = TrecQrel(gold_fpath)
    prediction = TrecRun(pred_fpath)
    results = TrecEval(prediction, gold_labels)

    # Calculate Metrics
    maps = [results.get_map(depth=i) for i in MAIN_THRESHOLDS]
    mrr = results.get_reciprocal_rank()
    precisions = [results.get_precision(depth=i) for i in MAIN_THRESHOLDS]

    return maps, mrr, precisions


def validate_files(pred_file, subtask):
    if not check_format(pred_file, subtask):
        logging.error('Bad format for pred file {}. Cannot score.'.format(pred_file))
        return False
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-file-path", "-g", required=True, type=str,
                        help="Path to files with gold annotations.")
    
    parser.add_argument("--pred-file-path", "-p", required=True, type=str,
                        help="Path to files with ranked line_numbers.")

    args = parser.parse_args()

    line_separator = '=' * 120
    pred_file = args.pred_file_path
    gold_file = args.gold_file_path

    maps, mrr, precisions = evaluate(gold_file, pred_file)
    filename = os.path.basename(pred_file)
    logging.info('{:=^120}'.format(' RESULTS for {} '.format(filename)))
    print_single_metric('RECIPROCAL RANK:', mrr)
    print_thresholded_metric('PRECISION@N:', MAIN_THRESHOLDS, precisions)
    print_thresholded_metric('MAP@N:', MAIN_THRESHOLDS, maps)

