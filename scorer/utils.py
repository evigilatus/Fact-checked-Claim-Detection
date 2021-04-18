import pdb
import logging
import argparse
import os

import sys

def get_threshold_line_format(thresholds, last_entry_name):
    threshold_line_format = '{:<30}' + "".join(['@{:<9}'.format(ind) for ind in thresholds])
    if last_entry_name:
        threshold_line_format = threshold_line_format + '{:<9}'.format(last_entry_name)
    return threshold_line_format

def print_thresholded_metric(title, thresholds, data, last_entry_name=None, last_entry_value=None):
    line_separator = '=' * 120
    threshold_line_format = get_threshold_line_format(thresholds, last_entry_name)
    items = data
    if last_entry_value is not None:
        items = items + [last_entry_value]
    logging.info(threshold_line_format.format(title))
    logging.info('{:<30}'.format("") + "".join(['{0:<10.4f}'.format(item) for item in items]))
    logging.info(line_separator)

def print_single_metric(title, value):
    line_separator = '=' * 120
    logging.info('{:<30}'.format(title) + '{0:<10.4f}'.format(value))
    logging.info(line_separator)