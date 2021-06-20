import json
import re
from glob import glob

import pandas as pd
import enum


class Subtask(enum.Enum):
    A = '2a'
    B = '2b'


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


def remove_new_lines(text):
    return text.replace("\n", "")


def preprocess_iclaim(subtask, iclaim):
    if subtask == Subtask.A:
        return separate_words(delete_link(iclaim))
    if subtask == Subtask.B:
        return separate_words(iclaim)


def preprocess_vclaim(subtask, vclaim):
    if subtask == Subtask.A:
        remove_last_punctuation(
            separate_words(vclaim['title'] + " " + vclaim['subtitle'] + " " + vclaim['vclaim'])) + ' ' + vclaim['date']
    if subtask == Subtask.B:
        return separate_words(remove_new_lines(vclaim['title'] + " " + vclaim['text']))


def load_vclaims(dir):
    vclaims_fp = glob(f'{dir}/*.json')
    # TODO Fix logic for translated data and uncomment the following lines
    # if task == '2b':
    #     translated_dir = dir.replace("politifact", "translated")
    #     vclaims_fp.append(glob(f'{translated_dir}/*.json'))
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


def load_iclaims(args):
    all_iclaims = pd.read_csv(args.iclaims_file_path, sep='\t', names=['iclaim_id', 'iclaim'])
    wanted_iclaim_ids = pd.read_csv(args.dev_file_path, sep='\t', names=['iclaim_id', '0', 'vclaim_id', 'relevance'])
    wanted_iclaim_ids = wanted_iclaim_ids.iclaim_id.tolist()
    iclaims = []
    for iclaim_id in wanted_iclaim_ids:
        iclaim = all_iclaims.iclaim[all_iclaims.iclaim_id == iclaim_id].iloc[0]
        iclaims.append((iclaim_id, iclaim))

    return iclaims, all_iclaims


def parse_datasets(args):
    dev_dataset = pd.read_csv(args.dev_file_path, sep='\t', names=['iclaim_id', '0', 'vclaim_id', 'relevance'])
    train_dataset = pd.read_csv(args.train_file_path, sep='\t', names=['iclaim_id', '0', 'vclaim_id', 'relevance'])

    return dev_dataset, train_dataset

