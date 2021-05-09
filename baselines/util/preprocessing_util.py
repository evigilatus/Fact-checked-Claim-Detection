import json
import os
import re
from glob import glob
from os.path import exists

import pandas as pd


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


def preprocess_iclaims(iclaim):
    return separate_words(delete_link(iclaim))


def preprocess_vclaims(vclaim):
    return remove_last_punctuation(
        separate_words(vclaim['title'] + " " + vclaim['subtitle'] + " " + vclaim['vclaim'])) + ' ' + vclaim['date']


def preprocess_iclaim(iclaim):
    return separate_words(iclaim)


def preprocess_vclaim(vclaim):
    return separate_words(remove_new_lines(vclaim['title'] + " " + vclaim['text']))


def load_vclaims(task, dir):
    vclaims_fp = glob(f'{dir}/*.json')
    if task == '2b':
        translated_dir = dir.replace("politifact", "translated")
        vclaims_fp.append(glob(f'{translated_dir}/*.json'))
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


def parse_claims(args):
    if not exists('baselines/data'):
        os.mkdir('baselines/data')
    vclaims, vclaims_list = load_vclaims(args.subtask, args.vclaims_dir_path)
    all_iclaims = pd.read_csv(args.iclaims_file_path, sep='\t', names=['iclaim_id', 'iclaim'])
    wanted_iclaim_ids = all_iclaims.iclaim_id.tolist()
    iclaims = []
    for iclaim_id in wanted_iclaim_ids:
        iclaim = all_iclaims.iclaim[all_iclaims.iclaim_id == iclaim_id].iloc[0]
        iclaims.append((iclaim_id, iclaim))

    return vclaims, vclaims_list, iclaims, all_iclaims


def parse_datasets(args):
    dev_dataset = pd.read_csv(args.dev_file_path, sep='\t', names=['iclaim_id', '0', 'vclaim_id', 'relevance'])
    train_dataset = pd.read_csv(args.train_file_path, sep='\t', names=['iclaim_id', '0', 'vclaim_id', 'relevance'])

    return dev_dataset, train_dataset

