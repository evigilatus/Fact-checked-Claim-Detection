import argparse
import re
import logging
import re
import argparse
from functools import partial

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

COLUMNS = ['qid', 'Q0', 'docno', 'rank', 'score', 'tag']

is_float = partial(re.match, r'^-?\d+(?:\.\d+)?$')

LINE_CHECKS = [
    lambda line: 'Wrong column delimiter' if len(line) == 1 else None,
    lambda line: 'Less columns than expected' if len(line) < len(COLUMNS) else None,
    lambda line: 'More columns than expected' if len(line) > len(COLUMNS) else None,
    lambda line: 'Wrong Q0' if line[COLUMNS.index('Q0')] != 'Q0' else None,
    lambda line: 'The score is not a float' if not is_float(line[COLUMNS.index('score')]) else None,
]

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def check_format(preditions_file_path):
    with open(preditions_file_path) as tsvfile:
        pair_ids = {}
        for line_no, line_str in enumerate(tsvfile, start=1):
            line = line_str.split('\t')
            for check in LINE_CHECKS:
                error = check(line)
                if error is not None:
                    return f'{error} on line {line_no} in file: {preditions_file_path}'

            tweet_id, vclaim_id = line[COLUMNS.index('qid')], line[COLUMNS.index('docno')]
            duplication = pair_ids.get((tweet_id, vclaim_id), False)
            if duplication:
                return f'Duplication of pair(tweet_id={tweet_id}, vclaim_id={vclaim_id}) ' \
                    f'on lines {duplication} and {line_no} in file: {preditions_file_path}'
            else:
                pair_ids[(tweet_id, vclaim_id)] = line_no
    return

def run_checks(prediction_file):
    error = check_format(prediction_file)
    if error:
        print(f"Format check: {bcolors.FAIL}Failed{bcolors.ENDC}")
        print(f"Cause: {bcolors.BOLD}{error}{bcolors.ENDC}")
        return False
    else:
        print(f"Format check: {bcolors.OKGREEN}Passed{bcolors.ENDC}")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-files-path", "-p", required=True, type=str, nargs='+',
                        help="The absolute pathes to the files you want to check.")
    args = parser.parse_args()
    for pred_file_path in args.pred_files_path:
        logging.info(f"Subtask 2: Checking file: {pred_file_path}")
        run_checks(pred_file_path)