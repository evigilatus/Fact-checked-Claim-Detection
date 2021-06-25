import json
import numpy as np

from fever_parse import parse_wiki


def parse_mappings():
    coordinates = set()
    with open("data/fever/train.jsonl") as f:
        for line in f:
            record = json.loads(line)
            evidences = get_evidences(record["evidence"])
            for evidence in evidences:
                if len(evidence) != 4:
                    raise ValueError(record)
                coordinates.add((record["id"], evidence[2], evidence[3]))
    print(len(coordinates))
    return coordinates


def get_evidences(evidence_tree):
    evidences = []
    if isinstance(evidence_tree[0], int):
        evidences.append(evidence_tree)
    elif isinstance(evidence_tree, list):
        for l in evidence_tree:
            evidence_list = get_evidences(l)
            evidences.extend(evidence_list)
    else:
        raise ValueError(evidence_tree)
    return evidences


def parse_train():
    records = []
    with open("data/fever/train.jsonl") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def sample_records(records):
    verifiable = [i for i in records if i["verifiable"] == "VERIFIABLE"]
    s = [i for i in verifiable if i["label"] == "SUPPORTS"]
    s = np.random.choice(s, 20000, replace=False)
    r = [i for i in verifiable if i["label"] == "REFUTES"]
    r = np.random.choice(r, 20000, replace=False)
    return np.concatenate((r, s), axis=0)


def shuffle(records):
    rng = np.random.default_rng()
    for _ in range(12):
        rng.shuffle(records)

    return records


def main():
    wiki = parse_wiki()
    coordinates = parse_mappings()
    records = sample_records(parse_train())
    records = shuffle(records)


def dump_wiki_vclaims():
    mappings = parse_mappings()
    wiki = parse_wiki()
    for id, t, s in mappings:
        try:
            v = {"title": t, "vclaim": wiki[t][s], "vclaim_id": str(t) + "-" + str(s)}
        except KeyError:
            pass
        with open(
            "clef_fever/wiki_vclaims/{}.json".format(
                v["vclaim_id"].replace("/", "_").replace("?", "")
            ),
            "w",
        ) as f:
            json.dump(v, f)


if __name__ == "__main__":
    parse_train()
