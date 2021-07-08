import json


def parse_lines(cnt, index):
    parsed = []
    try:
        jnt = json.loads(cnt)
    except Exception as e:
        print(index, cnt, e.args[0])
        return "", []
    for line in jnt["lines"].split("\n"):
        q = line.split("\t")
        if len(q) > 1:
            parsed.append(q[0])
        else:
            parsed.append("")
    return jnt["id"], parsed


def parse_wiki():
    wiki = {}
    for i in range(1, 110):
        with open(f"data/fever/wiki-pages/wiki-{i:03d}.jsonl") as f:
            for line in f:
                title, lines = parse_lines(line, i)
                wiki[title] = lines
    return wiki

