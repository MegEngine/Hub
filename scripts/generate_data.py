import os
import json
import argparse

import attr

import parser
import tags


p = argparse.ArgumentParser(description="Process some integers.")
p.add_argument("--source", type=str, help=".md source directory", required=True)
p.add_argument("--output", type=str, help="output directory", required=True)


if __name__ == "__main__":
    args = p.parse_args()
    os.makedirs(os.path.join(args.output, "api/v1/models"), exist_ok=True)

    models = []
    for name in os.listdir(args.source):
        if not name.endswith(".md"):
            continue
        if name in ("README.md",):
            continue

        id = name[:-3]
        with open(os.path.join(args.source, name)) as f:
            model = parser.parse_file(id, f)
        data = model.to_jsondict()
        with open(
            os.path.join(args.output, "api/v1/models", "{}.json".format(id)), "w"
        ) as f:
            json.dump(data, f)
        models.append(data["meta"])

    tags_dict = {}
    for t in tags.TAGS:
        tags_dict[t.id] = t.name

    with open(os.path.join(args.output, "api/v1/models.json"), "w") as f:
        data = {
            "models": models,
            "tags": tags_dict,
        }
        json.dump(data, f)
