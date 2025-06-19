import json

with open(f"data/t_search/{format}/train.json") as f:
        train = json.load(f)

with open(f"data/t_search/{format}/test.json") as f:
        test = json.load(f)

