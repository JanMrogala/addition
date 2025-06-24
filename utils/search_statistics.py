import json
import argparse
import pickle
import os

parser = argparse.ArgumentParser()
parser.add_argument("--num_of_samples", type=str)
parser.add_argument("--num_of_chains", type=int)
parser.add_argument("--num_of_nodes", type=int)
parser.add_argument("--max_rules", type=int)
parser.add_argument("--duration", type=int)

args = parser.parse_args()

with open(f"data/t_search/A/train.json") as f:
        train = json.load(f)

with open(f"data/t_search/A/test.json") as f:
        test = json.load(f)

if not os.path.exists("data/stats.pkl"):
    stats_list = []
else:
    with open("data/stats.pkl", "rb") as f:
        stats_list = pickle.load(f)

train_len = len(train)
test_len = len(test)

combined_len = train_len + test_len
num_of_samples = args.num_of_samples
num_of_chains = args.num_of_chains
num_of_nodes = args.num_of_nodes
max_rules = args.max_rules
duration = args.duration

stats_list.append({"combined_len": combined_len, "duration_seconds": duration, "num_of_samples": num_of_samples, "num_of_chains": num_of_chains, "num_of_nodes": num_of_nodes, "max_rules": max_rules})

with open("data/stats.pkl", "wb") as f:
    pickle.dump(stats_list, f)

