num_of_chains=20
num_of_nodes=20
file="../../data/automata1.pkl"
# file="../../data/automata2.pkl"
# file="../../data/res_G.pkl"

python 0_generate_chains.py \
    --n $num_of_chains \
    --m $num_of_nodes \
    --max_rules 2 \
    --cross_ratio1 0.5 \

python 1_generate_traces.py \
    --path $file \
    --num_of_samples 50000 \

python 2_generate_snapshot_batch.py \
    --max_nodes $num_of_nodes \

python 3_generate_data.py

python 4_convert_to_json.py



