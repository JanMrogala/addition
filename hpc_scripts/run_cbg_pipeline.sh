samples=50000
num_of_chains=20
num_of_nodes=20
max_rules=2
cross_ratio1=0.5

function run_cbg_pipeline {
    local samples=$1
    local num_of_chains=$2
    local num_of_nodes=$3
    local max_rules=$4
    local cross_ratio1=$5
    local file=$6
    local out_name=$7

    python data_generation/0_generate_chains.py \
        --n $num_of_chains \
        --m $num_of_nodes \
        --max_rules $max_rules \
        --cross_ratio1 $cross_ratio1 \

    python data_generation/1_generate_traces.py \
        --file_name $file \
        --num_of_samples $samples \

    python data_generation/2_generate_snapshot_batch.py \
        --max_nodes $num_of_nodes \

    python data_generation/3_generate_data.py

    python data_generation/4_convert_to_json.py

    zip data/${out_name} data/*.json
}

# Call the function with the parameters
run_cbg_pipeline $samples $num_of_chains $num_of_nodes $max_rules $cross_ratio1 "automata1.pkl" "automata1.zip"
run_cbg_pipeline $samples $num_of_chains $num_of_nodes $max_rules $cross_ratio1 "automata2.pkl" "automata2.zip"
run_cbg_pipeline $samples $num_of_chains $num_of_nodes $max_rules $cross_ratio1 "res_G.pkl" "res_G.zip"

