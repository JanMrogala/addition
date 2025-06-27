samples=100
num_of_chains=8
num_of_nodes=8
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
    local indexing_margin=$8

    python data_generation/0_generate_chains.py \
        --n $num_of_chains \
        --m $num_of_nodes \
        --max_rules $max_rules \
        --cross_ratio1 $cross_ratio1 \
        --indexing_margin $indexing_margin \

    python data_generation/1_generate_traces.py \
        --file_name $file \
        --num_of_samples $samples \
        --indexing_margin $indexing_margin \

    python data_generation/2_generate_snapshot_batch.py \
        --max_nodes $num_of_nodes \
        --max_rules $max_rules \
        --indexing_margin $indexing_margin \

    python data_generation/3_generate_data.py

    python data_generation/4_convert_to_json.py

    # move files to directory
    mkdir -p data/t_search/$out_name
    mv data/*.json data/t_search/$out_name
}

# Call the function with the parameters
run_cbg_pipeline $samples $num_of_chains $num_of_nodes $max_rules $cross_ratio1 "automata1.pkl" "A" 0
run_cbg_pipeline $samples $num_of_chains $num_of_nodes $max_rules $cross_ratio1 "automata2.pkl" "B" 100
run_cbg_pipeline $samples $num_of_chains $num_of_nodes $max_rules $cross_ratio1 "res_G.pkl" "C" 200

python data_generation/t_postprocess.py

python utils/create_tokenizer.py

python utils/validate_generated_data.py

python data_generation/t_postprocess.py