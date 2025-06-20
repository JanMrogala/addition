samples=100
# num_of_chains=5
# num_of_nodes=5
# max_rules=4
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
        --max_rules $max_rules \

    python data_generation/3_generate_data.py

    python data_generation/4_convert_to_json.py

    # move files to directory
    mkdir -p data/t_search/$out_name
    mv data/*.json data/t_search/$out_name
}

for chains in {8..10}
do

    for nodes in {8..10}
    do

        for rule in {2..4}
        do

        start_time=$(date +%s)

        # Call the function with the parameters
        run_cbg_pipeline $samples $chains $nodes $rule $cross_ratio1 "automata1.pkl" "A"
        run_cbg_pipeline $samples $chains $nodes $rule $cross_ratio1 "automata2.pkl" "B"
        run_cbg_pipeline $samples $chains $nodes $rule $cross_ratio1 "res_G.pkl" "C"

        python data_generation/t_postprocess.py

        python utils/create_tokenizer.py

        python utils/validate_generated_data.py

        python data_generation/t_postprocess.py

        end_time=$(date +%s)
        duration=$((end_time - start_time))

        python utils/statistics.py \
            --num_of_samples $samples \
            --num_of_chains $chains \
            --num_of_nodes $nodes \
            --max_rules $rule \
            --duration $duration

        done

    done

done

