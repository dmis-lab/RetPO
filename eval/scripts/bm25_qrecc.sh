export ROOT_DIR="<REPO_DIR>"

export dataset=qrecc
export MODEL=llama2-7b
export split=dev
export ret=bm25

if [ "$split" = "test" ]; then
    export gold_qrel_path=$ROOT_DIR/datasets/$dataset/qrecc_qrel.tsv
else
    export gold_qrel_path=$ROOT_DIR/datasets/$dataset/qrecc_qrel_train.tsv
fi

export DISTILL=oqf-union-bm25

for method in "dpo/one_pair"
do
    export input_path=$ROOT_DIR/distill_outputs/$dataset/$MODEL/$DISTILL/$method/outputs_qrecc_test/cands_meta_list.json
    export output_path=$ROOT_DIR/results/$dataset/$ret/$MODEL/$DISTILL/$method/$split

    export inst_type_path=$ROOT_DIR/datasets/$dataset/$split"_type.json"

    python bm25_qrecc.py \
    --input_query_path $input_path \
    --output_dir_path $output_path \
    --inst_type_path $inst_type_path \
    --index_dir_path $ROOT_DIR/datasets/$dataset/pyserini_index \
    --gold_qrel_file_path $gold_qrel_path \
    --query_type rewrite \

done