export ROOT_DIR="<REPO_DIR>"

export dataset=topiocqa
export MODEL=llama2-7b
export split=dev
export ret=bm25

if [ "$split" = "dev" ] || [ "$split" = "dev_100" ]; then
    export gold_qrel_path=$ROOT_DIR/datasets/$dataset/dev_gold.trec
else
    export gold_qrel_path=$ROOT_DIR/datasets/$dataset/train_gold.trec
fi


export DISTILL=oqf-qr-bm25
# for method in "dpo/adaptive_random8_from_qe"
for method in "sft_from_qe"
do
    export input_path=$ROOT_DIR/outputs/$dataset/$MODEL/$DISTILL/$method/$split".json"
    export output_path=$ROOT_DIR/results/$dataset/$ret/$MODEL/$DISTILL/$method/$split

    export inst_type_path=$ROOT_DIR/datasets/$dataset/$split"_type.json"

    python bm25_topiocqa.py \
    --split $split \
    --input_query_path $input_path \
    --output_dir_path $output_path \
    --inst_type_path $inst_type_path \
    --index_dir_path $ROOT_DIR/datasets/topiocqa/pyserini_index \
    --gold_qrel_file_path $gold_qrel_path \
    --query_type rewrite \
    # --use_wandb \
    # --wandb_project_name "topiocqa_"$split"_bm25" \
    # --wandb_run_name $MODEL"_"$DISTILL"_"$method
done