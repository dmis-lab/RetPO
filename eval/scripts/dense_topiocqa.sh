export ROOT_DIR="<REPO_DIR>"

export dataset=topiocqa
export split=dev
export ret=ance
export model=llama2-7b

if [ "$split" = "dev" ] || [ "$split" = "dev_100" ]; then
    export gold_qrel_path=$ROOT_DIR/datasets/$dataset/dev_gold.trec
else
    export gold_qrel_path=$ROOT_DIR/datasets/$dataset/train_gold.trec
fi

if [ -n "$ret" ]; then
  suffix="_$ret"
else
  suffix=""
fi

export mode=oqf-union-bm25
export method=dpo/adaptive_random8_from_qe
export input_path=$ROOT_DIR/distill_outputs/$dataset/$model/$mode/$method/outputs_qrecc_test/$split.json
export output_path=$ROOT_DIR/results/$dataset/$model/$mode/$method/outputs_qrecc_test/

CUDA_VISIBLE_DEVICES=0,1,2,3 python dense_topiocqa.py \
--query_encoder_checkpoint $ROOT_DIR/datasets/checkpoints/ad-hoc-ance-msmarco \
--pretrained_passage_encoder $ROOT_DIR/datasets/checkpoints/ad-hoc-ance-msmarco \
--max_concat_length 128 \
--max_query_length 128 \
--use_gpu \
--n_gpu 4 \
--trec_gold_qrel_file_path $gold_qrel_path \
--passage_collection_path $ROOT_DIR/datasets/$dataset/full_wiki_segments.tsv \
--passage_embeddings_dir_path $ROOT_DIR/datasets/$dataset/embeddings \
--passage_offset2pid_path $ROOT_DIR/datasets/$dataset/tokenized/offset2pid.pickle \
--test_file_path $input_path \
--qrel_output_path $output_path \
# --use_wandb \
# --wandb_project_name $dataset"_"$split"_ance" \
# --wandb_run_name $mode$suffix \
# --wandb_run_name $MODEL"_"$DISTILL"_"$method