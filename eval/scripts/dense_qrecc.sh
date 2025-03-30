export ROOT_DIR="<REPO_DIR>"

export ROOT_DIR=/hdd0/chanwoong/conv2doc
export dataset=qrecc
export MODEL=llama2-7b
export split=test


if [ "$split" = "test" ]; then
    export gold_qrel_path=$ROOT_DIR/datasets/$dataset/qrecc_qrel.tsv
else
    export gold_qrel_path=$ROOT_DIR/datasets/$dataset/qrecc_qrel_train.tsv
fi


export DISTILL=oqf-union-ance
export method=dpo/one_pair

export input_path=$ROOT_DIR/distill_outputs/$dataset/$MODEL/$DISTILL/$method/outputs_qrecc_test/cands_meta_list.json
export output_path=$ROOT_DIR/outputs/$dataset/$MODEL/$DISTILL/$method/$split/

CUDA_VISIBLE_DEVICES=0,1,2,3 python test_qrecc.py \
--query_encoder_checkpoint $ROOT_DIR/datasets/checkpoints/ad-hoc-ance-msmarco \
--pretrained_passage_encoder $ROOT_DIR/datasets/checkpoints/ad-hoc-ance-msmarco \
--use_gpu \
--n_gpu 4 \
--trec_gold_qrel_file_path $gold_qrel_path \
--passage_collection_path $ROOT_DIR/datasets/$dataset/qrecc_collection.tsv \
--passage_embeddings_dir_path $ROOT_DIR/datasets/$dataset/embeddings_ance \
--passage_offset2pid_path $ROOT_DIR/datasets/$dataset/tokenized/offset2pid.pickle \
--test_file_path $input_path \
--qrel_output_path $output_path \