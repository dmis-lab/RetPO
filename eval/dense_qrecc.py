from IPython import embed
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import sys
sys.path.append('..')
sys.path.append('.')

import csv
import argparse
from src.models import load_model
from utils import check_dir_exist_or_build, pstore, pload, split_and_padding_neighbor, set_seed, load_collection
from data_structure import ConvDataset_rewrite
import os
from os import path
from os.path import join as oj
import json
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
import faiss
import time
import copy
import pickle
import toml
import torch
import numpy as np
import pytrec_eval
import wandb

from eval import print_trec_res

'''
Test process, perform dense retrieval on collection (e.g., MS MARCO):
1. get args
2. establish index with Faiss on GPU for fast dense retrieval
3. load the model, build the test query dataset/dataloader, and get the query embeddings. 
4. iteratively searched on each passage block one by one to got the retrieved scores and passge ids for each query.
5. merge the results on all pasage blocks
6. output the result
'''



def build_faiss_index(args):
    logger.info("Building index...")
    # ngpu = faiss.get_num_gpus()
    ngpu = args.n_gpu
    gpu_resources = []
    tempmem = -1

    for i in range(ngpu):
        res = faiss.StandardGpuResources()
        if tempmem >= 0:
            res.setTempMemory(tempmem)
        gpu_resources.append(res)
        #embed()
        #input()

    cpu_index = faiss.IndexFlatIP(768)  
    index = None
    if args.use_gpu:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.usePrecomputed = False
        # gpu_vector_resources, gpu_devices_vector
        vres = faiss.GpuResourcesVector()
        vdev = faiss.Int32Vector()
        for i in range(0, ngpu):
            vdev.push_back(i)
            vres.push_back(gpu_resources[i])
        #embed()
        #input()
        gpu_index = faiss.index_cpu_to_gpu_multiple(vres,
                                                    vdev,
                                                    cpu_index, co)
        index = gpu_index
    else:
        index = cpu_index

    return index


def search_one_by_one_with_faiss(args, passge_embeddings_dir, index, query_embeddings, topN):
    merged_candidate_matrix = None

    for block_id in range(args.passage_block_num):
        logger.info("Loading passage block " + str(block_id))
        passage_embedding = None
        passage_embedding2id = None
        try:
            with open(
                    oj(
                        passge_embeddings_dir,
                        "passage_emb_block_" + str(block_id) + ".pb"),
                    'rb') as handle:
                passage_embedding = pickle.load(handle)
            with open(
                    oj(
                        passge_embeddings_dir,
                        "passage_embid_block_" + str(block_id) + ".pb"),
                    'rb') as handle:
                passage_embedding2id = pickle.load(handle)
        except:
            break
        logger.info('passage embedding shape: ' + str(passage_embedding.shape))
        logger.info("query embedding shape: " + str(query_embeddings.shape))
        index.add(passage_embedding)

        # ann search
        tb = time.time()
        D, I = index.search(query_embeddings, topN)
        elapse = time.time() - tb
        logger.info({
            'time cost': elapse,
            'query num': query_embeddings.shape[0],
            'time cost per query': elapse / query_embeddings.shape[0]
        })

        candidate_id_matrix = passage_embedding2id[I] # passage_idx -> passage_id
        D = D.tolist()
        candidate_id_matrix = candidate_id_matrix.tolist()
        candidate_matrix = []

        for score_list, passage_list in zip(D, candidate_id_matrix):
            candidate_matrix.append([])
            for score, passage in zip(score_list, passage_list):
                candidate_matrix[-1].append((score, passage))
            assert len(candidate_matrix[-1]) == len(passage_list)
        assert len(candidate_matrix) == I.shape[0]

        index.reset()
        del passage_embedding
        del passage_embedding2id

        if merged_candidate_matrix == None:
            merged_candidate_matrix = candidate_matrix
            continue
        
        # merge
        merged_candidate_matrix_tmp = copy.deepcopy(merged_candidate_matrix)
        merged_candidate_matrix = []
        for merged_list, cur_list in zip(merged_candidate_matrix_tmp,
                                         candidate_matrix):
            p1, p2 = 0, 0
            merged_candidate_matrix.append([])
            while p1 < topN and p2 < topN:
                if merged_list[p1][0] >= cur_list[p2][0]:
                    merged_candidate_matrix[-1].append(merged_list[p1])
                    p1 += 1
                else:
                    merged_candidate_matrix[-1].append(cur_list[p2])
                    p2 += 1
            while p1 < topN:
                merged_candidate_matrix[-1].append(merged_list[p1])
                p1 += 1
            while p2 < topN:
                merged_candidate_matrix[-1].append(cur_list[p2])
                p2 += 1
        
        # for id in range(block_id+1):        
            # logger.info(f"block_id-id: {block_id}-{id}, merged_candidate_matrix length: {len(merged_candidate_matrix)}, inner list length: {len(merged_candidate_matrix[id])}")
            # logger.info(f"block_id-id: {block_id}-{id}, merged_candidate_matrix shape: {np.array(merged_candidate_matrix[id]).shape}")


    merged_D, merged_I = [], []
    for merged_list in merged_candidate_matrix: # len(merged_candidate_matrix) = query_nums len([0]) = query_num * topk
        merged_D.append([])
        merged_I.append([])
        for candidate in merged_list: # len(merged_list) = query_num * topk
            merged_D[-1].append(candidate[0])
            merged_I[-1].append(candidate[1])
    
    merged_D, merged_I = np.array(merged_D), np.array(merged_I)
    logger.info(merged_I)
    logger.info(f"merged_D.shape: {merged_D.shape}")
    logger.info(f"merged_I.shape: {merged_I.shape}")
    
    return merged_D, merged_I


def get_test_query_embedding(args):
    set_seed(args)

    passage_tokenizer, _ = load_model(args.model_type + "_Passage", args.pretrained_passage_encoder)
    query_tokenizer, query_encoder = load_model(args.model_type + "_Query", args.query_encoder_checkpoint)
    query_encoder = query_encoder.to(args.device)

    # test dataset/dataloader
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    logger.info("Buidling test dataset...")

    test_dataset = ConvDataset_rewrite(args, query_tokenizer, args.test_file_path)
    test_loader = DataLoader(test_dataset, 
                            batch_size = args.eval_batch_size, 
                            shuffle=False, 
                            collate_fn=test_dataset.get_collate_fn(args))


    '''
    test_dataset = ConvDataset_qrecc(args, query_tokenizer, passage_tokenizer, args.test_file_path, add_doc_info=False)
    test_loader = DataLoader(test_dataset, 
                            batch_size = args.eval_batch_size, 
                            shuffle=False, 
                            collate_fn=test_dataset.get_collate_fn(args, add_doc_info = False, mode = "test"))
    '''

    logger.info("Generating query embeddings for testing...")
    query_encoder.zero_grad()

    embeddings = []
    embedding2id = []

    with torch.no_grad():
        for batch in tqdm(test_loader, disable=args.disable_tqdm):
            query_encoder.eval()
            batch_sample_id = batch["bt_sample_id"]
            
            # test type
            if args.test_type == "oracle":
                input_ids = batch["bt_oracle_query"].to(args.device)
                input_masks = batch["bt_oracle_query_mask"].to(args.device)
            elif args.test_type == "rewrite":
                input_ids = batch["bt_rewrite"].to(args.device)
                input_masks = batch["bt_rewrite_mask"].to(args.device)
            elif args.test_type == "raw":
                input_ids = batch["bt_raw_query"].to(args.device)
                input_masks = batch["bt_raw_query_mask"].to(args.device)
            elif args.test_type == "convq":
                input_ids = batch["bt_conv_query"].to(args.device)
                input_masks = batch["bt_conv_query_mask"].to(args.device)
            elif args.test_type == "convqa":
                input_ids = batch["bt_conv_query_ans"].to(args.device)
                input_masks = batch["bt_conv_query_ans_mask"].to(args.device)
            # elif args.test_type == "custom":
            #elif args.test_type == "convqp":
            #    input_ids = batch["bt_conv_query_last_res"].to(args.device)
            #    input_masks = batch["bt_conv_query_last_res_mask"].to(args.device)
            else:
                raise ValueError("test type:{}, has not been implemented.".format(args.test_type))
            
            query_embs = query_encoder(input_ids, input_masks)
            query_embs = query_embs.detach().cpu().numpy()
            embeddings.append(query_embs)
            embedding2id.extend(batch_sample_id)

    embeddings = np.concatenate(embeddings, axis = 0)
    torch.cuda.empty_cache()

    return embeddings, embedding2id


def output_test_res(query_embedding2id,
                    retrieved_scores_mat, # score_mat: score matrix, test_query_num * (top_n * block_num)
                    retrieved_pid_mat, # pid_mat: corresponding passage ids
                    offset2pid,
                    args):
    

    qids_to_ranked_candidate_passages = {}
    topN = args.top_n

    for query_idx in range(len(retrieved_pid_mat)):
        seen_pid = set()
        query_id = query_embedding2id[query_idx]

        top_ann_pid = retrieved_pid_mat[query_idx].copy()
        top_ann_score = retrieved_scores_mat[query_idx].copy()
        selected_ann_idx = top_ann_pid[:topN]
        selected_ann_score = top_ann_score[:topN].tolist()
        rank = 0

        if query_id in qids_to_ranked_candidate_passages:
            pass
        else:
            tmp = [(0, 0)] * topN
            tmp_ori = [0] * topN
            qids_to_ranked_candidate_passages[query_id] = tmp

        for idx, score in zip(selected_ann_idx, selected_ann_score):
            pred_pid = offset2pid[idx]

            if not pred_pid in seen_pid:
                qids_to_ranked_candidate_passages[query_id][rank] = (pred_pid, score)
                rank += 1
                seen_pid.add(pred_pid)


    # for case study and more intuitive observation
    logger.info('Loading query and passages\' real text...')
    
    # query
    qid2query = {}
    qid2convid = {}
    qid2turnid = {}
        #qid2query[record["sample_id"]] = record["query"]
        #qid2query[record["sample_id"]] = record["cur_utt_text"]
        #qid2convid[record["id"]] = record["conv_id"]
        #qid2turnid[record["id"]] = record["turn_id"]        
 
    # all passages
    # all_passages = load_collection(args.passage_collection_path)

    # write to file
    logger.info('begin to write the output...')

    #output_file = oj(args.qrel_output_path, "ANCE_q_res_.json")
    output_trec_file = oj(args.qrel_output_path, "ANCE_GPT_QR_res.trec")
    #merged_data = []
    #with open(output_file, "w") as f, open(output_trec_file, "w") as g:
    with open(output_trec_file, "w") as g:
        for qid, passages in qids_to_ranked_candidate_passages.items():
            #query = qid2query[qid]
            #conv_id = qid2convid[qid]
            #turn_id = qid2turnid[qid]
            rank_list = []
            for i in range(topN):
                pid, score = passages[i]
                # passage = all_passages[pid]
                rank_list.append(
                    {
                        "doc_id": str(pid),
                        "rank": i+1,
                        "retrieval_score": score,
                    }
                )
                g.write(str(qid) + " Q0 " + str(pid) + " " + str(i + 1) + " " + str(-i - 1 + 200) + " " + str(score) + " ance\n")
            
            #merged_data.append(
            #    {
            #        "query": query,
            #        "query_id": str(qid),
                    #"conv_id": str(conv_id),
                    #"turn_id": str(turn_id),
            #        "ctxs": rank_list,
            #    })

        #f.write(json.dumps(merged_data, indent=4) + "\n")

    logger.info("output file write ok at {}".format(args.qrel_output_path))
    
    with open(args.test_file_path, 'r') as f:
        data = f.readlines()
    for record in data:
        record = json.loads(record.strip())
    
    queries = []
    for query in data:
        query_json = json.loads(query.strip())
        queries.append(query_json['oracle_utt_text'])
        
    trec_res = print_trec_res(run_file=output_trec_file, 
                              qrel_file=args.trec_gold_qrel_file_path,
                              data=queries,
                              output_path=args.qrel_output_path,
                              type_path=args.inst_type_path,
                              rel_threshold=args.rel_threshold,
                              use_wandb=args.use_wandb
                              )
    
    output_metric_file = os.path.join(args.qrel_output_path, "metrics.txt")
    with open(output_metric_file, 'w') as f:
        f.write(json.dumps(trec_res, indent=4))
   
    return trec_res


def gen_metric_score_and_save(args, index, query_embeddings, query_embedding2id):
    # score_mat: score matrix, test_query_num * (top_n * block_num)
    # pid_mat: corresponding passage ids
    retrieved_scores_mat, retrieved_pid_mat = search_one_by_one_with_faiss(
                                                     args,
                                                     args.passage_embeddings_dir_path, 
                                                     index, 
                                                     query_embeddings, 
                                                     args.top_n) 

    with open(args.passage_offset2pid_path, "rb") as f:
        offset2pid = pickle.load(f)
    
    output_test_res(query_embedding2id,
                    retrieved_scores_mat,
                    retrieved_pid_mat,
                    offset2pid,
                    args)


def main():
    args = get_args()
    set_seed(args)
    
    if args.use_wandb:
        ## wandb setup
        os.environ["WANDB_API_KEY"]=CONFIG['wandb_api_key']
        os.environ["WANDB_PROJECT"]=args.wandb_project_name
        os.environ["WANDB_WATCH"]='false'
        os.environ["WANDB_START_METHOD"]='thread'
        os.environ["WANDB_USER_EMAIL"]=CONFIG['wandb_user_email']
        os.environ["WANDB_USERNAME"]=CONFIG['wandb_username']

        wandb.init()
        wandb.config.update(args)
        wandb.run.name = args.wandb_run_name
    
    output_trec_file = oj(args.qrel_output_path, "ANCE_GPT_QR_res.trec")
    
    if os.path.isfile(output_trec_file):
        logger.info("retrieval result already exist! load ...")

        with open(args.test_file_path, 'r') as f:
            data = f.readlines()
        for record in data:
            record = json.loads(record.strip())
        
        queries = []
        for query in data:
            query_json = json.loads(query.strip())
            queries.append(query_json['oracle_utt_text'])
            
        trec_res = print_trec_res(run_file=output_trec_file, 
                                qrel_file=args.trec_gold_qrel_file_path,
                                data=queries,
                                output_path=args.qrel_output_path, 
                                type_path=args.inst_type_path,
                                rel_threshold=args.rel_threshold,
                                use_wandb=args.use_wandb
                                )
        
        output_metric_file = os.path.join(args.qrel_output_path, "metrics.txt")
        with open(output_metric_file, 'w') as f:
            f.write(json.dumps(trec_res, indent=4))
    
    else:
        logger.info("no retrieval result ...")
        
        index = build_faiss_index(args)

        if not args.cross_validate:
            # args.test_model_path = args.test_model_path + '/epoch - test epoch'
            query_embeddings, query_embedding2id = get_test_query_embedding(args)
        else:
            base_test_file = args.test_file
            base_model_path = args.test_model_path
            NUM_FOLD = 5
            
            total_query_embeddings = []
            total_query_embedding2id = []
            for i in range(NUM_FOLD):
                args.test_file = base_test_file + '.{}'.format(i)
                args.test_model_path = base_model_path + '/fold_{}/epoch-{}'.format(i, args.test_epoch)

                query_embeddings, query_embedding2id = get_test_query_embedding(args)
                total_query_embeddings.append(query_embeddings)
                total_query_embedding2id.extend(query_embedding2id)

            total_query_embeddings = np.concatenate(total_query_embeddings, axis = 0)
            query_embeddings = total_query_embeddings
            query_embedding2id = total_query_embedding2id
            args.test_file = base_test_file


        gen_metric_score_and_save(args, index, query_embeddings, query_embedding2id)

    logger.info("Test finish!")
    

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--query_encoder_checkpoint", type = str)
    parser.add_argument("--pretrained_passage_encoder", type = str)
    parser.add_argument("--max_concat_length", type = int, default=128)
    parser.add_argument("--max_query_length", type = int, default=128)
    parser.add_argument("--max_doc_length", type = int, default=384)
    parser.add_argument("--seed", type = int, default=42)
    parser.add_argument("--model_type", type = str, default="ANCE")
    parser.add_argument("--passage_block_num", type = int, default=22)
    parser.add_argument("--per_gpu_eval_batch_size", type = int, default = 64)
    parser.add_argument("--n_gpu", type = int, default=1)
    parser.add_argument("--use_gpu", action='store_true', default=False)
    parser.add_argument("--top_n", type = str, default=100)
    parser.add_argument("--use_data_percent", type = str, default=1)
    parser.add_argument("--rel_threshold", type = str, default=1)
    parser.add_argument("--cross_validate", action='store_true', default=False)
    parser.add_argument("--disable_tqdm", action='store_true', default=False)
    parser.add_argument("--use_last_response", action='store_true', default=False)
    parser.add_argument("--test_type", type=str, default="rewrite")
    parser.add_argument("--use_wandb", action='store_true', default=False)
    parser.add_argument("--wandb_project_name", type = str)
    parser.add_argument("--wandb_run_name", type = str)
    
    parser.add_argument("--test_file_path", type = str)
    parser.add_argument("--qrel_output_path", type = str)
    parser.add_argument("--inst_type_path", type=str, default='')
    parser.add_argument("--trec_gold_qrel_file_path", type = str)
    parser.add_argument("--passage_collection_path", type = str)
    parser.add_argument("--passage_embeddings_dir_path", type = str)
    parser.add_argument("--passage_offset2pid_path", type = str)

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device

    check_dir_exist_or_build([args.qrel_output_path])
    logger.info("---------------------The arguments are:---------------------")
    logger.info(args)
    
    logger.info(args.test_file_path)
    logger.info(args.qrel_output_path)
    return args


if __name__ == '__main__':
    main()
