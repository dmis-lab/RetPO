"""
https://github.com/fengranMark/ConvGQR/blob/main/bm25/bm25_topiocqa.py
We modify the code from ConvGQR.
"""

from re import T
from IPython import embed
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import sys
sys.path.append('..')
sys.path.append('.')
import argparse
import os
from utils import check_dir_exist_or_build
from os import path
from os.path import join as oj
import toml
import numpy as np
import json
from pyserini.search.lucene import LuceneSearcher
import pytrec_eval
import wandb

# from api_config import CONFIG you can use this
from eval import print_trec_res


def main():
    args = get_args()
    
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
    
    # else:
    query_list = []
    qid_list = []
    
    with open(args.input_query_path, "r") as f:
        data = f.readlines()

    if args.input_query_path_2:
        with open(args.input_query_path_2, "r") as f2:
            data_2 = f2.readlines()

    n = len(data)

    if args.use_PRF:
        with open(args.PRF_file, 'r') as f:
            PRF = f.readlines()
        assert(len(data) == len(PRF))
        
    for i in range(n):
        data[i] = json.loads(data[i])
        if args.query_type == "raw":
            query = data[i]["query"]
        elif args.query_type == "rewrite":
            query = data[i]['rewrite'] #+ ' ' + data[i]['answer']

        elif args.query_type == "decode":
            query = data[i]['oracle_utt_text']
            if args.eval_type == "answer":
                data_2[i] = json.loads(data_2[i])
                query = data_2[i]['answer_utt_text']
            elif args.eval_type == "oracle+answer":
                data_2[i] = json.loads(data_2[i])
                query = query + ' ' + data_2[i]['answer_utt_text']

        query_list.append(query)
        if "sample_id" in data[i]:
            qid_list.append(data[i]['sample_id'])
        else:
            qid_list.append(data[i]['id'])  
    logger.info('load query_list and qid_list')
    
    logger.info(f"start search ... n: {len(data)}")
    # pyserini search
    searcher = LuceneSearcher(args.index_dir_path)
    searcher.set_bm25(args.bm25_k1, args.bm25_b)
    hits = searcher.batch_search(query_list, qid_list, k = args.top_k, threads = args.n_thread)
    logger.info(f"search done! save ... n: {len(data)}")
    
    os.makedirs(args.output_dir_path, exist_ok=True)
    cnt = 0
    logger.info(f"len(qid): {len(qid_list)}")
    logger.info(f"len(query): {len(query_list)}")
    logger.info(f"hits: {len(list(hits.keys()))}")
    
    # filter
    # Occasionally, bm25 retriever does not return #{args.top-k} number of search results due to the incompletion of query format.
    exclude_qid_list = []
    for qid in qid_list:
        if len(hits[qid]) != args.top_k:
            exclude_qid_list.append(qid)
    
    for i in exclude_qid_list:
        print(i)
        
    valid_idx = [idx for idx, i in enumerate(qid_list) if i not in exclude_qid_list]
    qid_list_filter = [qid_list[i] for i in valid_idx] 
    query_list_filter = [query_list[i] for i in valid_idx]
    logger.info(f"exlude qid list ... n: {len(exclude_qid_list)}")
    logger.info(f"len(qid_filter): {len(qid_list_filter)}")
    logger.info(f"len(query_filter): {len(query_list_filter)}")
    
    len_orig = len(set([i.rsplit("-", 1)[0] for i in qid_list]))
    len_filter = len(set([i.rsplit("-", 1)[0] for i in qid_list_filter]))
    logger.info(f"len_orig: {len_orig}")
    logger.info(f"len_filter: {len_filter}")
    assert len_orig == len_filter
    
    # save
    with open(oj(args.output_dir_path, "bm25_t5_oracle+answer_res.trec"), "w") as f:
        for qid in qid_list_filter:
            # if not hits[qid]:
            #     print(qid)
            #     raise AssertionError(f"hits error")

            if len(hits[qid]) != args.top_k:
                print(f"error qid: {qid}, len(hits): {len(hits[qid])}")
                raise AssertionError(f"not top-{args.top_k} result")
            
            for i, item in enumerate(hits[qid]):
                f.write("{} {} {} {} {} {} {}".format(qid,
                                                "Q0",
                                                item.docid[3:],
                                                i+1,
                                                -i - 1 + 200,
                                                item.score,
                                                "bm25"
                                                ))
                f.write('\n')
                cnt += 1
    print(f"cnt: {cnt}")

    # res = print_res(oj(args.output_dir_path, "bm25_t5_oracle+answer_res.trec"), args.gold_qrel_file_path, args.rel_threshold)
    res = print_trec_res(run_file=oj(args.output_dir_path, "bm25_t5_oracle+answer_res.trec"),
                        qrel_file=args.gold_qrel_file_path,
                        data=query_list_filter,
                        output_path=args.output_dir_path,
                        type_path=args.inst_type_path,
                        rel_threshold=args.rel_threshold,
                        use_wandb=args.use_wandb
                        )
    return res


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str)
    parser.add_argument("--input_query_path", type=str)
    parser.add_argument("--input_query_path_2", type=str)
    parser.add_argument("--index_dir_path", type=str)
    parser.add_argument("--output_dir_path", type=str)
    parser.add_argument("--inst_type_path", type=str, default='')
    parser.add_argument("--gold_qrel_file_path", type=str)
    
    parser.add_argument("--query_type", type=str, default="decode")
    parser.add_argument("--eval_type", type=str, default="oracle+answer")
    parser.add_argument("--bm25_k1", type=float, default="0.9")
    parser.add_argument("--bm25_b", type=float, default="0.4")
    parser.add_argument("--top_k", type=int, default="100")
    parser.add_argument("--rel_threshold", type=int, default="1")
    parser.add_argument("--use_PRF", action='store_true', default=False)
    parser.add_argument("--n_thread", type=int, default=40)
    parser.add_argument(
        "--use_wandb", action="store_true", default=False, help=""
    )
    parser.add_argument(
        "--wandb_project_name", type=str, default='', help=""
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default='test', help=""
    )
    args = parser.parse_args()
    
    logger.info(args.input_query_path)
    logger.info(args.output_dir_path)
    if args.inst_type_path:
        logger.info(f"inst type exists ...")
        logger.info(args.inst_type_path)

    return args

if __name__ == '__main__':
    main()
