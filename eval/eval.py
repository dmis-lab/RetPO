from re import T
from IPython import embed
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import sys
sys.path.append('..')
sys.path.append('.')
import os
from os.path import join as oj
import toml
import numpy as np
import json
import pytrec_eval
import wandb

def print_trec_res(run_file, qrel_file, data, output_path, type_path, rel_threshold=1, use_wandb=True):
    with open(run_file, 'r' )as f:
        run_data = f.readlines()
    with open(qrel_file, 'r') as f:
        qrel_data = f.readlines()
    
    qrels = {}
    qrels_ndcg = {}
    runs = {}
    
    for line in qrel_data:
        line = line.split("\t") if len(line.split("\t")) > 1 else line.split(" ") 
        query = line[0]
        passage = line[2]
        rel = int(line[3])
        if query not in qrels:
            qrels[query] = {}
        if query not in qrels_ndcg:
            qrels_ndcg[query] = {}

        # for NDCG
        qrels_ndcg[query][passage] = rel
        # for MAP, MRR, Recall
        if rel >= rel_threshold:
            rel = 1
        else:
            rel = 0
        qrels[query][passage] = rel
    
    for line in run_data:
        line = line.split(" ")
        query = line[0]
        passage = line[2]
        rel = int(line[4])
        if query not in runs:
            runs[query] = {}
        runs[query][passage] = rel
        
    print(f"len(runs): {len(list(runs.keys()))}")

    # EVALUATION
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"map", "recip_rank", "recall"})

    orig_key = ""
    cands_meta = {}
    cands_meta_list = []
    assert len(data) == len(list(runs.keys()))
    for query, (q_key, q_v) in zip(data, runs.items()):
        eval_q_key = q_key.rsplit("-", 1)[0] if q_key.count("-") >= 2 else q_key
        
        run = {eval_q_key: q_v}
        res = evaluator.evaluate(run)
        map_list = [v['map'] for v in res.values()]
        mrr_list = [v['recip_rank'] for v in res.values()]
        recall_5_list = [v['recall_5'] for v in res.values()]
        recall_10_list = [v['recall_10'] for v in res.values()]
        recall_20_list = [v['recall_20'] for v in res.values()]
        recall_100_list = [v['recall_100'] for v in res.values()]
        
        # print(q_key)
        # print(query)
        # print(len(list(run.keys())))
        # print('\n')
        
        mrr = [v['recip_rank'] for v in res.values()][0]
        rank = int(1/mrr) if mrr != 0 else 101
        
        q_meta = {
            'query': [query],
            'rank': rank,
        }
        
        if not orig_key:
            orig_key = eval_q_key
        
        if eval_q_key != orig_key:
            cands_meta_list.append(cands_meta)
            cands_meta = {}
            orig_key = eval_q_key

        cands_meta[q_key] = q_meta

    cands_meta_list.append(cands_meta)

    json.dump(runs, open(os.path.join(output_path, 'scores.json'),'w'), indent=4)
    json.dump(cands_meta_list, open(os.path.join(output_path, 'cands_meta_list.json'),'w'), indent=4)
    
    # select optimal query among candidates
    opt_dict = {}
    for cands_meta in cands_meta_list:
        min_rank = 101
        
        for idx, (cand_k, cand_v) in enumerate(cands_meta.items()):
            if idx == 0:
                min_qid = cand_k

            if cand_v['rank'] < min_rank:
                min_rank = cand_v['rank']
                min_qid = cand_k
        
        if min_qid.count("-") >= 2:
            opt_dict[min_qid.rsplit("-", 1)[0]] = runs[min_qid]
        else:
            opt_dict[min_qid] = runs[min_qid]
    
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"map", "recip_rank", "recall"})
    res = evaluator.evaluate(opt_dict)
    map_list = [v['map'] for v in res.values()]
    mrr_list = [v['recip_rank'] for v in res.values()]
    recall_5_list = [v['recall_5'] for v in res.values()]
    recall_10_list = [v['recall_10'] for v in res.values()]
    recall_20_list = [v['recall_20'] for v in res.values()]
    recall_100_list = [v['recall_100'] for v in res.values()]
    
    evaluator = pytrec_eval.RelevanceEvaluator(qrels_ndcg, {"ndcg_cut.3"})
    res = evaluator.evaluate(opt_dict)
    ndcg_3_list = [v['ndcg_cut_3'] for v in res.values()]

    res = {
            "MAP": np.average(map_list),
            "MRR": np.average(mrr_list),
            "NDCG@3": np.average(ndcg_3_list),
            "Recall@5": np.average(recall_5_list),
            "Recall@10": np.average(recall_10_list),
            "Recall@20": np.average(recall_20_list),
            "Recall@100": np.average(recall_100_list),
        }

    logger.info("---------------------Evaluation results:---------------------")
    logger.info(res)
    json.dump(res, open(os.path.join(output_path, 'opt_metrics.txt'),'w'), indent=4)
    
    if use_wandb:
        wandb.log(res)
    
    # first query
    cnt = 0
    first_dict = {}
    for cands_meta in cands_meta_list:
        min_rank = 101
        
        for idx, (cand_k, cand_v) in enumerate(cands_meta.items()):
            if idx == 0:
            # if idx == 1:
                min_qid = cand_k
                break
        cnt += 1
        
        if min_qid.count("-") >= 2:
            first_dict[min_qid.rsplit("-", 1)[0]] = runs[min_qid]
        else:
            first_dict[min_qid] = runs[min_qid]
        
    logger.info(f"n_first: {cnt}")
    
    if type_path:
        insts_type = json.load(open(type_path))
        
        for inst_type, qid_list in insts_type.items():
            evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"map", "recip_rank", "recall"})
            
            inst_dict = {key: first_dict[key] for key in qid_list if key in first_dict}
            res = evaluator.evaluate(inst_dict)
            
            map_list = [v['map'] for v in res.values()]
            mrr_list = [v['recip_rank'] for v in res.values()]
            recall_5_list = [v['recall_5'] for v in res.values()]
            recall_10_list = [v['recall_10'] for v in res.values()]
            recall_20_list = [v['recall_20'] for v in res.values()]
            recall_100_list = [v['recall_100'] for v in res.values()]

            evaluator = pytrec_eval.RelevanceEvaluator(qrels_ndcg, {"ndcg_cut.3"})
            res = evaluator.evaluate(inst_dict)
            ndcg_3_list = [v['ndcg_cut_3'] for v in res.values()]
            
            res = {
                    f"{inst_type}_MAP": np.average(map_list),
                    f"{inst_type}_MRR": np.average(mrr_list),
                    f"{inst_type}_NDCG@3": np.average(ndcg_3_list), 
                    f"{inst_type}_Recall@5": np.average(recall_5_list),
                    f"{inst_type}_Recall@10": np.average(recall_10_list),
                    f"{inst_type}_Recall@20": np.average(recall_20_list),
                    f"{inst_type}_Recall@100": np.average(recall_100_list),
                }
            
            json.dump(res, open(os.path.join(output_path, f'{inst_type}_metrics.txt'),'w'), indent=4)
            
            if use_wandb:
                wandb.log(res)
    
            
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"map", "recip_rank", "recall"})  
    res = evaluator.evaluate(first_dict)
    map_list = [v['map'] for v in res.values()]
    mrr_list = [v['recip_rank'] for v in res.values()]
    recall_5_list = [v['recall_5'] for v in res.values()]
    recall_10_list = [v['recall_10'] for v in res.values()]
    recall_20_list = [v['recall_20'] for v in res.values()]
    recall_100_list = [v['recall_100'] for v in res.values()]

    evaluator = pytrec_eval.RelevanceEvaluator(qrels_ndcg, {"ndcg_cut.3"})
    res = evaluator.evaluate(first_dict)
    ndcg_3_list = [v['ndcg_cut_3'] for v in res.values()]
    
    res = {
            "first_MAP": np.average(map_list),
            "first_MRR": np.average(mrr_list),
            "first_NDCG@3": np.average(ndcg_3_list), 
            "first_Recall@5": np.average(recall_5_list),
            "first_Recall@10": np.average(recall_10_list),
            "first_Recall@20": np.average(recall_20_list),
            "first_Recall@100": np.average(recall_100_list),
        }

    logger.info(res)
    json.dump(res, open(os.path.join(output_path, 'first_metrics.txt'),'w'), indent=4)
    
    if use_wandb:
        wandb.log(res)
    
    return res
