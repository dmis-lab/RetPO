# ref: https://github.com/huggingface/transformers/blob/v4.35.0/examples/pytorch/summarization/run_summarization_no_trainer.py

import os
import re
import json
import sys
import logging 
import numpy as np

from dataclasses import dataclass, field
from typing import List, Optional
from tqdm.auto import tqdm

import evaluate
import nltk
import torch
from torch.utils.data import DataLoader

import transformers
from transformers import (
                    AutoModelForCausalLM, 
                    AutoTokenizer,
                    DataCollatorForSeq2Seq,
)

from alignment import (
                    H4ArgumentParser,
                    get_tokenizer,
)

from cqr.prep.preprocess import (
    apply_cqr_template,
    get_qrecc,                                 
)

logger = logging.getLogger(__name__)

# Define and parse arguments.
@dataclass
class Arguments:

    dataset_dir: Optional[str] = field(
        default="", metadata={"help": "the directory path to the dataset"}
    )
    prompt_path: Optional[str] = field(
        default="", metadata={"help": "the directory path to the dataset"}
    )
    output_dir: Optional[str] = field(
        default="", metadata={"help": "the directory path to the outputs"}
    )
    prompt_dir: Optional[str] = field(
        default="", metadata={"help": "the directory path to the prompt"}
    )
    device: Optional[str] = field(
        default="cuda", metadata={"help": ""}
    )
    label_token: Optional[str] = field(
        default="<|rewrite|>", metadata={"help": ""}
    )
    target_key: Optional[str] = field(
        default="Truth_rewrite", metadata={"help": ""}
    )
    prefix: Optional[str] = field(
        default="", metadata={"help": ""}
    )
    n_toy: Optional[int] = field(
        default=-1, metadata={"help": "the number of datasets to be used"}
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=1, metadata={"help": "the number of batches"}
    )
    eval_split: Optional[str] = field(
        default="test",
        metadata={"help": ("List of train test splits to use in the dataset")},
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    do_eval: bool = field(
        default=False
    )
    preprocessing_num_workers: Optional[int] = field(
        default=20, metadata={"help": ""}
    )
    max_target_length: Optional[int] = field(
        default=400, metadata={"help": ""}
    )
    max_source_length: Optional[int] = field(
        default=2048, metadata={"help": ""}
    )
    chat: bool = field(
        default=False, metadata={"help": "whether to use chat version or not"}
    )

    
    # arguments for generation
    # val_max_target_length: Optional[int] = field(
    #     default=100, metadata={"help": ""}
    # )
    num_beams: Optional[int] = field(
        default=5, metadata={"help": ""}
    )
    
def postprocess_text(preds, labels, pick_one=False):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    
    if pick_one:
        preds = [pred.split("\n")[0] for pred in preds]

    return preds, labels

def parse_preds(string, sep="oqf", n_out=10):
    if sep == "line_sep":
        lst_split = string.split("\n")
    elif sep == "oqf": 
        lst_split = re.split(r"(?:\n)?Rewrite \d+: ", string)
        lst_split = [s.strip() for s in lst_split if s]
    else:
        raise ValueError(f"{sep} is not supported")
        
    lst_out = lst_split[:min(len(lst_split), n_out)]    
        
    return lst_out


def main():
    parser = H4ArgumentParser(Arguments)
    args = parser.parse()
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    logger.info(f"arguments {args}")
    
    raw_datasets = get_qrecc(args, [args.eval_split + "_sft"], target_key=args.target_key)
    column_names = raw_datasets[args.eval_split].column_names
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    prompt = ""
    if args.prompt_dir:
        prompt = json.load(open(args.prompt_dir))["1-shot"]
    raw_datasets = raw_datasets.map(apply_cqr_template, 
                                    fn_kwargs={"tokenizer": tokenizer, 
                                               "task": "generation",
                                               "prompt" : prompt,
                                               "label_prefix" : args.label_token + "\n",
                                            })
    

    def preprocess_function(examples, target_key='targets'):
        inputs = examples['text']
        targets = examples[target_key]
        inputs = [inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=True, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=args.max_target_length, padding=True, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
        
    eval_dataset = raw_datasets[args.eval_split].map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names + ['text'],
            desc="Running tokenizer on dataset",
        )
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.to(args.device)
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
    )
    
    eval_dataloader = DataLoader( 
        eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )
    
    model.eval()
    gen_kwargs = {
            # "max_length": args.max_source_length,
            "max_new_tokens" : args.max_target_length,
            "num_beams": args.num_beams,
        }
    
    # Metric
    if args.do_eval:
        metric = evaluate.load("rouge")
    logger.info("***** Running evaluating *****")
    progress_bar = tqdm(range(len(eval_dataloader)))
    preds = []
    str_preds = []
    for step, batch in enumerate(eval_dataloader):
            
        with torch.no_grad():
            batch = {k: v.to(args.device) for k,v in batch.items()}
            
            generated_tokens = model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
            )
            labels = batch['labels']
            labels = labels.cpu()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            
            generated_tokens = generated_tokens.cpu()

            decoded_gen = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True, 
                                                   max_length=args.max_target_length, truncation=True)
            decoded_preds = [gen.split(args.label_token)[-1] for gen in decoded_gen]
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels, pick_one=False)
            
            
            if args.do_eval:
                metric.add_batch(
                    predictions=decoded_preds,
                    references=decoded_labels,
                )
            str_preds += decoded_preds
            
            for pred in decoded_preds:
                preds.append(parse_preds(pred))
            
            progress_bar.update(1)
            if args.n_toy == step:
                break
    
    os.makedirs(args.output_dir, exist_ok=True)
    prefix = ""
    if args.prefix:
        prefix = "_" + args.prefix
    
    with open(os.path.join(args.output_dir, args.eval_split + prefix + "_rewrites.json"), "w") as f:
        json.dump(preds, f)
    
    with open(os.path.join(args.output_dir, args.eval_split + prefix + "_rewrites_str.json"), "w") as f:
        json.dump(str_preds, f)
        
    if args.do_eval:
        result = metric.compute(use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        with open(os.path.join(args.output_dir, args.eval_split + prefix + "_results.json"), "w") as f:
            json.dump(result, f)
        
        print(result)
    
if __name__ == "__main__":
    main()