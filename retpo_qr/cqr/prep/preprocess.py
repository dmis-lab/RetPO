import re
from typing import Literal

import os
import json
from datasets import Dataset, DatasetDict

def get_qrecc_qa(script_args, splits, target_key="Truth_answer", whole_conv=False):
    qrecc = {}
    answer_role = 'assistant' if script_args.chat else 'system'
    str_instruction = ''
    if script_args.prompt_path:
        str_instruction = json.load(open(script_args.prompt_path))['instruction']
        
    for split_str in splits:
        d_split, task = split_str.split("_")
        orig_qrecc = json.load(open(os.path.join(script_args.dataset_dir, f'{d_split}.json')))
        chat_qrecc = []
        for idx, ins in enumerate(orig_qrecc):
            if idx+1 < len(orig_qrecc) and whole_conv:
                if ins['Conversation_no'] == orig_qrecc[idx+1]['Conversation_no']: continue
            if len(chat_qrecc) > script_args.n_toy and script_args.n_toy > 0: break
            
            msgs = [{'content' : '', 'role' : 'system'}]
            if type(ins['Context']) == list:
                lst_context = ins['Context']
            elif type(ins['Context']) == str:
                lst_context = ins['Context'].split('\n')
                lst_context = [re.sub(f"[QA]\d+: ", "", s) for s in lst_context]
            
            for qa_idx, qa in enumerate(lst_context):
                role = 'user' if qa_idx % 2 == 0 else answer_role
                if qa_idx == 0 and role == 'user': 
                    qa = str_instruction + "\n" + qa
                msgs.append({'content' : qa, 'role' : role})
            msgs.append({'content' : ins['Question'], 'role' : 'user'})
            
            chat_ins = {'prompt'    : ins['Question'] if str_instruction == "" else str_instruction,
                        'prompt_id' : str (ins['Conversation_no']) + "_" + str (ins['Turn_no']),
                        'messages'  : msgs,
                        }
            
            if task == "sft":
                ans_key = "Truth_answer" if "Truth_answer" in ins.keys() else "Answer"
                target = ins[target_key] if d_split == 'train' else ins[ans_key]
                chat_ins.update({'targets'   : target,})
            
            chat_qrecc.append(chat_ins)
        qrecc[d_split] = chat_qrecc
    
    raw_datasets = DatasetDict({split : Dataset.from_list(dset) for split, dset in qrecc.items()})
    
    return raw_datasets


def apply_cqa_template(
    example, tokenizer, task: Literal["sft", "generation", "rm", "dpo"] = "sft", 
    assistant_prefix="<|assistant|>\n", label_prefix="<|system|>\n", prompt="",
):
    def _strip_prefix(s, pattern):
        # Use re.escape to escape any special characters in the pattern
        return re.sub(f"^{re.escape(pattern)}", "", s)

    if task in ["sft", "generation"]:
        messages = example["messages"]
        example["text"] = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        example["text"] += label_prefix
        if task == "sft":
            example["text"] += example["targets"] + tokenizer.eos_token
        
    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided task is one of {['sft', 'generation', 'rm', 'dpo']}"
        )
    return example


def get_qrecc(script_args, splits, target_key="Truth_rewrite"):
    qrecc = {}
    answer_role = 'assistant' if script_args.chat else 'system'
    for split_str in splits:
        d_split, task = split_str.split("_")
        orig_qrecc = json.load(open(os.path.join(script_args.dataset_dir, f'{d_split}.json')))
        chat_qrecc = []
        for ins in orig_qrecc:
            if len(chat_qrecc) > script_args.n_toy and script_args.n_toy > 0: break
            if ins['Turn_no'] == 1 and d_split == 'train' : continue
            
            msgs = [{'content' : '', 'role' : 'system'}]
            if type(ins['Context']) == list:
                lst_context = ins['Context']  
            elif type(ins['Context']) == str:
                lst_context = ins['Context'].split('\n')
                lst_context = [re.sub(f"[QA]\d+: ", "", s) for s in lst_context]
            
            for qa_idx, qa in enumerate(lst_context):
                role = 'user' if qa_idx % 2 == 0 else answer_role
                msgs.append({'content' : qa, 'role' : role})
            msgs.append({'content' : ins['Question'], 'role' : 'user'})
            
            chat_ins = {'prompt'    : ins['Question'],
                        'prompt_id' : str (ins['Conversation_no']) + "_" + str (ins['Turn_no']),
                        'messages'  : msgs,
                        }
            
            if task == "sft" and target_key in ins.keys():
                target = ins[target_key] if d_split == 'train' else ins["Question"]
                chat_ins.update({'targets'   : target,})
            if task == "prefs":
                chat_ins.update({'chosen' : ins['chosen'],
                                 'rejected' : ins['rejected'],
                                 })
            
            chat_qrecc.append(chat_ins)
        qrecc[d_split] = chat_qrecc
    
    raw_datasets = DatasetDict({split : Dataset.from_list(dset) for split, dset in qrecc.items()})
    
    return raw_datasets


def apply_cqr_template(
    example, tokenizer, task: Literal["sft", "generation", "rm", "dpo"] = "sft", 
    assistant_prefix="<|assistant|>\n", label_prefix="<|rewrite|>\n", prompt="",
):
    def _strip_prefix(s, pattern):
        # Use re.escape to escape any special characters in the pattern
        return re.sub(f"^{re.escape(pattern)}", "", s)

    if task in ["sft", "generation"]:
        messages = example["messages"]
        example["text"] = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        example["text"] += label_prefix
        if task == "sft":
            example["text"] += example["targets"] + tokenizer.eos_token
        
    elif task == "dpo":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            # Compared to reward modeling, we filter out the prompt, so the text is everything after the last assistant token
            prompt_messages = example["messages"]
            
            # TODO: handle case where chosen/rejected also have system messages
            example["text_chosen"] = example["chosen"]
            example["text_rejected"] = example["rejected"]
            example["text_prompt"] = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=False
            )
            example["text_prompt"] += label_prefix
            
        else:
            raise ValueError(
                f"Could not format example as dialogue for `dpo` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided task is one of {['sft', 'generation', 'rm', 'dpo']}"
        )
    return example
