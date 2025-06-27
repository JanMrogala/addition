import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from transformers import TrainerCallback
from lightning.pytorch.callbacks import Callback
import os
import json
import torch
import numpy as np
import wandb
import pandas as pd
from datetime import datetime
from tqdm import trange
from pathlib import Path
from transformers import AutoConfig, AutoModelForCausalLM
from litgpt.scripts.convert_lit_checkpoint import convert_lit_checkpoint
from litgpt.utils import copy_config_files, auto_download_checkpoint
import torch
from pathlib import Path
from datetime import datetime
import re
import ast
from collections import OrderedDict
from typing import List, Dict, Any
from transformers import PreTrainedTokenizerFast

import torch
from litgpt import LLM
from litgpt.data import Alpaca2k
import lightning as L
from utils.data import *
import hydra
from config import hf_config
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from utils.search_evaluator import Evaluator
from litgpt.config import configs, Config, name_to_config
from litgpt.model import GPT
from litgpt.api import Preprocessor
import json
import os
import wandb
import numpy as np
from transformers import get_cosine_schedule_with_warmup
from typing import Optional

import logging

def parse_data_string(data_str: str):
    """
    Parse a string of the form:
      Init_state: [ 0 : 6 , 1 : 5 , ... ] Stack: [  { 0 : 2 } , { 1 : 3 , 2 : 5 } ]
    into:
      init_state = {0: 6, 1: 5, ...}
      stack = [ {0: 2}, {1: 3}, ...]
    """
    # --- Extract the init_state portion
    init_pat = r"Init_state:\s*\[(.*?)\]"
    init_match = re.search(init_pat, data_str)
    if not init_match:
        raise ValueError("Could not find 'Init_state: [...]' in string.")
    init_content = init_match.group(1).strip()

    # --- Extract the stack portion
    stack_pat = r"Stack:\s*\[(.*?)\]$"
    stack_match = re.search(stack_pat, data_str)
    if not stack_match:
        raise ValueError("Could not find 'Stack: [...]' in string.")
    stack_content = stack_match.group(1).strip()

    # --- Parse init_state into a dict
    init_state = {}
    for pair_str in init_content.split(","):
        pair_str = pair_str.strip()
        if not pair_str:
            continue
        k_str, v_str = pair_str.split(":")
        k = int(k_str.strip())
        v = int(v_str.strip())
        init_state[k] = v

    # --- Parse the stack (list of dicts)
    raw_dicts = re.findall(r"\{(.*?)\}", stack_content)
    stack_list = []
    for rd in raw_dicts:
        d = {}
        pairs = rd.split(",")
        for p in pairs:
            p = p.strip()
            if not p:
                continue
            k_str, v_str = p.split(":")
            d[int(k_str.strip())] = int(v_str.strip())
        stack_list.append(d)

    return init_state, stack_list

def cleanup_stack(stack: List[Dict[int,int]]):
    """Remove any empty dictionaries from the beginning of the stack."""
    while stack and not stack[0]:
        stack.pop(0)

def apply_command(op: str, init_state: Dict[int,int], stack: List[Dict[int,int]], prompt):
    """
    Apply a command to init_state & stack.
    This includes:
     - Movement commands: "X D" or "X U"
     - TF, RL, N {...} stubs (you can adapt your own rules here).
    """

    # Movement commands: "X D" or "X U"
    if op.endswith(" D") or op.endswith(" U"):
        parts = op.split()
        if len(parts) != 2:
            print(f"Error: Malformed movement command '{op}'")
            return
        automat_index, direction = parts
        automat_index = int(automat_index)
        if direction == "D":
            if init_state[automat_index] > 0:
                init_state[automat_index] -= 1
            else:
                print(f"Warning: Automat {automat_index} is already at 0.")
        elif direction == "U":
            if init_state[automat_index] < 12:
                init_state[automat_index] += 1
            else:
                print(f"Warning: Automat {automat_index} is already at max 12.")
        return

    # TF (Take First)
    if op == "TF":
        cleanup_stack(stack)
        if not stack or not stack[0]:
            print("Warning: TF on empty stack or empty first dict.")
            return
        first_dict = stack[0]
        # pop the first key-value
        key = next(iter(first_dict))
        value = first_dict.pop(key)
        from collections import OrderedDict
        new_dict = OrderedDict([(key, value)])
        # Insert the new OrderedDict at the beginning
        stack.insert(0, new_dict)
        return

    # RL (Remove if matches init_state)
    if op == "RL":
        cleanup_stack(stack)
        if not stack:
            print("Warning: RL on empty stack.")
            return
        first_dict = stack[0]
        valid = True
        for k, v in first_dict.items():
            if init_state.get(k) != v:
                valid = False
                break
        if valid:
            stack.pop(0)
        else:
            print("Warning: RL mismatch between stack[0] and init_state.")
        return

    # N {...} (Add new dict at the beginning of the stack)
    if op.startswith("N "):
        dict_str = op[2:].strip()  # everything after "N "
        from collections import OrderedDict
        try:
            new_pair = ast.literal_eval(dict_str)
            if not isinstance(new_pair, dict) or not (1 <= len(new_pair) <= 12):
                print("Warning: N operation requires 1-10 key-value pairs.")
                return
            cleanup_stack(stack)
            # Insert a new OrderedDict at the beginning
            stack.insert(0, OrderedDict(new_pair.items()))
        except Exception as e:
            print(f"Warning: error parsing 'N' dictionary: {e}")
        return

    # If we get here, we have an unknown operation

    print(f"init_state: {init_state}")
    print(f"stack: {stack}")
    print(f"Warning: Unknown operation '{op}'.")

def generate_command_for_full_eval(model, tokenizer, prompt: str, max_length=256) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Remove token_type_ids (if present) before generating
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=prompt_len + max_length,
            pad_token_id=tokenizer.eos_token_id,
            num_beams=1,  # greedy
            do_sample=False
        )

    # Decode
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    generated = full_text[len(prompt)-1:].strip()
    return generated

def evaluate_full_problem(input_text: str, model, tokenizer, max_iters: int = 200) -> bool:
    """
    Evaluate a full problem by running the iterative inference loop.
    Returns True if the loop terminates by matching init_state to the target_dict
    within max_iters, False otherwise.
    """
    try:
        # Extract automata prefix (A, B, or C) from the beginning and keep it
        automata_prefix = ""
        if input_text.startswith(('A ', 'B ', 'C ')):
            automata_prefix = input_text[:2]  # Keep "A ", "B ", or "C "
            data_str = input_text[2:]  # Remove the prefix and space for parsing
        else:
            data_str = input_text
            
        original_init_state, original_stack = parse_data_string(data_str)
    except Exception as e:
        print(f"[ERROR] parsing example: {e}")
        return False

    if not original_stack:
        print("[ERROR] example has empty original stack")
        return False

    target_dict = original_stack[0].copy()
    current_init_state = original_init_state.copy()
    current_stack = [d.copy() for d in original_stack]

    def data_to_string(init_state: Dict[int,int], stack_list: List[Dict[int,int]]) -> str:
        """Convert init_state + stack_list back to the same string format"""
        # Build init_state string
        init_entries = [f"{k} : {v}" for k, v in init_state.items()]
        init_str = " , ".join(init_entries)

        # Build stack string
        stack_strs = []
        for d in stack_list:
            pairs = [f"{k} : {v}" for k, v in d.items()]
            dict_str = "{ " + " , ".join(pairs) + " }"
            stack_strs.append(dict_str)
        total_stack_str = " , ".join(stack_strs)

        return f"Init_state: [ {init_str} ] Stack: [  {total_stack_str} ]"

    for i in range(max_iters):
        current_data_str = data_to_string(current_init_state, current_stack)
        bos = getattr(tokenizer, "bos_token", "")
        prompt = f"{bos} {automata_prefix} {current_data_str} Command:"
        command_pred = generate_command_for_full_eval(model, tokenizer, prompt)
        print("command_pred pre split: ", command_pred)

        command_pred = command_pred.split('[EOS]')[0].strip()
        print("command_pred post split: ", command_pred)
        apply_command(command_pred, current_init_state, current_stack, prompt)
        if current_init_state == target_dict:
            return True
    print("Current init state:", current_init_state)
    print("Target dict:", target_dict)
    return False

def evaluate_full_problems(test_full_data, hf_model, tokenizer):
    """Evaluate full problems and return accuracy"""
    correct = 0
    total = 0
    split_str = "Command:"

    print(f"Evaluating {len(test_full_data)} full problems...")
    
    for sample in test_full_data:
        # Get the original input text (before tokenization)
        input_ids = sample["input_ids"]
        # Decode the full input to get the original text
        full_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        
        # Extract just the problem part (before the split_str delimiter)
        if split_str in full_text:
            input_text = full_text.split(split_str)[0].strip()
        else:
            # Fallback: if split_str not found, try "Command:" for backward compatibility
            if "Command:" in full_text:
                input_text = full_text.split("Command:")[0].strip()
            else:
                input_text = full_text
        
        try:
            result = evaluate_full_problem(input_text, hf_model, tokenizer, max_iters=130)
            if result:
                correct += 1
            total += 1
        except Exception as e:
            print(f"Error evaluating problem: {e}")
            total += 1
            
    full_problem_accuracy = correct / total if total > 0 else 0.0
    print(f"Full problem accuracy: {correct}/{total} = {full_problem_accuracy:.4f}")
    
    return full_problem_accuracy

def get_data_tmp(tokenizer, test_path):
    
    empty_dataset = Dataset.from_dict({"input": [], "output": []})
    
    hf_dataset = load_dataset(
            "json",
            data_files={
                "test": test_path,
            },
        )
    hf_dataset = DatasetDict({
        "train": empty_dataset,
        "val": empty_dataset,
        "test": hf_dataset["test"]
    })
    
    

    def tokenize(examples):
        # Format input and output with delimiter between them
        texts = []
        
        for i in range(len(examples["input"])):
            input_text = examples["input"][i]

            # Full test files: just input + split_str (no output)
            full_text = tokenizer.bos_token + " " + input_text + " " + "Command:"
                
            texts.append(full_text)
        
        outputs = tokenizer(
            texts,
            truncation=True,
            max_length=4096,
            padding='longest',
            return_overflowing_tokens=False,
        )
        return {"input_ids": outputs["input_ids"]}

    # Remove both "input" and "output" columns after tokenization
    # Remove columns after tokenization - handle both cases
    columns_to_remove = ["input"]

    tokenized_dataset = hf_dataset.map(
        tokenize, batched=True, remove_columns=columns_to_remove
    )

    return tokenized_dataset

tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="temp/tmp/tokenizer.json",
    )
tokenizer.eos_token = "[EOS]"
tokenizer.unk_token = "[UNK]"
tokenizer.pad_token = "[PAD]"
tokenizer.mask_token = "[MASK]"
tokenizer.bos_token = "[BOS]"

datasets = get_data_tmp(tokenizer, "temp/test_only_first.json")
data = Datamodule(datasets, 16, 16, tokenizer)
data.connect(max_seq_length=4096)
data.setup()

test_data = data.dataset["test"]

hf_model = AutoModelForCausalLM.from_pretrained(
        "temp/tmp/",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        # state_dict=state_dict,
        # attn_implementation="flash_attention_2",
    )

a = evaluate_full_problems(test_data, hf_model, tokenizer)