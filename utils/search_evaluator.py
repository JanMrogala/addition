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


def convert_litgpt_to_hf(cfg):
    out_dir = Path(cfg.convert_hf.out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(cfg.convert_hf.in_path)
    model_path = out_dir / "pytorch_model.bin"
    model_path = Path(model_path)

    copy_config_files(source_dir=source_dir, out_dir=out_dir)
    convert_lit_checkpoint(checkpoint_dir=source_dir, output_dir=out_dir)

    state_dict = torch.load(out_dir / "model.pth")
    torch.save(state_dict, model_path)
    hf_model = AutoModelForCausalLM.from_pretrained(
        out_dir,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        state_dict=state_dict,
        attn_implementation="flash_attention_2",
    )
    return hf_model


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
            if init_state[automat_index] > (0 + nodes_indexing_margin):
                init_state[automat_index] -= 1
            else:
                print(f"Warning: Automat {automat_index} is already at 0.")
        elif direction == "U":
            if init_state[automat_index] < (nodes_num + nodes_indexing_margin):
                init_state[automat_index] += 1
            else:
                print(f"Warning: Automat {automat_index} is already at max {nodes_num}.")
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
            if not isinstance(new_pair, dict) or not ((1 + nodes_indexing_margin) <= len(new_pair) <= (nodes_num + nodes_indexing_margin)):
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


class Evaluator:
    def __init__(self, config, test_set, test_full_data, tokenizer, split_str, step=None, model=None, dataset_name=None):
        self.config = config
        self.num_examples = config.eval.num_examples
        self.batch_size = config.eval.batch_size
        self.global_step = step
        self.test_full_data = test_full_data
        self.tokenizer = tokenizer
        self.results_dir = config.eval.results_dir
        self.model = model
        self.hf_model = convert_litgpt_to_hf(config)
        self.test_set = test_set
        self.step = step
        self.split_str = split_str
        os.makedirs(self.results_dir, exist_ok=True)

        self.prompts, self.gts = self.get_prompts()
        self.full_predictions = None
        self.predictions_after_delimiter = None

        global nodes_num
        global nodes_indexing_margin

        nodes_num = config.data.nodes_num

        if dataset_name is not None:
            if 'A' in dataset_name:
                nodes_indexing_margin = config.data.nodes_indexing_margin_A
            elif 'B' in dataset_name:
                nodes_indexing_margin = config.data.nodes_indexing_margin_B
            elif 'C' in dataset_name:
                nodes_indexing_margin = config.data.nodes_indexing_margin_C
        print(f"Using nodes_indexing_margin: {nodes_indexing_margin} for dataset {dataset_name}")

    def get_prompts(self):
        search_token_id = self.tokenizer.encode(self.split_str, add_special_tokens=False)[0]

        gts = []
        prompts = []
        for sample in self.test_set:
            input_ids = sample["input_ids"]
            try:
                split_index = input_ids.index(search_token_id)
                # Find the EOS token
                end_index = input_ids.index(self.tokenizer.eos_token_id) if self.tokenizer.eos_token_id in input_ids else len(input_ids)
            except Exception as e:
                print("ERROR")
                print(e)
                print(input_ids)
                print(sample)
                print(self.tokenizer.decode(input_ids, skip_special_tokens=True))
                continue
                
            # Take everything up to Search: token
            prompt_ids = input_ids[: split_index + 1]

            # Decode to text, add BOS token at start
            prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
            full_prompt = self.tokenizer.bos_token + " " + prompt_text
            
            # Ground truth is everything AFTER the delimiter up to EOS
            gt_ids = input_ids[split_index+1:end_index]
            gt = self.tokenizer.decode(gt_ids, skip_special_tokens=True)

            # Re-encode with BOS token
            prompt_with_bos = self.tokenizer.encode(
                full_prompt, add_special_tokens=False
            )
            prompts.append(prompt_with_bos)
            gts.append(gt)

        return prompts, gts

    def get_preds(self):
        batch_size = self.batch_size
        data = self.prompts
        tokenizer = self.tokenizer
        output_texts_concat = []
        predictions_after_delimiter = []
        
        search_token_id = self.tokenizer.encode(self.split_str, add_special_tokens=False)[0]

        self.hf_model.cuda()
        self.hf_model.eval()

        for b in trange(0, len(data), batch_size):
            batch = data[b : min(b + batch_size, len(data))]
            batch_text = [tokenizer.decode(x, skip_special_tokens=False) for x in batch]
            tokenizer.padding_side = "left"
            inputs = tokenizer(batch_text, return_tensors="pt", padding=True).to("cuda")
            input_prompt = inputs["input_ids"]

            outputs = self.hf_model.generate(
                input_ids=input_prompt,
                pad_token_id=tokenizer.pad_token_id,
                attention_mask=inputs["attention_mask"].to("cuda"),
                max_length=self.config.model.block_size,
                num_beams=1,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
            )

            # Process each generated sequence
            for output_ids in outputs.tolist():
                # Find the delimiter token in the output
                try:
                    split_index = output_ids.index(search_token_id)
                    # Find the EOS token
                    end_index = output_ids.index(tokenizer.eos_token_id) if tokenizer.eos_token_id in output_ids else len(output_ids)
                    
                    # Get full generated text
                    full_text = tokenizer.decode(output_ids, skip_special_tokens=False)
                    
                    # Get just the part after the delimiter
                    after_delimiter = tokenizer.decode(output_ids[split_index+1:end_index], skip_special_tokens=True)
                    
                except ValueError:
                    full_text = tokenizer.decode(output_ids, skip_special_tokens=False)
                    after_delimiter = ""
                
                output_texts_concat.append(full_text)
                predictions_after_delimiter.append(after_delimiter)

        return output_texts_concat, predictions_after_delimiter

    def evaluate_full_problems(self):
        """Evaluate full problems and return accuracy"""
        correct = 0
        total = 0
        
        print(f"Evaluating {len(self.test_full_data)} full problems...")
        
        for sample in self.test_full_data:
            # Get the original input text (before tokenization)
            input_ids = sample["input_ids"]
            # Decode the full input to get the original text
            full_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
            
            # Extract just the problem part (before the split_str delimiter)
            if self.split_str in full_text:
                input_text = full_text.split(self.split_str)[0].strip()
            else:
                # Fallback: if split_str not found, try "Command:" for backward compatibility
                if "Command:" in full_text:
                    input_text = full_text.split("Command:")[0].strip()
                else:
                    input_text = full_text
            
            try:
                result = evaluate_full_problem(input_text, self.hf_model, self.tokenizer, max_iters=50)
                if result:
                    correct += 1
                total += 1
            except Exception as e:
                print(f"Error evaluating problem: {e}")
                total += 1
                
        full_problem_accuracy = correct / total if total > 0 else 0.0
        print(f"Full problem accuracy: {correct}/{total} = {full_problem_accuracy:.4f}")
        
        return full_problem_accuracy

    def calculate_metrics(self, predictions, gts):
        """Calculate token-level and exact match accuracy"""
        token_accuracies = []
        exact_matches = []
        
        for pred, gt in zip(predictions, gts):
            # Tokenize prediction and ground truth
            pred_tokens = self.tokenizer.encode(pred, add_special_tokens=False)
            gt_tokens = self.tokenizer.encode(gt, add_special_tokens=False)
            
            # Calculate token-by-token accuracy
            min_len = min(len(pred_tokens), len(gt_tokens))
            matches = 0
            for i in range(min_len):
                if pred_tokens[i] == gt_tokens[i]:
                    matches += 1
                    
            token_acc = matches / max(len(pred_tokens), len(gt_tokens)) if max(len(pred_tokens), len(gt_tokens)) > 0 else 1.0
            token_accuracies.append(token_acc)
            
            # Calculate exact match
            exact_match = 1.0 if pred == gt else 0.0
            exact_matches.append(exact_match)
        
        metrics = {
            "token_full_accuracy": np.mean(token_accuracies),
            "exact_match_accuracy": np.mean(exact_matches)
        }
        
        return metrics

    def save(self, full_predictions, predictions_after_delimiter, gts, metrics=None):
        eval_dir = os.path.join(self.config.eval.results_dir, f"step_{self.step}")
        os.makedirs(eval_dir, exist_ok=True)
        results_file = os.path.join(eval_dir, f"results_{self.num_examples}.json")
        
        results = {
            "full_predictions": full_predictions,
            "predictions_after_delimiter": predictions_after_delimiter,
            "ground_truths": gts
        }
        
        if metrics:
            results["metrics"] = metrics
            
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)

    def evaluate(self, run_full_problem_eval: bool = True):
        # Get single-step predictions
        full_preds, preds_after_delimiter = self.get_preds()
        
        # Save predictions as attributes so they can be accessed later
        self.full_predictions = full_preds
        self.predictions_after_delimiter = preds_after_delimiter
        
        # Calculate single-step metrics
        step_metrics = self.calculate_metrics(preds_after_delimiter, self.gts)
        
        # Evaluate full problems conditionally
        metrics = {**step_metrics} # Initialize metrics with single-step ones
        if run_full_problem_eval: # Modified line
            full_problem_accuracy = self.evaluate_full_problems()
            metrics["full_problem_accuracy"] = full_problem_accuracy # Only add if run
        
        # Print metrics
        print("\nEvaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Clean up model to free memory
        del self.hf_model
        torch.cuda.empty_cache()

        return metrics