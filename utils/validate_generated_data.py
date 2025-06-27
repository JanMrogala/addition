from tqdm import tqdm
from transformers import PreTrainedTokenizerFast
from omegaconf import DictConfig
import os
import hydra
import json

def get_input_ids(texts, tokenizer):
    """
    Args:
        texts: List of text strings
        tokenizer: Tokenizer from get_tokenizer function
    
    Returns:
        List of lists of input_ids (raw tokenization)
    """
    outputs = tokenizer(
        texts,
        truncation=False,    
        padding=False,       
        add_special_tokens=False,  
        return_overflowing_tokens=False,
    )
    return outputs["input_ids"]

def filter_data(data, tokenizer):
    res = []

    for item in tqdm(data, desc="Tokenizing data"):
        input_ids = get_input_ids(item['text'], tokenizer)
        res.append(input_ids)


    # duplicate_idxs = []
    # spacer = 1
    # for x, i in tqdm(enumerate(res), desc="Finding duplicates"):
    #     for idx, n in enumerate(res[spacer:]):
    #         if i == n:
    #             duplicate_idxs.append(idx+spacer)
    #     spacer += 1
    
    # for duplicate in duplicate_idxs:
    #     data[duplicate] = None

    # data = [item for item in data if item is not None]

    print("Removing duplicates...")
    res_set = set(tuple(x) for x in res)
    res_list = [list(x) for x in res_set]
    print("duplicates removed:", len(res) - len(res_list))

    # based on the res_list, filter the original data
    filtered_data = []
    for item in tqdm(data, desc="Filtering data"):
        input_ids = get_input_ids(item['text'], tokenizer)
        if input_ids in res_list:
            filtered_data.append(item)
            res_list.remove(input_ids)  # Remove to ensure uniqueness

    return filtered_data

@hydra.main(config_path="../config", config_name="search", version_base=None)
def main(tok_data: DictConfig):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.abspath(tok_data.data.tokenizer_path)
    )

    formats = ["A", "B", "C"]
    for format in formats:
        print(f"Processing format: {format}")
        with open(f"data/t_search/{format}/train.json") as f:
            train = json.load(f)
            train_len = len(train)

        with open(f"data/t_search/{format}/test.json") as f:
            test = json.load(f)
            test_len = len(test)
        
        train = filter_data(train, tokenizer)
        test = filter_data(test, tokenizer)

        with open(f"data/t_search/{format}/train.json", "w") as f:
            json.dump(train, f, indent=4)
            train_after_len = len(train)
        # with open(f"data/t_search/{format}/test.json", "w") as f:
        #     json.dump(test, f, indent=4)
        test_after_len = len(test)

        print(f"Train set: {train_len} -> {train_after_len} ({train_len - train_after_len} duplicates removed)")
        print(f"Test set: {test_len} -> {test_after_len} ({test_len - test_after_len} duplicates removed)")


        # compare train and test sets for duplicates
        train_texts = {item['text'] for item in train}
        test_texts = {item['text'] for item in test}

        print(f"Train and test sets have {len(train_texts.intersection(test_texts))} overlapping items")

        # remove overlapping items from test set
        test = [item for item in test if item['text'] not in train_texts]

        print(f"Final test set length after removing overlaps: {len(test)}")

        # save the final test set
        with open(f"data/t_search/{format}/test.json", "w") as f:
            json.dump(test, f, indent=4)


if __name__ == "__main__":
    main()


