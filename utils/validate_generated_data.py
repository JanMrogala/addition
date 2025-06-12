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

@hydra.main(config_path="../config", config_name="search", version_base=None)
def main(tok_data: DictConfig):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.abspath(tok_data.data.tokenizer_path)
    )

    with open(f"data/t_search/{tok_data.data.format}/train.json") as f:
        train = json.load(f)
    with open(f"data/t_search/{tok_data.data.format}/test.json") as f:
        test = json.load(f)

    res = []
    for item in train:
        input_ids = get_input_ids(item['text'], tokenizer)
        res.append(input_ids)

    train_set = set(tuple(i) for i in res)
    print(len((res)), len(train_set))


if __name__ == "__main__":
    main()


