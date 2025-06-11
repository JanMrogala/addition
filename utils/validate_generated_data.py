from transformers import PreTrainedTokenizerFast
from omegaconf import DictConfig
import os
import hydra
from typing import Optional

@hydra.main(config_path="../config", config_name="search", version_base=None)
def get_tokenizer(tok_data: DictConfig):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.abspath(tok_data.data.tokenizer_path)
    )
    return tokenizer

tokenizer = get_tokenizer()
text = "This is an example sentence."
inputs = tokenizer.encode(text, return_tensors="pt")
print(inputs)

