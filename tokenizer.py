import pandas as pd
from transformers import GPT2Tokenizer

def tokenized_data(data, model_name):
	seq_data=data.tolist()
	tokenizer = GPT2Tokenizer.from_pretrained(model_name)
	tokenized_data = [tokenizer.encode(dataval, add_special_tokens=True) for dataval in seq_data]
	return tokenized_data
