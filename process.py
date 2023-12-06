import torch


def processed_tokenized_data(data):
	max_len=max(len(block) for block in data)
	padding_data=[block+[0]*(max_len-len(block)) for block in data]
	tensor_data=torch.tensor(padding_data)
	return tensor_data
	
	
	







