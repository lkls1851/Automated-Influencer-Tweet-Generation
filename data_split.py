import torch
from torch.utils.data import TensorDataset,random_split

def split(tensor_data, train_size):
	tensor_dataset=TensorDataset(tensor_data)
	train_len=int(train_size*len(tensor_dataset))
	test_len=len(tensor_dataset) - train_len
	train,test=random_split(tensor_dataset, [train_len,test_len])
	return train,test
