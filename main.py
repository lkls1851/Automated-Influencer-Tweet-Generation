import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel
from dataset import processed_data
from tokenizer import tokenized_data
from process import processed_tokenized_data
from output import user_interface
from data_split import split

path='data.csv'
model_name='gpt2'
batch_size=4
train_size=0.8


model = GPT2LMHeadModel.from_pretrained(model_name)


data=processed_data(path)
token_data=tokenized_data(data, model_name)
final_input=processed_tokenized_data(token_data)


train, test=split(final_input, train_size)

train_dataloader=DataLoader(train, batch_size, shuffle=True)
test_dataloader=DataLoader(test, batch_size, shuffle=True)




model.eval()
total_loss=torch.tensor(0.0)
total_tokens=torch.tensor(0)
for blocks in test_dataloader:
    input_data=blocks[0]
    labels=input_data.clone()
    with torch.no_grad():
        outputs=model(input_data, labels=labels)
        loss=outputs.loss
    total_loss+=loss.item()
    total_tokens+=labels.numel()

perplexity=torch.exp(total_loss / total_tokens)

print(f"Perplexity: {perplexity.item()}")


user_interface()



