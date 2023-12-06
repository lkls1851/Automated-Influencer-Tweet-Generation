import gradio as gr
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name="gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)  

def user_interface():
	def function(prompt, length):
    		input_ids = tokenizer.encode(prompt, return_tensors="pt")
    		output = model.generate(
        		input_ids,
        		max_length=length,
        		num_return_sequences=1,
        		no_repeat_ngram_size=2,
        		pad_token_id=tokenizer.eos_token_id,
        		attention_mask=torch.ones(input_ids.shape)
    		)
    		generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    		return generated_text
    		
    	
	demo = gr.Interface(
    	fn=function,
    	inputs=["text", "number"],
    	outputs=["text"],
	)
	demo.launch()
