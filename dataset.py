import pandas as pd

def processed_data(path):
	df=pd.read_csv(path)
	df=df['full_text']
	return df
