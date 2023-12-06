
## Steps

### 1. Data Gathering (data.csv)

- **Description:** This step involves collecting social media posts from platform, in this example Twitter, respecting legal and ethical guidelines.
- **Methodology:** We use open source website **apify** to scrape Twitter posts from user **@thejustinwelsh**, respecting legal and privacy policies of Twitter.
- **File:** `data.csv`
- **Contents:** Contains scraped data with fields like post_text, date, retweets, etc.
  
### 2. Data Preprocessing (dataset.py)

- **Description:** Clean and preprocess the scraped data to make it suitable for training a language model.
- **Processing:** Data Preprocessing done to remove less important categories like retweets, etc.
- **File:** `dataset.py`
- **Methods:**
  - `processed_data(data)`: Removes less useful categories from the dataset.
  
### 3. Tokenization (tokenizer.py)

- **Description:** Tokenize the preprocessed data to prepare it for training.
- **File:** `tokenizer.py`
- **Method:** We use pre trained GPT2 Tokenizer for tokenizing the cleaned data and special tokens wherever necessary. 
- **Methods:**
  - `tokenized_data(data)`: Tokenizes the cleaned dataset.

### 4. Data Processing (process.py)

- **Description:** Apply padding to ensure uniform sequence length and convert tokenized data to PyTorch tensors.
- **File:** `process.py`
- **Methods:**
  - `processed_tokenized_data(data)`: Returns the padded data after converting it to torch.tensors

### 5. Output Generation (output.py)

- **Description:** Generate posts in the style of influencers based on user prompts and preferences. It creates an user interface using Gradio, where user can input prompts and desired length of tweets and get output.
- **File:** `output.py`
- **Methods:**
  - `user_interface()`: Generates model output using fine-tuned language model.

### 6. Data Splitting (data_split.py)

- **Description:** Randomly split the processed data into training and testing sets, based on input train ratio.
- **File:** `data_split.py`
- **Methods:**
  - `split(tensor_data, train_size)`: Splits the data into training and testing datasets.

### 7. Main Program (main.py)

- **Description:** Control flow and execution of the entire project. Calculates perplexity as an evaluation metric.
- **File:** `main.py`

## Evaluation

### Perplexity

Perplexity is a metric used to evaluate the quality of language models. It measures how well the model predicts a given sequence of words. In this project, perplexity is calculated on the test dataset as an evaluation metric. A lower perplexity indicates better model performance.
For our Fine-Tuned Model of GPT2, we have achieved a Perplexity Score of 1.0044 on test dataset.


