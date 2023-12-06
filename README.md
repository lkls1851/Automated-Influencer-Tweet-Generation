# Automated-Influencer-Tweet-Generation
In this project, we build a model which can generate tweets similar to the style of tweets of influencers.
This project involves fine tuning a pre trained LLM,in this case GPT2, and leveraging its capabilities to generate tweets based on prompts, similar to the style of influencers on Twitter.

For this project, we have taken the example of Influencer Justin Welsh.

Architecture of Model:
- Project Root
  - data.csv
  - src
    - dataset.py
    - tokenizer.py
    - process.py
    - output.py
    - data_split.py
    - main.py
  - docs
    - documentation.md
   
Steps for running the program:
- Download the files
- Run python3 main.py
- Click the local link to go to the Gradio Interface
