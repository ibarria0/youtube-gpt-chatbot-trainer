# youtube-gpt-chatbot-trainer

Question Answering using Embeddings for any YouTube Channel. This project is based off this tutorial: https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb

## Installation
```
#clone repo
git clone https://github.com/ibarria0/youtube-gpt-chatbot-trainer.git
cd youtube-gpt-chatbot-trainer

#setup venv
python3 -m venv venv
source venv/bin/activate

#dependencies
pip install -r requirements.txt
```

## Config
You need to setup environment variables for Google Cloud and OPENAI.
```
export OPENAI_API_KEY="XXXXXXXXXXXXXXXXXXXXXXXXXXXX"
export GOOGLE_API_KEY="XXXXXXXXXXXXXXXXXXXXXXXXXXXX"
```

## Training
This will iterate through the channels videos, download all the transcripts (only en, es working for now), split them up into sections of default 120 tokens (see SECTION_TOKENS in utils.py), calculate embeddings and save both the embeddings and the content into csv files in ./train-data/
```
python ygct/main.py train lacapital
```
## Question answering
This will load the formerly mentioned csv files and build prompts from questions you can input from the console.
```
python ygct/main.py ask lacapital
```
## Todo
- Lots of work in the breaking content into sections part
- Work with any language and manually generated transcripts
- Fine tunning
-  ????
- Profit
