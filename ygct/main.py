import pandas as pd
import argparse
import downloader
import utils
import os

def check_env():
    for var in ["OPENAI_API_KEY"]:
        if not os.getenv(var):
            raise Exception(f"Missing ENV {var}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "ask"])
    parser.add_argument("channel")
    args = parser.parse_args()

    channel = list(filter(None, args.channel.split('/')))[-1]
    check_env()

    if not os.path.exists('train-data'):
        # Create the directory
        os.makedirs('train-data')

    if args.command == "train":
        print(f'training with channel {args.channel}')	
        content = downloader.extract_content(args.channel)
        utils.train_embeddings(content, channel)
        print('done')
        
    elif args.command == "ask":
        print('loading....')
        df = pd.read_csv(f'train-data/{channel}_sections.csv')
        document_embeddings = utils.load_embeddings(f'train-data/{channel}_embeddings.csv')
        while True:
            print('enter question\n')
            p = input()
            a = utils.answer_query_with_context(p, df, document_embeddings)
            print(f'\n{a}\n')

if __name__ == "__main__":
    main()
