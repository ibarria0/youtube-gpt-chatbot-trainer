import pandas as pd
import argparse
import downloader
import utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "test"])
    parser.add_argument("channel")
    args = parser.parse_args()

    if args.command == "train":
        print(f'training with channel {args.channel}')	
        content = downloader.extract_content(args.channel)
        utils.train_embeddings(content, args.channel)
        print('done')
        
    elif args.command == "test":
        df = pd.read_csv(f'train-data/{args.channel}_sections.csv')
        document_embeddings = utils.load_embeddings(f'train-data/{args.channel}_embeddings.csv')
        while True:
            print('enter question')
            p = input()
            a = utils.answer_query_with_context(p, df, document_embeddings)
            print(a)

if __name__ == "__main__":
    main()
