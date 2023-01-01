import argparse
import downloader
import utils
from transformers import GPT2TokenizerFast
import pandas as pd
import numpy as np
import openai

MAX_SECTION_LEN = 500
SEPARATOR = "\n* "

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
separator_len = len(tokenizer.tokenize(SEPARATOR))


DOC_EMBEDDINGS_MODEL = "text-embedding-ada-002"
QUERY_EMBEDDINGS_MODEL = "text-embedding-ada-002"

COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 300,
    "model": "text-davinci-002",
}

def get_embedding(text: str, model: str) -> list[float]:
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def get_doc_embedding(text: str) -> list[float]:
    return get_embedding(text, DOC_EMBEDDINGS_MODEL)

def get_query_embedding(text: str) -> list[float]:
    return get_embedding(text, QUERY_EMBEDDINGS_MODEL)

def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.

    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    return pd.concat(
        [
            df,
            pd.DataFrame(
                [get_doc_embedding(r.content) for idx, r in df.iterrows()], 
                index=df.index, 
                columns=list(range(0, 1536))
            )
        ], axis=1
    )

def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    document_embeddings: dict[(str, str), np.array],
    show_prompt: bool = False
) -> str:
    prompt = construct_prompt(
        query,
        document_embeddings,
        df
    )
    
    if show_prompt:
        print(prompt)

    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )

    return response["choices"][0]["text"].strip(" \n")

def load_embeddings(fname: str) -> dict[tuple[str, str], list[float]]:
    """
    Read the document embeddings and their keys from a CSV.
    
    fname is the path to a CSV with exactly these named columns: 
        "title", "content", "tokens", "0", "1", ... up to the length of the embedding vectors.
    """
    
    df = pd.read_csv(fname, header=0)
    max_dim = max([int(c) for c in df.columns if c != "title" and c != "content" and c != "tokens"])
    return {
           (r.title, r.content, r.tokens): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
    }

def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    We could use cosine similarity or dot product to calculate the similarity between vectors.
    In practice, we have found it makes little difference.
    """
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[(float, (str, str))]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections.

    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_query_embedding(query)

    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)

    return document_similarities


def construct_prompt(question: str, context_embeddings: dict) -> str:
    """
    Fetch relevant
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)

    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []

    for _, section_index in most_relevant_document_sections:
        print(_, section_index)
        # Add contexts until we run out of space.
        document_section = section_index[1]

        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break

        chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))

    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections:")
    print("\n".join(chosen_sections_indexes))

    header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""

    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "test"])
    parser.add_argument("channel")
    args = parser.parse_args()

    if args.command == "train":
        print(f'training with channel {args.channel}')	
        channel_id = downloader.get_channel_id(args.channel)
        print(f'channel id: {channel_id}')	
        content = []
        videos = downloader.get_channel_videos(channel_id)
        for video_id,title in videos[:3]:
            raw_transcript = downloader.get_transcript_for_video(video_id)
            sections = [(title, body, tokens) for body, tokens in utils.get_sections(raw_transcript)]
            content.extend(sections)
        df = pd.DataFrame(content, columns=["title", "content", "tokens"])
        df = df[df.tokens>40]
        df.to_csv(f'train-data/{args.channel}_sections.csv')
        context_embeddings = compute_doc_embeddings(df)
        context_embeddings.to_csv(f'train-data/{args.channel}_embeddings.csv')
        print(context_embeddings.head())
    elif args.command == "test":
        df = pd.read_csv(f'train-data/{args.channel}_sections.csv')
        document_embeddings = load_embeddings(f'train-data/{args.channel}_embeddings.csv')
        while True:
            print('enter question')
            p = input()
            prompt = construct_prompt(
                p,
                document_embeddings,
            )
            print("===\n", prompt)

if __name__ == "__main__":
    main()
