from src.collection.collector import collect_articles
from src.preprocessing import cleaner, tokenizer, token_processor, vectorizer
from src.modeling.topic_modeling import topic_modeling, mk_output_path, save_result
from tqdm import tqdm
tqdm.pandas()


def main():
    # collect articles
    df = collect_articles()
    print(df)

    # remove noise
    print("Removing noise from description...")
    df['description'] = df['description'].progress_apply(lambda x: cleaner.remove_noise(x))

    # tokenize
    print("Tokenizing description...")
    df['desc_tokens'] = df['description'].progress_apply(lambda x: tokenizer.tokenize(x))

    # remove stopwords
    print("Removing stopwords from description...")
    custom_stopwords = ['뉴스데일리', '속보', '9632', '진짜']
    df['desc_tokens'] = df['desc_tokens'].progress_apply(lambda x: token_processor.process_tokens(x, custom_stopwords))

    # join tokens
    print("Joining tokens in description...")
    df["description"] = df["description"].progress_apply(lambda x: " ".join(x) if isinstance(x, list) else x)

    # Documnet: Combine title and description
    df['text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
    documents = df['text'].tolist()
    print("Documents shape:", len(documents))


    # Embedding
    print("Loading embedding model...")
    embedding_model = vectorizer.load_embedding_model("jinaai/jina-embeddings-v3")

    # Topic Modeling
    print("Performing topic modeling...")
    topic_model, topics, probs = topic_modeling(documents, embedding_model)

    # Save result
    output_path = mk_output_path()
    save_result(df, documents, topic_model, topics, probs, output_path)




if __name__ == "__main__":
    main()