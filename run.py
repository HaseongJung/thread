from src.collection.collector import collect_articles
from src.preprocessing import cleaner, tokenizer, token_processor, vectorizer
from src.modeling.topic_modeling import topic_modeling, save_result
from src.util.path_manager import make_output_path
from src.util.email_sender import send_topic_results
from src.summarize.post_maker import PostMaker
import polars as pl
import time
import glob
from alive_progress import alive_it



def main():
    # collect articles
    df = collect_articles()

    # remove noise
    print("Removing noise...")
    df = df.with_columns(
        title = pl.col('title').map_elements(cleaner.remove_noise, skip_nulls=True, return_dtype=pl.Utf8),
        description = pl.col('description').map_elements(cleaner.remove_noise, skip_nulls=True, return_dtype=pl.Utf8)
    )
    # df['title'] = df['title'].apply(lambda x: cleaner.remove_noise(x))
    # df['description'] = df['description'].apply(lambda x: cleaner.remove_noise(x))

    # tokenize
    print("Tokenizing...")
    df = df.with_columns(
        title_tokens = pl.col('title').map_elements(tokenizer.tokenize, skip_nulls=True, return_dtype=pl.List(pl.Utf8)),
        desc_tokens = pl.col('description').map_elements(tokenizer.tokenize, skip_nulls=True, return_dtype=pl.List(pl.Utf8))
    )
    # df['title_tokens'] = df['title'].apply(lambda x: tokenizer.tokenize(x))
    # df['desc_tokens'] = df['description'].apply(lambda x: tokenizer.tokenize(x))

    # remove stopwords
    print("Removing stopwords...")
    df = df.with_columns(
        title_tokens = pl.col('title_tokens').map_elements(token_processor.remove_stopwords, return_dtype=pl.List(pl.Utf8)),
        desc_tokens = pl.col('desc_tokens').map_elements(token_processor.remove_stopwords, return_dtype=pl.List(pl.Utf8))
    )
    
    # df['title_tokens'] = df['title_tokens'].apply(lambda x: token_processor.process_tokens(x, custom_stopwords))
    # df['desc_tokens'] = df['desc_tokens'].apply(lambda x: token_processor.process_tokens(x, custom_stopwords))

    # Join tokens
    print("Joining tokens...")
    df = df.with_columns(
        title_tokens = pl.col('title_tokens').map_elements(lambda x: ' '.join(x), return_dtype=pl.Utf8),
        desc_tokens = pl.col('desc_tokens').map_elements(lambda x: ' '.join(x), return_dtype=pl.Utf8)
    )
    # df['title_tokens'] = df["title_tokens"].apply(lambda x: " ".join(x) if isinstance(x, list) else x)
    # df['desc_tokens'] = df["desc_tokens"].apply(lambda x: " ".join(x) if isinstance(x, list) else x)

    # Documnet: Combine title and description
    df = df.with_columns(
        text = pl.concat_str([pl.col('title_tokens'), pl.col('desc_tokens')], separator=' ', ignore_nulls=True)
    )
    documents = [document for document in df['text'].to_list() if document is not None]
    print("Documents shape:", len(documents))

    # Embedding
    print("Loading embedding model...")
    embedding_model = vectorizer.load_embedding_model("jinaai/jina-embeddings-v3")

    # Topic Modeling
    print("Performing topic modeling...")
    topic_model, topics, probs = topic_modeling(documents, embedding_model)

    # Save result
    output_path = make_output_path()
    save_result(df, documents, topic_model, topics, probs, output_path)

    # 결과를 이메일로 전송
    # send_topic_results(output_path)

    # Generate thread Posts
    documents_dir = f"{output_path}Documents/"
    documents_paths = sorted(glob.glob(f"{documents_dir}*.csv"))[1:]

    post_maker = PostMaker(model_name="gemini-2.5-flash-preview-04-17")
    
    bar = alive_it(documents_paths, title="Generating posts")
    for document_path in bar:
        # Generate post
        post = post_maker.generate_post(document_path)
        # Save post
        post_file_path = f"{output_path}Posts/{document_path.split('/')[-1].replace('.csv', '.txt')}"
        post_maker.save_post(post, post_file_path)
        time.sleep(7)
        



if __name__ == "__main__":
    main()




