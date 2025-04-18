from tqdm import tqdm
from src.collection.collector import collect_articles
from src.preprocessing import cleaner, tokenizer, token_processor, vectorizer
from src.modeling.topic_modeling import topic_modeling, save_result
from src.util.path_manager import make_output_path
from src.util.email_sender import send_topic_results
from src.summarize.post_maker import PostMaker
import glob
from alive_progress import alive_it

tqdm.pandas()


def main():
    # collect articles
    df = collect_articles()

    # remove noise
    print("Removing noise...")
    df['title'] = df['title'].progress_apply(lambda x: cleaner.remove_noise(x))
    df['description'] = df['description'].progress_apply(lambda x: cleaner.remove_noise(x))

    # tokenize
    print("Tokenizing...")
    df['title_tokens'] = df['title'].progress_apply(lambda x: tokenizer.tokenize(x))
    df['desc_tokens'] = df['description'].progress_apply(lambda x: tokenizer.tokenize(x))

    # remove stopwords
    print("Removing stopwords...")
    custom_stopwords = ['뉴스데일리', '속보', '9632', '진짜', '여담', '야담', '9650', 'SBS', 'lt', '편상욱', '뉴스', '브리핑', 'gt', '여담야담', '단독', '방송', 'JTBC', '영상']
    df['title_tokens'] = df['title_tokens'].progress_apply(lambda x: token_processor.process_tokens(x, custom_stopwords))
    df['desc_tokens'] = df['desc_tokens'].progress_apply(lambda x: token_processor.process_tokens(x, custom_stopwords))

    # Join tokens
    print("Joining tokens...")
    df['title_tokens'] = df["title_tokens"].progress_apply(lambda x: " ".join(x) if isinstance(x, list) else x)
    df['desc_tokens'] = df["desc_tokens"].progress_apply(lambda x: " ".join(x) if isinstance(x, list) else x)

    # Documnet: Combine title and description
    df['text'] = df['title_tokens'].fillna('') + ' ' + df['desc_tokens'].fillna('')
    documents = df['text'].tolist()
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

    post_maker = PostMaker(model_name="gemma-3-12b-it")
    
    bar = alive_it(documents_paths, title="Generating posts")
    for document_path in bar:
        # Generate post
        post = post_maker.generate_post(document_path)
        # Save post
        post_file_path = f"{output_path}Posts/{document_path.split('/')[-1].replace('.csv', '.txt')}"
        post_maker.save_post(post, post_file_path)



if __name__ == "__main__":
    main()




