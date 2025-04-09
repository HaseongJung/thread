import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import silhouette_score
# from sklearn.feature_extraction.text import CountVectorizer
# from konlpy.tag import Mecab
from bertopic import BERTopic

def load_data(path):
    """
    주어진 경로에서 데이터를 로드합니다.
    
    Args:
        path (str): 데이터 파일 경로
    
    Returns:
        pd.DataFrame: 로드된 데이터프레임
    """
    df = pd.read_csv(path, encoding='utf-8')
    print("Data loaded successfully")
    return df



def main():
    data_path = "./data/political_news/political_news_20250409_1145.csv"
    df = load_data(data_path)

    df['text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
    documents = df['text'].tolist()


    # Embeddings 모델 로드
    # model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')   # BERT Sentence Embeddings 모델 로드
    embedding_model = SentenceTransformer(
        "jinaai/jina-embeddings-v3",
        trust_remote_code=True,
        device="cuda"
        )
    
    # emedding
    embeddings = embedding_model.encode(
        df['text'].tolist(),
        task="separation",
        prompt_name="separation",
        show_progress_bar=True
    )  
    print("임베딩 결과 shape:", embeddings.shape)


    MODEL_NAME = "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens"
    topic_model = BERTopic(embedding_model=embedding_model, calculate_probabilities=True)
    # model = BERTopic(embedding_model= MODEL_NAME, vectorizer_model=vectorizer, nr_topics=50, top_n_words=10, calculate_probabilities=True)
    topics, probs = topic_model.fit_transform(documents)


if __name__ == "__main__":
    main()