from sentence_transformers import SentenceTransformer

# Embedding
embedding_model = SentenceTransformer(  # load embedding model
    "jinaai/jina-embeddings-v3",
    trust_remote_code=True, 
    device="cuda"
    )

embeddings = embedding_model.encode(    # emedding
    df['text'].tolist(),
    task="separation",
    prompt_name="separation",
    show_progress_bar=True
)  
print("임베딩 결과 shape:", embeddings.shape)


def load_embedding_model(model_name: str):
    """
    주어진 모델 이름으로 SentenceTransformer 모델을 로드합니다.

    Args:
        model_name (str): 로드할 모델의 이름

    Returns:
        SentenceTransformer: 로드된 모델
    """
    return SentenceTransformer(model_name, device="cuda")