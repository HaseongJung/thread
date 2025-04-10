from sentence_transformers import SentenceTransformer


def load_embedding_model(model_name: str):
    """
    주어진 모델 이름으로 SentenceTransformer 모델을 로드합니다.

    Args:
        model_name (str): 로드할 모델의 이름

    Returns:
        SentenceTransformer: 로드된 모델
    """
    embedding_model = SentenceTransformer(  # load embedding model
    "jinaai/jina-embeddings-v3",
    trust_remote_code=True, 
    device="cuda"
    )

    return embedding_model


def embed_documents(embedding_model, documents: list):
    """
    주어진 문서 리스트를 임베딩합니다.

    Args:
        embedding_model (SentenceTransformer): 임베딩 모델
        documents (list): 임베딩할 문서 리스트

    Returns:
        np.ndarray: 임베딩된 문서 리스트
    """
    embeddings = embedding_model.encode(    # emedding
        documents,
        task="separation",
        prompt_name="separation",
        show_progress_bar=True
    )  
    print("임베딩 결과 shape:", embeddings.shape)
    
    return embeddings

