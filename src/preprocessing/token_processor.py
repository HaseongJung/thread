import requests
from typing import List

def load_stopwords() -> List[str]:
    """
    불용어 사전을 로드합니다.
    
    Returns:
        list: 불용어 리스트
    """
    url = "https://gist.githubusercontent.com/chulgil/d10b18575a73778da4bc83853385465c/raw/a1a451421097fa9a93179cb1f1f0dc392f1f9da9/stopwords.txt"
    response = requests.get(url)
    data = response.content.decode("utf-8")
    stopwords = data.split("\n")
    stopwords = [word for word in stopwords if word]
    return stopwords


def remove_stopwords(tokens: List[str], stopwords: List[str]) -> List[str]:
    """
    토큰에서 불용어를 제거합니다.

    Args:
        tokens (list): 입력 토큰 리스트
        stopwords (list): 불용어 리스트
    Returns:
        list: 불용어가 제거된 토큰 리스트
    """
    if type(tokens) == list:
        return [token for token in tokens if (token not in stopwords) and (len(token) > 1)]


def process_tokens(tokens: List[str], custom_stopwords: List[str] = None) -> List[str]:
    """
    토큰 처리를 위한 통합 함수
    Args:
        tokens (list): 입력 토큰 리스트
        custom_stopwords (list): 사용자 정의 불용어 리스트
    Returns:
        list: 처리된 토큰 리스트
    """
    stopwords = load_stopwords()
    if custom_stopwords:
        stopwords.extend(custom_stopwords)
    processed_tokens = remove_stopwords(tokens, stopwords)
    return processed_tokens