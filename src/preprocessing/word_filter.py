import requests

def load_stopwords():
    """
    불용어 사전을 다운로드합니다.

    Returns:
        list: 불용어 리스트
    """
    url = "https://gist.githubusercontent.com/chulgil/d10b18575a73778da4bc83853385465c/raw/a1a451421097fa9a93179cb1f1f0dc392f1f9da9/stopwords.txt"  # 불용어 사전
    response = requests.get(url)    # 불용어 사전 다운로드
    data = response.content.decode("utf-8") # 불용어 사전을 utf-8로 디코딩
    stopwords = data.split("\n")    # 불용어 사전을 줄바꿈을 기준으로 분리
    stopwords = [word for word in stopwords if word]    # 빈 문자열 제거

    return stopwords


def remove_stopwords(tokens: list, stopwords):
    """
    주어진 토큰 리스트에서 불용어를 제거합니다.
    Args:
        tokens (list): 입력 토큰 리스트
        stopwords (list): 불용어 리스트
    Returns:
        list: 불용어가 제거된 토큰 리스트
    """
    if type(tokens) == list:
        return [token for token in tokens if (token not in stopwords) and (len(token) > 1)] # 한 글자 초과인 단어만 추출, 불용어 사전으로 제거
    
