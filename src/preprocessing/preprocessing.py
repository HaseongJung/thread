import pandas as pd
import re
import requests
from tqdm import tqdm;  tqdm.pandas()
from tokenizer import tokenize



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

def remove_noise(text):
    if type(text) == str:
        text = re.sub(r'<[^>]+>', ' ', text)     # HTML 태그 제거
        text = re.sub(r'http\S+|www\.\S+', ' ', text)   # URL 제거
        text = re.sub(r'\S+@\S+', ' ', text)    # 이메일 주소 제거
        text = re.sub(r'\s+', ' ', text).strip()    # 여러 개의 공백을 단일 공백으로 변환
        text = re.sub(r"[^0-9a-zA-Zㄱ-ㅎㅏ-ㅣ가-힣 ]", '', str(text))   # 특수문자 제거

    return text


def load_stopwords():
    url = "https://gist.githubusercontent.com/chulgil/d10b18575a73778da4bc83853385465c/raw/a1a451421097fa9a93179cb1f1f0dc392f1f9da9/stopwords.txt"  # 불용어 사전
    response = requests.get(url)    # 불용어 사전 다운로드
    data = response.content.decode("utf-8") # 불용어 사전을 utf-8로 디코딩
    stopwords = data.split("\n")    # 불용어 사전을 줄바꿈을 기준으로 분리
    stopwords = [word for word in stopwords if word]    # 빈 문자열 제거

    return stopwords

def remove_stopwords(tokens: list, stopwords):
    if type(tokens) == list:
        return [token for token in tokens if (token not in stopwords) and (len(token) > 1)] # 한 글자 초과인 단어만 추출, 불용어 사전으로 제거


def main():
    # laod data
    data_path = "./data/political_news/political_news_20250409_2109.csv"
    df = load_data(data_path)

    # remove noise
    df['description'] = df['description'].progress_apply(lambda x: remove_noise(x))
    
    # tokenize
    df['description'] = df['description'].progress_apply(lambda x: tokenize(x))


    # remove stopwords
    stopwords = load_stopwords()
    my_stopwords = ['뉴스데일리', '9650']
    stopwords.extend(my_stopwords)  # 추가 불용어
    df["description"] = df["description"].progress_apply(lambda x: remove_stopwords(x, stopwords))

    # join tokens
    df["description"] = df["description"].progress_apply(lambda x: " ".join(x) if isinstance(x, list) else x)

    print(df)

    # save the cleaned data
    datetime_ = data_path.split('/')[-1][-17:]
    df.to_csv(f"./data/preprocessed/text_cleaned_{datetime_}.csv", index=False, encoding='utf-8')

if __name__ == "__main__":
    main()