import re
import pandas as pd
from tqdm import tqdm
tqdm.pandas()


def remove_noise(text):
    """
    주어진 텍스트에서 노이즈를 제거합니다.
    Args:
        text (str): 입력 텍스트
    Returns:
        str: 노이즈가 제거된 텍스트
    """
    if type(text) == str:
        text = re.sub(r'<[^>]+>', ' ', text)     # HTML 태그 제거
        text = re.sub(r'http\S+|www\.\S+', ' ', text)   # URL 제거
        text = re.sub(r'\S+@\S+', ' ', text)    # 이메일 주소 제거
        text = re.sub(r'\s+', ' ', text).strip()    # 여러 개의 공백을 단일 공백으로 변환
        text = re.sub(r"[^0-9a-zA-Zㄱ-ㅎㅏ-ㅣ가-힣 ]", '', str(text))   # 특수문자 제거
    
    return text