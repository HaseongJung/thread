import os
import numpy as np
from dotenv import load_dotenv
from bareunpy import Tokenizer
# nltk.download('stopwords')

# load API KEy
load_dotenv('./')   # load .env
API_KEY = os.environ.get('BAREUN_API_KEY') 




def tokenize(text: str) -> list:
    """
    Tokenize the text using the Bareun tokenizer.
    """
    # load API Key
    load_dotenv('./')   # load .env
    API_KEY = os.environ.get('BAREUN_API_KEY') 

    # load tokenizer
    tokenizer = Tokenizer(API_KEY, 'localhost', port=5757)

    # tokenize
    if isinstance(text, str) and text.strip():  # 문자열인지 확인하고 비어 있지 않은지 확인
        result = tokenizer.tokenize(text).segments()
        return result
    return np.nan  # text가 유효하지 않으면 np.nan 반환



