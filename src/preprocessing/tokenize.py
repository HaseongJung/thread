import pandas as pd
import sys
import google.protobuf.text_format as tf
from bareunpy import Tokenizer
from konlpy.tag import Mecab
import multiprocessing as mp
import parmap
num_workers = mp.cpu_count()    # 병렬 처리를 위한 worker 수 설정


if __name__ == "__main__":


    # load dataset
    data_path = "../data/political_news/political_news_20250409_1652.csv"
    dataset = pd.read_feather(data_path)
    print("Dataset loaded!")

    # tokenize
    mecab = Mecab()

    dataset["Title"] = parmap.map(mecab.morphs, dataset["Title"], pm_pbar=True, pm_processes=num_workers)  # parmap.map 함수를 사용하여 mecab.morphs 함수를 병렬로 적용
    dataset["Text"] = parmap.map(mecab.morphs, dataset["Text"], pm_pbar=True, pm_processes=num_workers)
    print(dataset)

    # save tokenized dataset
    dataset.to_feather("src/data/processed/tokenized.feather")
    print("Tokenized dataset saved!")