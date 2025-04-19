import os
from datetime import datetime
from alive_progress import alive_it
import feedparser
import json
import pandas as pd
import polars as pl
import numpy as np

def create_df():
    schema = {
        'title': pl.Utf8,
        'description': pl.Utf8,
        'published': pl.Datetime,
        'link': pl.Utf8,
        'media': pl.Utf8
    }
    return pl.DataFrame(schema=schema)
    # df = pl.DataFrame(schema=['title', 'description', 'published', 'link', 'media'])
    # return df


def bar_ending(bar):
    bar.title = "RSS Parsing and DataFrame Creation"


def load_json(data_path):
    with open(data_path, 'r', encoding="utf-8") as file:
        data = json.load(file)
    return data


def extract_article_date(article: dict, media: str) -> dict:
    """
    뉴스 기사에서 제목, 값, 설명, 게시일, 링크를 등 추출

    args:
        article (dict): 뉴스 기사 객체
    
    returns:
        dict: 제목, 값, 설명, 게시일, 링크를 포함하는 딕셔너리
    """ 
    try:
        title = article.title
    except:
        title = None

    try:
        desc = article.description
    except KeyError:
        desc = article.summary
    except:
        desc = None
    if desc == '':
        try:
            desc = article['content'][0]['value']
        except KeyError:
            pass

    try:
        published = article.published
    except:
        published = article.updated
    published = pd.to_datetime(published) 
    published = published.strftime("%Y%m%d-%H%M")
    published = datetime.strptime(published, "%Y%m%d-%H%M")

    try:
        link = article.link
    except:
        link = None

    return {'title': title, 'description': desc, 'published': published, 'link': link, 'media': media}


def save_df(df, save_path):
    '''
    DataFrame을 CSV 파일로 저장합니다.

    Args:
        df (pd.DataFrame): 저장할 DataFrame
        save_path (str): 저장할 경로
    '''
    datetime_ = datetime.now().strftime("%Y%m%d_%H%M")
    file_name = f'{datetime_}.csv'

    df.write_csv(os.path.join(save_path, file_name))
    print(f"{file_name} file saved successfully.")


def collect_articles(data_path="./data/political_rss.json", save_path="./data/raw/"):
    # initialize the pandas DataFrame
    df = create_df()

    # load the JSON data
    data = load_json(data_path)

    # RSS별로 뉴스 기사 가져오기
    bar = alive_it(data, finalize=bar_ending, force_tty=True)
    for rss in bar:
        media = rss['media']
        url = rss['url']
        feed = feedparser.parse(url)

        # article 정보 추출
        articles = feed.entries
        for article in articles:
            article_data = extract_article_date(article, media)

            # 새로운 행(뉴스 기사) 추가
            df = pl.concat([df, pl.DataFrame([article_data])])

    print(f"{len(df)} articles collected from {len(data)} RSS feeds.")
    save_df(df, save_path)
    return df