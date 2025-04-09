import os
from datetime import datetime
from alive_progress import alive_it
import feedparser
import json
import pandas as pd
from bs4 import BeautifulSoup

def create_df():
    df = pd.DataFrame(columns=['title', 'description', 'published', 'link', 'media'])
    return df


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
        desc = 'None'
    if desc == '':
        try:
            desc = article['content'][0]['value']
        except KeyError:
            pass
    # cleaning the description
    # try:
    #     desc = ''.join(article['content'][0]['value'])
    #     soup = BeautifulSoup(desc, 'html.parser')
    #     paragraphs = soup.find_all('p')
    #     desc = paragraphs[0].get_text()

    try:
        published = article.published
    except:
        published = article.updated

    try:
        link = article.link
    except:
        link = None

    return {'title': title, 'description': desc, 'published': published, 'link': link, 'media': media}


def save_df(df):
    datetime_ = datetime.now().strftime("%Y%m%d_%H%M")
    file_path = "./data/political_news/"
    file_name = f'political_news_{datetime_}.csv'

    df.to_csv(os.path.join(file_path, file_name), index=False, encoding='utf-8')
    print(f"{file_name} file saved successfully.")


def main():
    # initialize the pandas DataFrame
    df = create_df()

    # load the JSON data
    data_path = "./data/political_rss.json"
    data = load_json(data_path)

    # RSS별로 뉴스 기사 가져오기
    bar = alive_it(data, finalize=bar_ending, force_tty=True)
    bar.title = "test"
    for rss in bar:
        media = rss['media']
        url = rss['url']
        feed = feedparser.parse(url)

        # article 정보 추출
        articles = feed.entries
        for article in articles:
            article_data = extract_article_date(article, media)

            # 새로운 행(뉴스 기사) 추가
            df = pd.concat([df, pd.DataFrame([article_data])], ignore_index=True)


    print(df)
    save_df(df)




if __name__ == "__main__":
    main()