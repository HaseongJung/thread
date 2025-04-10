import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from pathlib import Path
from datetime import datetime

def load_data(path):
    """
    주어진 경로에서 데이터를 로드합니다.
    
    Args:
        path (str): 데이터 파일 경로
    
    Returns:
        pd.DataFrame: 로드된 데이터프레임
    """
    df = pd.read_csv(path, encoding='utf-8')
    print(f"Data loaded successfully. Shape: {df.shape}")
    
    return df



def mk_output_path():
    '''
    Create output directory for topic modeling results.
    The directory is named with the current date and time.
    '''
    
    datetime_ = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = f"./output/topic_modeling/{datetime_}/"
    os.makedirs(output_path, exist_ok=True)
    
    return output_path



def save_result(df, documents, topic_model, topics, probs, output_path: str):
    """
    Save the topic model result to CSV and visualize the results.

    Args:
        topic_model (BERTopic): The trained BERTopic model.
        probs (list): The probabilities of the topics.
        output_path (str): The path to save the results.
    """

    datetime_ = datetime.now().strftime("%Y%m%d_%H%M")  # 현재 날짜와 시간

    output_path = f"./output/topic_modeling/{datetime_}/"   # 결과 저장 경로
    chart_path = f"{output_path}Chart/"  # 차트 저장 경로
    documents_path = f"{output_path}Documents/"  # 문서 저장 경로
    os.makedirs(chart_path, exist_ok=True)
    os.makedirs(documents_path, exist_ok=True)

    # Topic model result
    tp_result = topic_model.get_topic_info()
    tp_result.to_csv(output_path + "Result.csv", index=False, encoding='utf-8')

    # Topic model result: title, description, pulbished, link, media, Topic
    # df.drop(columns=["desc_tokens", "text"], inplace=True)  # desc_tokens, text 열 삭제
    df['topic'] = topics    # topic 번호 추가


    print("Saving topic model result...")
    # Topic별 뉴스기사 저장 -> .csv
    for i in range(len(tp_result)):
        topic_num = tp_result['Topic'][i]
        topic_df = df[df['topic'] == topic_num]

        topic_name = '_'.join([word[0] for word in topic_model.get_topic(tp_result['Topic'][i])[:5]])
        mean_publish_date = topic_df['published'].mean().strftime('%Y%m%d_%H%M') # 평균 게시일
        filename = f'{mean_publish_date}_Topic{(str(i-1).zfill(2))}_{topic_name}.csv'
        # filename 숫자 두자리수로 정렬

        topic_df.to_csv(documents_path + filename, index=False, encoding='utf-8')

    # Intertopic Distance Map (Bubble chart) 
    fig = topic_model.visualize_topics()
    fig.write_image(chart_path + "Intertopic_Distance_Map.png")

    # Topic Probability Distribution: Bar chart
    fig = topic_model.visualize_distribution(probs[300], min_probability=0.015)
    fig.write_image(chart_path + "Topic_Probability_Distribution.png")

    # Topic Hierarchy: Dendrogram
    fig = topic_model.visualize_hierarchy()
    fig.write_image(chart_path + "Topic_Hierarchy.png")

    # Topic Word Scores: Bar chart
    fig = topic_model.visualize_barchart(top_n_topics=10, n_words=10)
    fig.write_image(chart_path + "Topic_Word_Scores.png")

    # Topic Similarity: Heatmap
    fig = topic_model.visualize_heatmap()
    fig.write_image(chart_path + "Topic_Similarity.png")

    # Document Distribution: Scatter plot
    fig = topic_model.visualize_documents(docs=documents, topics=topics)
    fig.write_image(chart_path + "Document_Distribution.png")
    
    # 우측 정렬된 출력}} -> {path}")
    print(f"Documents saved to {documents_path}")
    print(f"Charts saved to {chart_path}")


def topic_modeling(documents, embedding_model):
    """
    Perform topic modeling using BERTopic.

    Args:
        documents (list): List of documents to be modeled.
        embedding_model (SentenceTransformer): Pre-trained embedding model.

    Returns:
        tuple: topics and probabilities
    """
    # Topic modeling
    topic_model = BERTopic(embedding_model=embedding_model, calculate_probabilities=True)
    topics, probs = topic_model.fit_transform(documents)
    
    return topic_model, topics, probs





