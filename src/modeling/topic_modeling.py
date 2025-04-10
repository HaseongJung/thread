import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from pathlib import Path

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

def save_tp_result(df, documents, topic_model, topics, probs, output_path: str):
    """
    Save the topic model result to CSV and visualize the results.

    Args:
        topic_model (BERTopic): The trained BERTopic model.
        probs (list): The probabilities of the topics.
        output_path (str): The path to save the results.
    """
    logs = []

    # Topic model result
    tp_result = topic_model.get_topic_info()
    tp_result.to_csv(output_path + "Result.csv", index=False, encoding='utf-8')
    logs.append(("Topic model result saved to", output_path + "Result.csv"))


    # # Intertopic Distance Map (Bubble chart) 
    # fig = topic_model.visualize_topics()
    # fig.write_image(output_path + "Intertopic_Distance_Map.png")
    # logs.append(("Intertopic Distance Map saved to", output_path + "Intertopic_Distance_Map.png"))

    # Topic별 뉴스기사 저장 -> .csv
    for i in range(len(tp_result)):
        df['topic'] = topics
        topic_name = '_'.join([word[0] for word in topic_model.get_topic(tp_result['Topic'][i])])
        topic_num = tp_result['Topic'][i]
        filename = f'Topic{str(i-1)}_{topic_name}.csv'
        topic_df = df[df['topic'] == topic_num]
        topic_df.to_csv(output_path + filename, index=False, encoding='utf-8')
        logs.append((f"Topic {topic_num} saved to", output_path + filename))

    # Topic Probability Distribution: Bar chart
    fig = topic_model.visualize_distribution(probs[300], min_probability=0.015)
    fig.write_image(output_path + "Topic_Probability_Distribution.png")
    logs.append(("Topic Probability Distribution saved to", output_path + "Topic_Probability_Distribution.png"))

    # Topic Hierarchy: Dendrogram
    fig = topic_model.visualize_hierarchy()
    fig.write_image(output_path + "Topic_Hierarchy.png")
    logs.append(("Topic Hierarchy saved to", output_path + "Topic_Hierarchy.png"))

    # Topic Word Scores: Bar chart
    fig = topic_model.visualize_barchart(top_n_topics=10, n_words=10)
    fig.write_image(output_path + "Topic_Word_Scores.png")
    logs.append(("Topic Word Scores saved to", output_path + "Topic_Word_Scores.png"))

    # Topic Similarity: Heatmap
    fig = topic_model.visualize_heatmap()
    fig.write_image(output_path + "Topic_Similarity.png")
    logs.append(("Topic Similarity saved to", output_path + "Topic_Similarity.png"))

    # Document Distribution: Scatter plot
    fig = topic_model.visualize_documents(docs=documents, topics=topics)
    fig.write_image(output_path + "Document_Distribution.png")
    logs.append(("Document Distribution saved to", output_path + "Document_Distribution.png"))
    
    # 우측 정렬된 출력
    max_label_width = max(len(label) for label, _ in logs)
    for label, path in logs:
        print(f"{label:<{max_label_width}} -> {path}")





def main():
    # data load
    data_path = "./data/preprocessed/20250409_2109.csv"
    df = load_data(data_path)

    # Preprocessing: Combine title and description
    df['text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
    documents = df['text'].tolist()
    print("Documents shape:", len(documents))


    # Load the embedding model
    embedding_model = SentenceTransformer(  # load embedding model
        "jinaai/jina-embeddings-v3",
        trust_remote_code=True,
        device="cuda"
        )

    # Embedding
    embeddings = embedding_model.encode(    # emedding
        df['text'].tolist(),
        task="separation",
        prompt_name="separation",
        show_progress_bar=True
    )  
    print("Embedding result shape:", embeddings.shape)


    # Topic modeling
    topic_model = BERTopic(embedding_model=embedding_model, calculate_probabilities=True)
    topics, probs = topic_model.fit_transform(documents)
    print("Topic modeling completed.")
    
    # Save the topic model result
    datetime_ = Path(data_path).stem.split('/')[-1][-17:]
    output_path = f"./output/topic_modeling/{datetime_}/"
    os.makedirs(output_path, exist_ok=True)
    print(f"Output directory created: {output_path}")
    save_tp_result(df, documents, topic_model, topics, probs, output_path)



if __name__ == "__main__":
    main()