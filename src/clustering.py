import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import silhouette_score

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




def get_main_news(df, num_clusters, embeddings, kmeans):
    """
    각 군집의 대표(중요) 뉴스를 추출합니다.
    
    Args:
        num_clusters (int): 군집의 개수
        embeddings (list): 뉴스 기사에 대한 임베딩 리스트
        kmeans (KMeans): KMeans 클러스터링 모델
    
    Returns:
        pd.DataFrame: 각 군집의 대표 뉴스 데이터프레임
    """
    representative_indices = [] # 각 군집(클러스터)에서 대표하는 (가장 중심에 가까운) 뉴스의 인덱스 저장할 list
    for cluster_id in range(num_clusters):
        cluster_indices = df.index[df['cluster'] == cluster_id].tolist()    # 군집에 속한 뉴스의 인덱스
        cluster_embeddings = [embeddings[i] for i in cluster_indices]   # 군집에 속한 뉴스의 임베딩
        centroid = kmeans.cluster_centers_[cluster_id]  # 군집의 중심(centroid) 임베딩
        closest, _ = pairwise_distances_argmin_min([centroid], cluster_embeddings, metric='cosine') # 군집의 중심과 가장 가까운 뉴스의 인덱스
        representative_index = cluster_indices[closest[0]]  # 군집의 중심과 가장 가까운 뉴스의 인덱스
        representative_indices.append(representative_index)

    return representative_indices


def select_optimal_clusters(embeddings):
    silhouette_scores = []
    k_range = range(2, 11)  # k=1은 계산 불가 → 2부터 시작
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        # metric은 유클리드 거리(Euclidean)가 기본이지만, 
        # 텍스트 임베딩의 경우 코사인 유사도를 사용해도 좋습니다.
        score = silhouette_score(embeddings, labels, metric='cosine')
        silhouette_scores.append(score)
        print(f"Cluster: {k}, Silhouette Score: {score:.4f}")

    # 실루엣 점수 플롯 생성
    plt.figure()
    plt.plot(k_range, silhouette_scores, marker='o')
    plt.xlabel("Cluster Num (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Analysis")
    plt.savefig("silhouette_analysis.png")
    plt.show()

    # 평균 실루엣 점수가 최대인 k 선택
    optimal_k = k_range[silhouette_scores.index(max(silhouette_scores))]
    print(f"최적의 클러스터 개수: {optimal_k}")
    return optimal_k



# 각 군집의 대표(중요) 뉴스 추출cription', 'published', 'link', 'media']])  # 대표 뉴스 출력

def main():
    data_path = "./data/political_news/political_news_20250409_1145.csv"
    df = load_data(data_path)

    df['text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')


    # Embeddings 모델 로드
    # model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')   # BERT Sentence Embeddings 모델 로드
    model = SentenceTransformer(
        "jinaai/jina-embeddings-v3",
        trust_remote_code=True,
        device="cuda"
        )
    
    # emedding
    embeddings = model.encode(
        df['text'].tolist(),
        task="separation",
        prompt_name="separation",
        show_progress_bar=True
    )  
    print("임베딩 결과 shape:", embeddings.shape)

    # 여기서 실루엣 점수를 활용해 최적의 클러스터 개수 결정
    optimal_clusters = select_optimal_clusters(embeddings)

    # KMeans 클러스터링
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(embeddings)

    # 클러스터링 결과 시각화 (2차원으로 단순화한 경우)
    fig = plt.figure(figsize=(10, 8))
    fig = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=df['cluster'], cmap='viridis', marker='o', s=10)
    plt.title('KMeans Clustering of News Articles')
    plt.xlabel('Embedding Dimension 1')
    plt.ylabel('Embedding Dimension 2')
    plt.colorbar(label='Cluster ID')
    plt.savefig('kmeans_clustering.png')
    plt.show()

    
    # 각 군집의 대표(중요) 뉴스 추출
    main_news_indexes = get_main_news(df, optimal_clusters, embeddings, kmeans) # 각 군집의 대표 뉴스 인덱스
    main_news = df.loc[main_news_indexes]  # 각 군집의 대표 뉴스 데이터프레임
    print("각 군집의 대표 뉴스:")
    print(main_news[['title', 'description', 'published', 'link', 'media']])  # 대표 뉴스 출력


if __name__ == "__main__":
    main()