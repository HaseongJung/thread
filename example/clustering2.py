from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os
from datetime import datetime, timezone, timedelta
from sklearn.metrics.pairwise import cosine_similarity
from dateutil import parser as date_parser
from key_bert import extract_keywords_keybert

def load_bert_model(model_path):
    """
    BERT Sentence Embeddings 모델을 로드합니다.
    
    Args:
        model_path (str): 모델 경로
    
    Returns:
        SentenceTransformer: 로드된 BERT 모델
    """
    return SentenceTransformer(model_path)

def load_news_data(json_path):
    """
    JSON 파일에서 뉴스 데이터를 로드합니다.
    
    Args:
        json_path (str): 뉴스 데이터 JSON 파일 경로
    
    Returns:
        tuple: (news_titles, published_list, continent_map) 또는 None(에러 발생시)
    """
    news_titles = []        # 뉴스 제목 리스트
    published_list = []     # 발행일 리스트 (문자열)
    continent_map = {}      # {뉴스 제목: 대륙명} 매핑
    
    if not os.path.exists(json_path):
        print(f"❌ {json_path} 파일을 찾을 수 없습니다. 뉴스를 불러오지 못했습니다.")
        return None
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 각 대륙의 뉴스 데이터에서 제목과 발행일, 대륙명을 추출
    for continent, cont_data in data.get("continents", {}).items():
        for source in cont_data.get("sources", []):
            for article in source.get("articles", []):
                title = article.get("title", "No Title")
                published = article.get("published", "")
                news_titles.append(title)
                published_list.append(published)
                continent_map[title] = continent
    
    if not news_titles:
        print("❌ 뉴스 데이터가 없습니다. 유사도 분석을 수행할 수 없습니다.")
        return None
        
    return news_titles, published_list, continent_map

def calculate_similarity_matrix(model, news_titles):
    """
    BERT를 사용하여 뉴스 제목 간의 유사도 행렬을 계산합니다.
    
    Args:
        model: BERT 모델
        news_titles (list): 뉴스 제목 목록
    
    Returns:
        numpy.ndarray: 유사도 행렬
    """
    embeddings = model.encode(news_titles)
    return cosine_similarity(embeddings)

def perform_clustering(similarity_matrix, threshold=0.5):
    """
    유사도 행렬을 기반으로 클러스터링을 수행합니다.
    
    Args:
        similarity_matrix (numpy.ndarray): 유사도 행렬
        threshold (float): 클러스터링 임계값 (기본값: 0.5)
    
    Returns:
        tuple: (clusters, processed_indices)
            - clusters: 클러스터 목록
            - processed_indices: 처리된 뉴스 인덱스 집합
    """
    clusters = []
    processed_indices = set()
    
    for i, row in enumerate(similarity_matrix):
        if i in processed_indices:
            continue
        cluster = {i}
        for j, sim in enumerate(row):
            if i != j and sim > threshold and j not in processed_indices:
                cluster.add(j)
        if len(cluster) > 1:
            clusters.append(cluster)
            processed_indices.update(cluster)
            
    return clusters, processed_indices

def categorize_clusters(clusters, news_titles, continent_map):
    """
    클러스터를 대륙 기준으로 글로벌/지역 클러스터로 분류합니다.
    
    Args:
        clusters (list): 클러스터 목록
        news_titles (list): 뉴스 제목 목록
        continent_map (dict): 뉴스 제목별 대륙 정보
    
    Returns:
        tuple: (global_clusters, regional_clusters)
            - global_clusters: 글로벌 클러스터 목록 (2개 이상 대륙 포함)
            - regional_clusters: 지역 클러스터 목록 (단일 대륙)
    """
    global_clusters = []  # 2개 이상의 대륙이 포함된 클러스터 (글로벌 트렌드)
    regional_clusters = []  # 한 대륙만 포함된 클러스터 (지역 트렌드)
    
    for cluster in clusters:
        continents_in_cluster = set()
        for idx in cluster:
            title = news_titles[idx]
            continents_in_cluster.add(continent_map.get(title, "Unknown"))
        if len(continents_in_cluster) >= 2:
            global_clusters.append((cluster, continents_in_cluster))
        else:
            regional_clusters.append((cluster, continents_in_cluster))
            
    return global_clusters, regional_clusters

def extract_leftover_news(news_count, processed_indices, continent_map, news_titles, published_list):
    """
    클러스터에 포함되지 않은 잔여 뉴스를 추출하고 대륙별로 정리합니다.
    
    Args:
        news_count (int): 총 뉴스 개수
        processed_indices (set): 클러스터에 포함된 뉴스 인덱스
        continent_map (dict): 뉴스 제목별 대륙 정보
        news_titles (list): 뉴스 제목 목록
        published_list (list): 발행일 목록
    
    Returns:
        dict: 대륙별 주요 뉴스 인덱스 (최신순 최대 3개)
    """
    all_indices = set(range(news_count))
    leftover_indices = all_indices - processed_indices
    
    # 각 대륙별로 잔여 뉴스 인덱스 그룹화
    leftover_by_continent = {}
    for idx in leftover_indices:
        cont = continent_map.get(news_titles[idx], "Unknown")
        leftover_by_continent.setdefault(cont, []).append(idx)
    
    # 각 대륙별로 최신 뉴스 3개 선택 (발행일 기준 내림차순)
    highlight_by_continent = {}
    for cont, indices in leftover_by_continent.items():
        def parse_date(idx):
            try:
                return date_parser.parse(published_list[idx])
            except Exception:
                return datetime.min
        sorted_indices = sorted(indices, key=parse_date, reverse=True)
        highlight_by_continent[cont] = sorted_indices[:3]
        
    return highlight_by_continent

def select_most_important_news(cluster_news, news_titles, model):
    """
    클러스터에서 가장 중요한 뉴스 1개를 선정합니다.
    BERT 임베딩을 활용하여 클러스터 중심(centroid)과 가장 유사한 뉴스 선택합니다.
    
    Args:
        cluster_news (set): 클러스터에 포함된 뉴스 인덱스 집합
        news_titles (list): 뉴스 제목 목록
        model: BERT 모델
    
    Returns:
        int: 선정된 중요 뉴스의 인덱스
    """
    cluster_news = list(cluster_news)  # set을 리스트로 변환
    
    if len(cluster_news) == 1:
        return cluster_news[0]  # 뉴스가 하나뿐이면 그대로 반환
    
    embeddings = model.encode([news_titles[idx] for idx in cluster_news])
    centroid = np.mean(embeddings, axis=0)
    similarities = np.dot(embeddings, centroid) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(centroid))
    best_idx = np.argmax(similarities)
    
    return cluster_news[best_idx]  # 가장 중심적인 뉴스 반환

def print_global_trends(global_clusters, news_titles, continent_map, model):
    """
    글로벌 트렌드 클러스터 정보를 출력합니다.
    
    Args:
        global_clusters (list): 글로벌 클러스터 목록
        news_titles (list): 뉴스 제목 목록
        continent_map (dict): 뉴스 제목별 대륙 정보
        model: BERT 모델
    """
    print("\nGlobal economic trends (multi-continent clusters)")
    if global_clusters:
        for i, (cluster, conts) in enumerate(global_clusters, start=1):
            print(f"\nGlobal Cluster {i} [Continent: {', '.join(conts)}]:")
            continent_news = {}
            for idx in cluster:
                cont = continent_map.get(news_titles[idx], "Unknown")
                continent_news.setdefault(cont, []).append(idx)

            for cont, indices in continent_news.items():
                important_idx = select_most_important_news(indices, news_titles, model)
                print(f"- {news_titles[important_idx]} [{cont}]")
    else:
        print("글로벌 트렌드 클러스터가 없습니다.")

def print_regional_trends(regional_clusters, news_titles, model):
    """
    지역 트렌드 클러스터 정보를 출력합니다.
    
    Args:
        regional_clusters (list): 지역 클러스터 목록
        news_titles (list): 뉴스 제목 목록
        model: BERT 모델
    """
    print("\nRegionally significant trends (single continent clusters)")
    if regional_clusters:
        for i, (cluster, conts) in enumerate(regional_clusters, start=1):
            cont_label = list(conts)[0] if conts else "Unknown"
            print(f"\nRegional clusters {i} [{cont_label}]:")
            
            # 중요 뉴스 1개 선택
            important_idx = select_most_important_news(cluster, news_titles, model)  
            print(f"- {news_titles[important_idx]} [{cont_label}]")  

            # 중요 뉴스 제외한 나머지 뉴스 제목 모음
            remaining_news = [news_titles[idx] for idx in cluster if idx != important_idx]

            if remaining_news:
                # KeyBERT 기반 키워드 추출 (한 글자 키워드 제거됨)
                keywords = extract_keywords_keybert(" ".join(remaining_news), top_n=5)  
                
                # 키워드 리스트를 쉼표로 연결
                keywords_str = ", ".join(keywords) if keywords else "키워드 없음"
                print(f"Other news keywords: {keywords_str}")
            else:
                print("  🔍 주요 키워드: 없음")

def print_regional_highlights(highlight_by_continent, news_titles, published_list):
    """
    대륙별 주요 뉴스 하이라이트를 출력합니다.
    
    Args:
        highlight_by_continent (dict): 대륙별 주요 뉴스 인덱스
        news_titles (list): 뉴스 제목 목록
        published_list (list): 발행일 목록
    """
    print("\nHighlights by region (unclustered residual news)")
    for cont, indices in highlight_by_continent.items():
        print(f"\n{cont} Highlight:")
        for idx in indices:
            print(f"- {news_titles[idx]} (Publication Date: {published_list[idx]})")

def print_timestamp():
    """
    현재 한국 시간을 포맷에 맞게 출력합니다.
    """
    kst = timezone(timedelta(hours=9))
    kst_time = datetime.now()
    formatted_time = kst_time.strftime('%Y-%m-%d %H:%M UTC')
    print(f"🌍 글로벌 경제 트렌드 분석\n{formatted_time}\n#글로벌경제\n\n")

def main():
    """
    메인 실행 함수: 뉴스 데이터 로드부터 분석 결과 출력까지의 전체 과정을 처리합니다.
    """
    # 1. 모델 및 데이터 로드
    model_path = "/Users/rokkie/Python_Project/Telegram_API/binance/future/News/similarity-search/Model/all-MiniLM-L6-v2"
    json_path = "/Users/rokkie/Python_Project/Telegram_API/binance/future/News/RSS/result.json"
    
    model = load_bert_model(model_path)
    data_result = load_news_data(json_path)
    
    if data_result is None:
        return
    
    news_titles, published_list, continent_map = data_result
    
    # 2. BERT 임베딩 및 유사도 행렬 계산
    similarity_matrix = calculate_similarity_matrix(model, news_titles)
    print("🔍 BERT 기반 유사도 행렬 계산 완료")
    
    # 3. 유사도 기준으로 클러스터링
    clusters, processed_indices = perform_clustering(similarity_matrix, threshold=0.5)
    
    # 4. 클러스터별 분류
    global_clusters, regional_clusters = categorize_clusters(clusters, news_titles, continent_map)
    
    # 5. 잔여 뉴스 추출 및 대륙별 정리
    highlight_by_continent = extract_leftover_news(
        len(news_titles), processed_indices, continent_map, news_titles, published_list
    )
    
    # 6. 결과 출력
    print_global_trends(global_clusters, news_titles, continent_map, model)
    print_regional_trends(regional_clusters, news_titles, model)
    print_regional_highlights(highlight_by_continent, news_titles, published_list)
    print_timestamp()

if __name__ == "__main__":
    main()