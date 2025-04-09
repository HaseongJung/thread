from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os
import pandas as pd
from datetime import datetime, timezone, timedelta
from sklearn.metrics.pairwise import cosine_similarity
from dateutil import parser as date_parser  # 날짜 파싱용
from key_bert import extract_keywords_keybert  # KeyBERT 함수 import

# 🔹 BERT Sentence Embeddings 모델 로드
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

news_data_path = "./data/political_rss.csv"
news_data = pd.read_csv(news_data_path, encoding='utf-8')

# 결과 데이터 및 뉴스 목록 초기화
news_titles = []         # 뉴스 제목 리스트
published_list = []      # 발행일 리스트 (문자열)
continent_map = {}       # {뉴스 제목: 대륙명} 매핑


# 2. BERT 임베딩 및 유사도 행렬 계산
embeddings = model.encode(news_titles)
similarity_matrix = cosine_similarity(embeddings)
print("🔍 BERT 기반 유사도 행렬:")

# 3. 유사도 기준으로 클러스터링 (임계값: 0.5)
threshold = 0.5
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

# 4. 클러스터별 분류 (다중 대륙 클러스터 vs 단일 대륙 클러스터)
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

# 5. 잔여(클러스터에 속하지 않은) 뉴스 추출 및 대륙별 정리
all_indices = set(range(len(news_titles)))
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

# 6. 중요 뉴스 선정 함수 (BERT 기반)
def select_most_important_news(cluster_news):
    """
    클러스터에서 가장 중요한 뉴스 1개를 선정하는 함수
    - 방법: BERT 임베딩을 활용하여 클러스터 중심(centroid)과 가장 유사한 뉴스 선택
    """
    cluster_news = list(cluster_news)  # 🔹 set을 리스트로 변환 (중요)
    
    if len(cluster_news) == 1:
        return cluster_news[0]  # 뉴스가 하나뿐이면 그대로 반환
    
    embeddings = model.encode([news_titles[idx] for idx in cluster_news])
    centroid = np.mean(embeddings, axis=0)
    similarities = np.dot(embeddings, centroid) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(centroid))
    best_idx = np.argmax(similarities)
    
    return cluster_news[best_idx]  # 가장 중심적인 뉴스 반환

# 7. 결과 출력
print("\nGlobal economic trends (multi-continent clusters)")
if global_clusters:
    for i, (cluster, conts) in enumerate(global_clusters, start=1):
        print(f"\nGlobal Cluster {i} [Continent: {', '.join(conts)}]:")
        continent_news = {}
        for idx in cluster:
            cont = continent_map.get(news_titles[idx], "Unknown")
            continent_news.setdefault(cont, []).append(idx)

        for cont, indices in continent_news.items():
            important_idx = select_most_important_news(indices)
            print(f"- {news_titles[important_idx]} [{cont}]")
else:
    print("글로벌 트렌드 클러스터가 없습니다.")

print("\nRegionally significant trends (single continent clusters)")
if regional_clusters:
    for i, (cluster, conts) in enumerate(regional_clusters, start=1):
        cont_label = list(conts)[0] if conts else "Unknown"
        print(f"\nRegional clusters {i} [{cont_label}]:")
        
        # 🔹 중요 뉴스 1개 선택
        important_idx = select_most_important_news(cluster)  
        print(f"- {news_titles[important_idx]} [{cont_label}]")  

        # 🔹 중요 뉴스 제외한 나머지 뉴스 제목 모음
        remaining_news = [news_titles[idx] for idx in cluster if idx != important_idx]

        if remaining_news:
            # 🔹 KeyBERT 기반 키워드 추출 (한 글자 키워드 제거됨)
            keywords = extract_keywords_keybert(" ".join(remaining_news), top_n=5)  
            
            # 📌 키워드 리스트를 쉼표로 연결
            keywords_str = ", ".join(keywords) if keywords else "키워드 없음"
            print(f"Other news keywords: {keywords_str}")
        else:
            print("  🔍 주요 키워드: 없음")

print("\nHighlights by region (unclustered residual news)")
for cont, indices in highlight_by_continent.items():
    print(f"\n{cont} Highlight:")
    for idx in indices:
        print(f"- {news_titles[idx]} (Publication Date: {published_list[idx]})")

# 한국 시간 (KST, UTC+9) 정의
kst = timezone(timedelta(hours=9))

# KST 기준 날짜 생성
kst_time = datetime.now()

# 원하는 형식으로 변환하여 출력
formatted_time = kst_time.strftime('%Y-%m-%d %H:%M UTC')
print(f"🌍 글로벌 경제 트렌드 분석\n{formatted_time}\n#글로벌경제\n\n")