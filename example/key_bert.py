from keybert import KeyBERT

kw_model = KeyBERT()

# 키워드 추출 함수
def extract_keywords_keybert(text, top_n=5):
    keywords = kw_model.extract_keywords(
        text, 
        keyphrase_ngram_range=(1, 2),  # 단어 조합(n-gram) 고려
        stop_words='english',  # 기본 불용어 제거
        use_maxsum=True,  # 중복 키워드 방지
        diversity=0.8,  # 키워드 다양성 증가 (기본값 0.5, 추천: 0.7~0.9)
        top_n=top_n
    )

    # 한 글자 키워드 필터링
    keywords_filtered = [kw[0] for kw in keywords if len(kw[0]) > 1]

    return keywords_filtered

if __name__ == "__main__":
    article = """
    ‘Catastrophe’: Why we should ‘sell to Trump’
    Trump says 25% tariffs on EU will be announced soon
    The U.S. imported $1.2 trillion more in goods in 2024 than it exported
    """
    print(extract_keywords_keybert(article))  # ['trade war', 'tariffs', 'Trump policy', 'market impact']