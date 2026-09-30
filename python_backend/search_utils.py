import re
from typing import List

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as",
    "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can",
    "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further",
    "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other", "our", "ours",
    "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such", "than", "that",
    "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "would", "you", "your", "yours", "yourself",
    "yourselves",
    "та", "і", "в", "у", "на", "з", "за", "до", "по", "про", "для", "від", "без", "або", "але", "як", "що",
    "це", "цей", "ця", "ці", "ти", "ми", "ви", "він", "вона", "вони", "воно", "так", "не", "є", "й", "лише",
    "тобто", "коли", "якщо", "тоді", "чи", "можна", "могти", "буде", "був", "була", "було", "були", "бувши",
    "під", "над", "при", "через", "після", "перед", "між", "серед", "кожен", "будь", "деякий", "деякі"
}

TECHNICAL_SHORT_TOKENS = {
    "ai", "ml", "ui", "ux", "api", "ocr", "rag", "llm", "nlp", "gpt", "sql", "etl", "pdf",
    "csv", "json", "xml", "yaml", "url", "cpu", "gpu", "gpu", "cdn", "io", "db", "dbs"
}


def clean_markdown(text: str) -> str:
    text = re.sub(r'```.*?```', ' ', text, flags=re.DOTALL)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'!\[.*?\]\(.*?\)', ' ', text)
    text = re.sub(r'\|.*?\|', ' ', text)
    text = re.sub(r'[-*]{3,}', ' ', text)
    text = re.sub(r'#+\s+', '', text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    text = re.sub(r'[*_~`>]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    cleaned = clean_markdown(str(text))
    tokens = re.findall(r'(?u)[A-Za-zА-Яа-яЁё0-9]+(?:[-_][A-Za-zА-Яа-яЁё0-9]+)*', cleaned.lower())
    result = []
    for token in tokens:
        if token in STOPWORDS:
            continue
        if len(token) <= 2 and token not in TECHNICAL_SHORT_TOKENS:
            continue
        result.append(token)
    return result


def extract_keywords(text: str, limit: int = 12) -> List[str]:
    tokens = tokenize(text)
    seen = set()
    keywords = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        keywords.append(token)
        if len(keywords) >= limit:
            break
    return keywords


def keyword_overlap_score(query: str, text: str) -> float:
    query_terms = set(extract_keywords(query))
    if not query_terms:
        return 0.0
    doc_terms = set(extract_keywords(text))
    if not doc_terms:
        return 0.0
    overlap = sum(1 for term in query_terms if term in doc_terms)
    return overlap / len(query_terms)
