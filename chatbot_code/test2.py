# debug_crawl2.py
import requests
from bs4 import BeautifulSoup

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

url = "https://search.naver.com/search.naver"
params = {
    "where": "news",
    "query": "삼성전자",
    "sort": 1,
    "ds": "2026.03.15",
    "de": "2026.03.21",
    "nso": "so:dd,p:from20260315to20260321",
    "start": 1,
}

resp = requests.get(url, params=params, headers=headers, timeout=10)
soup = BeautifulSoup(resp.text, 'html.parser')

# .group_news a 중에서 실제 뉴스 제목 찾기
links = soup.select('.group_news a')
for a in links[:30]:
    text = a.get_text(strip=True)
    href = a.get('href', '')
    classes = a.get('class', [])
    parent_classes = a.parent.get('class', []) if a.parent else []
    
    if len(text) > 10:  # 짧은 건 제목 아님
        print(f"class={classes} | parent={parent_classes} | href={href[:50]}")
        print(f"  → {text[:80]}")
        print()