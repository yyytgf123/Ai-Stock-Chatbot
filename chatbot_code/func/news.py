import requests
from bs4 import BeautifulSoup

def get_news(keyword="경제"):
    url = f"https://news.google.com/rss/search?q={keyword}&hl=ko&gl=KR&ceid=KR:ko"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(resp.text, "xml")

    articles = []
    for item in soup.find_all("item")[:10]:
        articles.append({
            "title": item.title.text,
            "link": item.link.text,
            "source": item.source.text if item.source else "",
            "date": item.pubDate.text
        })
    return articles

