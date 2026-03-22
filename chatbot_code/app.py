from flask import Flask, request, jsonify, render_template, redirect, url_for, send_file
import os
from dotenv import load_dotenv
import boto3
import json
from func.stock_price import get_currency, get_stock_price, get_stock_symbol, find_company_symbol
import yfinance as yf
import pandas as pd
import time
from func.sp_predict import predict_stock_price
import requests

# 세션 생성
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"
})

### bedrock setting ###
load_dotenv()
inferenceProfileArn = os.getenv("BEDROCK_INFERENCE_PROFILE_ARN")
app = Flask(__name__)
bedrock_client = boto3.client("bedrock-runtime", region_name="ap-northeast-2")

# 공통 시스템 지시
FORMAT_INSTRUCTION = (
    "\n\n[응답 형식 규칙]\n"
    "- 마크다운 형식으로 답변해. **굵은글씨**, - 리스트 등을 적절히 활용해.\n"
    "- 이모지는 사용하지 마.\n"
    "- 깔끔하고 읽기 쉽게 구조화해서 답변해.\n"
)


### Bedrock 공통 호출 함수 ###
def invoke_bedrock(prompt, max_tokens=200, temperature=0.7):
    """Bedrock 모델 호출 공통 함수"""
    prompt = prompt + FORMAT_INSTRUCTION

    response = bedrock_client.invoke_model(
        modelId=inferenceProfileArn,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }
            ]
        }),
    )
    return json.loads(response["body"].read())["content"][0]["text"].strip()


### 일반 평문 대답 ###
def chatbot_response(user_input):
    prompt = (
        "너는 주식/금융 전문 AI 비서야. 질문에 대해 친절하고 유익한 답변을 해줘.\n"
        "200자 이내로 간결하게 답변해.\n"
        f"질문: {user_input}"
    )
    return invoke_bedrock(prompt, max_tokens=200)


### 주식 가격 출력 ###
def chatbot_response2(user_input):
    company_name = find_company_symbol(user_input)
    symbol = get_stock_symbol(company_name)

    stock_info = None
    if symbol:
        stock_price = get_stock_price(symbol)
        currency = get_currency(symbol)
        stock_info = f"**{company_name}** ({symbol})의 현재 주가는 **{stock_price} {currency}** 입니다."

    prompt = (
        "너는 주식 전문 AI 비서야.\n"
        "주식 가격 정보를 포함해서 200자 이내로 답변해.\n"
        f"질문: {user_input}\n"
        f"참고 데이터: {company_name}({symbol})의 주가는 {stock_price} {currency}"
    )
    return stock_info if stock_info else invoke_bedrock(prompt, max_tokens=200)

### 경제 뉴스 출력 ###
from func.news import get_news

def chatbot_response3(user_input):
    try:
        articles = get_news()
        print(f"[DEBUG] type={type(articles)}, value={articles[:2]}")
        
        news_text = ""
        for i, article in enumerate(articles, 1):
            news_text += f"{i}. [{article['source']}] {article['title']}\n"

        prompt = (
            "너는 경제 뉴스 전문 AI 비서야.\n"
            "아래 뉴스 데이터를 바탕으로 오늘의 주요 경제 뉴스를 요약해줘.\n"
            "각 뉴스는 **제목**과 한줄 요약으로 정리해.\n"
            "500자 이내로 답변해.\n"
            f"질문: {user_input}\n"
            f"뉴스 데이터:\n{news_text}"
        )
        return invoke_bedrock(prompt, max_tokens=500)
    except Exception as e:
        print(f"[ERROR] chatbot_response3: {type(e).__name__}: {e}")
        return "뉴스 정보를 가져올 수 없습니다."


### 주가 예측 ###
def chatbot_response4(user_input):
    company_name = find_company_symbol(user_input)
    symbol = get_stock_symbol(company_name)

    predict_sp = predict_stock_price(user_input)
    print(predict_sp)

    prompt = (
        "너는 주식 분석 AI 비서야.\n"
        "500자 이내로 아래 형식에 맞춰 답변해.\n\n"
        f"1. 첫 줄: '내일 예측 가격: **{predict_sp:,}원**' 형태로 출력\n"
        f"2. 분석 방법: XGBoost 모델과 기술적 지표(RSI, MACD, MA, 볼린저밴드)를 사용했다고 간단히 설명\n"
        f"3. {symbol} 회사 전망을 간단히 설명\n"
        f"4. 마지막에 '본 예측은 참고용이며, 투자 판단은 본인 책임입니다.'라는 면책 문구 추가\n"
    )
    return invoke_bedrock(prompt, max_tokens=500)


### 재무제표 ###
import func.f_statement as fs

def chatbot_response5(user_input):
    data = fs.find_f_statement(user_input)

    prompt = (
        "너는 재무 분석 전문 AI 비서야.\n"
        "500자 이내로 답변해.\n"
        f"아래 재무제표 데이터를 분석해서 해당 회사의 재무 상태를 평가해줘.\n\n"
        f"데이터: {data}\n"
        f"(순서: 영업이익률, 순이익률, 부채비율, ROE, ROA)\n\n"
        f"각 지표별로 간단한 평가와 종합 의견을 제시해줘.\n"
        f"질문: {user_input}"
    )
    return invoke_bedrock(prompt, max_tokens=500)


### 설명서 출력 ###
def chatbot_response6(user_input):
    # 설명서는 모델 호출 없이 직접 반환 (불필요한 API 호출 절약)
    guide = (
        "### 주식 챗봇 가이드\n\n"
        "**1. 일반 질문**\n"
        "- 예시: 안녕, 주식이 뭐야?, 상장된 회사 설명해줘\n\n"
        "**2. 주가 정보**\n"
        "- 예시: 삼성 주가 알려줘, 애플 주식가격 제공해줘\n\n"
        "**3. 최신 경제 뉴스 (금일)**\n"
        "- 예시: 경제 뉴스 알려줘, 금일 뉴스 알려줘\n\n"
        "**4. 재무제표 분석**\n"
        "- 예시: 삼성 재무제표 분석해줘, 테슬라 재무제표 분석해줘\n\n"
        "**5. 주가 예측 (내일)**\n"
        "- 예시: 삼성 주가예측 해줘\n"
        "- 참고: 예측 모델 실행에 수 초 소요\n"
    )
    return guide


#### Flask 엔드포인트 ####
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"error": "메시지가 없습니다."}), 400

    user_input = data["message"].strip()

    if not user_input:
        return jsonify({"error": "빈 메시지입니다."}), 400

    sp_predict_keywords = ["주가예측", "주가 예측"]
    stock_price_keywords = ["주가", "가격", "주식가격", "주식 가격", "현재가"]
    news_keywords = ["경제뉴스", "경제 뉴스", "최신 경제", "최신뉴스", "오늘 뉴스", "금일 뉴스"]
    f_statement_keywords = ["재무제표", "재무 제표"]
    user_manual_keywords = ["기능", "사용서", "설명서", "사용법", "어떻게 사용", "도움말", "방법", "가이드"]

    if any(keyword in user_input for keyword in sp_predict_keywords):
        response = chatbot_response4(user_input)
    elif any(keyword in user_input for keyword in stock_price_keywords):
        response = chatbot_response2(user_input)
    elif any(keyword in user_input for keyword in news_keywords):
        response = chatbot_response3(user_input)
    elif any(keyword in user_input for keyword in f_statement_keywords):
        response = chatbot_response5(user_input)
    elif any(keyword in user_input for keyword in user_manual_keywords):
        response = chatbot_response6(user_input)
    else:
        response = chatbot_response(user_input)

    return jsonify({"response": response})


## ------ homepage ------ ##

### 주가 그래프 ###
def get_stock_data(symbol="AAPL", period="1d", interval="1m"):
    try:
        data = yf.download(symbol, period=period, interval=interval)

        if data.empty:
            return {"error": f"'{symbol}'에 대한 데이터를 찾을 수 없습니다."}

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        data = data.reset_index()

        date_col = "Datetime" if "Datetime" in data.columns else "Date"
        data["Date"] = pd.to_datetime(data[date_col])
        if data["Date"].dt.tz is None:
            data["Date"] = data["Date"].dt.tz_localize("UTC")

        data["Date"] = data["Date"].dt.tz_convert("Asia/Seoul")
        data["Date"] = data["Date"].dt.strftime("%Y-%m-%d %H:%M")

        volumes = data["Volume"].fillna(0).astype(int).tolist()

        return {
            "dates": data["Date"].tolist(),
            "prices": data["Close"].tolist(),
            "opens": data["Open"].tolist(),
            "highs": data["High"].tolist(),
            "lows": data["Low"].tolist(),
            "closes": data["Close"].tolist(),
            "volumes": volumes
        }

    except Exception as e:
        return {"error": str(e)}


### 한줄 뉴스 ###
from func.web.news import get_economic_news

@app.route("/get_news", methods=["GET"])
def get_news_route():
    news = get_economic_news()
    return jsonify(news)


### 주요 자산 page ###
from func.web.asset_price import get_asset_prices

@app.route("/get_asset_prices", methods=["GET"])
def asset_prices():
    prices = get_asset_prices()
    return jsonify(prices)


### 차트 그래프 page ###
@app.route("/get_stock_data", methods=["GET"])
def stock_data():
    symbol = request.args.get("symbol", "AAPL").upper()
    period = request.args.get("period", "1d")
    interval = request.args.get("interval", "1m")
    stock_data = get_stock_data(symbol, period, interval)
    return jsonify(stock_data)


### 차트 그래프 symbol 변환 page ###
@app.route("/resolve_symbol", methods=["GET"])
def resolve_symbol():
    name = request.args.get("name", "")
    symbol = find_company_symbol(name)
    company_name = ""
    if symbol:
        try:
            stock = yf.Ticker(symbol)
            company_name = stock.info.get("longName", "")
        except Exception as e:
            print("회사명 조회 실패:", e)

    return jsonify({
        "symbol": symbol or name,
        "name": company_name or name
    })


### chatbot ###
@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    time.sleep(10)
    app.run(host="0.0.0.0", port=5000, debug=True)