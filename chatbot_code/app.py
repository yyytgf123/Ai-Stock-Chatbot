from flask import Flask, request, jsonify, render_template, redirect, url_for, send_file
import os
from dotenv import load_dotenv
import boto3
import json
import re
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


# ============================================================
#  Bedrock 공통 호출 함수
# ============================================================
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


# ============================================================
#  LLM 기반 의도 분류 + 개체 추출 (키워드 매칭 대체)
# ============================================================
INTENT_PROMPT_TEMPLATE = """당신은 주식 챗봇의 의도 분류기입니다.
사용자 입력을 분석하여 아래 JSON 형식으로만 응답하세요.
JSON 외 다른 텍스트는 절대 포함하지 마세요.

## 의도(intent) 종류
- stock_prediction : 주가 예측, 내일/다음주 전망, 오를까/내릴까, 살만할까, 어떨까
- stock_price     : 현재가 조회, 시세, 주가 확인, 얼마야
- news            : 경제 뉴스, 최신 소식, 오늘 뉴스, 이슈
- financial       : 재무제표, 실적, 매출, 영업이익, PER, PBR, 재무 분석
- user_manual     : 사용법, 기능, 설명서, 도움말, 가이드, 어떻게 사용
- general         : 위에 해당하지 않는 일반 질문/인사/잡담

## 응답 형식 (반드시 이 JSON 구조만 출력)
{{"intent": "intent종류", "company": "회사명 또는 null", "period": "기간 또는 null", "detail": "추가 키워드 또는 null"}}

## 분류 예시
입력: "삼성전자 내일 오를까?"         → {{"intent": "stock_prediction", "company": "삼성전자", "period": null, "detail": null}}
입력: "테슬라 주가 알려줘"            → {{"intent": "stock_price", "company": "테슬라", "period": null, "detail": null}}
입력: "애플 3개월 시세"               → {{"intent": "stock_price", "company": "애플", "period": "3개월", "detail": null}}
입력: "오늘 경제 뉴스 뭐 있어?"       → {{"intent": "news", "company": null, "period": null, "detail": null}}
입력: "반도체 관련 뉴스"              → {{"intent": "news", "company": null, "period": null, "detail": "반도체"}}
입력: "삼성 재무제표 분석해줘"        → {{"intent": "financial", "company": "삼성", "period": null, "detail": null}}
입력: "뭐 할 수 있어?"               → {{"intent": "user_manual", "company": null, "period": null, "detail": null}}
입력: "안녕 주식이 뭐야?"            → {{"intent": "general", "company": null, "period": null, "detail": null}}
입력: "SK하이닉스 살만해?"           → {{"intent": "stock_prediction", "company": "SK하이닉스", "period": null, "detail": null}}
입력: "카카오 어떻게 될 것 같아?"     → {{"intent": "stock_prediction", "company": "카카오", "period": null, "detail": null}}

사용자 입력: "{user_input}"
"""


def parse_json_safe(text):
    """LLM 응답에서 JSON 부분만 안전하게 추출"""
    # 1차: 중괄호 블록 추출
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # 2차: 코드블록 안의 JSON
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    return None


def classify_intent(user_input):
    """Claude를 이용한 의도 분류 + 개체 추출
    
    Returns:
        dict: {"intent": str, "company": str|None, "period": str|None, "detail": str|None}
    """
    prompt = INTENT_PROMPT_TEMPLATE.format(user_input=user_input)

    # 라우팅 전용 호출: FORMAT_INSTRUCTION 제외, temperature=0 (일관된 분류)
    response = bedrock_client.invoke_model(
        modelId=inferenceProfileArn,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 150,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }
            ]
        }),
    )
    raw = json.loads(response["body"].read())["content"][0]["text"].strip()
    print(f"[INTENT] raw response: {raw}")

    result = parse_json_safe(raw)
    if result and "intent" in result:
        # intent 유효성 검증
        valid_intents = {"stock_prediction", "stock_price", "news", "financial", "user_manual", "general"}
        if result["intent"] not in valid_intents:
            result["intent"] = "general"
        return result

    # 파싱 실패 시 폴백
    print(f"[INTENT] JSON 파싱 실패, general로 폴백")
    return {"intent": "general", "company": None, "period": None, "detail": None}


# ============================================================
#  핸들러 함수 (기존 chatbot_response 1~6 리팩토링)
# ============================================================

### 일반 대화 ###
def handle_general(user_input):
    prompt = (
        "너는 주식/금융 전문 AI 비서야. 질문에 대해 친절하고 유익한 답변을 해줘.\n"
        "200자 이내로 간결하게 답변해.\n"
        f"질문: {user_input}"
    )
    return invoke_bedrock(prompt, max_tokens=200)


### 주식 가격 조회 ###
def handle_stock_price(user_input, company=None):
    search_input = company if company else user_input
    company_name = find_company_symbol(search_input)
    symbol = get_stock_symbol(company_name)

    stock_info = None
    if symbol:
        stock_price = get_stock_price(symbol)
        currency = get_currency(symbol)
        stock_info = f"**{company_name}** ({symbol})의 현재 주가는 **{stock_price} {currency}** 입니다."

    if stock_info:
        return stock_info

    prompt = (
        "너는 주식 전문 AI 비서야.\n"
        "주식 가격 정보를 포함해서 200자 이내로 답변해.\n"
        f"질문: {user_input}\n"
    )
    return invoke_bedrock(prompt, max_tokens=200)


### 경제 뉴스 ###
from func.news import get_news

def handle_news(user_input, detail=None):
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
        print(f"[ERROR] handle_news: {type(e).__name__}: {e}")
        return "뉴스 정보를 가져올 수 없습니다."


### 주가 예측 ###
def handle_prediction(user_input, company=None):
    search_input = company if company else user_input

    predict_sp = predict_stock_price(search_input)
    if not predict_sp:
        return "종목 정보를 찾지 못했습니다."

    company_name = predict_sp["stock_name"]
    symbol = predict_sp["symbol"]
    last_close = predict_sp["last_close"]
    predicted_price = predict_sp["predicted_price"]
    predicted_return = predict_sp["predicted_return"] * 100
    direction = predict_sp["direction"]
    signal = predict_sp["signal"]
    direction_acc = predict_sp["cv_summary"]["Direction_Accuracy"]["mean"] * 100
    mae = predict_sp["cv_summary"]["MAE"]["mean"] * 100

    prompt = (
        "너는 주식 분석 AI 비서야.\n"
        "아래 예측 결과를 바탕으로 500자 이내로 자연스럽고 간결하게 답변해.\n\n"
        f"- 종목명: {company_name}\n"
        f"- 종목 코드: {symbol}\n"
        f"- 현재가: {last_close:,.0f}원\n"
        f"- 예측가: {predicted_price:,.0f}원\n"
        f"- 예상 등락률: {predicted_return:.2f}%\n"
        f"- 방향: {direction}\n"
        f"- 시그널: {signal}\n"
        f"- 방향성 정확도: {direction_acc:.2f}%\n"
        f"- 평균 오차(MAE): {mae:.2f}%p\n\n"
        "출력 형식:\n"
        f"1. 첫 줄은 반드시 '내일 예측 가격: **{predicted_price:,.0f}원**'으로 시작\n"
        f"2. 다음 줄에서 현재가 대비 예상 등락률과 방향({direction})을 짧게 설명\n"
        "3. 분석 방법은 XGBoost, 기술적 지표(RSI, MACD, MA, 볼린저밴드), 뉴스 감성을 활용했다고 설명\n"
        f"4. {company_name}({symbol}) 전망을 한두 문장으로 설명\n"
        f"5. 시그널은 '{signal}'이라고 반영\n"
        "6. 마지막에 '본 예측은 참고용이며, 투자 판단은 본인 책임입니다.'를 추가\n"
    )

    return invoke_bedrock(prompt, max_tokens=500)


### 재무제표 ###
import func.f_statement as fs

def handle_financial(user_input, company=None):
    search_input = company if company else user_input
    data = fs.find_f_statement(search_input)

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


### 사용 가이드 ###
def handle_user_manual():
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
        "- 예시: 삼성 주가예측 해줘, SK하이닉스 내일 어떨까?\n"
        "- 참고: 예측 모델 실행에 수 초 소요\n"
    )
    return guide


# ============================================================
#  메인 라우터 (LLM 의도 분류 기반)
# ============================================================
def route_request(user_input):
    """LLM 의도 분류 결과를 기반으로 핸들러 분기"""
    classified = classify_intent(user_input)

    intent = classified["intent"]
    company = classified.get("company")
    detail = classified.get("detail")

    print(f"[ROUTER] intent={intent}, company={company}, detail={detail}")

    if intent == "stock_prediction":
        return handle_prediction(user_input, company)
    elif intent == "stock_price":
        return handle_stock_price(user_input, company)
    elif intent == "news":
        return handle_news(user_input, detail)
    elif intent == "financial":
        return handle_financial(user_input, company)
    elif intent == "user_manual":
        return handle_user_manual()
    else:
        return handle_general(user_input)


# ============================================================
#  Flask 엔드포인트
# ============================================================
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"error": "메시지가 없습니다."}), 400

    user_input = data["message"].strip()

    if not user_input:
        return jsonify({"error": "빈 메시지입니다."}), 400

    response = route_request(user_input)
    return jsonify({"response": response})


# ============================================================
#  홈페이지 관련 엔드포인트 (기존 유지)
# ============================================================

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