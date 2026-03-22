# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-Stock-Chatbot is a Korean-language stock analysis chatbot powered by AWS Bedrock (Claude 3.5 Sonnet), AWS SageMaker (XGBoost), and real-time financial data APIs. It runs as a Flask web app deployed on AWS EC2.

## Running the App

```bash
# Activate virtual environment (Windows)
source chatbot_code/chatbot/Scripts/activate

# Run Flask app (from chatbot_code/)
cd chatbot_code && python app.py
# → http://localhost:5000
```

Environment variables are loaded from `chatbot_code/.env`:
- `BEDROCK_INFERENCE_PROFILE_ARN` — Bedrock inference profile for Claude 3.5 Sonnet
- `NEWS_API_KEY` — News API key

## Architecture

### Request Flow

```
User input (Korean)
  → POST /chat
  → Keyword routing (주가/가격, 경제뉴스, 주가예측, 재무제표)
  → Handler fetches data (yfinance / web scrape / SageMaker)
  → AWS Bedrock Claude generates natural language response
  → JSON back to frontend
```

### Key Modules (`chatbot_code/func/`)

| File | Purpose |
|------|---------|
| `stock_price.py` | Resolve Korean company names → ticker symbols, fetch live prices |
| `sp_predict.py` | Train XGBoost on SageMaker and predict next-day close price |
| `f_statement.py` | Pull financial statements via yfinance, compute ratios |
| `news.py` | Scrape economic news from sedaily.com |
| `web/news.py` | Naver Finance top headlines |
| `web/asset_price.py` | Gold, silver, oil, bitcoin, USD/KRW prices |

### Chatbot Handlers (in `app.py`)

- `chatbot_response()` — General Q&A via Bedrock (200 char max)
- `chatbot_response2()` — Live stock price + AI commentary (triggers: 주가, 가격, 주식가격)
- `chatbot_response3()` — Economic news summary (triggers: 경제뉴스, 금일 뉴스)
- `chatbot_response4()` — SageMaker price prediction (triggers: 주가예측) — takes ~15 min
- `chatbot_response5()` — Financial statement analysis (triggers: 재무제표)
- `chatbot_response6()` — Usage guide

### AWS Services

| Service | Usage |
|---------|-------|
| Bedrock | Claude 3.5 Sonnet (`apac.anthropic.claude-3-5-sonnet-20241022-v2:0`) in ap-northeast-2 |
| SageMaker | XGBoost training + endpoint for stock prediction |
| S3 | Training data storage (`chatbot-sagemaker-s3` bucket) |
| EC2 | t3.micro (Ubuntu) hosting the Flask app |
| CloudFormation | Infrastructure in `cloudformation/` — deploy via `master.yaml` |

### SageMaker Prediction Pipeline (`sp_predict.py`)

1. Fetch 6-month history via yfinance
2. Engineer features: MA9, MA12, MA26, Vol_MA5, Pct_change
3. 80/20 train-test split → upload CSVs to S3
4. Train XGBoost estimator (ml.m5.xlarge, ~15 min)
5. Deploy endpoint → invoke with today's indicators → return predicted price

### CloudFormation Stack Order

`master.yaml` deploys nested stacks in dependency order:
VPC → Security Group + IAM Policy → EC2 → Route53

## Korean Language Handling

Company names from user input are automatically translated Korean→English via `deep-translator` (GoogleTranslator) before querying Yahoo Finance APIs.
