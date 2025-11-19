<h1>💼 Ai-Stock-Chatbot</h1>

<!-- 프로젝트 대표 이미지 -->
<img width="1230" height="581" alt="캡스톤사진" src="https://github.com/user-attachments/assets/d02fba82-e4c4-4a5b-999f-705890f6d01a" />

<br>
<br>

<h2>📘 Notion Page</h2>
<h3><a href="https://sable-mars-102.notion.site/Capstone-1b2cb42f28df80158e69c3b250f5bbbe?pvs=4" target="_blank">Project Notion</a></h3>
<br>


<h2>🧠 Project Overview</h2>

* 본 프로젝트는 AWS Bedrock과 SageMaker를 활용한 AI 기반 주식 예측 시스템

* Yahoo Finance 및 Yahooquery API를 통해 실시간 주식 데이터를 수집하고, SageMaker(XGBoost 모델)로 미래 주가를 예측

* 주가 예측 이외에 일반 평문, 주식 관련 정보(주가, 회사 분석, 뉴스)를 제공
 
* Flask 웹 UI를 통해 결과를 시각화하고, AWS 인프라 상에서 자동화된 배포 및 운영 환경을 구성

<br>

<h2>🧩 Tech Stack</h2>

<h3>🤖 AI / ML</h3>
<ul>
  <li><strong>Bedrock</strong> – <a href="https://www.notion.so/1b2cb42f28df8174a918d2f0bf59a2b0?pvs=21" target="_blank">공식 문서</a></li>
  <li><strong>SageMaker</strong> – <a href="https://www.notion.so/1bbcb42f28df807cbaf8f08658c7cfa5?pvs=21" target="_blank">공식 문서</a></li>
</ul>

<h3>📈 Stock API</h3>
<ul>
  <li><a href="https://www.notion.so/Yahooquery-1b2cb42f28df815e9b9fedf021cbda43?pvs=21" target="_blank">Yahooquery</a></li>
  <li><a href="https://www.notion.so/Yahoo-Finance-1b2cb42f28df812fa77cdd4583d76cdc?pvs=21" target="_blank">Yahoo Finance</a></li>
  <li><a href="https://www.notion.so/FinanceDataReader-1b2cb42f28df81e4b367ce5c64235dd0?pvs=21" target="_blank">FinanceDataReader</a></li>
  <li>mplfinance</li>
</ul>

<h3>🌐 Web UI</h3>
<ul>
  <li>Flask (Python Web Framework)</li>
</ul>

<h3>🗄 Database</h3>
<ul>
  <li>flask_sqlalchemy (ORM for Flask)</li>
</ul>

<h3>🐍 Python Libraries</h3>
<ul>
  <li>GoogleTranslator</li>
  <li>os</li>
  <li>json</li>
  <li>matplotlib</li>
  <li>Plotly</li>
  <li>pandas</li>
</ul>

<h3>🧰 AWS SDK</h3>
<ul>
  <li>boto3 (AWS SDK for Python)</li>
</ul>

<h3>☁️ Infrastructure</h3>
<ul>
  <li><strong>CloudFormation</strong></li>
  <ul>
    <li>VPC</li>
    <li>EC2</li>
    <li>Security Group</li>
    <li>Route53</li>
    <li>IAM Policy</li>
  </ul>
</ul>
