<img src="logo.png" alt="PlayStation Support Agent" width="120"/>

# 🎮 PlayStation Support Agent

### Grounded AI Customer Support · Hiver SDE Intern Take-Home

> An auditable AI support agent that classifies customer issues, retrieves
> historical evidence, generates grounded replies, validates response safety,
> and decides whether a case should be auto-handled or escalated.

[🚀 Live Demo](https://playstation-support-agent.streamlit.app/) ·
[📖 Project Report](REPORT.md) ·
[🧪 Review Guide](REVIEW_GUIDE.md)

---

## ✨ Overview

**PlayStation Support Agent** is an end-to-end AI customer-support system built
around the public **AskPlayStation** support data from Kaggle's
**Customer Support on Twitter** dataset.

The system is designed around one principle:

> **Ground before generating.**

Instead of allowing an LLM to freely answer a customer, the pipeline first
classifies the issue, retrieves relevant historical support evidence, generates
a grounded response, checks that response for safety, and finally determines
whether the case can be handled automatically or should be escalated.

The project is intentionally auditable: every major stage is independently
testable and evaluated, and known limitations are explicitly documented.

---

## 🚀 Live Demo

### [▶ Try the deployed Streamlit app](https://playstation-support-agent.streamlit.app/)

The application provides three views:

| View                        | Purpose                                                          |
| --------------------------- | ---------------------------------------------------------------- |
| 💬 **Try the Agent**        | Run customer queries through the complete pipeline               |
| 📊 **Evaluation Dashboard** | Inspect classification, retrieval, safety and escalation results |
| 🗺️ **What's Next**          | Review completed work, limitations and planned improvements      |

---

## 🧠 How It Works

```text
                     ┌─────────────────────┐
                     │   Customer Message  │
                     └──────────┬──────────┘
                                │
                                ▼
              ┌────────────────────────────┐
              │ Intent Classification      │
              │ TF-IDF + Logistic Reg.     │
              └────────────┬───────────────┘
                           │
                           ▼
              ┌────────────────────────────┐
              │ Evidence Retrieval         │
              │ TF-IDF + Intent-aware      │
              └────────────┬───────────────┘
                           │
                           ▼
              ┌────────────────────────────┐
              │ Grounded Reply Generation  │
              │ Mock / Live LLM            │
              └────────────┬───────────────┘
                           │
                           ▼
              ┌────────────────────────────┐
              │ Response Safety            │
              │ Fail-closed checks         │
              └────────────┬───────────────┘
                           │
                           ▼
              ┌────────────────────────────┐
              │ Escalation Engine          │
              │ Explicit reason codes      │
              └────────────┬───────────────┘
                           │
                    ┌──────┴──────┐
                    ▼             ▼
                AUTO-HANDLE    ESCALATE

📊 Evaluation Snapshot
Metric	Result
Intent Accuracy	76.1%
Intent Macro F1	0.760
Retrieval Recall@1	53.2%
Retrieval Recall@3	79.8%
Escalation Accuracy	70.0%
Escalation F1	0.55
Golden Evaluation Set	188 examples
Intent Categories	9
Escalation Tests	12/12 passing


🔍 Intent Classification
The agent uses a TF-IDF + Logistic Regression classifier to route incoming
customer messages into one of 9 support intents.
The classifier was trained on weak/silver labels generated from the source
dataset. The evaluation set is kept separate from training and retrieval.
Supported Intents
Intent	Description
account_access	Login, account access and authentication issues
billing_payment	Charges, payments and billing-related issues
console_hardware	Console hardware and device problems
game_software	Game installation, launch and software issues
network_connectivity	Internet, connection and online-service problems
subscription_services	PlayStation Plus and subscription-related issues
refund_cancellation	Refund, cancellation and purchase-reversal requests
security_compromise	Account security and suspicious-access concerns
out_of_scope	Requests outside the supported support domain


Classifier Evaluation
- Accuracy: 76.1%
- Macro F1: 0.760
- Training labels: weak/silver labels
- Evaluation: stratified held-out evaluation set
- Human validation: not yet completed
The classifier numbers should be interpreted as an engineering baseline,
because the training labels are not manually verified ground truth.

🔎 Retrieval
After classification, the agent retrieves relevant historical support
examples using TF-IDF similarity.
The retrieval stage is intentionally separated from the classifier so that
the generated response can be grounded in actual historical support evidence.
Retrieval Flow
Customer Query
      │
      ▼
Predicted Intent
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Similarity Search
      │
      ▼
Top-K Historical Examples
      │
      ▼
Relevant Evidence

<table>
<tr>
<td width="50%" valign="top">

🤖 Grounded Reply
Retrieved support evidence is passed to the response generator.
Generation rules
- Stay grounded in retrieved evidence
- No unsupported policies, refunds or guarantees
- Don't expose internal system details
- Don't repeat already-tried fixes
- Ask for clarification when evidence is weak
- Keep responses concise
Evidence
   ↓
Prompt
   ↓
LLM
   ↓
Response

Modes: Mock / Template · Live LLM
If live generation is unavailable, the system uses a safe deterministic
fallback instead of fabricating an answer.
</td>

<td width="50%" valign="top">

🛡️ Response Safety
Every generated response passes through safety checks.
Checks include
- Unsupported claims
- Weak or missing evidence
- Risky responses
- Sensitive-request handling
- Cases requiring escalation
Response
   ↓
Safety Check
 ┌─┴─┐
PASS FAIL
 │    │
 ↓    ↓
Next Fallback

45 safety evaluation examples
</td>
</tr>
</table>

<table>
<tr>
<td width="50%" valign="top">

🚨 Escalation Engine
The final decision determines whether a ticket can be auto-handled or needs
human review.
Auto-Handle	Escalate
High confidence	Low confidence
Useful evidence	Insufficient evidence
Safety passed	Safety failure
Supported request	Sensitive / out-of-scope


Every escalation includes explicit reason codes.
Metric	Score
Accuracy	70.0%
Precision	62.0%
Recall	49.0%
F1	0.55


</td>

<td width="50%" valign="top">

🧪 Evaluation & Testing
Independent evaluation covers:
- Intent classification
- Retrieval Recall@K
- Response safety
- Escalation metrics
- Escalation tests
- End-to-end pipeline
- API endpoints
LLM-as-a-judge, human-review agreement and golden-set human sign-off
are still pending.

</td>
</tr>
</table>

🔄 End-to-End Pipeline
Customer Query
      │
      ▼
Intent Classification
(TF-IDF + Logistic Regression)
      │
      ▼
Evidence Retrieval
(TF-IDF + Intent-aware)
      │
      ▼
Grounded LLM Reply
      │
      ▼
Safety Checks
      │
      ▼
Escalation + Reason Codes
      │
   ┌──┴───┐
   ▼      ▼
HANDLE  ESCALATE

<table>
<tr>
<td width="50%" valign="top">

🌐 REST API
Method	Endpoint	Purpose
GET	/health	Service health
POST	/tickets	Process ticket
GET	/tickets/{id}	Retrieve ticket


POST /tickets
      ↓
Classify → Retrieve → Generate
      ↓
Safety → Escalation
      ↓
Persist

Processing: Synchronous
Database: SQLite by default
Config: DATABASE_URL
</td>

<td width="50%" valign="top">

⚙️ Local Setup
git clone https://github.com/A-verse/PlayStation-Support-Agent.git
cd PlayStation-Support-Agent

python -m venv .venv
.venv\Scripts\Activate.ps1

pip install -r requirements.txt

Create .env:
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_api_key
REPLY_GEN_MODEL=gemini-3.7-flash

Never commit .env.

</td>
</tr>
</table>

<table>
<tr>
<td width="50%" valign="top">

▶️ Run
Streamlit
streamlit run app.py

http://localhost:8501
FastAPI
uvicorn api.main:app --reload

http://127.0.0.1:8000
Tests
pytest -q

</td>

<td width="50%" valign="top">

🧩 Design Principles
Ground before generating
Evidence constrains the response.
Fail closed
Uncertainty leads to fallback or escalation.
Auditable decisions
Each stage produces inspectable outputs.
Honest evaluation
Weak labels and validation gaps are documented.
Modular architecture
Components can be evaluated independently.
</td>
</tr>
</table>

📁 Project Structure
PlayStation-Support-Agent/
├── api/
│   ├── database.py
│   ├── db_models.py
│   ├── dependencies.py
│   ├── main.py
│   └── schemas.py
│
├── src/
│   ├── agent.py
│   ├── classifier.py
│   ├── retrieval.py
│   ├── reply_generator.py
│   ├── safety.py
│   ├── escalation.py
│   ├── llm_providers.py
│   └── evaluation.py
│
├── tests/
├── app.py
├── REPORT.md
├── REVIEW_GUIDE.md
├── decision_log.md
├── requirements.txt
├── .env.example
├── logo.png
└── README.md

<table>
<tr>
<td width="50%" valign="top">

🚧 Known Limitations
- Golden-set human sign-off pending
- Retrieval human validation pending
- LLM-as-a-judge pending
- Human-review agreement unavailable
- Hardware retrieval needs improvement
- Escalation performance can improve
- No authentication / rate limiting
- Synchronous ticket processing
- No background queue / retry worker
- No real PlayStation account/order/payment integration
</td>

<td width="50%" valign="top">

🗺️ What's Next
Evaluation
- Human golden-set review
- Human-review agreement
- Retrieval validation
- Live LLM quality evaluation
- Threshold tuning
Product
- Authentication & rate limiting
- Async processing
- Background workers
- Human-agent overrides
- Production database
Model
- Better weak labels
- Hardware retrieval
- Confidence calibration
- Expanded safety evaluation
- Alternative retrieval methods
</td>
</tr>
</table>

📚 Documentation
Document	Purpose
[`REPORT.md`](REPORT.md)	Technical report & evaluation
[`REVIEW_GUIDE.md`](REVIEW_GUIDE.md)	Reviewer walkthrough
[`decision_log.md`](decision_log.md)	Engineering decisions & trade-offs


🎯 Assignment Context
Built as an SDE Intern take-home project for Hiver.
Data
  ↓
Classification
  ↓
Retrieval
  ↓
LLM
  ↓
Safety
  ↓
Escalation
  ↓
API
  ↓
Evaluation

The goal is not only to generate a plausible answer, but to build a support
system whose decisions can be tested, inspected and challenged.
👨‍💻 Author
A-verse
Python · scikit-learn · Google Gemini · Streamlit · FastAPI
Ground before generating. · Escalate when uncertain. · Measure what matters.
```
