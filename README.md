<table>
<tr>
<td width="50%" valign="top">

## 🤖 Grounded Reply

Retrieved support evidence is passed to the response generator.

**Generation rules**

- Stay grounded in retrieved evidence
- No unsupported policies, refunds or guarantees
- Don't expose internal system details
- Don't repeat already-tried fixes
- Ask for clarification when evidence is weak
- Keep responses concise

```text
Evidence
   ↓
Prompt
   ↓
LLM
   ↓
Response

Supports mock/template mode and live LLM mode.
If live generation fails, the system uses a safe deterministic fallback.
</td>

<td width="50%" valign="top">

🛡️ Response Safety
Every generated response passes through safety checks.
Checks include
- Unsupported claims
- Weak/missing evidence
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
Escalation triggers
- Low confidence
- Insufficient evidence
- Safety failure
- Sensitive account/security issues
- Out-of-scope requests
- Explicit escalation rules
Every escalation includes reason codes.
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
LLM-as-a-judge, human-review agreement and golden-set human sign-off are
still pending.

</td>
</tr>
</table>

🔄 End-to-End Pipeline
Customer Query
      │
      ▼
Intent Classification
(TF-IDF + LogReg)
      │
      ▼
Evidence Retrieval
(TF-IDF Similarity)
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
   ┌──┴──┐
   ▼     ▼
HANDLE ESCALATE

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
Classify
     ↓
Retrieve
     ↓
Generate
     ↓
Safety
     ↓
Escalate
     ↓
Persist

Synchronous processing. SQLite by default.
Database connection is configurable through DATABASE_URL.
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
Components can be tested independently.
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
- No authentication/rate limiting
- Synchronous ticket processing
- No background queue/retry worker
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
[`decision_log.md`](decision_log.md)	Engineering decisions


🎯 Assignment Context
Built as an SDE Intern take-home project:
Data → Classification → Retrieval → LLM → Safety → Escalation → API → Evaluation

The goal is not only to generate a plausible answer, but to build a support
system whose decisions can be tested, inspected and challenged.
👨‍💻 Author
A-verse
Python · scikit-learn · Google Gemini · Streamlit · FastAPI
Ground before generating. · Escalate when uncertain. · Measure what matters.
```
