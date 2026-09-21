# AskPlayStation Support Agent

An AI-powered customer support agent designed to understand customer issues, retrieve relevant support information, generate grounded responses, and identify cases that should be escalated to a human support representative.

## Overview

AskPlayStation Support Agent processes customer support messages through a structured pipeline that combines intent classification, retrieval, response generation, safety checks, and escalation logic.

The system is designed to:

- Classify customer messages into support intents.
- Retrieve relevant information from historical support conversations.
- Generate responses grounded in retrieved context.
- Detect low-confidence or potentially unsafe responses.
- Escalate cases that require human attention.
- Provide evaluation artifacts to inspect system behavior and limitations.

## Key Features

### Intent Classification

Uses a TF-IDF-based text representation and Logistic Regression classifier to predict the intent of a customer message.

### Context-Aware Retrieval

Retrieves similar historical support messages using TF-IDF cosine similarity. Intent-aware reranking helps prioritize results when classifier confidence is sufficiently high.

### Grounded Response Generation

Generates responses using retrieved support context, with configurable response-generation modes.

### Safety and Quality Checks

Applies message-quality checks and response safety validation before returning a response.

### Escalation Engine

Uses scoring and reason codes to identify conversations that may require human support.

### Evaluation and Auditability

Includes evaluation scripts, decision logs, review guidance, and reports to make system behavior easier to inspect.

## System Architecture

```text
Customer Message
       |
       v
Message Quality Guard
       |
       v
Intent Classification
       |
       v
Confidence-Gated Retrieval
       |
       v
Grounded Response Generation
       |
       v
Safety Checks
       |
       v
Escalation Scoring
       |
       v
Final Response / Human Escalation
```

## Technology Stack

- **Language:** Python
- **Machine Learning:** scikit-learn
- **Text Processing:** TF-IDF vectorization
- **Classification:** Logistic Regression
- **Retrieval:** Cosine similarity with intent-aware reranking
- **Interface:** Streamlit
- **Testing:** pytest

## Dataset

The project uses the [Customer Support on Twitter dataset](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter).

The dataset contains customer support conversations from Twitter. The raw dataset is not included in this repository due to its size.

Expected raw data file:

```text
data/raw/twcs.csv
```

## Getting Started

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_DIRECTORY>
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Place the dataset at:

```text
data/raw/twcs.csv
```

### 5. Run the application

```bash
streamlit run app.py
```

## Usage

The application provides an interface for submitting customer support messages and viewing the agent's response and decision-making output.

The agent can also be run from the command line:

```bash
python src/agent.py
```

## Evaluation

The project includes evaluation artifacts for classification, retrieval, response quality, safety, and escalation behavior.

| Component             | Evaluation                                                |
| --------------------- | --------------------------------------------------------- |
| Intent classification | Compared against a baseline using accuracy and macro F1   |
| Retrieval             | Evaluated using retrieval proxies and relevance judgments |
| Response pipeline     | Evaluated for response quality and safety failures        |
| Escalation engine     | Evaluated using escalation and auto-handling metrics      |

### Classification Results

| Model                        | Accuracy | Macro F1 |
| ---------------------------- | -------: | -------: |
| Baseline                     |     6.4% |    0.013 |
| TF-IDF + Logistic Regression |    76.1% |    0.760 |

The classifier was trained using weak silver labels. The separate golden evaluation set contains 188 examples.

### Evaluation Notes

- Some response-pipeline metrics were measured using mock/template response mode.
- The golden set was created with LLM assistance and is pending independent human sign-off.
- Retrieval relevance judgments were conducted on a limited sample and have not been independently validated.
- LLM-as-judge and human agreement evaluations have not been completed.
- Evaluation results should be interpreted alongside the documented dataset, sampling, and validation limitations.

## Project Structure

```text
.
├── app.py
├── src/
│   └── agent.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── tests/
├── decision_log.md
├── REPORT.md
├── REVIEW_GUIDE.md
└── requirements.txt
```

## Testing

Run the test suite with:

```bash
pytest
```

## Documentation

- **`REPORT.md`**: Evaluation results, analysis, and limitations.
- **`decision_log.md`**: Design decisions and implementation notes.
- **`REVIEW_GUIDE.md`**: Guidance for reviewing the golden evaluation set.

## Response Generation Modes

The project supports mock/template response generation by default. A live language model can be configured through supported provider settings, including Anthropic, OpenAI, or an OpenAI-compatible endpoint.

Reported pipeline evaluation figures should be interpreted in the context of the response mode used during evaluation.

## Support Intent Categories

The classifier predicts customer support intents, while unclear or insufficiently specified messages can be handled as a routing state rather than treated as a dedicated intent label.

## Limitations

- The quality of classification and retrieval depends on the dataset and its labels.
- Historical support conversations may not fully represent current product or policy information.
- LLM-assisted evaluation labels require independent human review.
- Limited-sample retrieval judgments should not be interpreted as comprehensive validation.
- Generated responses should be reviewed carefully in cases involving uncertainty, safety, or escalation.

## Future Improvements

- Complete independent human review of the golden evaluation set.
- Expand retrieval relevance evaluation with a larger, human-validated sample.
- Evaluate response quality across live model configurations.
- Measure agreement between automated judgments and human reviewers.
- Improve escalation thresholds using additional validation data.
