# 🎮 PlayStation Support Agent

### AI-Powered Customer Support Agent | Hiver SDE Intern Take-Home

An end-to-end AI customer-support pipeline inspired by the public **AskPlayStation** support dataset.

The system takes a customer message, identifies the issue, retrieves relevant historical support evidence, drafts a grounded response, checks the response for safety, and decides whether the case can be auto-handled or should be escalated.

> **Public-data prototype:** This project uses publicly available AskPlayStation support data. It does not connect to or perform actions on real PlayStation/Sony customer accounts.

---

## 🚀 Live Demo

### [▶ Try the deployed Streamlit app](https://playstation-support-agent.streamlit.app/)

The application provides three views:

| Tab | What it shows |
|---|---|
| 💬 **Try the Agent** | Run real customer queries through the complete pipeline |
| 📊 **Evaluation Dashboard** | Classification, retrieval, safety and escalation evaluation |
| 🗺️ **What's Next** | Completed work, known limitations and next steps |

---

## ✨ What the Agent Does

```text
Customer Message
       │
       ▼
┌─────────────────────┐
│ Intent Classification│
│ TF-IDF + LogReg      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Evidence Retrieval   │
│ TF-IDF Similarity    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Grounded Reply       │
│ Generation           │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Response Safety      │
│ Checks               │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Escalation Engine    │
│ + Reason Codes       │
└──────────┬──────────┘
           │
      ┌────┴────┐
      ▼         ▼
  Auto-handle  Escalate