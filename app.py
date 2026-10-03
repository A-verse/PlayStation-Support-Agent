"""
app.py — Streamlit frontend for the AskPlayStation support agent.

Three tabs:

1. "Try the Agent" — enter a message (+ optional prior context), see the
   real pipeline's intent, confidence, retrieved evidence, drafted reply,
   safety-check outcomes, and escalation decision with reason codes.

2. "Evaluation Dashboard" — the eval harness's actual saved results.

3. "What's Next" — honest, current status of what's complete vs. open,
   pulled from the same source as REPORT.md/README so it can't drift.

Connects to the REAL pipeline (Agent -> ReplyGenerator -> retrieval -> classifier),
not mock data.

Reply generation runs in mock/template mode unless an LLM provider is configured.
The configured provider and model are always shown explicitly in the UI.

Run:
    streamlit run app.py
"""

import json
import sys
from pathlib import Path

import streamlit as st


# ---------------------------------------------------------------------------
# Import project modules
# ---------------------------------------------------------------------------

sys.path.insert(
    0,
    str(Path(__file__).parent / "src")
)

from agent import Agent


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AskPlayStation Support Agent",
    page_icon="🎮",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>

    :root {
        --ps-blue: #0070d1;
        --ps-dark: #10141c;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    h1, h2, h3 {
        letter-spacing: -0.01em;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        flex-wrap: wrap;
    }

    .stTabs [data-baseweb="tab"] {
        background: #f0f2f6;
        border-radius: 8px 8px 0 0;
        padding: 8px 18px;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background: var(--ps-blue);
        color: white;
    }

    div[data-testid="stMetric"] {
        background: #f7f9fc;
        border: 1px solid #e6e9ef;
        border-radius: 10px;
        padding: 12px 16px;
    }

    .status-pill {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .status-mock {
        background: #fff3cd;
        color: #7a5c00;
    }

    .status-live {
        background: #d4edda;
        color: #1e6b32;
    }

    @media (max-width: 640px) {

        .block-container {
            padding-left: 0.8rem;
            padding-right: 0.8rem;
        }

        h1 {
            font-size: 1.6rem !important;
        }

        .stTabs [data-baseweb="tab"] {
            padding: 6px 10px;
            font-size: 0.85rem;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Agent loading
# ---------------------------------------------------------------------------

@st.cache_resource
def load_agent():
    return Agent()


# ---------------------------------------------------------------------------
# Quick examples
# ---------------------------------------------------------------------------

EXAMPLE_MESSAGES = {
    "Hardware issue":
        "my ps4 controller won't charge and keeps turning off",

    "Account compromise (high-risk)":
        "my account was hacked and I lost all my items",

    "Ambiguous/vague":
        "i need help",

    "Frustrated but resolvable":
        "this is ridiculous, my download keeps freezing at 90%!!",

    "Error code":
        "getting error code CE-34878-0 when I try to launch the game",
}


# ---------------------------------------------------------------------------
# Provider status
# ---------------------------------------------------------------------------

def provider_status_html(agent):

    if agent.reply_generator.mock_mode:

        return (
            '<span class="status-pill status-mock">'
            '⚙️ MOCK MODE — no LLM key configured, '
            'replies built directly from retrieved evidence'
            '</span>'
        )

    name = agent.reply_generator.provider.name
    model = agent.reply_generator.provider.model

    return (
        '<span class="status-pill status-live">'
        f'🟢 LIVE LLM — provider: {name} · model: {model}'
        '</span>'
    )


# ---------------------------------------------------------------------------
# Evidence renderer
# ---------------------------------------------------------------------------

def render_evidence(evidence_list):

    if not evidence_list:

        st.info(
            "No retrieved historical evidence was found."
        )

        return

    for i, evidence in enumerate(
        evidence_list,
        1
    ):

        with st.container(border=True):

            st.markdown(
                f"**Evidence #{i}** · "
                f"similarity `{evidence['similarity_score']:.2f}` · "
                f"intent tag `{evidence['predicted_intent']}` · "
                f"`{evidence['pair_id']}`"
            )

            st.markdown(
                f"🗨️ *Historical customer message:* "
                f"{evidence['historical_customer_text']}"
            )

            st.markdown(
                f"↳ *Historical brand reply:* "
                f"{evidence['historical_brand_reply']}"
            )


# ---------------------------------------------------------------------------
# Try the Agent tab
# ---------------------------------------------------------------------------

def render_agent_tab():

    agent = load_agent()

    st.markdown(
        provider_status_html(agent),
        unsafe_allow_html=True
    )

    # -----------------------------------------------------------------------
    # Quick examples
    # -----------------------------------------------------------------------

    st.markdown("##### Quick examples")

    ex_cols = st.columns(
        len(EXAMPLE_MESSAGES)
    )

    for col, (label, message) in zip(
        ex_cols,
        EXAMPLE_MESSAGES.items()
    ):

        if col.button(
            label,
            width="stretch"
        ):

            st.session_state[
                "customer_text_input"
            ] = message

    # -----------------------------------------------------------------------
    # Customer message
    # -----------------------------------------------------------------------

    st.subheader(
        "Customer message"
    )

    customer_text = st.text_area(
        "Message",
        key="customer_text_input",
        value=st.session_state.get(
            "customer_text_input",
            "my ps4 controller won't charge and keeps turning off"
        ),
        height=80,
    )

    # -----------------------------------------------------------------------
    # Prior context
    # -----------------------------------------------------------------------

    ctx_raw = ""

    with st.expander(
        "➕ Add prior conversation context (optional)"
    ):

        ctx_raw = st.text_area(
            "One turn per line, format: role: text "
            "(role = customer or brand)",
            value="",
            height=80,
            placeholder=(
                "customer: I tried restarting it already\n"
                "brand: Please try a different USB cable"
            ),
        )

    prior_context = []

    for line in ctx_raw.strip().splitlines():

        if ":" in line:

            role, text = line.split(
                ":",
                1
            )

            role = role.strip().lower()

            if role in (
                "customer",
                "brand"
            ):

                prior_context.append(
                    {
                        "role": role,
                        "text": text.strip(),
                    }
                )

    # -----------------------------------------------------------------------
    # Run agent
    # -----------------------------------------------------------------------

    if st.button(
        "▶ Run agent",
        type="primary",
        width="stretch"
    ):

        with st.spinner(
            "Running pipeline..."
        ):

            result = agent.handle(
                customer_text,
                prior_context
            )

        # -------------------------------------------------------------------
        # Two-column main result area
        # -------------------------------------------------------------------

        col1, col2 = st.columns(
            [1, 1]
        )

        # ===================================================================
        # LEFT COLUMN
        # ===================================================================

        with col1:

            # ---------------------------------------------------------------
            # Intent classification
            # ---------------------------------------------------------------

            st.subheader(
                "Intent classification"
            )

            m1, m2 = st.columns(2)

            m1.metric(
                "Predicted intent",
                result["predicted_intent"]
            )

            m2.metric(
                "Confidence",
                f"{result['classifier_confidence']:.1%}"
            )

            st.caption(
                f"Retrieval mode used: "
                f"**{result['retrieval_mode']}** "
                f"(confidence-gated at 0.6)"
            )

            # ---------------------------------------------------------------
            # Drafted reply
            # ---------------------------------------------------------------

            st.subheader(
                "Drafted reply"
            )

            st.success(
                result["generated_reply"]
            )

            # ---------------------------------------------------------------
            # Generation / fallback status
            # ---------------------------------------------------------------

            fallback_reason = result.get(
                "final_fallback_reason"
            )

            generation_metadata = result.get(
                "generation_metadata",
                {}
            )

            if fallback_reason:

                if fallback_reason.startswith(
                    "llm_error:"
                ):

                    st.warning(
                        "⚠️ AI drafting service temporarily "
                        "unavailable. A safe fallback response "
                        "was used."
                    )

                elif fallback_reason.startswith(
                    "response_safety_failure:"
                ):

                    st.warning(
                        "⚠️ The generated draft did not pass "
                        "the response-safety checks, so a safe "
                        "fallback response was used."
                    )

                elif fallback_reason.startswith(
                    "evidence_failure:"
                ):

                    st.warning(
                        "⚠️ The retrieved evidence was not strong "
                        "enough for a reliable answer, so a safe "
                        "fallback response was used."
                    )

                else:

                    st.warning(
                        "⚠️ A safe fallback response was used."
                    )

            # ---------------------------------------------------------------
            # Generation method
            # ---------------------------------------------------------------

            generation_method = generation_metadata.get(
                "method"
            )

            if generation_method:

                if generation_method.startswith(
                    "llm:"
                ):

                    st.caption(
                        f"Generation: **Live LLM** · "
                        f"`{generation_method}`"
                    )

                elif generation_method.startswith(
                    "llm_error_fallback:"
                ):

                    st.caption(
                        "Generation: **Safe fallback "
                        "after LLM error**"
                    )

                elif generation_method == "mock_template":

                    st.caption(
                        "Generation: **Mock/template mode**"
                    )

                else:

                    st.caption(
                        f"Generation: `{generation_method}`"
                    )

        # ===================================================================
        # RIGHT COLUMN
        # ===================================================================

        with col2:

            # ---------------------------------------------------------------
            # Escalation decision
            # ---------------------------------------------------------------

            st.subheader(
                "Escalation decision"
            )

            decision = result[
                "escalation_decision"
            ]

            if decision["action"] == "escalate":

                st.error(
                    f"🚨 **ESCALATE** — score "
                    f"{decision['escalation_score']} / "
                    f"threshold "
                    f"{decision['escalation_threshold']}"
                )

            else:

                st.success(
                    f"✅ **AUTO-HANDLE** "
                    f"({decision['sub_mode']}) — score "
                    f"{decision['escalation_score']} / "
                    f"threshold "
                    f"{decision['escalation_threshold']}"
                )

            st.markdown(
                "**Reason codes:**"
            )

            if decision["reason_codes"]:

                st.markdown(
                    ", ".join(
                        f"`{code}`"
                        for code in decision[
                            "reason_codes"
                        ]
                    )
                )

            else:

                st.markdown(
                    "_(none fired)_"
                )

            # ---------------------------------------------------------------
            # Safety checks
            # ---------------------------------------------------------------

            st.subheader(
                "Safety checks"
            )

            st.json(
                {
                    "evidence_failure":
                        result["evidence_failure"],

                    "evidence_failure_reasons":
                        result[
                            "evidence_failure_reasons"
                        ],

                    "polarity_check_passed":
                        result[
                            "polarity_check"
                        ]["passed"],

                    "responsiveness_check_passed":
                        result[
                            "responsiveness_check"
                        ]["passed"],

                    "context_consistency_passed":
                        result[
                            "context_consistency_check"
                        ]["passed"],

                    "already_tried_detected":
                        result[
                            "already_tried_detected"
                        ],
                },
                expanded=False
            )

        # -------------------------------------------------------------------
        # Retrieved evidence
        # -------------------------------------------------------------------

        st.subheader(
            "Retrieved evidence (traceability)"
        )

        render_evidence(
            result[
                "retrieved_evidence"
            ]
        )

        # -------------------------------------------------------------------
        # Raw pipeline output
        # -------------------------------------------------------------------

        with st.expander(
            "🔍 Full raw pipeline output "
            "(for debugging/audit)"
        ):

            st.json(
                result
            )


# ---------------------------------------------------------------------------
# Evaluation Dashboard
# ---------------------------------------------------------------------------

def render_dashboard_tab():

    st.subheader(
        "Evaluation summary"
    )

    st.caption(
        "Loaded from "
        "`data/processed/eval_harness_summary.json` — "
        "generated by `python3 src/eval_harness.py`. "
        "Every number carries the same honesty caveats as "
        "the underlying stages: proxy vs. Claude-judged vs. "
        "(not yet available) human-verified."
    )

    try:

        summary = json.loads(
            Path(
                "data/processed/eval_harness_summary.json"
            ).read_text()
        )

    except FileNotFoundError:

        st.error(
            "Run `python3 src/eval_harness.py` first "
            "to generate the evaluation summary."
        )

        return

    # -----------------------------------------------------------------------
    # Intent classification
    # -----------------------------------------------------------------------

    st.markdown(
        "### Intent classification"
    )

    st.caption(
        summary[
            "intent_classification"
        ][
            "training_label_caveat"
        ]
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Majority baseline accuracy",
        f"{summary['intent_classification']['majority_baseline']['accuracy']:.1%}"
    )

    c2.metric(
        "TF-IDF+LogReg accuracy",
        f"{summary['intent_classification']['tfidf_logreg']['accuracy']:.1%}",
        delta=(
            f"macro F1 "
            f"{summary['intent_classification']['tfidf_logreg']['macro_f1']:.2f}"
        )
    )

    if Path(
        "data/processed/confusion_matrix_tfidf_logreg.png"
    ).exists():

        st.image(
            "data/processed/confusion_matrix_tfidf_logreg.png",
            caption=(
                "TF-IDF+LogReg confusion matrix "
                "(golden set)"
            ),
            width="stretch",
        )

    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------

    st.markdown(
        "### Retrieval"
    )

    st.caption(
        summary[
            "retrieval"
        ][
            "caveat"
        ]
    )

    ret_col1, ret_col2 = st.columns(2)

    with ret_col1:

        st.markdown(
            "**Automatic proxy recall (n=188)**"
        )

        st.json(
            summary[
                "retrieval"
            ][
                "automatic_proxy"
            ]
        )

    with ret_col2:

        st.markdown(
            "**Claude-judged subset "
            "(n=36, NOT human-validated)**"
        )

        st.json(
            summary[
                "retrieval"
            ][
                "claude_judged_subset_n36"
            ]
        )

    # -----------------------------------------------------------------------
    # Escalation engine
    # -----------------------------------------------------------------------

    st.markdown(
        "### Escalation engine vs. trivial baselines"
    )

    st.caption(
        summary[
            "escalation_vs_baselines"
        ][
            "caveat"
        ]
    )

    esc = summary[
        "escalation_vs_baselines"
    ]

    import pandas as pd

    rows = []

    for name, key in [
        (
            "Always auto-handle",
            "always_auto_handle"
        ),
        (
            "Always escalate",
            "always_escalate"
        ),
        (
            "Escalation engine",
            "escalation_engine"
        ),
    ]:

        metrics = esc[key]

        rows.append(
            {
                "Approach":
                    name,

                "Accuracy":
                    metrics["accuracy"],

                "Precision":
                    metrics["precision"],

                "Recall":
                    metrics["recall"],

                "F1":
                    metrics["f1"],

                "False escalation %":
                    metrics[
                        "false_escalation_rate"
                    ],

                "False auto-handle %":
                    metrics[
                        "false_auto_handle_rate"
                    ],
            }
        )

    
    st.dataframe(
        pd.DataFrame(rows).style.format(
            {
                "Accuracy": "{:.2f}",
                "Precision": "{:.2f}",
                "Recall": "{:.2f}",
                "F1": "{:.2f}",
                "False escalation %": "{:.1%}",
                "False auto-handle %": "{:.1%}",
            }
        ),
        width="stretch",
    )
    # -----------------------------------------------------------------------
    # Pipeline-wide rates
    # -----------------------------------------------------------------------

    st.markdown(
        "### Pipeline-wide rates (n=188)"
    )

    pr = summary[
        "pipeline_run"
    ]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Evidence-failure rate",
        f"{pr['evidence_failure_rate']:.1%}"
    )

    c2.metric(
        "Response-safety-failure rate",
        f"{pr['response_safety_failure_rate']:.1%}"
    )

    escalate_rate = (
    pr["action_counts"].get("escalate", 0)
    / summary["golden_eval_size"]
)

    c3.metric(
        "Escalate rate",
        f"{escalate_rate:.1%}"
    )
    
    st.markdown(
        "**Reason-code frequency:**"
    )

    st.bar_chart(
        pr["reason_code_counts"]
    )

    # -----------------------------------------------------------------------
    # LLM-as-judge and human review
    # -----------------------------------------------------------------------

    st.markdown(
        "### LLM-as-judge & human-review agreement"
    )

    st.warning(
        f"LLM-as-judge: "
        f"{summary['llm_as_judge']}"
    )

    st.warning(
        f"Human-review agreement: "
        f"{summary['human_review_agreement']}"
    )

    # -----------------------------------------------------------------------
    # Failure cases
    # -----------------------------------------------------------------------

    st.markdown(
        "### Browse failure cases"
    )

    try:

        full = json.loads(
            Path(
                "data/processed/"
                "eval_full_pipeline_results.json"
            ).read_text()
        )

    except FileNotFoundError:

        st.info(
            "Run the eval harness to populate this section."
        )

        return

    filter_choice = st.selectbox(
        "Show examples where:",
        [
            "evidence_failure == True",
            "response_safety_failure == True",
            "escalation_decision.action == escalate",
            "all examples",
        ],
    )

    if filter_choice == (
        "evidence_failure == True"
    ):

        filtered = [
            record
            for record in full
            if record["evidence_failure"]
        ]

    elif filter_choice == (
        "response_safety_failure == True"
    ):

        filtered = [
            record
            for record in full
            if record[
                "response_safety_failure"
            ]
        ]

    elif filter_choice == (
        "escalation_decision.action == escalate"
    ):

        filtered = [
            record
            for record in full
            if record[
                "escalation_decision"
            ]["action"] == "escalate"
        ]

    else:

        filtered = full

    st.caption(
        f"{len(filtered)} matching examples"
    )

    for record in filtered[:20]:

        with st.expander(
            f"{record['candidate_id']} — "
            f"{record['customer_text'][:80]}"
        ):

            st.json(
                {
                    "predicted_intent":
                        record[
                            "predicted_intent"
                        ],

                    "golden_true_intent":
                        record[
                            "golden_true_intent"
                        ],

                    "classifier_confidence":
                        record[
                            "classifier_confidence"
                        ],

                    "evidence_failure_reasons":
                        record[
                            "evidence_failure_reasons"
                        ],

                    "escalation_decision":
                        record[
                            "escalation_decision"
                        ],

                    "generated_reply":
                        record[
                            "generated_reply"
                        ],
                }
            )


# ---------------------------------------------------------------------------
# What's Next / Roadmap
# ---------------------------------------------------------------------------

def render_roadmap_tab():

    st.subheader(
        "What's complete"
    )

    st.markdown(
        """
        - ✅ Dataset ingestion, brand selection (AskPlayStation),
          language filtering, EDA
        - ✅ Frozen 9-intent taxonomy, derived bottom-up from real data
        - ✅ 189-example golden set (sampling + labeling + QA) —
          **see caveat below**
        - ✅ TF-IDF + Logistic Regression intent classifier,
          evaluated vs. majority baseline
        - ✅ TF-IDF retrieval (global + confidence-gated
          intent-aware rerank), evaluated
        - ✅ Grounded reply generation with evidence traceability,
          mock or **any LLM provider**
        - ✅ Response-safety checks
          (polarity, responsiveness, context-consistency)
        - ✅ Escalation engine with auditable reason codes,
          20 passing tests
        - ✅ Full end-to-end evaluation harness with honest
          proxy/LLM/human labeling
        - ✅ This dashboard, connected to the real pipeline
        """
    )

    st.subheader(
        "What's still open"
    )

    st.markdown(
        """
        1. 🔲 **Golden-set human sign-off** — the single biggest
           open item. Every label is Claude-judged, not yet reviewed
           by a human. See `REVIEW_GUIDE.md`.

        2. 🔲 **LLM-as-judge pass** — not run without a live LLM
           provider configured.

        3. 🔲 **Human-review agreement** — depends on #1.

        4. 🔲 **Live-LLM reply quality** — everything measured so far
           is mock/template mode.

        5. 🔲 **Rare-intent (Hardware) classifier recall is still weak
           (0.62)** — needs either augmented training data or an
           embedding-based fallback for low-confidence cases.

        6. 🔲 Response-safety `idf_overlap_threshold`
           (currently 5.0) is a single-anecdote-tuned value —
           needs validation against a larger, human-reviewed test set.
        """
    )

    st.subheader(
        "Next-week plan (from REPORT.md)"
    )

    try:

        report_text = Path(
            "REPORT.md"
        ).read_text()

        section = report_text.split(
            "## 7. What we'd do with one more week"
        )[1]

        st.markdown(
            "## What we'd do with one more week"
            + section.split("\n## ")[0]
        )

    except Exception:

        st.info(
            "See REPORT.md section 7."
        )


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

st.title(
    "🎮 AskPlayStation Support Agent"
)

tab1, tab2, tab3 = st.tabs(
    [
        "💬 Try the Agent",
        "📊 Evaluation Dashboard",
        "🗺️ What's Next",
    ]
)

with tab1:

    render_agent_tab()

with tab2:

    render_dashboard_tab()

with tab3:

    render_roadmap_tab()