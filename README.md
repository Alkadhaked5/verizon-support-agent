# Verizon AI Support Agent

An AI-powered customer support prototype built using the Customer Support on Twitter dataset.

The system focuses on `VerizonSupport` and performs three tasks:

1. **Intent Classification** — classifies an incoming customer message into one of 11 support intents.
2. **Historical Retrieval + Reply Drafting** — retrieves similar historical Verizon customer-support interactions and uses them as evidence for drafting a response.
3. **Escalation Decision** — decides whether the message can be auto-handled or should be escalated to a human, with an explicit reason.

## Problem Framing

Customer support conversations on Twitter are often short, noisy, and context-dependent. A useful support agent therefore needs to understand the customer's likely intent, use evidence from previously resolved support cases, and avoid automatically handling cases where the available evidence is insufficient.

This project intentionally focuses on a **small, reproducible prototype** rather than attempting to build a production-ready customer support system.

## What is Built

- 200-example manually labelled golden evaluation set
- 11 customer-support intents
- Majority and word-level TF-IDF baselines
- Character TF-IDF + Logistic Regression classifier
- Historical Verizon support case retrieval using TF-IDF cosine similarity
- Grounded reply drafting using historical responses
- Rule-based escalation with reasons
- Human audit of retrieval/reply quality
- LLM-as-judge evaluation of reply quality
- FastAPI backend
- Simple browser-based frontend

## Headline Results

The intent classifier was evaluated using 5-fold stratified cross-validation on the 200-example golden set.

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Majority baseline | 18.5% | 2.8% | 5.8% |
| Word TF-IDF + Logistic Regression | 33.5% | 29.9% | 33.4% |
| **Character TF-IDF + Logistic Regression** | **48.5%** | **45.7%** | **47.9%** |

The character-level model improved accuracy by **15.0 percentage points** over the word-level TF-IDF baseline and by **30.0 percentage points** over the majority baseline.

### Important caveat

The 48.5% headline accuracy should not be interpreted as production-level support accuracy. The evaluation uses only 200 manually labelled examples and 11 intents. Many customer messages are short or depend on previous conversation context, which is not fully represented in the single-message classifier.

The dataset is also noisy and contains overlapping support categories. Therefore, the result demonstrates improvement over simple baselines on this evaluation set rather than production readiness.

## System Architecture

The support agent follows this pipeline:

```text
Customer Message
       |
       v
+----------------------+
| Intent Classification|
| Character TF-IDF + LR|
+----------+-----------+
           |
           v
+----------------------+
| Historical Retrieval |
| TF-IDF + Cosine Sim. |
+----------+-----------+
           |
           v
+----------------------+
| Escalation Decision  |
| Rules + Evidence     |
+----------+-----------+
           |
           +------------------+
           |                  |
           v                  v
      AUTO_HANDLE          ESCALATE
           |
           v
+----------------------+
| Reply Drafting       |
| Historical Evidence  |
| + Gemini (optional)  |
+----------------------+
           |
           v
      Draft Reply

## Evaluation Setup

### Golden Set

A manually labelled golden evaluation set of **200 customer messages** was created from VerizonSupport customer interactions.

The examples were sampled randomly using a fixed random seed (`42`) to make the evaluation reproducible.

Each example was assigned one of 11 support intents based on the customer's message and the intent definitions established during dataset exploration.

The golden set is used for classifier evaluation through **5-fold stratified cross-validation**.

### Intent Classification Evaluation

Three models were compared:

1. Majority-class baseline
2. Word TF-IDF + Logistic Regression
3. Character TF-IDF + Logistic Regression

The primary metrics are:

- Accuracy
- Macro F1
- Weighted F1

Macro F1 is particularly useful here because the 11 intents are not equally represented in the golden set.

### Historical Retrieval Evaluation

A 20-case manual audit was performed on retrieved historical support cases.

Results:

- Clearly relevant: **8/20 (40%)**
- At least partially relevant: **16/20 (80%)**
- Irrelevant: **4/20 (20%)**

The audit showed that lexical similarity is useful for finding related cases, but the cosine similarity score should not be treated as a guarantee of semantic relevance.

### Reply Quality Evaluation

The top retrieved historical response was evaluated on 20 cases using four criteria:

- Relevance
- Groundedness
- Helpfulness
- Safety

Each criterion was scored from 0 to 2.

Human evaluation results:

- Good: **7/20 (35%)**
- Needs improvement: **7/20 (35%)**
- Poor: **6/20 (30%)**
- Average total score: **5.4/8**

### LLM-as-Judge

The same 20 cases were also evaluated using an LLM judge with the same four dimensions.

Agreement between human evaluation and the LLM judge was measured using both raw agreement and Cohen's kappa.

| Dimension | Raw Agreement | Cohen's Kappa |
|---|---:|---:|
| Relevance | 40% | 0.114 |
| Groundedness | 60% | 0.192 |
| Helpfulness | 35% | 0.085 |
| Safety | 95% | 0.000 |

The low agreement on relevance and helpfulness indicates that the LLM judge should not be treated as a replacement for human evaluation.

The high raw safety agreement is less informative because almost all examples received the same safety score, resulting in near-zero variance and a Cohen's kappa of 0.

### Escalation Evaluation

A 20-case human audit was used to compare the escalation policy against human decisions.

The revised rule-based policy achieved **65% agreement** with human escalation labels on this audit, compared with **50% agreement** for the earlier policy.

This is reported as agreement on the audited sample, not as a production-level escalation accuracy estimate.

## What Is Misleading About My Headline Number?

The headline number is **48.5% intent classification accuracy** on the 200-example golden set.

This number is useful for comparing the proposed classifier against the baselines, but it has several important limitations:

1. **Small evaluation set** — The result is based on only 200 manually labelled examples, not the full Verizon dataset.

2. **Single-message evaluation** — Many Twitter support messages are short and depend on previous conversation turns. The classifier currently receives only the current customer message.

3. **Ambiguous intent boundaries** — Some categories overlap in practice, such as internet issues vs router issues, support requests vs complaints, and account vs billing issues.

4. **Class imbalance** — The 11 intents are not equally represented in the golden set, so accuracy alone can hide weak performance on smaller classes. This is why Macro F1 is also reported.

5. **Not a production accuracy estimate** — The evaluation demonstrates improvement over the selected baselines on this benchmark. It does not establish how the system would perform on live customer traffic.

6. **Confidence is not reliability** — The classifier's predicted probability should not be interpreted as a calibrated probability of correctness.

Therefore, the main claim is:

> The character-level TF-IDF classifier substantially outperforms the selected simple baselines on the manually labelled evaluation set, but the result should not be interpreted as production-ready intent classification accuracy.

## Top Failure Modes

### 1. Short and Context-Dependent Messages

Examples include messages such as `DM?`, `South Jersey`, `ty`, and `Hello...`.

**Hypothesis:** The classifier receives only the current message and cannot reliably infer what these fragments refer to. Adding previous conversation turns should improve classification.

### 2. Internet, Router, and Outage Confusion

Messages involving internet connectivity, routers, and service availability are frequently confused across these categories.

**Hypothesis:** These intents have overlapping vocabulary. A hierarchical classifier or explicit signals for device, connectivity, and area-wide outage could improve separation.

### 3. Support Request vs Complaint/Feedback

Messages can simultaneously describe a support problem and express frustration with the support experience.

**Hypothesis:** Emotional language can dominate lexical features even when the underlying customer intent is requesting assistance. Separating sentiment from intent could help.

### 4. Account, Billing, Mobile, and Installation Overlap

Short messages such as questions about deals, passwords, accounts, or service upgrades can contain insufficient information to distinguish the correct category.

**Hypothesis:** Entity and action information, combined with conversation context, could provide stronger signals than surface-level text alone.

### 5. Twitter Noise and Ambiguous Labels

The dataset contains usernames, URLs, fragments, conversational replies, and messages whose meaning depends heavily on preceding turns.

**Hypothesis:** Better conversation reconstruction and multi-turn annotation would reduce ambiguity and establish a more reliable performance ceiling.

## Next Week Plan

The next iteration would focus on improving the parts of the system that currently have the largest limitations.

1. **Add conversation context**  
   Include the previous 1–3 customer/support turns when classifying short or context-dependent messages.

2. **Improve intent definitions**  
   Refine boundaries between overlapping intents such as internet/router/outage and support-request/complaint.

3. **Evaluate better retrieval methods**  
   Compare TF-IDF retrieval against embedding-based retrieval while keeping the evaluation set fixed.

4. **Improve reply generation**  
   Generate responses from multiple relevant historical cases rather than relying primarily on the highest-ranked case.

5. **Calibrate escalation**  
   Expand the human escalation audit and tune the policy using more labelled examples.

6. **Increase evaluation coverage**  
   Expand the golden set beyond 200 examples and include conversation-level evaluation rather than only single-message evaluation.

7. **Add stronger safety controls**  
   Introduce explicit handling for sensitive account/payment requests and cases where historical evidence is insufficient.

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Alkadhaked5/verizon-support-agent.git
cd verizon-support-agent


