# Verizon AI Support Agent — Assignment Report

## 1. Problem Framing

The goal of this project is to build a lightweight AI customer-support agent for VerizonSupport using historical customer-support conversations from the Customer Support on Twitter dataset.

The agent is designed to handle three core decisions:

1. **Intent classification:** Determine what type of support issue the customer is describing.
2. **Grounded reply drafting:** Retrieve historically similar Verizon support interactions and use their responses as evidence for drafting a reply.
3. **Escalation:** Decide whether the case is suitable for automated handling or should be sent to a human, along with a reason for the decision.

### Scope

The project focuses on a single brand, `VerizonSupport`, rather than attempting to model all brands in the dataset.

The prototype uses 11 support intents:

- Internet / Wi-Fi issues
- Service outages
- Router / equipment issues
- Mobile phone issues
- TV / channel issues
- Billing / payment issues
- Account issues
- Installation / availability
- Support requests
- Complaints / feedback
- Other

### What is not built

This is intentionally a prototype rather than a production customer-support system.

It does not attempt to:

- access or modify real customer accounts;
- process real payments;
- perform account-specific actions;
- guarantee that an automatically drafted response is correct;
- reconstruct every multi-turn conversation;
- estimate production-level support accuracy.

The main design goal is to demonstrate a reproducible support-agent pipeline using historical evidence, while making uncertainty and escalation explicit.


## 2. Approach

### 2.1 Data Selection

The Customer Support on Twitter dataset contains approximately 2.8 million tweets across multiple customer-support brands.

I selected `VerizonSupport` as the target brand. From the dataset, customer messages directly associated with VerizonSupport were extracted, resulting in approximately 12,953 customer messages.

Historical customer-agent pairs were constructed from messages where a customer tweet had a linked VerizonSupport response.

### 2.2 Intent Classification

A manually labelled golden set of 200 customer messages was created using 11 support intents.

Two lexical approaches were compared:

- Word-level TF-IDF with unigrams/bigrams + Logistic Regression
- Character-level TF-IDF with 3–5 character n-grams + Logistic Regression

The character-level model was selected as the final classifier because it performed better on the noisy and fragmented Twitter support messages.

Class balancing was enabled using `class_weight="balanced"`.

### 2.3 Historical Retrieval

The historical Verizon customer-agent pairs were cleaned to remove Twitter-specific artifacts such as URLs and usernames.

Customer messages were represented using word-level TF-IDF vectors. Given a new customer message, cosine similarity is used to retrieve the most similar historical support cases.

The final retrieval corpus contains approximately 8,523 historical customer-agent pairs after removing evaluation examples.

The retrieved historical responses are used as evidence for reply drafting.

### 2.4 Reply Drafting

The reply component uses the retrieved historical cases as context.

When Gemini is available, it is prompted to produce a concise response grounded in the retrieved Verizon support examples.

The prompt explicitly instructs the model not to invent unsupported troubleshooting steps.

If Gemini is unavailable or generation fails, the system falls back to the highest-ranked historical Verizon response.

### 2.5 Escalation

A rule-based escalation layer determines whether the case should be automatically handled or escalated.

The policy considers:

- Predicted intent
- Historical retrieval similarity
- Message length
- Signals of unresolved or recurring problems
- Account and payment-related intents

Account and payment-related cases are escalated because they may require secure handling that is outside the prototype's scope.

Low-evidence and highly context-dependent messages are also escalated.

### 2.6 End-to-End Pipeline

The complete system is:

```text
Customer Message
       |
       v
Intent Classification
       |
       v
Historical Case Retrieval
       |
       v
Escalation Decision
       |
       +----> ESCALATE
       |
       +----> AUTO_HANDLE
                    |
                    v
              Reply Drafting
                    |
                    v
                Draft Reply


## 3. Evaluation Setup

### Golden Evaluation Set

A golden set of **200 customer messages** was manually labelled from the VerizonSupport customer-message pool.

The sample was selected randomly using `random_state=42`. Each message was assigned one of the 11 defined intents.

The intent classifier was evaluated using **5-fold stratified cross-validation**, so every example is evaluated out-of-fold rather than being evaluated on the same data used to fit its fold's model.

### Baselines

Two baselines were used:

1. **Majority baseline** — always predicts the most frequent intent.
2. **Word TF-IDF + Logistic Regression** — a simple lexical NLP baseline using word unigrams and bigrams.

The proposed model is:

**Character TF-IDF (3–5 character n-grams) + Logistic Regression**

### Metrics

For intent classification, the following metrics are reported:

- Accuracy
- Macro F1
- Weighted F1

Macro F1 is emphasized because the intent classes have different numbers of examples.

### Retrieval Audit

A manual audit of **20 retrieved cases** was conducted.

Each retrieved case was classified as:

- Relevant
- Partial
- Irrelevant

### Reply Quality Audit

The drafted replies were manually evaluated on 20 cases using four dimensions:

- Relevance
- Groundedness
- Helpfulness
- Safety

Each dimension was scored from 0–2, giving a maximum total score of 8.

### LLM-as-Judge

The same 20 cases were evaluated by an LLM judge using the same four dimensions.

Human-vs-LLM agreement was measured using:

- Raw percentage agreement
- Cohen's kappa

This comparison is included to test whether automated judging is a reliable substitute for human evaluation.

### Escalation Audit

A separate 20-case audit compared the rule-based escalation decision with human labels.

The revised escalation policy achieved **65% agreement** with the human audit, compared with **50% agreement** for the earlier policy.


## 4. Results

### 4.1 Intent Classification

The proposed character-level TF-IDF classifier outperformed both baselines on the 200-example golden set.

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Majority baseline | 18.5% | 2.8% | 5.8% |
| Word TF-IDF + Logistic Regression | 33.5% | 29.9% | 33.4% |
| **Character TF-IDF + Logistic Regression** | **48.5%** | **45.7%** | **47.9%** |

The proposed model improves accuracy by **15.0 percentage points** over the word-level baseline and **30.0 percentage points** over the majority baseline.

The final model's per-class performance is uneven. The strongest F1 scores were observed for `other` (0.64), `mobile_phone_issue` (0.55), `tv_channel_issue` (0.53), and `internet_wifi_issue` (0.51). Lower performance was observed for `router_equipment_issue` (0.33), `service_outage` (0.32), and `installation_availability` (0.36).

### 4.2 Retrieval

The retrieval system was evaluated through a 20-case manual audit.

| Retrieval outcome | Cases | Percentage |
|---|---:|---:|
| Clearly relevant | 8 | 40% |
| Partially relevant | 8 | 40% |
| Irrelevant | 4 | 20% |

Overall, **80% of audited cases were at least partially relevant**.

This audit also showed that a high cosine-similarity score does not necessarily mean that the retrieved case is semantically relevant. Therefore, retrieval similarity is treated as evidence for ranking rather than as a guaranteed relevance score.

### 4.3 Reply Quality

Human evaluation of 20 drafted replies produced:

| Quality category | Cases | Percentage |
|---|---:|---:|
| Good | 7 | 35% |
| Needs improvement | 7 | 35% |
| Poor | 6 | 30% |

The average human score was **5.4/8**.

Average scores by dimension were:

- Relevance: **1.05/2**
- Groundedness: **1.30/2**
- Helpfulness: **1.10/2**
- Safety: **1.95/2**

The results suggest that grounding and safety were stronger than relevance and helpfulness. Retrieval quality is therefore a major bottleneck for improving the generated replies.

### 4.4 LLM-as-Judge Agreement

The LLM judge was compared against human judgments on the same 20 cases.

| Dimension | Raw Agreement | Cohen's Kappa |
|---|---:|---:|
| Relevance | 40% | 0.114 |
| Groundedness | 60% | 0.192 |
| Helpfulness | 35% | 0.085 |
| Safety | 95% | 0.000 |

Agreement was low for relevance and helpfulness, suggesting that the LLM judge should not be considered a replacement for human evaluation.

Safety had high raw agreement, but the kappa value was 0 because the safety labels had almost no variation. In this situation, raw agreement is more intuitive than kappa alone.

### 4.5 Escalation

In a 20-case human audit, the revised rule-based escalation policy achieved **65% agreement** with human escalation decisions.

The earlier policy achieved **50% agreement** on the same audit.

This result indicates improvement in the escalation heuristic, but the small audit size means it should not be interpreted as production-level escalation accuracy.


## 5. Failure Modes

The classifier's errors were inspected using the out-of-fold predictions from the 200-example golden set. Five recurring failure patterns were identified.

### 1. Short and Context-Dependent Messages

Examples include `DM?`, `South Jersey`, `ty`, `Hello...`, and product names such as `Actiontec wcb6200q`.

These messages contain too little standalone information to reliably infer intent.

**Hypothesis:** Including the previous 1–3 conversation turns would provide the missing context and improve classification.

### 2. Internet, Router, and Service-Outage Confusion

Several errors occurred between `internet_wifi_issue`, `router_equipment_issue`, and `service_outage`.

For example, messages describing recurring connectivity problems or router behaviour can contain similar vocabulary across all three categories.

**Hypothesis:** A hierarchical approach separating the broad problem type from the specific device/state could improve these boundaries.

### 3. Support Request vs Complaint/Feedback

Customers frequently combine a request for assistance with frustration about their support experience.

For example, messages about being repeatedly transferred between departments may reasonably look like either a support request or a complaint.

**Hypothesis:** Separating sentiment from the underlying customer intent could reduce this type of confusion.

### 4. Account, Billing, Mobile, and Installation Overlap

Short messages about accounts, passwords, deals, phone numbers, or service upgrades can lack enough information to distinguish the appropriate category.

**Hypothesis:** Extracting entities and requested actions, followed by hierarchical intent classification, could provide stronger signals than surface-level lexical features.

### 5. Noisy and Ambiguous Twitter Data

The dataset contains usernames, URLs, fragments, conversational responses, and messages whose meaning depends on previous turns.

Some examples are inherently difficult to label from a single message.

**Hypothesis:** Conversation-level reconstruction and multi-turn annotation would reduce ambiguity and provide a more realistic evaluation.

## 6. What Is Misleading About My Headline Number?

The headline result is **48.5% accuracy for intent classification**.

This number is useful for demonstrating improvement over the selected baselines, but it should not be interpreted as the accuracy of the complete AI support agent.

There are several reasons:

- The evaluation contains only **200 manually labelled examples**.
- The classifier evaluates the **current customer message**, while many Twitter messages depend on previous conversation turns.
- The 11 intents have overlapping boundaries, particularly technical issues, outages, support requests, and complaints.
- The classes are imbalanced, so accuracy alone does not represent performance across all intents.
- The classifier's confidence score is not calibrated and should not be interpreted as the probability that a prediction is correct.
- Reply quality and escalation performance are separate from intent classification accuracy.
- Retrieval evaluation showed that only **40% of audited retrieved cases were clearly relevant**, demonstrating that retrieval remains a significant limitation.
- Human evaluation rated only **35% of audited replies as Good**, so correct intent classification does not automatically imply a good customer-support response.

Therefore, the 48.5% figure should be interpreted as:

> **Performance of the proposed intent classifier on a small manually labelled benchmark, compared with simple baselines — not production-level accuracy of the complete support agent.**

## 7. What I Would Do With One More Week

If given another week, I would prioritize improvements based on the observed failure modes and evaluation results.

### 1. Add conversation-aware classification

The largest limitation is that many Twitter messages are context-dependent. I would reconstruct conversation threads and provide the previous few turns to the classifier.

### 2. Improve retrieval

The current TF-IDF retriever provides at least partially relevant evidence for 80% of audited cases, but only 40% are clearly relevant.

I would compare the current retriever against embedding-based retrieval and evaluate both using the same manually audited examples.

### 3. Improve reply generation

Instead of relying primarily on the top retrieved response, I would provide multiple high-quality retrieved cases to the generator and require the generated answer to be supported by those examples.

### 4. Refine intent boundaries

I would review ambiguous examples in the golden set and refine the definitions of overlapping categories such as:

- Internet vs router issues
- Service outage vs installation/availability
- Support request vs complaint/feedback
- Account vs billing/mobile issues

### 5. Expand human evaluation

The current human audits contain only 20 cases each. I would increase the sample size and evaluate complete conversation-level interactions rather than isolated messages.

### 6. Improve escalation

I would collect more human escalation labels and tune the escalation policy using those examples. The goal would be to optimize for safe handling rather than simply maximizing the percentage of cases automatically handled.

### 7. Add confidence-aware handling

The classifier currently produces low-confidence predictions for many difficult examples. I would calibrate confidence and use it as an additional signal for escalation, rather than treating the raw maximum probability as a reliable correctness probability.

## 8. Decision Summary

The prototype demonstrates a complete customer-support workflow built around historical support evidence.

The main decisions were:

- Focus on a single brand, `VerizonSupport`, to keep the problem well-defined.
- Use a manually labelled 200-example golden set for reproducible evaluation.
- Compare the proposed classifier against both a trivial majority baseline and a simple word-level baseline.
- Use character-level TF-IDF because the dataset contains noisy, abbreviated, and fragmented Twitter messages.
- Use historical Verizon customer-agent pairs as retrieval evidence for reply drafting.
- Keep retrieval transparent and reproducible using TF-IDF cosine similarity.
- Escalate cases where the available evidence is weak or the issue may require secure human handling.
- Treat LLM-generated replies as drafts rather than guaranteed correct answers.
- Evaluate different parts of the system separately instead of using a single misleading end-to-end metric.
- Explicitly document failure modes and limitations rather than presenting the headline classifier accuracy as production performance.

The resulting system is a working prototype that improves substantially over simple intent-classification baselines while also exposing the main limitations of single-message classification and lexical retrieval.


