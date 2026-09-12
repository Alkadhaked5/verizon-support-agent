# Decision Log

This document records non-obvious decisions made while building the Verizon AI Support Agent.

## 1. Selected VerizonSupport as the target brand

**Decision:** Use `VerizonSupport` as the single brand for the project.

**Why:** The dataset contains many brands. Focusing on one brand makes the historical response patterns more consistent and keeps the project scope manageable.

---

## 2. Used direct customer-to-Verizon reply pairs for historical retrieval

**Decision:** Build the retrieval corpus from customer messages that have a direct VerizonSupport response.

**Why:** These pairs provide actual examples of how the support team responded to customers, which is more useful for grounded reply generation than using unrelated tweets.

---

## 3. Used `created_at` rather than `tweet_id` for chronology

**Decision:** Use `created_at` when reasoning about message order.

**Why:** Tweet IDs are not a reliable representation of chronological order in this dataset. The timestamp provides the actual temporal ordering.

---

## 4. Created a manually labelled golden evaluation set

**Decision:** Create a 200-example hand-labelled golden set.

**Why:** The original dataset does not provide the required intent labels. A manually labelled evaluation set gives us a fixed benchmark for measuring classifier performance.

---

## 5. Defined 11 intents from the Verizon support data

**Decision:** Use 11 practical intents:
- `internet_wifi_issue`
- `service_outage`
- `router_equipment_issue`
- `mobile_phone_issue`
- `tv_channel_issue`
- `billing_payment_issue`
- `account_issue`
- `installation_availability`
- `support_request`
- `complaint_feedback`
- `other`

**Why:** These categories cover the major support themes observed in the sampled Verizon conversations while keeping the classification problem manageable.

---

## 6. Kept an `other` category

**Decision:** Include `other` rather than forcing every message into a specific support category.

**Why:** The dataset contains many short, contextual, conversational, or ambiguous messages that do not provide enough evidence for a specific intent.

---

## 7. Removed golden examples from the retrieval corpus

**Decision:** Historical retrieval does not use the golden examples used for evaluation.

**Why:** This reduces direct evaluation leakage. Otherwise, the retriever could return the exact evaluation message and make retrieval quality appear artificially strong.

---

## 8. Cleaned Twitter-specific noise before retrieval

**Decision:** Remove URLs, Twitter usernames, certain Twitter markers, and excessive whitespace from retrieval text.

**Why:** These artifacts generally do not describe the customer's support problem and can distort lexical similarity.

---

## 9. Used word TF-IDF + Logistic Regression as a simple baseline

**Decision:** Use word-level TF-IDF with unigrams/bigrams and Logistic Regression as the simple classifier baseline.

**Why:** It is a lightweight and reproducible classical NLP baseline that provides a meaningful comparison against the final classifier.

---

## 10. Selected character TF-IDF + Logistic Regression as the final classifier

**Decision:** Use character n-grams (`3–5`) with Logistic Regression for the final intent classifier.

**Why:** Customer support tweets contain spelling variations, abbreviations, fragments, usernames, product terms, and noisy text. Character n-grams are better suited to capturing these patterns than word features alone.

---

## 11. Used class balancing during classifier training

**Decision:** Set `class_weight="balanced"` in Logistic Regression.

**Why:** The 11 intent classes are unevenly represented in the golden set. Class balancing reduces the tendency of the classifier to favour the more frequent classes.

---

## 12. Used TF-IDF cosine similarity for historical retrieval

**Decision:** Retrieve historical cases using TF-IDF vectors and cosine similarity.

**Why:** This provides a simple, transparent and reproducible retrieval mechanism that can run locally without requiring a separate embedding service or vector database.

---

## 13. Used historical responses as fallback replies

**Decision:** If the LLM reply generation is unavailable or fails, return the highest-ranked historical agent response.

**Why:** The fallback keeps the prototype functional and ensures that the response remains grounded in an actual Verizon support example rather than generating unsupported content.

---

## 14. Added rule-based escalation

**Decision:** Use explicit escalation rules based on intent, retrieval similarity, message length, and unresolved/recurring issue language.

**Why:** Some cases should not be automatically handled, especially account/payment issues, low-evidence cases, very short messages, and recurring or unresolved problems.

---

## 15. Evaluated reply quality using a human audit and LLM judge

**Decision:** Evaluate 20 generated replies using both human scoring and an LLM judge.

**Why:** Automated classifier accuracy alone does not tell us whether a drafted support reply is relevant, grounded, helpful, and safe. Comparing human and LLM judgments also provides evidence about the reliability of the automated evaluation.

---

## Summary

The main design principle was to prefer simple, reproducible methods with explicit evidence from historical Verizon support conversations. The system deliberately avoids claiming that retrieval similarity or classifier confidence alone guarantees a correct or safe support response.