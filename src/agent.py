from .classifier import predict_intent
from .retrieval import retrieve_similar_cases
from .reply import generate_grounded_reply
from .escalation import decide_handling


def support_agent(
    customer_message,
    intent_model,
    historical_pairs,
    retriever_vectorizer,
    historical_matrix,
    gemini_client=None,
    top_k=3,
    conversation_history=None,
):
    """
    Run the complete customer-support pipeline.

    conversation_history:
        List of previous conversation messages.
        Example:
        [
            {"role": "user", "content": "My internet is not working"},
            {"role": "assistant", "content": "Are all devices affected?"}
        ]
    """

    if conversation_history is None:
        conversation_history = []

    # 1. Build context from previous conversation
    conversation_context = ""

    for message in conversation_history:
        role = message.get("role", "")
        content = message.get("content", "")

        if role and content:
            conversation_context += f"{role}: {content}\n"

    # Use current message + previous context for conversational understanding
    if conversation_context:
        classifier_input = (
            f"Previous conversation:\n"
            f"{conversation_context}\n"
            f"Current customer message:\n"
            f"{customer_message}"
        )
    else:
        classifier_input = customer_message

    # 2. Classify intent
    intent, intent_confidence = predict_intent(
        intent_model,
        classifier_input
    )

    # 3. Retrieve similar historical cases
    retrieved_cases = retrieve_similar_cases(
        classifier_input,
        historical_pairs,
        retriever_vectorizer,
        historical_matrix,
        top_k=top_k
    )

    best_similarity = float(
        retrieved_cases.iloc[0]["similarity"]
    )

    # 4. Decide AUTO_HANDLE vs ESCALATE
    decision, reason = decide_handling(
        customer_message,
        intent,
        best_similarity
    )

    # 5. Generate grounded reply
    draft_reply, reply_source = generate_grounded_reply(
        customer_message,
        retrieved_cases,
        client=gemini_client,
        conversation_history=conversation_history
    )

    return {
        "customer_message": customer_message,
        "intent": intent,
        "intent_confidence": round(intent_confidence, 3),
        "retrieved_cases": retrieved_cases,
        "retrieval_similarity": round(best_similarity, 3),
        "draft_reply": draft_reply,
        "reply_source": reply_source,
        "decision": decision,
        "reason": reason,
        "conversation_history": conversation_history,
    }