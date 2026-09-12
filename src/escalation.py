def decide_handling(customer_message, intent, retrieval_similarity):
    """
    Decide whether a customer message should be auto-handled
    or escalated to a human.
    """

    text = customer_message.lower().strip()

    # 1. Account/payment issues require secure human handling.
    if intent in ["billing_payment_issue", "account_issue"]:
        return (
            "ESCALATE",
            "Account or payment-related issue may require secure human handling."
        )

    # 2. Weak historical evidence.
    if retrieval_similarity < 0.40:
        return (
            "ESCALATE",
            "No sufficiently similar historical support case was found."
        )

    # 3. Very short/context-dependent messages.
    words = text.split()

    if len(words) <= 4:
        return (
            "ESCALATE",
            "The message is too short or context-dependent for safe automated handling."
        )

    # 4. Explicit unresolved or recurring problems.
    escalation_phrases = [
        "every week",
        "all the time",
        "keeps cutting",
        "cuts out",
        "tried everything",
        "still no service",
        "no service",
        "return my money",
        "rerouted",
        "not working"
    ]

    if any(phrase in text for phrase in escalation_phrases):
        return (
            "ESCALATE",
            "The issue appears unresolved or recurring and may require human investigation."
        )

    # 5. Otherwise, historical evidence supports automated handling.
    return (
        "AUTO_HANDLE",
        "The issue appears suitable for automated handling based on historical support evidence."
    )