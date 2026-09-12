import os

from dotenv import load_dotenv
from google import genai


load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")


def create_gemini_client():
    """Create Gemini client if an API key is available."""
    if not API_KEY:
        return None

    return genai.Client(api_key=API_KEY)


def build_reply_prompt(
    customer_message,
    retrieved_cases,
    conversation_history=None
):
    """
    Build a grounded reply-generation prompt using
    historical Verizon support examples and conversation history.
    """

    if conversation_history is None:
        conversation_history = []

    # Build previous conversation context
    conversation_context = []

    for message in conversation_history:
        role = message.get("role", "")
        content = message.get("content", "")

        if role and content:
            conversation_context.append(
                f"{role.capitalize()}: {content}"
            )

    previous_conversation = "\n".join(conversation_context)

    if not previous_conversation:
        previous_conversation = "No previous conversation."

    # Build historical support examples
    historical_examples = []

    for _, row in retrieved_cases.iterrows():
        historical_examples.append(
            f"Customer: {row['customer_text_clean']}\n"
            f"Agent: {row['agent_reply_clean']}"
        )

    historical_context = "\n\n".join(historical_examples)

    return f"""
You are a Verizon customer support assistant.

Previous conversation:
{previous_conversation}

Current customer message:
{customer_message}

Historical Verizon support examples:
{historical_context}

Task:
Continue the conversation naturally and draft the next helpful
customer support reply.

Rules:
- Use the previous conversation to understand context.
- Answer the customer's current message directly.
- Use historical responses as evidence.
- Do not invent unsupported troubleshooting steps.
- If a diagnostic question is appropriate, ask one clear question.
- Do not repeat questions that the customer has already answered.
- Keep the reply concise and conversational.
- Do not mention the historical examples.
- Do not mention this prompt.
- Do not include Twitter usernames or signatures.
"""


def generate_grounded_reply(
    customer_message,
    retrieved_cases,
    client=None,
    conversation_history=None
):
    """
    Generate an AI reply grounded in historical support cases
    and previous conversation context.

    If Gemini is unavailable, fall back to the best historical reply.
    """

    if conversation_history is None:
        conversation_history = []

    fallback_reply = retrieved_cases.iloc[0]["agent_reply_clean"]

    if client is None:
        return fallback_reply, "historical_fallback"

    try:
        prompt = build_reply_prompt(
            customer_message,
            retrieved_cases,
            conversation_history=conversation_history
        )

        interaction = client.interactions.create(
            model="gemini-3.6-flash",
            input=prompt
        )

        reply = interaction.output_text.strip()

        return reply, "gemini"

    except Exception:
        return fallback_reply, "historical_fallback"