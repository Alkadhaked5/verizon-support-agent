import re


def clean_tweet(text):
    """
    Clean Twitter-specific noise from a customer message.
    """
    text = str(text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remove Twitter usernames
    text = re.sub(r"@\w+", "", text)

    # Remove dataset-specific markers such as ^US
    text = re.sub(r"\^[A-Z]{2,5}\b", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

def prepare_historical_pairs(historical_pairs):
    """
    Clean customer messages and agent replies for retrieval.
    """
    pairs = historical_pairs.copy()

    pairs["customer_text_clean"] = pairs["customer_text"].apply(
        clean_tweet
    )

    pairs["agent_reply_clean"] = pairs["agent_reply"].apply(
        clean_tweet
    )

    pairs = pairs.dropna(
        subset=["customer_text_clean", "agent_reply_clean"]
    )

    return pairs

