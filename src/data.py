import pandas as pd


DATA_PATH = "twcs.csv"


def load_dataset():
    """Load the required columns from the Twitter customer-support dataset."""
    return pd.read_csv(
        DATA_PATH,
        usecols=[
            "tweet_id",
            "author_id",
            "inbound",
            "created_at",
            "text",
            "response_tweet_id",
            "in_response_to_tweet_id",
        ],
    )


def get_brand_data(df, brand="VerizonSupport"):
    """Return tweets belonging to the selected support brand."""
    return df[df["author_id"] == brand].copy()


def get_customer_messages(df, brand_tweets):
    """Return inbound customer messages that directly reply to brand tweets."""
    brand_tweet_ids = set(brand_tweets["tweet_id"])

    return df[
        (df["inbound"] == True)
        & (df["in_response_to_tweet_id"].isin(brand_tweet_ids))
    ].copy()

def build_historical_pairs(customer_messages, brand_tweets):
    """
    Build direct customer -> brand-agent reply pairs.
    """

    agent_replies = brand_tweets[
        brand_tweets["in_response_to_tweet_id"].isin(
            customer_messages["tweet_id"]
        )
    ].copy()

    pairs = customer_messages.merge(
        agent_replies[
            ["tweet_id", "in_response_to_tweet_id", "text"]
        ],
        left_on="tweet_id",
        right_on="in_response_to_tweet_id",
        suffixes=("_customer", "_agent")
    )

    historical_pairs = pairs[
        [
            "tweet_id_customer",
            "text_customer",
            "text_agent"
        ]
    ].copy()

    historical_pairs.columns = [
        "customer_tweet_id",
        "customer_text",
        "agent_reply"
    ]

    historical_pairs = historical_pairs.dropna(
        subset=["customer_text", "agent_reply"]
    )

    return historical_pairs
