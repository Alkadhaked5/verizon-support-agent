from .data import (
    load_dataset,
    get_brand_data,
    get_customer_messages,
    build_historical_pairs
)

from .data_processing import prepare_historical_pairs
from .classifier import train_intent_classifier
from .retrieval import build_retriever
from .agent import support_agent

def main():
    # 1. Load dataset
    df = load_dataset()

    # 2. Select Verizon support data
    brand_data = get_brand_data(df, "VerizonSupport")

    # 3. Get customer messages
    customer_messages = get_customer_messages(df, brand_data)

    # 4. Build historical customer-agent pairs
    historical_pairs = build_historical_pairs(
        customer_messages,
        brand_data
    )

    # 5. Prepare historical pairs
    historical_pairs = prepare_historical_pairs(
        historical_pairs
    )

    # 6. Load golden set
    golden_set = __import__("pandas").read_csv("golden_set.csv")

    # 7. Train intent classifier
    intent_model = train_intent_classifier(
        golden_set["text"],
        golden_set["intent"]
    )

    # 8. Build retrieval index
    historical_pairs, retriever_vectorizer, historical_matrix = build_retriever(
        historical_pairs
    )

    # 9. Run support agent
    customer_message = "My Fios internet keeps disconnecting"

    result = support_agent(
        customer_message,
        intent_model,
        historical_pairs,
        retriever_vectorizer,
        historical_matrix,
        gemini_client=None,
        top_k=3
    )

    # 10. Display result
    print("\nCustomer:", result["customer_message"])
    print("Intent:", result["intent"])
    print("Intent confidence:", result["intent_confidence"])
    print("Retrieval similarity:", result["retrieval_similarity"])
    print("Decision:", result["decision"])
    print("Reason:", result["reason"])
    print("Reply source:", result["reply_source"])
    print("Draft reply:", result["draft_reply"])


if __name__ == "__main__":
    main()