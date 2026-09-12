from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .data_processing import prepare_historical_pairs


def build_retriever(historical_pairs):
    """
    Clean historical pairs and build a TF-IDF retrieval index.
    """
    historical_pairs = prepare_historical_pairs(historical_pairs)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        stop_words="english"
    )

    matrix = vectorizer.fit_transform(
        historical_pairs["customer_text_clean"]
    )

    return historical_pairs, vectorizer, matrix


def retrieve_similar_cases(
    query,
    historical_pairs,
    vectorizer,
    matrix,
    top_k=3
):
    """
    Retrieve the most similar historical customer-support cases.
    """
    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        matrix
    ).flatten()

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = historical_pairs.iloc[top_indices].copy()
    results["similarity"] = similarities[top_indices]

    return results[
        [
            "customer_text_clean",
            "agent_reply_clean",
            "similarity"
        ]
    ]