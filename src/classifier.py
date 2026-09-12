from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_intent_classifier():
    """
    Build the character-level TF-IDF + Logistic Regression
    intent classifier used by the support agent.
    """
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            )
        )
    ])


def train_intent_classifier(texts, intents):
    """
    Train the intent classifier on labelled examples.
    """
    model = build_intent_classifier()
    model.fit(texts, intents)
    return model


def predict_intent(model, message):
    """
    Predict intent and return its confidence.
    """
    probabilities = model.predict_proba([message])[0]

    predicted_intent = model.classes_[probabilities.argmax()]
    confidence = probabilities.max()

    return predicted_intent, float(confidence)