import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, classification_report

from .classifier import build_intent_classifier


def evaluate_majority_baseline(golden_set):
    """
    Evaluate a trivial majority-class baseline.
    """

    y = golden_set["intent"]

    majority_class = y.value_counts().idxmax()
    predictions = [majority_class] * len(y)

    return {
        "model": "Majority baseline",
        "accuracy": accuracy_score(y, predictions),
        "macro_f1": f1_score(
            y,
            predictions,
            average="macro",
            zero_division=0
        ),
        "weighted_f1": f1_score(
            y,
            predictions,
            average="weighted",
            zero_division=0
        )
    }


def evaluate_word_tfidf_baseline(golden_set, cv):
    """
    Evaluate word-level TF-IDF + Logistic Regression.
    """

    X = golden_set["text"]
    y = golden_set["intent"]

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            )
        )
    ])

    predictions = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict"
    )

    return {
        "model": "Word TF-IDF + Logistic Regression",
        "accuracy": accuracy_score(y, predictions),
        "macro_f1": f1_score(
            y,
            predictions,
            average="macro",
            zero_division=0
        ),
        "weighted_f1": f1_score(
            y,
            predictions,
            average="weighted",
            zero_division=0
        )
    }


def evaluate_final_classifier(golden_set, cv):
    """
    Evaluate the final character-level TF-IDF classifier.
    """

    X = golden_set["text"]
    y = golden_set["intent"]

    model = build_intent_classifier()

    predictions = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict"
    )

    print("\nFinal Classifier Report")
    print("-----------------------")

    print(
        classification_report(
            y,
            predictions,
            zero_division=0
        )
    )

    return {
        "model": "Character TF-IDF + Logistic Regression",
        "accuracy": accuracy_score(y, predictions),
        "macro_f1": f1_score(
            y,
            predictions,
            average="macro",
            zero_division=0
        ),
        "weighted_f1": f1_score(
            y,
            predictions,
            average="weighted",
            zero_division=0
        )
    }


def evaluate_intent_classifier(golden_set):
    """
    Run all intent classification baselines and compare results.
    """

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    results = [
        evaluate_majority_baseline(golden_set),
        evaluate_word_tfidf_baseline(golden_set, cv),
        evaluate_final_classifier(golden_set, cv)
    ]

    results_df = pd.DataFrame(results)

    print("\nIntent Classification Comparison")
    print("--------------------------------")
    print(
        results_df[
            [
                "model",
                "accuracy",
                "macro_f1",
                "weighted_f1"
            ]
        ].to_string(index=False)
    )

    return results_df
def show_misclassified_examples(golden_set):
    X = golden_set["text"]
    y = golden_set["intent"]

    model = build_intent_classifier()

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    predictions = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict"
    )

    errors = golden_set.copy()
    errors["predicted_intent"] = predictions

    errors = errors[
        errors["intent"] != errors["predicted_intent"]
    ]

    print("\nMisclassified Examples")
    print("----------------------")

    print(
        errors[
            [
                "tweet_id",
                "text",
                "intent",
                "predicted_intent"
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    golden_set = pd.read_csv("golden_set.csv")

    evaluate_intent_classifier(golden_set)
    show_misclassified_examples(golden_set)