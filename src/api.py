from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from .data import (
    load_dataset,
    get_brand_data,
    get_customer_messages,
    build_historical_pairs,
)
from .classifier import train_intent_classifier
from .retrieval import build_retriever
from .agent import support_agent
from .reply import create_gemini_client


app = FastAPI(
    title="Verizon AI Support Agent",
    description="AI customer support agent using historical Verizon support data.",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Request models
# -----------------------------

class ConversationMessage(BaseModel):
    role: str
    content: str


class CustomerRequest(BaseModel):
    message: str
    conversation_history: list[ConversationMessage] = []


# -----------------------------
# Load and prepare pipeline
# -----------------------------

df = load_dataset()

brand_data = get_brand_data(
    df,
    "VerizonSupport"
)

customer_messages = get_customer_messages(
    df,
    brand_data
)

historical_pairs = build_historical_pairs(
    customer_messages,
    brand_data
)

golden_set = __import__("pandas").read_csv(
    "golden_set.csv"
)

intent_model = train_intent_classifier(
    golden_set["text"],
    golden_set["intent"]
)

historical_pairs, retriever_vectorizer, historical_matrix = build_retriever(
    historical_pairs
)
gemini_client = create_gemini_client()

# -----------------------------
# Routes
# -----------------------------

@app.get("/")
def root():
    return {
        "message": "Verizon AI Support Agent is running"
    }


@app.post("/predict")
def predict(request: CustomerRequest):

    result = support_agent(
        request.message,
        intent_model,
        historical_pairs,
        retriever_vectorizer,
        historical_matrix,
        gemini_client=gemini_client,
        top_k=3,
        conversation_history=[
            {
                "role": message.role,
                "content": message.content
            }
            for message in request.conversation_history
        ]
    )

    retrieved_cases = result["retrieved_cases"]

    return {
        "customer_message": result["customer_message"],
        "intent": result["intent"],
        "intent_confidence": result["intent_confidence"],
        "retrieval_similarity": result["retrieval_similarity"],
        "decision": result["decision"],
        "reason": result["reason"],
        "draft_reply": result["draft_reply"],
        "reply_source": result["reply_source"],
        "retrieved_cases": retrieved_cases.to_dict(
            orient="records"
        ),
        "conversation_history": result["conversation_history"],
    }