from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

# Initialize FastAPI app
app = FastAPI()

# Allow all CORS (so frontend can call backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and tokenizer
model_name = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Prometheus counter
REQUEST_COUNT = Counter("app_requests_total", "Total requests to API")

@app.post("/predict")
async def predict(request: Request):
    REQUEST_COUNT.inc()  # count each request

    data = await request.json()
    code_snippet = data.get("code", "")

    inputs = tokenizer(code_snippet, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1)

    label = predictions.item()
    return {"prediction": "Buggy" if label == 1 else "Clean"}

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
