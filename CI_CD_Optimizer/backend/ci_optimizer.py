from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

model_name = "mrm8488/codebert-base-finetuned-detect-insecure3-code"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

@app.post("/optimize-ci")
async def optimize_ci(request: Request):
    data = await request.json()
    pipeline_code = data.get("pipeline", "")

    inputs = tokenizer(pipeline_code, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1)

    label = predictions.item()
    risk = "High" if label == 1 else "Low"

    return {
        "build_risk": risk,
        "suggestions": [
            "Use parallel stages",
            "Cache dependencies",
            "Add lint step",
            "Run tests in parallel"
        ],
        "optimized_pipeline": {
            "parallel_jobs": True,
            "estimated_time": "4m 30s"
        }
    }
