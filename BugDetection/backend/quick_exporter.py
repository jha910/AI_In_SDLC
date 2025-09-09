from fastapi import FastAPI
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import random

app = FastAPI()
ALERTS = Counter("alert_events_total", "Fake alert events")

@app.get("/metrics")
def metrics():
    # bump the counter a little so you’ll see movement
    if random.random() < 0.5:
        ALERTS.inc()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
