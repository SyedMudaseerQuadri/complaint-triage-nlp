"""FastAPI service: POST /predict routes a complaint narrative to a product team."""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from api.predictor import load_predictor
from src.preprocess import clean_text


class PredictRequest(BaseModel):
    text: str = Field(..., description="Free-text consumer complaint narrative.")


class PredictResponse(BaseModel):
    label: str
    probabilities: dict[str, float]
    model_version: str


def create_app(predictor=None) -> FastAPI:
    """Build the app. Pass a predictor to skip loading from disk (used by tests)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor = predictor if predictor is not None else load_predictor()
        yield

    app = FastAPI(title="Complaint Triage", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        p = app.state.predictor
        return {"status": "ok", "model_version": p.model_version, "n_classes": len(p.labels)}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        if len(clean_text(req.text).split()) < 3:
            raise HTTPException(status_code=422, detail="text must contain at least 3 words")
        p = app.state.predictor
        probs = p.predict_proba(req.text)
        probs_by_label = {lab: round(float(x), 6) for lab, x in zip(p.labels, probs, strict=True)}
        label = max(probs_by_label, key=probs_by_label.get)
        return PredictResponse(label=label, probabilities=probs_by_label, model_version=p.model_version)

    return app


app = create_app()
