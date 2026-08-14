from datetime import date
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.ecowatch_inference import MiningPredictor, earth_engine_preview, earth_engine_tile

app = FastAPI(title="EcoWatch Mining Inference")
predictor = MiningPredictor()
logger = logging.getLogger(__name__)

class CoordinateRequest(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    date_start: date | None = None
    date_end: date | None = None

@app.get("/health")
def health(): return {"status": "ok", "model_version": predictor.version}

@app.post("/v1/predict/coordinate")
def predict_at_coordinate(request: CoordinateRequest):
    try:
        start = str(request.date_start) if request.date_start else None
        end = str(request.date_end) if request.date_end else None
        result = predictor.predict(earth_engine_tile(request.latitude, request.longitude, start, end))
        return {**result, "latitude": request.latitude, "longitude": request.longitude, "date_start": start, "date_end": end}
    except (KeyError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        logger.exception("Earth Engine prediction failed")
        raise HTTPException(status_code=502, detail=f"Satellite imagery request failed: {error}") from error


@app.post("/v1/imagery/coordinate")
def imagery_at_coordinate(request: CoordinateRequest):
    try:
        start = str(request.date_start) if request.date_start else None
        end = str(request.date_end) if request.date_end else None
        return {
            **earth_engine_preview(request.latitude, request.longitude, start, end),
            "latitude": request.latitude, "longitude": request.longitude,
            "date_start": start, "date_end": end,
        }
    except (KeyError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        logger.exception("Earth Engine imagery request failed")
        raise HTTPException(status_code=502, detail=f"Satellite imagery request failed: {error}") from error
