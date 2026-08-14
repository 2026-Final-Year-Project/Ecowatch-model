# Illegal Mining Detection using U-net

This project fine-tunes a U-net segmentation model on the SmallMinesDS dataset to detect illegal mining sites from satellite imagery.

## Project Structure

- notebooks/
- src/
- configs/
- assets/

## Client-facing inference

`best_model.pt` is the checkpoint produced by `03_unet_training_evaluation.ipynb`. It is intentionally ignored by Git and loaded by the FastAPI service in `app.py`.

1. Create an Earth-Engine-enabled Google Cloud project and set `EE_PROJECT` (copy `.env.example` to `.env`). Authenticate locally with `earthengine authenticate`, or use Application Default Credentials when deployed to Google Cloud.
2. Install the service dependencies: `python -m pip install -r requirements.txt`.
3. Run `uvicorn app:app --reload --port 8000`.
4. In EcoWatch's `backend/.env`, set `MODEL_API_URL=http://localhost:8000`, then run its Express server.
5. Click a location in the live map. The UI calls `/api/predictions/coordinate`; Express proxies the request to this service, which fetches a Sentinel-2 tile and returns the U-Net result.

The checkpoint was trained on a 13-band, 128×128 Sentinel-2 input. It cannot make a scientifically valid prediction from the RGB map display tiles alone.
