"""Earth Engine tile retrieval and U-Net inference."""
import os
import math
from datetime import datetime, timedelta, timezone
import numpy as np
import torch
from src.models.unet import UNet

MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.pt")
THRESHOLD = float(os.getenv("MODEL_THRESHOLD", "0.5"))
TILE_SIZE = 128
PIXEL_AREA_M2 = 100
CONFIRMED_MINING_PIXELS = 25  # 0.25 hectares at 10 m resolution
HIGH_CONFIDENCE_PIXELS = 100  # 1 hectare at 10 m resolution


def connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 8-connected positive-pixel components without an extra SciPy dependency."""
    visited = np.zeros(mask.shape, dtype=bool)
    components = []
    height, width = mask.shape
    for row, column in np.argwhere(mask):
        row, column = int(row), int(column)
        if visited[row, column]:
            continue
        component, stack = [], [(row, column)]
        visited[row, column] = True
        while stack:
            current_row, current_column = stack.pop()
            component.append((current_row, current_column))
            for row_delta in (-1, 0, 1):
                for column_delta in (-1, 0, 1):
                    next_row, next_column = current_row + row_delta, current_column + column_delta
                    if (row_delta or column_delta) and 0 <= next_row < height and 0 <= next_column < width and mask[next_row, next_column] and not visited[next_row, next_column]:
                        visited[next_row, next_column] = True
                        stack.append((next_row, next_column))
        components.append(component)
    return components


class MiningPredictor:
    def __init__(self):
        self.model = UNet()
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.version = f"unet-epoch-{checkpoint.get('epoch', 'unknown')}"

    def predict(self, bands: np.ndarray):
        if bands.shape != (13, TILE_SIZE, TILE_SIZE):
            raise ValueError(f"Expected 13x{TILE_SIZE}x{TILE_SIZE} bands, received {bands.shape}.")
        low, high = bands.min(), bands.max()
        normalized = (bands - low) / (high - low) if high > low else bands
        with torch.inference_mode():
            probabilities = torch.sigmoid(self.model(torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0)))[0, 0].numpy()
        raw_mask = probabilities >= THRESHOLD
        components = connected_components(raw_mask)
        component_sizes = [len(component) for component in components]
        largest_component_pixels = max(component_sizes, default=0)
        confirmed_mask = np.zeros(raw_mask.shape, dtype=np.uint8)
        for component in components:
            if len(component) >= CONFIRMED_MINING_PIXELS:
                rows, columns = zip(*component)
                confirmed_mask[rows, columns] = 1
        if largest_component_pixels >= HIGH_CONFIDENCE_PIXELS:
            detection_level = "high_confidence"
        elif largest_component_pixels >= CONFIRMED_MINING_PIXELS:
            detection_level = "detected"
        else:
            detection_level = "none"
        confirmed_pixels = int(confirmed_mask.sum())
        return {
            "probability": float(probabilities.mean()),
            "mining_detected": detection_level in {"detected", "high_confidence"},
            "detection_level": detection_level,
            "mining_fraction": confirmed_pixels / raw_mask.size,
            "affected_area_m2": confirmed_pixels * PIXEL_AREA_M2,
            "largest_component_pixels": largest_component_pixels,
            "mask": confirmed_mask.tolist(),
            "threshold": THRESHOLD, "model_version": self.version,
        }


def earth_engine_tile(latitude: float, longitude: float, date_start: str | None = None, date_end: str | None = None) -> np.ndarray:
    """Fetch a 128-pixel, 13-band Sentinel-2 tile from Google Earth Engine."""
    import ee
    project = os.environ["EE_PROJECT"]
    ee.Initialize(project=project)
    point = ee.Geometry.Point([longitude, latitude])
    # Use Level-1C harmonized imagery because it includes all 13 Sentinel-2 bands,
    # including B10. The SR collection omits B10 and cannot match this checkpoint.
    collection = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(point)
    if date_start and date_end:
        collection = collection.filterDate(date_start, date_end)
    # Avoid a brittle client-date default. With no dates supplied, select the newest
    # available scene; a requested date range is respected when explicitly provided.
    if collection.size().getInfo() == 0:
        raise ValueError("No Sentinel-2 image is available for this location and requested date range.")
    image = collection.sort("system:time_start", False).first()
    # The model was trained on all 13 bands in this exact Earth Engine order.
    bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
    # The REST computePixels API accepts a grid, not region/dimensions. Clip the
    # expression first, then specify a 1.28 km (128 pixels × 10 m) Web-Mercator tile.
    half_width_m = TILE_SIZE * 10 / 2
    earth_radius_m = 6378137.0
    x = earth_radius_m * math.radians(longitude)
    latitude_radians = math.radians(max(min(latitude, 85.05112878), -85.05112878))
    y = earth_radius_m * math.log(math.tan(math.pi / 4 + latitude_radians / 2))
    region = point.buffer(half_width_m).bounds()
    grid = {
        "crsCode": "EPSG:3857",
        "dimensions": {"width": TILE_SIZE, "height": TILE_SIZE},
        "affineTransform": {
            "scaleX": 10, "shearX": 0, "translateX": x - half_width_m,
            "shearY": 0, "scaleY": -10, "translateY": y + half_width_m,
        },
    }
    payload = ee.data.computePixels({
        "expression": image.select(bands).resample("bilinear").clip(region),
        "fileFormat": "NUMPY_NDARRAY", "grid": grid,
    })
    # Current Earth Engine Python clients return an in-memory structured ndarray
    # for NUMPY_NDARRAY; older clients returned bytes that required np.load.
    structured = payload
    return np.stack([structured[band] for band in bands])


def earth_engine_preview(latitude: float, longitude: float, date_start: str | None = None, date_end: str | None = None) -> dict:
    """Return a contrast-stretched Sentinel-2 natural-colour thumbnail for one period."""
    import ee
    project = os.environ["EE_PROJECT"]
    ee.Initialize(project=project)
    point = ee.Geometry.Point([longitude, latitude])
    region = point.buffer(TILE_SIZE * 10 / 2).bounds()
    collection = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(point)
    if date_start and date_end:
        collection = collection.filterDate(date_start, date_end)
    else:
        # A "latest" comparison should be recent but not necessarily the most
        # recently ingested (and often cloudiest) scene. Limit the search to the
        # past year, then select the least-cloudy available scene.
        today = datetime.now(timezone.utc)
        recent = collection.filterDate(
            (today - timedelta(days=365)).strftime("%Y-%m-%d"),
            (today + timedelta(days=1)).strftime("%Y-%m-%d"),
        )
        if recent.size().getInfo() > 0:
            collection = recent
    if collection.size().getInfo() == 0:
        raise ValueError("No Sentinel-2 image is available for this location and requested date range.")
    image = collection.sort("CLOUDY_PIXEL_PERCENTAGE").first()
    rgb = image.select(["B4", "B3", "B2"])
    # Fixed visualisation bounds made dark historic scenes nearly black and
    # bright recent scenes clip to white. Stretch every preview independently
    # using local 2nd/98th percentiles, matching the model dataset's approach.
    stats = rgb.reduceRegion(
        reducer=ee.Reducer.percentile([2, 98]), geometry=region,
        scale=10, maxPixels=1_000_000, bestEffort=True,
    ).getInfo() or {}
    minimum = [stats.get(f"{band}_p2", 0) for band in ("B4", "B3", "B2")]
    maximum = [stats.get(f"{band}_p98", 3000) for band in ("B4", "B3", "B2")]
    # Avoid an invalid visualisation range for flat or sparsely sampled tiles.
    maximum = [high if high > low else low + 1 for low, high in zip(minimum, maximum)]
    visual = rgb.visualize(min=minimum, max=maximum, gamma=1.0).clip(region)
    return {
        "image_url": visual.getThumbURL({"region": region, "dimensions": 512, "format": "png"}),
        "captured_at": ee.Date(image.get("system:time_start")).format("YYYY-MM-dd").getInfo(),
    }
