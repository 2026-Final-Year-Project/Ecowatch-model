"""Verify a reference-candidate GeoJSON layer with the running EcoWatch API.

This is inference-only: it uses the saved checkpoint through the local service,
does not train the model, and writes only locations that EcoWatch itself detects.
Run in small batches to stay within your Earth Engine quota.
"""
import argparse
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def predict(url: str, longitude: float, latitude: float, date_start: str | None, date_end: str | None) -> dict:
    body = json.dumps({
        "latitude": latitude, "longitude": longitude,
        "date_start": date_start, "date_end": date_end,
    }).encode()
    request = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=90) as response:
            return json.loads(response.read())
    except (HTTPError, URLError) as error:
        return {"error": str(error)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-url", default="http://localhost:8000/v1/predict/coordinate")
    parser.add_argument("--limit", type=int, default=25, help="Maximum sites to inspect in this run.")
    parser.add_argument("--offset", type=int, default=0, help="Start at this candidate offset for the next batch.")
    parser.add_argument("--date-start", help="Optional YYYY-MM-DD start date for historical verification.")
    parser.add_argument("--date-end", help="Optional YYYY-MM-DD end date for historical verification.")
    arguments = parser.parse_args()

    collection = json.loads(arguments.input.read_text())
    candidates = collection.get("features", [])[arguments.offset:arguments.offset + arguments.limit]
    confirmed = []
    for number, feature in enumerate(candidates, start=arguments.offset + 1):
        longitude, latitude = feature["geometry"]["coordinates"]
        result = predict(arguments.model_url, longitude, latitude, arguments.date_start, arguments.date_end)
        if result.get("mining_detected"):
            feature = json.loads(json.dumps(feature))
            feature["properties"].update({
                "name": "EcoWatch model-verified mining site",
                "source": "Earthrise reference candidate; verified by EcoWatch deployed model",
                "color": "#dc2626",
                "status": "ecowatch_verified",
                "detection_level": result["detection_level"],
                "affected_area_m2": result["affected_area_m2"],
                "date_start": arguments.date_start,
                "date_end": arguments.date_end,
            })
            confirmed.append(feature)
            print(f"[{number}] confirmed level={result['detection_level']} area={result['affected_area_m2']}m²")
        else:
            if result.get("error"):
                print(f"[{number}] error: {result['error']}")
            else:
                print(
                    f"[{number}] not confirmed level={result.get('detection_level')} "
                    f"mean={result.get('probability', 0) * 100:.2f}% "
                    f"largest={result.get('largest_component_pixels', 0)}px"
                )

    existing = []
    if arguments.output.exists():
        existing = json.loads(arguments.output.read_text()).get("features", [])
    by_id = {feature["properties"].get("id"): feature for feature in existing}
    by_id.update({feature["properties"].get("id"): feature for feature in confirmed})
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps({"type": "FeatureCollection", "features": list(by_id.values())}, indent=2))
    print(f"Added {len(confirmed)} confirmed sites; {len(by_id)} total in {arguments.output}")


if __name__ == "__main__":
    main()
