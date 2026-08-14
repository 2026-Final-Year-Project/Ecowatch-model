"""Create verified demonstration-site GeoJSON from positive held-out masks.

Run this only against the test split used by the notebook.  It never invents
coordinates: every exported point is the centroid of a labelled mining region.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform as reproject


def components(mask):
    visited = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    for row, column in np.argwhere(mask):
        row, column = int(row), int(column)
        if visited[row, column]:
            continue
        group, stack = [], [(row, column)]
        visited[row, column] = True
        while stack:
            current_row, current_column = stack.pop()
            group.append((current_row, current_column))
            for row_delta in (-1, 0, 1):
                for column_delta in (-1, 0, 1):
                    next_row, next_column = current_row + row_delta, current_column + column_delta
                    if (row_delta or column_delta) and 0 <= next_row < height and 0 <= next_column < width and mask[next_row, next_column] and not visited[next_row, next_column]:
                        visited[next_row, next_column] = True
                        stack.append((next_row, next_column))
        yield group


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset-root", type=Path, help="Local SmallMinesDS root used to replace CSV paths created in Colab.")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--min-pixels", type=int, default=25)
    arguments = parser.parse_args()

    candidates = []
    with arguments.test_csv.open(newline="") as file:
        for record in csv.DictReader(file):
            mask_path = Path(record["mask_path"])
            image_path = Path(record["image_path"])
            if arguments.dataset_root:
                try:
                    mask_suffix = mask_path.parts[mask_path.parts.index("SmallMinesDS") + 1:]
                    image_suffix = image_path.parts[image_path.parts.index("SmallMinesDS") + 1:]
                except ValueError as error:
                    raise ValueError("CSV paths must contain a SmallMinesDS directory.") from error
                mask_path = arguments.dataset_root.joinpath(*mask_suffix)
                image_path = arguments.dataset_root.joinpath(*image_suffix)
            if not mask_path.exists():
                raise FileNotFoundError(f"Mask missing: {mask_path}. Run this where the dataset is mounted.")
            with rasterio.open(mask_path) as source:
                mask = source.read(1) > 0
                transform, crs = source.transform, source.crs
                for group in components(mask):
                    if len(group) < arguments.min_pixels:
                        continue
                    rows, columns = zip(*group)
                    longitude, latitude = rasterio.transform.xy(transform, float(np.mean(rows)), float(np.mean(columns)))
                    if crs and crs.to_string() != "EPSG:4326":
                        longitude, latitude = reproject(crs, "EPSG:4326", [longitude], [latitude])
                        longitude, latitude = longitude[0], latitude[0]
                    candidates.append((len(group), latitude, longitude, image_path.name, mask_path.name, mask_path.parent.parent.name, str(crs)))

    candidates.sort(reverse=True)
    features = []
    for index, (pixels, latitude, longitude, image_name, mask_name, year, crs) in enumerate(candidates[:arguments.limit], start=1):
        features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [longitude, latitude]}, "properties": {
            "id": f"smallmines-test-{index:02d}", "name": f"Validated demonstration site {index}",
            "source": "SmallMinesDS held-out test mask", "label": "Labelled artisanal/small-scale mining region",
            "component_pixels": pixels, "area_m2": pixels * 100, "color": "#dc2626",
            "image_file": image_name, "mask_file": mask_name, "year": year, "crs": crs,
        }})
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2))
    print(f"Wrote {len(features)} validated demonstration sites to {arguments.output}")


if __name__ == "__main__":
    main()
