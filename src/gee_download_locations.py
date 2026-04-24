import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import ee
import geemap
import matplotlib
import numpy as np
import pandas as pd
import rasterio as rio
from dotenv import load_dotenv

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CSV = ROOT / "locations.csv"
OUT_DIR = ROOT / "data" / "samples_generated"

START_DATE = "2025-05-01"
END_DATE = "2025-08-28"
BUFFER_M = 1280
MAX_CLOUD = 20
SCALE = 10
MIN_LABEL_CONFIDENCE = 0.45

S2_BANDS = [
    "B1", "B2", "B3", "B4", "B5", "B6",
    "B7", "B8", "B8A", "B9", "B11", "B12",
]

CLASS_PROB_BANDS = {
    "greenery": ["trees", "grass", "flooded_vegetation", "crops"],
    "sand": ["bare"],
    "water": ["water"],
    "cement": ["built"],
}

DW_PROB_BANDS = [band for bands in CLASS_PROB_BANDS.values() for band in bands]
COLORS = np.array([
    [0, 0, 0],
    [0, 100, 0],
    [184, 134, 11],
    [0, 0, 139],
    [105, 105, 105],
]) / 255.0

NAME_ALIASES = {
    "Cairo University": "CairoUniv",
    "Iconic Tower, New Administrative Capital": "IconicTower",
}


def sample_name(name):
    if name in NAME_ALIASES:
        return NAME_ALIASES[name]

    words = "".join(c if c.isalnum() else " " for c in name).split()
    return "".join(word[:1].upper() + word[1:] for word in words)


def init_ee():
    load_dotenv()
    project = os.getenv("EE_PROJECT_ID") or os.getenv("EE_PROJECT")

    try:
        ee.Initialize(project=project) if project else ee.Initialize()
    except Exception as e:
        raise SystemExit("Earth Engine is not ready. Run `earthengine authenticate` first.") from e


def mask_clouds(image):
    scl = image.select("SCL")
    clear = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return image.updateMask(clear)


def sentinel2(region, start, end, max_cloud):
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
        .map(mask_clouds)
    )


def dynamic_world(region, start, end):
    return ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(region).filterDate(start, end)


def confident_mask(dw, region):
    probs = dw.select(DW_PROB_BANDS).mean().clip(region)
    scores = ee.Image.cat(*[
        probs.select(bands).reduce(ee.Reducer.sum()).rename(name)
        for name, bands in CLASS_PROB_BANDS.items()
    ])

    best_score = scores.reduce(ee.Reducer.max())
    mask = scores.toArray().arrayArgmax().arrayGet([0]).add(1).rename("macro_class")
    return mask.where(best_score.lt(MIN_LABEL_CONFIDENCE), 0).toUint8()


def build_image(region, start, end, max_cloud):
    s2 = sentinel2(region, start, end, max_cloud)
    dw = dynamic_world(region, start, end)

    spectral = s2.select(S2_BANDS).median().clip(region)
    image = spectral.addBands(confident_mask(dw, region))

    return image, s2.size().getInfo(), dw.size().getInfo()


def export_image(image, path, region):
    geemap.ee_export_image(
        image,
        str(path),
        crs="EPSG:4326",
        region=region,
        scale=SCALE,
        file_per_band=False,
        verbose=False,
    )

    if not path.exists():
        raise RuntimeError(f"export failed: {path.name} was not created")


def split_stack(stack_path, spectral_path, mask_path):
    with rio.open(stack_path) as src:
        data = src.read()
        profile = src.profile

    if data.shape[0] != len(S2_BANDS) + 1:
        raise RuntimeError(f"expected {len(S2_BANDS) + 1} bands, found {data.shape[0]}")

    spectral_profile = {**profile, "count": len(S2_BANDS), "dtype": "float64", "nodata": None}
    mask_profile = {**profile, "count": 1, "dtype": "uint8", "nodata": 0}

    with rio.open(spectral_path, "w", **spectral_profile) as dst:
        dst.write(data[:len(S2_BANDS)].astype("float64"))

    with rio.open(mask_path, "w", **mask_profile) as dst:
        dst.write(data[len(S2_BANDS):].astype("uint8"))

    stack_path.unlink(missing_ok=True)


def rgb_preview(spectral_path):
    with rio.open(spectral_path) as src:
        rgb = np.moveaxis(src.read([4, 3, 2]).astype("float32"), 0, -1)

    low = np.percentile(rgb, 2, axis=(0, 1), keepdims=True)
    high = np.percentile(rgb, 98, axis=(0, 1), keepdims=True)
    return np.clip((rgb - low) / (high - low + 1e-6), 0, 1)


def save_preview(spectral_path, mask_path, preview_path):
    with rio.open(mask_path) as src:
        mask = src.read(1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(rgb_preview(spectral_path))
    axes[0].set_title("RGB")
    axes[1].imshow(mask, cmap=ListedColormap(COLORS), vmin=0, vmax=4, interpolation="nearest")
    axes[1].set_title("Mask")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(preview_path, dpi=122, facecolor="white")
    plt.close(fig)


def download_location(row, start, end, buffer_m, max_cloud):
    name = sample_name(row.name)
    folder = OUT_DIR / name
    folder.mkdir(parents=True, exist_ok=True)

    spectral_path = folder / f"{name}_Spectral.tif"
    mask_path = folder / f"{name}_Mask.tif"
    preview_path = folder / f"{name}_viz.png"
    stack_path = folder / f".{name}_stack.tif"

    point = ee.Geometry.Point([row.longitude, row.latitude])
    region = point.buffer(buffer_m).bounds()
    image, s2_count, dw_count = build_image(region, start, end, max_cloud)

    if not s2_count:
        print(f"skip {name}: no Sentinel-2 images")
        return "no_s2"
    if not dw_count:
        print(f"skip {name}: no Dynamic World images")
        return "no_dw"

    export_image(image, stack_path, region)
    split_stack(stack_path, spectral_path, mask_path)
    save_preview(spectral_path, mask_path, preview_path)
    print(f"saved {name}")
    return "saved"


def print_run_summary(stats, args):
    total = sum(stats.values())
    summary = ", ".join(f"{name}={count}" for name, count in stats.items())

    print("\nDiscussion")
    print(
        f"Processed {total} locations from {args.start} to {args.end} "
        f"with buffer={args.buffer} m, scale={SCALE} m, max_cloud={args.max_cloud}% "
        f"and label_confidence>={MIN_LABEL_CONFIDENCE:.2f}."
    )
    print(f"Outcome: {summary}.")
    print(
        "This setup favors cleaner labels by sending uncertain pixels to class 0, "
        "so the training notebooks can ignore ambiguous boundaries instead of learning noise."
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=END_DATE)
    parser.add_argument("--buffer", type=int, default=BUFFER_M)
    parser.add_argument("--max-cloud", type=int, default=MAX_CLOUD)
    parser.add_argument("--only", nargs="*")
    return parser.parse_args()


def main():
    args = parse_args()
    init_ee()

    rows = pd.read_csv(args.csv)
    if args.only:
        rows = rows[rows["name"].isin(args.only)]
    if rows.empty:
        raise SystemExit("No matching locations found.")

    locations = list(rows.itertuples(index=False))
    iterator = locations if len(locations) == 1 else tqdm(locations, desc="Locations", unit="loc")
    stats = {"saved": 0, "no_s2": 0, "no_dw": 0}

    for row in iterator:
        status = download_location(row, args.start, args.end, args.buffer, args.max_cloud)
        stats[status] += 1

    print_run_summary(stats, args)


if __name__ == "__main__":
    main()
