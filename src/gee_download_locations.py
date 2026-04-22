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
CSV = ROOT / "locations.csv"
OUT = ROOT / "data" / "samples_generated"

BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
COLORS = [(0, 0, 0), (0, 100, 0), (184, 134, 11), (0, 0, 139), (105, 105, 105)]
DW_FROM = [0, 1, 2, 3, 4, 5, 6, 7, 8]
DW_TO = [3, 1, 1, 0, 1, 0, 4, 2, 0]
ALIASES = {
    "Cairo University": "CairoUniv",
    "Iconic Tower, New Administrative Capital": "IconicTower",
}


def sample_name(name):
    if name in ALIASES:
        return ALIASES[name]

    text = "".join(c if c.isalnum() else " " for c in name)
    return "".join(word[:1].upper() + word[1:] for word in text.split())


def init_ee():
    load_dotenv()
    project = os.getenv("EE_PROJECT_ID") or os.getenv("EE_PROJECT")

    try:
        ee.Initialize(project=project) if project else ee.Initialize()
    except Exception as e:
        raise SystemExit("Earth Engine is not ready. Run `earthengine authenticate` first.") from e


def mask_clouds(image):
    scl = image.select("SCL")
    keep = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return image.updateMask(keep)


def build_sample(region, start, end, max_cloud):
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
        .map(mask_clouds)
    )

    dw = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(region).filterDate(start, end)

    spectral = s2.select(BANDS).median().clip(region)
    mask = dw.select("label").mode().clip(region).remap(DW_FROM, DW_TO).rename("macro_class")

    return spectral.addBands(mask), s2.size().getInfo(), dw.size().getInfo()


def export_stack(image, path, region, scale):
    geemap.ee_export_image(
        image,
        str(path),
        crs="EPSG:4326",
        region=region,
        scale=scale,
        file_per_band=False,
        verbose=False,
    )

    if not path.exists():
        raise RuntimeError(f"export failed: {path.name} was not created")

def save_viz(spectral_path, mask_path, viz_path):
    with rio.open(spectral_path) as src:
        rgb = np.moveaxis(src.read([4, 3, 2]).astype("float32"), 0, -1)

    with rio.open(mask_path) as src:
        mask = src.read(1)

    low = np.percentile(rgb, 2, axis=(0, 1), keepdims=True)
    high = np.percentile(rgb, 98, axis=(0, 1), keepdims=True)
    rgb = np.clip((rgb - low) / (high - low + 1e-6), 0, 1)

    cmap = ListedColormap(np.array(COLORS) / 255.0)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(rgb)
    ax[0].set_title("RGB")
    ax[1].imshow(mask, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
    ax[1].set_title("Mask v3")

    for a in ax:
        a.axis("off")

    fig.tight_layout()
    fig.savefig(viz_path, dpi=122, facecolor="white")
    plt.close(fig)


def split_stack(stack_path, spectral_path, mask_path):
    with rio.open(stack_path) as src:
        data = src.read()
        profile = src.profile

    if data.shape[0] != 13:
        raise RuntimeError(f"expected 13 bands, found {data.shape[0]}")

    spectral = data[:12].astype("float64")
    mask = data[12:13].astype("uint8")

    spectral_profile = profile.copy()
    spectral_profile.update(count=12, dtype="float64", nodata=None)

    mask_profile = profile.copy()
    mask_profile.update(count=1, dtype="uint8", nodata=0)

    with rio.open(spectral_path, "w", **spectral_profile) as dst:
        dst.write(spectral)

    with rio.open(mask_path, "w", **mask_profile) as dst:
        dst.write(mask)

    stack_path.unlink(missing_ok=True)


def download_one(row, start, end, buffer_m, max_cloud, leave):
    name = sample_name(row.name)
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)

    spectral_path = folder / f"{name}_Spectral.tif"
    mask_path = folder / f"{name}_Mask.tif"
    viz_path = folder / f"{name}_viz.png"
    stack_path = folder / f".{name}_stack.tif"

    point = ee.Geometry.Point([row.longitude, row.latitude])
    region = point.buffer(buffer_m).bounds()
    scale = 10

    bar = tqdm(total=5, desc=name, unit="step", leave=leave)
    bar.set_postfix_str("query")

    image, s2_count, dw_count = build_sample(region, start, end, max_cloud)
    bar.update()
    bar.update()

    if not s2_count:
        bar.close()
        print(f"skip {name}: no Sentinel-2 images")
        return

    if not dw_count:
        bar.close()
        print(f"skip {name}: no Dynamic World images")
        return

    bar.set_postfix_str("export")
    export_stack(image, stack_path, region, scale)
    bar.update()

    bar.set_postfix_str("split")
    split_stack(stack_path, spectral_path, mask_path)
    bar.update()

    bar.set_postfix_str("viz")
    save_viz(spectral_path, mask_path, viz_path)
    bar.update()
    bar.close()

    print(f"saved {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=CSV)
    parser.add_argument("--start", default="2025-05-01")
    parser.add_argument("--end", default="2025-08-28")
    parser.add_argument("--buffer", type=int, default=1280)
    parser.add_argument("--max-cloud", type=int, default=20)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()

    init_ee()
    rows = pd.read_csv(args.csv)

    if args.only:
        rows = rows[rows["name"].isin(args.only)]

    if rows.empty:
        raise SystemExit("No matching locations found.")

    items = list(rows.itertuples(index=False))
    total = len(items)

    iterator = items if total == 1 else tqdm(items, desc="Locations", unit="loc")

    for row in iterator:
        download_one(row, args.start, args.end, args.buffer, args.max_cloud, total == 1)


if __name__ == "__main__":
    main()
