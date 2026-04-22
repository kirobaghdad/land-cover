import ee
import numpy as np
import geemap
import os
import random
from dotenv import load_dotenv

# Authenticate once if you haven't: ee.Authenticate() or just run $ earthengine authenticate
# load_dotenv()
# ee.Initialize(project=os.getenv("EE_PROJECT_ID")) 

# Dates based on the project recommendation (summer to avoid clouds)
start_date = '2025-05-01'
end_date = '2025-08-28'

def mask_clouds(img):
    scl = img.select('SCL')
    # Keep pixels that are NOT shadows, clouds, or cirrus
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return img.updateMask(mask)

def get_13_band_image(region):
    # Sentinel-2 12-Band Spectral Data
    s2_col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterBounds(region)
              .filterDate(start_date, end_date)
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
              .map(mask_clouds))
    
    target_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
    s2_img = s2_col.select(target_bands).median().clip(region)
    
    # Dynamic World Mask Data
    dw_col = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
              .filterBounds(region)
              .filterDate(start_date, end_date))
    
    dw_label = dw_col.select('label').mode().clip(region)
    
    # Map Dynamic World (0-8) to Macro Classes (0-4)
    # DW: 0:water, 1:trees, 2:grass, 3:flooded_veg, 4:crops, 5:shrub, 6:built, 7:bare, 8:snow
    # Macro: 0:Unknown, 1:Greenery, 2:Sand, 3:Water, 4:Cement
    dw_from = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    dw_to   = [3, 1, 1, 0, 1, 0, 4, 2, 0] 
    
    macro_mask = dw_label.remap(dw_from, dw_to).rename('macro_class')
    
    # Stack the 12 S2 bands with the 13th mask band
    final_img = s2_img.addBands(macro_mask)
    return final_img

# 2. Targeted Coordinates (from document + a few highly diverse additions)
target_locations = [
    {'name': 'CairoUniv', 'lon': 31.20909, 'lat': 30.02633},       # Urban / Greenery
    {'name': 'IconicTower', 'lon': 31.7018369, 'lat': 30.0109181}, # Cement / Sand
    {'name': 'SiwaOasis', 'lon': 25.5195, 'lat': 29.2032},         # Water / Green / Sand
    {'name': 'KarnakLuxor', 'lon': 32.6583, 'lat': 25.7180},       # River / Sand / Urban
    {'name': 'PhilaeAswan', 'lon': 32.8844, 'lat': 24.0255},       # Water / Sand islands
    {'name': 'HawaraFayoum', 'lon': 30.8986, 'lat': 29.2747},      # Agriculture / Sand
    {'name': 'Alexandria', 'lon': 29.9187, 'lat': 31.2001},        # Deep Water / Dense Urban
    {'name': 'Mansoura', 'lon': 31.3807, 'lat': 31.0379}           # Deep Delta Agriculture
]

out_dir = "egypt_s2_diverse_dataset"
os.makedirs(out_dir, exist_ok=True)

samples_per_location = 70

# 1 degree of lat/lon is roughly 111 km.
# A jitter of 0.1 degrees means we sample within an ~11km radius of the target.
jitter_range = 0.1 

image_count = 0

for loc in target_locations:
    print(f"--- Sampling around {loc['name']} ---")
    
    for i in range(samples_per_location):
        # Apply random jitter to the center point
        rand_lon = loc['lon'] + random.uniform(-jitter_range, jitter_range)
        rand_lat = loc['lat'] + random.uniform(-jitter_range, jitter_range)
        
        pt = ee.Geometry.Point([rand_lon, rand_lat])
        
        # 2.5km buffer (5x5km tile)
        region = pt.buffer(2500).bounds()
        
        img = get_13_band_image(region)
        filename = f"{loc['name']}_{i:03d}_{rand_lon:.3f}_{rand_lat:.3f}.tif"
        out_path = os.path.join(out_dir, filename)
        
        try:
            print(f"Downloading {filename}...")
            geemap.ee_export_image(
                img, 
                filename=out_path, 
                scale=10, 
                region=region, 
                file_per_band=False
            )
            image_count += 1
        except Exception as e:
            print(f"Failed on {filename}: {e}")

print(f"Data collection complete! Successfully downloaded {image_count} diverse tiles.")