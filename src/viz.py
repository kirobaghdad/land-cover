import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import argparse as ap


def visualize_dataset_tile(tif_path):
    with rasterio.open(tif_path) as src:
        # Read the entire array
        data = src.read()
        
        # Extract RGB bands. B4 is index 3, B3 is index 2, B2 is index 1
        # Normalize the 16-bit data (usually 0-10000 range in S2 SR) to 0-1 for plotting
        r = data[3] / 10000.0
        g = data[2] / 10000.0
        b = data[1] / 10000.0
        
        # Clip values to 0-1 range to avoid artifacts
        rgb = np.dstack((r, g, b))
        rgb = np.clip(rgb, 0, 1)
        
        # Extract the 13th band (the mask)
        mask = data[12]

    # Define custom colormap matching project guidelines
    # 0: Unknown (Black), 1: Greenery (Dark Green), 2: Sand (Yellow), 3: Water (Dark Blue), 4: Cement (Grey)
    colors = ['black', 'darkgreen', 'orange', 'darkblue', 'grey']
    cmap = ListedColormap(colors)
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Plot RGB
    axes[0].imshow(rgb)
    axes[0].set_title("Sentinel-2 RGB (B4, B3, B2)")
    axes[0].axis('off')
    
    # Plot Mask
    # vmin=0, vmax=4 ensures the colormap aligns strictly with classes 0 through 4
    mask_plot = axes[1].imshow(mask, cmap=cmap, vmin=0, vmax=4, interpolation='nearest')
    axes[1].set_title("Ground Truth Macro Class")
    axes[1].axis('off')
    
    # Create a legend
    # cbar = plt.colorbar(mask_plot, ax=axes[1], ticks=[0, 1, 2, 3, 4], fraction=0.046, pad=0.04)
    # cbar.ax.set_yticklabels(['0: Unknown', '1: Greenery', '2: Sand', '3: Water', '4: Cement'])
    
    plt.tight_layout()
    plt.show()

# Usage:
# python data_vizualization_v2.py <image_path>

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("TIF_filepath", type=str)
    args = parser.parse_args()
    img = args.TIF_filepath
    print(f"visualizing {img}")
    visualize_dataset_tile(img)