import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

# --- GLOBAL VARIABLES ---
MAP_NAME = "1.txt"

# --- PATHS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

# Map paths
input_path = os.path.join(BASE_DIR, "inputs", MAP_NAME)
output_path = os.path.join(BASE_DIR, "outputs", MAP_NAME)

# Image paths
image_paths = {
    'X': os.path.join(BASE_DIR, 'images/residential.png'),
    'O': os.path.join(BASE_DIR, 'images/hospital.png'),
    'T': os.path.join(BASE_DIR, 'images/industry.png'),
    'E': os.path.join(BASE_DIR, 'images/substation.png'),
    'C': os.path.join(BASE_DIR, 'images/transformer.png'),
    '-': None  # Nothing is painted
}

# --- Read maps ---
with open(input_path, "r") as f:
    input_map = [list(line.strip()) for line in f.readlines()]

with open(output_path, "r") as f:
    output_map = [list(line.strip()) for line in f.readlines()]

# Normalize row width
input_width = max(len(row) for row in input_map)
output_width = max(len(row) for row in output_map)
width = max(input_width, output_width)

# Convert list to string, apply ljust, and convert back to list
input_map = [list(''.join(row).ljust(width)) for row in input_map]
output_map = [list(''.join(row).ljust(width)) for row in output_map]

rows = len(input_map)
cols = width

# --- Load images and rotate them 180 degrees ---
loaded_images = {}
for symbol, path in image_paths.items():
    if path and os.path.exists(path):
        img = mpimg.imread(path)
        # Rotate 180 degrees (rotate 90 twice or use flip)
        rotated_img = np.rot90(img, 2)  # rot90 twice = 180 degrees
        loaded_images[symbol] = rotated_img
    else:
        loaded_images[symbol] = None

# --- Function to draw map ---
def draw_map(ax, map_data, title):
    # Draw white background for the entire map
    background = plt.Rectangle((-0.5, -0.5), cols, rows, 
                         fill=True, facecolor='white', zorder=0)
    ax.add_patch(background)
    
    for i in range(rows):
        for j in range(cols):
            symbol = map_data[i][j]
            if symbol in loaded_images and loaded_images[symbol] is not None:
                # Draw image centered in the cell
                img = loaded_images[symbol]
                # Adjust image size (85% of cell size)
                extent = [j - 0.425, j + 0.425, i - 0.425, i + 0.425]
                ax.imshow(img, extent=extent, aspect='auto', zorder=2)
            # If it's '-', nothing is drawn (blank space)
            # Draw a subtle grid for better visualization
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, 
                               fill=False, edgecolor='lightgray', 
                               linewidth=0.3, zorder=1)
            ax.add_patch(rect)

# Create folder for generated images
GENERATED_IMAGES_DIR = os.path.join(BASE_DIR, "generated_images")
os.makedirs(GENERATED_IMAGES_DIR, exist_ok=True)

# --- Create and save input map ---
fig1, ax1 = plt.subplots(1, 1, figsize=(10, 10))
ax1.set_xlim(-0.5, cols - 0.5)
ax1.set_ylim(-0.5, rows - 0.5)
ax1.set_aspect('equal')
ax1.invert_yaxis()  # Invert so that (0,0) is top-left
ax1.axis('off')

draw_map(ax1, input_map, 'Input')

plt.tight_layout()

input_image_path = os.path.join(GENERATED_IMAGES_DIR, f"{MAP_NAME.replace('.txt', '_input.png')}")
plt.savefig(input_image_path, dpi=150, bbox_inches='tight', pad_inches=0)
print(f"Input map saved to: {input_image_path}")
plt.close(fig1)

# --- Create and save output map ---
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 10))
ax2.set_xlim(-0.5, cols - 0.5)
ax2.set_ylim(-0.5, rows - 0.5)
ax2.set_aspect('equal')
ax2.invert_yaxis()
ax2.axis('off')

draw_map(ax2, output_map, 'Output')

plt.tight_layout()

output_image_path = os.path.join(GENERATED_IMAGES_DIR, f"{MAP_NAME.replace('.txt', '_output.png')}")
plt.savefig(output_image_path, dpi=150, bbox_inches='tight', pad_inches=0)
print(f"Output map saved to: {output_image_path}")
plt.close(fig2)

print("\nVisualizations completed!")
