# Week: CNN -- the cat photo and its parts for the introduction figure (drawn in main.tex with TikZ)
#   * needs scikit-image for the photo (skimage.data.chelsea, 300x451 RGB)
# Figures are saved to ../figures/ (next to main.tex), wherever this file is run from

import os
import matplotlib.pyplot as plt
from skimage import data

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

img = data.chelsea()                                   # (300, 451, 3)
plt.imsave(os.path.join(FIG_DIR, "cat_full.png"), img)

# (x0, y0, x1, y1) in pixels; the same boxes are drawn on the photo in main.tex
parts = {
    "eye": (130, 80, 215, 165),
    "ear": (325, 0, 405, 80),
    "nose": (222, 205, 307, 290),
    "whisker": (20, 205, 105, 290),
}
for name, (x0, y0, x1, y1) in parts.items():
    plt.imsave(os.path.join(FIG_DIR, f"cat_{name}.png"), img[y0:y1, x0:x1])
print("saved to", os.path.abspath(FIG_DIR))
