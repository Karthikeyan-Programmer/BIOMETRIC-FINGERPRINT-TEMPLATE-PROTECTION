import numpy as np
from skimage import io, segmentation, filters
import matplotlib.pyplot as plt

def Regiongrowing():
    image = io.imread('binary_fingerprint.png')
    seed_x, seed_y = 100, 100
    threshold = 0.2
    gray_image = image[:, :, 0] 
    filtered_image = filters.gaussian(gray_image, sigma=1)
    region_segmentation = segmentation.flood(filtered_image, (seed_x, seed_y), tolerance=threshold)
    plt.imsave('segmented_image.png', region_segmentation, cmap='gray')
    plt.imshow(region_segmentation, cmap='gray')
    plt.title('Region Growing Segmentation')
    plt.axis('off')
    plt.show()
