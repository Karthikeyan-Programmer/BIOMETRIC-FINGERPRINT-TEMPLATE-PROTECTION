import cv2
import numpy as np
import matplotlib.pyplot as plt

def IS_SIFT():
    image = cv2.imread('segmented_image.png', cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)
    image_with_keypoints_rgb = cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB)
    plt.imshow(image_with_keypoints_rgb)
    plt.title('Fingerprint with SIFT Keypoints')
    plt.axis('off')
    plt.savefig('fingerprint_with_keypoints.png', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.show()
    image = cv2.imread('fingerprint_with_keypoints.png', cv2.IMREAD_GRAYSCALE)
    kernel_size = 11
    theta = np.pi / 4  
    sigma = 4.0  
    lambda_ = 10.0
    gamma = 0.5
    gabor_kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma, theta, lambda_, gamma)
    filtered_image = cv2.filter2D(image, cv2.CV_8U, gabor_kernel)
    threshold_value = 100
    ridge_mask = (filtered_image > threshold_value).astype(np.uint8)
    contours, _ = cv2.findContours(ridge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(result_image, contours, -1, (0, 0, 255), 2)
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title('Fingerprint Ridge Structures')
    plt.axis('off')
    plt.show()
    ridge_positions = [tuple(contour[0][0]) for contour in contours]
    print("Number of ridge structures:", len(ridge_positions))
    print("Local positions of ridge structures:", ridge_positions)

