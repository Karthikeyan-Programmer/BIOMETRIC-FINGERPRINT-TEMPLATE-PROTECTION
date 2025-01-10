import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, segmentation, filters
from skimage.morphology import skeletonize
from tkinter import filedialog
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pyswarm import pso
from cryptography.fernet import Fernet
import os
def fingerprint_processing_pipeline():
    try:
        # Image Selection
        print("Select Input Fingerprint Image:")
        fileName = filedialog.askopenfilename(filetypes=[("TIF", ".tif"), ("PNG", ".png")])
        print(fileName)
        img = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
        plt.imshow(img, cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.title('Input Image')
        plt.show()

        # Preprocessing
        print("Performing Preprocessing...")
        min_pixel_value = img.min()
        max_pixel_value = img.max()
        scaled_image = 255 * ((img - min_pixel_value) / (max_pixel_value - min_pixel_value))
        scaled_image = scaled_image.astype('uint8')
        cv2.imwrite('grayscale_image.png', scaled_image)
        plt.imshow(scaled_image, cmap='gray')
        plt.title('Grayscale Image')
        plt.axis('off')
        plt.show()

        # Segmentation
        print("Performing Segmentation...")
        seed_x, seed_y = 100, 100
        threshold = 0.2
        gray_image = scaled_image
        filtered_image = filters.gaussian(gray_image, sigma=1)
        region_segmentation = segmentation.flood(filtered_image, (seed_x, seed_y), tolerance=threshold)
        plt.imsave('segmented_image.png', region_segmentation, cmap='gray')
        plt.imshow(region_segmentation, cmap='gray')
        plt.title('Region Growing Segmentation')
        plt.axis('off')
        plt.show()

        # Feature Extraction
        print("Performing Feature Extraction...")
        image = cv2.imread('segmented_image.png', cv2.IMREAD_GRAYSCALE)
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)
        image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)
        image_with_keypoints_rgb = cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB)
        plt.imshow(image_with_keypoints_rgb)
        plt.title('Fingerprint with SIFT Keypoints')
        plt.axis('off')
        plt.show()

        # Feature Selection
        print("Performing Feature Selection...")
        X = np.array([[0, 1], [1, 0], [0, 1], [1, 0]])
        y = np.array([0, 1, 0, 1])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        def fitness_function(selected_features):
            selected_indices = np.where(selected_features)[0]
            if len(selected_indices) == 0:
                return 0.0
            X_train_selected = X_train[:, selected_indices]
            X_test_selected = X_test[:, selected_indices]
            clf = RandomForestClassifier(random_state=42)
            clf.fit(X_train_selected, y_train)
            y_pred = clf.predict(X_test_selected)
            return accuracy_score(y_test, y_pred)

        num_features = X_train.shape[1]
        lb = np.zeros(num_features)
        ub = np.ones(num_features)

        def is_pso(num_iterations, num_particles):
            def objective_function(selected_features):
                return -fitness_function(selected_features)

            best_features, _ = pso(objective_function, lb, ub, swarmsize=num_particles, maxiter=num_iterations)

            return best_features

        num_iterations = 50
        num_particles = 20

        selected_features = is_pso(num_iterations, num_particles)

        selected_indices = np.where(selected_features)[0]
        X_train_selected = X_train[:, selected_indices]
        X_test_selected = X_test[:, selected_indices]
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train_selected, y_train)
        y_pred = clf.predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy with selected features: {accuracy:.2f}")

        # Non-Invertible Transform
        print("Performing Non-Invertible Transform...")
        _, binary_image = cv2.threshold(scaled_image, 128, 255, cv2.THRESH_BINARY)
        biohash_code = binary_image.flatten()
        text_file_name = "Bio_Hash.txt"
        biohash_code_str = " ".join(map(str, biohash_code))
        with open(text_file_name, mode="w") as file:
            file.write(biohash_code_str)
        print(f'Biohash code has been written to "{text_file_name}".')

        # Encryption
        print("Performing Encryption...")
        secret_key = Fernet.generate_key()
        cipher_suite = Fernet(secret_key)
        encrypted_file_name = "Encrypted_Biohash.txt"
        with open(text_file_name, "rb") as file:
            biohash_code_str = file.read()
        encrypted_biohash_code = cipher_suite.encrypt(biohash_code_str)
        with open(encrypted_file_name, "wb") as file:
            file.write(encrypted_biohash_code)
        print(f"Biohash code in '{text_file_name}' has been encrypted and saved to '{encrypted_file_name}'")

        print("Processing completed successfully.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
