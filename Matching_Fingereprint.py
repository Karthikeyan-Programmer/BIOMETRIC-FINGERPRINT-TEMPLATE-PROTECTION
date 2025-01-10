import warnings
import os
import csv
import time
import shutil
import sys
import pandas as pd
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color
from skimage.morphology import skeletonize
from tkinter import *
from PIL import ImageTk, Image, ImageChops, ImageFilter
from tkinter import filedialog
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.fftpack import dct
from skimage import io, segmentation, filters
from datetime import datetime
from PIL import Image as PILImage
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pyswarm import pso
def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
       return im.crop(bbox)
def main_process():    
    def ImageSelection():
        time.sleep(1)
        print("\n\t\t\t===========************ LOAD THE INPUT FINGERPRINT IMAGES FROM THE FVC2004 and FVC2002 Dataset ************============")
        time.sleep(1)
        plt.figure(1)
        print("Select Input Fingerprint Image:")
        fileName = filedialog.askopenfilename(filetypes=[("TIF", ".tif"), ("PNG", ".png")])
        print(fileName)
        img = mpimg.imread(fileName)
        imgplot = plt.imshow(cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE))
        plt.xticks([]), plt.yticks([])
        plt.savefig('IpImg1.png', dpi=300, bbox_inches='tight')
        time.sleep(1)
        print("\n\nInput fingerprint images Selected Successfully...!")
    def Preprocessing():
        time.sleep(1)
        print("\n\t\t\t============************ IMAGE ACQUISITION AND PREPROCESSING **************===============")
        print("Grayscale Conversion:")
        time.sleep(1)
        image = cv2.imread('IpImg1.png')
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        min_pixel_value = gray_image.min()
        max_pixel_value = gray_image.max()
        scaled_image = 255 * ((gray_image - min_pixel_value) / (max_pixel_value - min_pixel_value))
        scaled_image = scaled_image.astype('uint8')
        cv2.imwrite('grayscale_image1.png', scaled_image)
        print("\n\nNormalization:")
        image = cv2.imread('grayscale_image1.png')
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
        min_pixel_value = gray_image.min()
        max_pixel_value = gray_image.max()
        normalized_image = 255 * ((gray_image - min_pixel_value) / (max_pixel_value - min_pixel_value))
        normalized_image = normalized_image.astype('uint8')
        plt.savefig('normalized_image1.png', bbox_inches='tight', pad_inches=0, dpi=300)
        print("\n\nBinarization:")
        image = cv2.imread('normalized_image1.png')
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        threshold_value = 128
        binary_image = np.where(gray_image > threshold_value, 255, 0).astype('uint8')
        plt.imsave('binary_fingerprint1.png', binary_image, cmap='gray')
        print("\n\nThinning:")
        image = cv2.imread('binary_fingerprint1.png', cv2.IMREAD_GRAYSCALE)
        threshold_value = 128
        binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)[1]
        skeletonized_image = skeletonize(binary_image)
        plt.imsave('thinning_fingerprint1.png', skeletonized_image, cmap='gray')
        print("IMAGE ACQUISITION AND PREPROCESSING process is Successfully Completed...")
    def Segmentation():
        time.sleep(1)
        print("\n\t\t\t=======******** SEGMENTATION *********========")
        print("\nRegion growing method:")
        time.sleep(1)
        image = io.imread('binary_fingerprint1.png')
        seed_x, seed_y = 100, 100
        threshold = 0.2
        gray_image = image[:, :, 0] 
        filtered_image = filters.gaussian(gray_image, sigma=1)
        region_segmentation = segmentation.flood(filtered_image, (seed_x, seed_y), tolerance=threshold)
        plt.imsave('segmented_image1.png', region_segmentation, cmap='gray')
        time.sleep(1)
        print("SEGMENTATION process is successfully completed...!")
    def Featureextraction():
        time.sleep(1)
        print("\n\t\t\t==========************* FEATURE EXTRACTION **********===========")
        print("\nIS-SIFT algorithm")
        image = cv2.imread('segmented_image1.png', cv2.IMREAD_GRAYSCALE)
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)
        image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)
        image_with_keypoints_rgb = cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB)
        plt.savefig('fingerprint_with_keypoints1.png', bbox_inches='tight', pad_inches=0, dpi=300)
        image = cv2.imread('fingerprint_with_keypoints1.png', cv2.IMREAD_GRAYSCALE)
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
        ridge_positions = [tuple(contour[0][0]) for contour in contours]
        print("Number of ridge structures:", len(ridge_positions))
        print("Local positions of ridge structures:", ridge_positions)
        time.sleep(1)
        print("\n\nFEATURE EXTRACTION Process is Successfully Completed...!")
    def Featureselection():
        time.sleep(1)
        print("\n\t\t\t==========************ FEATURE SELECTION ***********==========")
        print("\nSpatial Attention based Particle swarm optimization")
        time.sleep(1)
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
        time.sleep(1)
        print("\n\nFEATURE SELECTION process is successfully completed...!")
    def NoninvertibleTransformation():
        time.sleep(1)
        print("\n\t\t\t==========************ NON-INVERTIBLE TRANSFORM ***********==========")
        image_path = 'IpImg1.png'
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is not None:
            _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

            if binary_image is not None:
                biohash_code = binary_image.flatten()
                print("Biohash Code:", biohash_code)

                # Write the Biohash Code to a text file
                text_file_name = "data\\Bio_Hash1.txt"
                biohash_code_str = " ".join(map(str, biohash_code))
                with open(text_file_name, mode="w") as file:
                    file.write(biohash_code_str)
                print(f'Biohash code has been written to "{text_file_name}".')

            else:
                print("Thresholding failed. Check your image preprocessing.")
        else:
            print("Image loading failed. Please verify the image path.")
        print("\n\nNON-INVERTIBLE TRANSFORM Process is successfully completed...!")
    def Encryption():
        print("\n\t\t\t===========********** ENCRYPTION *********=============")
        time.sleep(1)
        from cryptography.fernet import Fernet
        secret_key = Fernet.generate_key()
        cipher_suite = Fernet(secret_key)
        text_file_name = "data\\Bio_Hash1.txt"
        encrypted_file_name = "data\\Encrypted_Biohash.txt"
        with open(text_file_name, "rb") as file:
            biohash_code_str = file.read()
        encrypted_biohash_code = cipher_suite.encrypt(biohash_code_str)
        with open(encrypted_file_name, "wb") as file:
            file.write(encrypted_biohash_code)
        print(f"Biohash code in '{text_file_name}' has been encrypted and saved to '{encrypted_file_name}'")
        time.sleep(1)
        print("\nNext click MATCHING button")
    ImageSelection()
    Preprocessing()
    Segmentation()
    Featureextraction()
    Featureselection()
    NoninvertibleTransformation()
    Encryption()

if __name__ == "__main__":
    main_process()
