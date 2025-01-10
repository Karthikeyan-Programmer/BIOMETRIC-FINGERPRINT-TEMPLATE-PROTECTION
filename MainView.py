import warnings
import os
import csv
import time
import shutil
import sys
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.morphology import skeletonize
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image, ImageChops, ImageFilter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import glob
from typing import Tuple
import png
import tkinter as tk
import tkinter.font as font
from itertools import count, cycle
from scipy.fftpack import dct
from datetime import datetime
from PIL import Image as PILImage
from Segmentation import Regiongrowing
from Feature_Extraction import IS_SIFT
from Feature_Selection import IS_PSO
import Matching_Fingereprint
from Matching_Fingereprint import main_process
from Metrics import *
def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
       return im.crop(bbox)
print("\n\t\t\t===========************ NON-INVERTIBLE TRANSFORMATION BASED TECHNIQUE FOR BIOMETRIC FINGERPRINT TEMPLATE PROTECTION ************============")
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
    plt.savefig('IpImg.png', dpi=300, bbox_inches='tight')
    plt.title('Input Image')
    plt.show()
    time.sleep(1)
    print("\n\nInput fingerprint images Selected Successfully...!")
    time.sleep(1)
    print("\nNext click IMAGE ACQUISITION AND PREPROCESSING button")
def Preprocessing():
    time.sleep(1)
    print("\n\t\t\t============************ IMAGE ACQUISITION AND PREPROCESSING **************===============")
    print("Grayscale Conversion:")
    time.sleep(1)
    image = cv2.imread('IpImg.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_pixel_value = gray_image.min()
    max_pixel_value = gray_image.max()
    scaled_image = 255 * ((gray_image - min_pixel_value) / (max_pixel_value - min_pixel_value))
    scaled_image = scaled_image.astype('uint8')
    cv2.imwrite('grayscale_image.png', scaled_image)
    plt.imshow(scaled_image, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')
    plt.show()
    print("\n\nNormalization:")
    image = cv2.imread('grayscale_image.png')
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    min_pixel_value = gray_image.min()
    max_pixel_value = gray_image.max()
    normalized_image = 255 * ((gray_image - min_pixel_value) / (max_pixel_value - min_pixel_value))
    normalized_image = normalized_image.astype('uint8')
    plt.imshow(normalized_image, cmap='gray')
    plt.title('Normalized Image')
    plt.axis('off')
    plt.savefig('normalized_image.png', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.show()
    print("\n\nBinarization:")
    image = cv2.imread('normalized_image.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold_value = 128
    binary_image = np.where(gray_image > threshold_value, 255, 0).astype('uint8')
    plt.imsave('binary_fingerprint.png', binary_image, cmap='gray')
    plt.imshow(binary_image, cmap='gray')
    plt.title('Binary Fingerprint Image')
    plt.axis('off')
    plt.show()
    print("\n\nThinning:")
    image = cv2.imread('binary_fingerprint.png', cv2.IMREAD_GRAYSCALE)
    threshold_value = 128
    binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)[1]
    skeletonized_image = skeletonize(binary_image)
    plt.imsave('thinning_fingerprint.png', skeletonized_image, cmap='gray')
    plt.imshow(skeletonized_image, cmap='gray')
    plt.title('Thinning Fingerprint Image')
    plt.axis('off')
    plt.show()
    print("IMAGE ACQUISITION AND PREPROCESSING process is Successfully Completed...")
    time.sleep(1)
    print("\nNext click SEGMENTATION button")
def Segmentation():
    time.sleep(1)
    print("\n\t\t\t=======******** SEGMENTATION *********========")
    print("\nRegion growing method:")
    time.sleep(1)
    Regiongrowing()
    time.sleep(1)
    print("SEGMENTATION process is successfully completed...!")
    time.sleep(1)
    print("\nNext click FEATURE EXTRACTION button")
def Featureextraction():
    time.sleep(1)
    print("\n\t\t\t==========************* FEATURE EXTRACTION **********===========")
    print("\nIS-SIFT algorithm")
    IS_SIFT()
    time.sleep(1)
    print("\n\nFEATURE EXTRACTION Process is Successfully Completed...!")
    time.sleep(1)
    print("\nNext click FEATURE SELECTION button")
def Featureselection():
    time.sleep(1)
    print("\n\t\t\t==========************ FEATURE SELECTION ***********==========")
    print("\nSpatial Attention based Particle swarm optimization")
    time.sleep(1)
    IS_PSO()
    time.sleep(1)
    print("\n\nFEATURE SELECTION process is successfully completed...!")
    time.sleep(1)
    print("\nNext click NON-INVERTIBLE TRANSFORM button")
def NoninvertibleTransformation():
    time.sleep(1)
    print("\n\t\t\t==========************ NON-INVERTIBLE TRANSFORM ***********==========")
    image_path = 'IpImg.png'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is not None:
        _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

        if binary_image is not None:
            biohash_code = binary_image.flatten()
            print("Biohash Code:", biohash_code)

            # Write the Biohash Code to a text file
            text_file_name = "data\\Bio_Hash.txt"
            biohash_code_str = " ".join(map(str, biohash_code))
            with open(text_file_name, mode="w") as file:
                file.write(biohash_code_str)
            print(f'Biohash code has been written to "{text_file_name}".')

        else:
            print("Thresholding failed. Check your image preprocessing.")
    else:
        print("Image loading failed. Please verify the image path.")
    print("\n\nNON-INVERTIBLE TRANSFORM Process is successfully completed...!")
    print("\nNext click ENCRYPTION button")
def Encryption():
    print("\n\t\t\t===========********** ENCRYPTION *********=============")
    time.sleep(1)
    from cryptography.fernet import Fernet
    secret_key = Fernet.generate_key()
    cipher_suite = Fernet(secret_key)
    text_file_name = "data\\Bio_Hash.txt"
    encrypted_file_name = "data\\Encrypted_Biohash.txt"
    with open(text_file_name, "rb") as file:
        biohash_code_str = file.read()
    encrypted_biohash_code = cipher_suite.encrypt(biohash_code_str)
    with open(encrypted_file_name, "wb") as file:
        file.write(encrypted_biohash_code)
    print(f"Biohash code in '{text_file_name}' has been encrypted and saved to '{encrypted_file_name}'")
    time.sleep(1)
    print("\nNext click MATCHING button")
def Matching():
    time.sleep(1)
    main_process()
    def compare_text_files(file1_path, file2_path):
        with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
            content1 = file1.read()
            content2 = file2.read()
            if content1 == content2:
                print("\n\nFingerprint Matched")
            else:
                print("\n\nFingerprint MisMatched")
    file1_path = "data\\Bio_Hash.txt"
    file2_path = "data\\Bio_Hash1.txt"
    compare_text_files(file1_path, file2_path)
    print("\nNext click PERFORMANCE METRICS button")
def Metrics():
    print("\n\t\t\t===========********** PERFORMANCE METRICS *********=============")
    time.sleep(1)
    PerformanceMetrics()
    time.sleep(1)
    print("\nImplementation is Completed...!")
    print("\n\n+++++++++++++++++++++++++++++++++++++++ END ++++++++++++++++++++++++++++++++++++")
def main_screen():
    global window
    window = Tk()
    window.title("Proposed")
    window_width = 650
    window_height = 450
    window.geometry(f"{window_width}x{window_height}")
    Label(window, text="NON-INVERTIBLE TRANSFORMATION BASED TECHNIQUE\nFOR BIOMETRIC FINGERPRINT TEMPLATE PROTECTION", bg="gray", fg="yellow", width="700", height="2", font=('Times New Roman', 15)).pack()
    Label(text="").pack()
    b1 = Button(text="START", height="2", width="20", bg="white", fg="blue", font=('Times New Roman', 12), command=ImageSelection)
    b1.pack(pady=10)
    b2 = Button(text="IMAGE ACQUISITION\n & PREPROCESSING", height="2", width="20", bg="white", fg="blue", font=('Times New Roman', 12), command=Preprocessing)
    b2.pack(pady=10)
    b3 = Button(text="SEGMENTATION", height="2", width="20", bg="white", fg="blue", font=('Times New Roman', 12), command=Segmentation)
    b3.pack(pady=10)
    b4 = Button(text="FEATURE EXTRACTION", height="2", width="20", bg="white", fg="blue", font=('Times New Roman', 12), command=Featureextraction)
    b4.pack(pady=10)
    b5 = Button(text="FEATURE SELECTION", height="2", width="20", bg="white", fg="blue", font=('Times New Roman', 12), command=Featureselection)
    b5.pack(pady=10)
    b6 = Button(text="NON-INVERTIBLE\nTRANSFORM", height="2", width="20", bg="white", fg="blue", font=('Times New Roman', 12), command=NoninvertibleTransformation)
    b6.pack(pady=10)
    b7 = Button(text="ENCRYPTION", height="2", width="20", bg="white", fg="blue", font=('Times New Roman', 12), command=Encryption)
    b7.pack(pady=10)
    b8 = Button(text="MATCHING", height="2", width="20", bg="white", fg="blue", font=('Times New Roman', 12), command=Matching)
    b8.pack(pady=10)
    b9 = Button(text="PERFORMANCE\nMETRICS", height="2", width="20", bg="white", fg="blue", font=('Times New Roman', 12), command=Metrics)
    b9.pack(pady=10)
    Label(text="").pack()
    window.mainloop()
main_screen()
