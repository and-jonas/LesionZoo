
# ======================================================================================================================
# Author: Jonas Anderegg, jonas.anderegg@usys.ethz.ch
# Project: LesionZoo
# Date: 15.03.2021
# Functions for training data extraction: Pixel-wise segmentation of scanned leaf images into necrotic lesion and
# surrounding healthy tissue
# ======================================================================================================================

import glob
import imageio
import numpy as np
import cv2
import os
import pandas as pd


def get_color_spaces(patch):

    # Scale to 0...1
    img_RGB = np.array(patch / 255, dtype=np.float32)

    # Images are in RGBA mode, but alpha seems to be a constant - remove to convert to simple RGB
    img_RGB = img_RGB[:, :, :3]

    # Convert to other color spaces
    img_HSV = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2HSV)
    img_Luv = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2Luv)
    img_Lab = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2Lab)
    img_YUV = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2YUV)
    img_YCbCr = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2YCrCb)

    # Calculate vegetation indices: ExR, ExG, TGI
    R, G, B = cv2.split(img_RGB)
    normalizer = np.array(R + G + B, dtype=np.float32)
    # Avoid division by zero
    normalizer[normalizer == 0] = 10
    r, g, b = (R, G, B) / normalizer

    # weights for TGI
    lambda_r = 670
    lambda_g = 550
    lambda_b = 480

    TGI = -0.5 * ((lambda_r - lambda_b) * (r - g) - (lambda_r - lambda_g) * (r - b))
    ExR = np.array(1.4 * r - b, dtype=np.float32)
    ExG = np.array(2.0 * g - r - b, dtype=np.float32)

    # Concatenate all
    descriptors = np.concatenate(
        [img_RGB, img_HSV, img_Lab, img_Luv, img_YUV, img_YCbCr, np.stack([ExG, ExR, TGI], axis=2)], axis=2)
    # Names
    descriptor_names = ['sR', 'sG', 'sB', 'H', 'S', 'V', 'L', 'a', 'b',
                        'L', 'u', 'v', 'Y', 'U', 'V', 'Y', 'Cb', 'Cr', 'ExG', 'ExR', 'TGI']

    # Return as tuple
    return (img_RGB, img_HSV, img_Lab, img_Luv, img_YUV, img_YCbCr, ExG, ExR, TGI), descriptors, descriptor_names


def extract_training_data(patch):

    color_spaces, descriptors, descriptor_names = get_color_spaces(patch)

    predictors = []

    # iterate over all pixels in the patch
    for x in range(patch.shape[0]):
        for y in range(patch.shape[1]):
            predictors_ = descriptors[x, y].tolist()
            # Append to training set
            predictors.append(predictors_)

    # Convert to numpy array
    a_predictors = np.array(predictors)

    return a_predictors, descriptor_names


def iterate_patches(dir_positives, dir_negatives):

    # POSITIVE patches
    all_files_pos = glob.glob(f'{dir_positives}/*.png')
    # iterate over patches
    for i, file in enumerate(all_files_pos):
        print(f'{i}/{len(all_files_pos)}')
        patch = imageio.imread(file)
        X, X_names = extract_training_data(patch)
        df = pd.DataFrame(X, columns=X_names)
        df['response'] = 'pos'
        f_name = os.path.splitext(os.path.basename(file))[0]
        df.to_csv(f'{dir_positives}/{f_name}_data.csv',  index=False)

    # NEGATIVE patches
    all_files_neg = glob.glob(f'{dir_negatives}/*.png')
    # iterate over patches
    for i, file in enumerate(all_files_neg):
        print(f'{i}/{len(all_files_neg)}')
        patch = imageio.imread(file)
        X, X_names = extract_training_data(patch)
        df = pd.DataFrame(X, columns=X_names)
        df['response'] = 'neg'
        f_name = os.path.splitext(os.path.basename(file))[0]
        df.to_csv(f'{dir_negatives}/{f_name}_data.csv',  index=False)

