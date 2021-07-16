
# ======================================================================================================================
# Author: Jonas Anderegg, jonas.anderegg@usys.ethz.ch
# Project: LesionZoo
# Date: 15.03.2021
# Tool for sampling training data for lesion classification
# ======================================================================================================================

import pickle
import copy
import glob
import os
from pathlib import Path
import numpy as np
import pandas as pd
import imageio
import cv2
from skimage import morphology
from matplotlib import pyplot as plt
import feature_extraction_functions_tr as fef
import utils_tr as utils


class ImageSegmentor:

    def __init__(self, dir_positives, dir_negatives, dir_model, save_output):
        self.dir_positives = dir_positives
        self.dir_negatives = dir_negatives
        self.dir_model = dir_model
        self.save_output = save_output

    def file_feed(self):
        # get all files and their paths
        files_pos = glob.glob(f'{self.dir_positives}/*.png')
        files_neg = glob.glob(f'{self.dir_negatives}/*.png')
        all_files = files_pos + files_neg

        return all_files

    def segment_image(self, img):

        # load model
        with open(self.dir_model, 'rb') as model:
            model = pickle.load(model)

        # extract pixel features
        color_spaces, descriptors, descriptor_names = fef.get_color_spaces(img)
        descriptors_flatten = descriptors.reshape(-1, descriptors.shape[-1])

        # predict pixel label
        a_segmented_flatten = model.predict(descriptors_flatten)

        # restore image, return as binary mask
        a_segmented = a_segmented_flatten.reshape((descriptors.shape[0], descriptors.shape[1]))
        a_segmented = np.where(a_segmented == 'pos', 1, 0)
        a_segmented = np.where(a_segmented == 0, 255, 0).astype("uint8")

        return a_segmented

    def post_process_segmentation_mask(self, img, mask):

        # reshape image
        img = img[:, :, :3]
        img = np.ascontiguousarray(img, dtype=np.uint8)
        # add border
        img = utils.add_image_border(img, intensity=255)
        mask = utils.add_image_border(mask, intensity=0)

        # detect leaf (remove white background)
        lower_white = np.array([250, 250, 250], dtype=np.uint8)
        upper_white = np.array([255, 255, 255], dtype=np.uint8)
        mask_leaf = cv2.inRange(img, lower_white, upper_white)  # could also use threshold
        mask_leaf = cv2.bitwise_not(mask_leaf)

        # erode leaf to remove segmentation artifacts along leaf edges
        kernel = np.ones((15, 15), np.uint8)
        mask1_erode = morphology.erosion(mask_leaf, kernel)
        # "shrink" leaf
        mask = mask * mask1_erode

        # median filter to remove noise without affecting edges
        mask_blur = cv2.medianBlur(mask, 17)

        # INSERTED 2021-04-20: Perform opening BEFORE filling holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        mask_blur = cv2.morphologyEx(mask_blur, cv2.MORPH_CLOSE, kernel)
        # END INSERTED

        # remove holes in green lesions (contour filling - FASTER!)
        mask_pp = copy.copy(mask_blur)
        _, contour, _ = cv2.findContours(mask_blur, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        for cnt in contour:
            cv2.drawContours(mask_pp, [cnt], 0, 255, -1)

        # remove holes in green leaf tissue (component filtering)
        # detect all components on the original mask
        _, output, stats, _ = cv2.connectedComponentsWithStats(mask_pp, connectivity=8)
        sizes = stats[1:, -1];
        idx = (np.where(sizes >= 500)[0] + 1).tolist()
        out = np.in1d(output, idx).reshape(output.shape)
        mask_pp = np.where(out, mask_pp, 0)
        mask_all_obj = mask_pp

        # filter out the main object (i.e. lesion of interest)
        n_comps, output, stats, centroids = cv2.connectedComponentsWithStats(mask_pp, connectivity=8)
        sizes = list(stats[:, 4][1:])
        index = sizes.index(np.max(sizes))
        lesion_mask = np.uint8(np.where(output == index + 1, 1, 0))

        # draw resulting contour
        _, contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_out = copy.copy(img)
        for cnt in contours:
            cv2.drawContours(img_out, [cnt], -1, (0, 0, 255), 3)

        # draw ALL contours
        _, contours, _ = cv2.findContours(mask_all_obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_out_all_obj = copy.copy(img)
        for cnt in contours:
            cv2.drawContours(img_out_all_obj, [cnt], -1, (0, 0, 255), 3)

        return img_out, img_out_all_obj, lesion_mask, mask_all_obj

    def iterate_images(self):

        files = self.file_feed()[961:]

        for i, file in enumerate(files):

            print(file)

            # get file basename
            dirname = os.path.dirname(file)
            basename = os.path.basename(file)
            trcname = os.path.splitext(basename)[0]

            # load image
            img = imageio.imread(file)
            # reshape image
            img = img[:, :, :3]
            img = np.ascontiguousarray(img, dtype=np.uint8)
            # output paths
            img_name_out = f'{dirname}/Segmented/Overlay/{basename}'
            mask_name_out = f'{dirname}/Segmented/Mask/{basename}'
            mask_all_name_out = f'{dirname}/Segmented/Mask/allObj/{basename}'

            # ==========================================================================================================
            # IMAGE SEGMENTATION
            # ==========================================================================================================

            # # if output already exists, load from disk
            # if Path(img_name_out).exists() and Path(mask_name_out).exists():
            #     print("Skipping Segmentation")
            #     mask_pp = imageio.imread(mask_name_out)
            #     img_out = imageio.imread(img_name_out)
            # # if not process images
            # else:
            print(f'processing {i+1}/{len(files)}')
            # predict pixel labels
            mask = self.segment_image(img)
            # post process mask
            img_out, img_out_all_obj, mask_pp, mask_all_obj = self.post_process_segmentation_mask(img, mask)

            # add border to mask
            # mask_pp = utils.add_image_border(mask_pp, intensity=0)
            # mask_all_obj = utils.add_image_border(mask_all_obj, intensity=0)
            mask_pp = mask_pp*255

            # fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
            # # Show RGB and segmentation mask
            # axs[0].imshow(mask_pp)
            # axs[0].set_title('original patch')
            # axs[1].imshow(new_img)
            # axs[1].set_title('original patch')
            # axs[2].imshow(img_out)
            # axs[2].set_title('seg')
            # plt.show(block=True)

            # save output
            if self.save_output:
                # save image with lesion boundary overlay
                Path(os.path.dirname(img_name_out)).mkdir(parents=True, exist_ok=True)
                imageio.imwrite(img_name_out, img_out)
                # save masks
                Path(os.path.dirname(mask_name_out)).mkdir(parents=True, exist_ok=True)
                imageio.imwrite(mask_name_out, mask_pp)
                Path(os.path.dirname(mask_all_name_out)).mkdir(parents=True, exist_ok=True)
                imageio.imwrite(mask_all_name_out, mask_all_obj)

            # ==========================================================================================================
            # EXTRACT SPLINE NORMALS
            # ==========================================================================================================

            img = utils.add_image_border(img, intensity=255)
            check_img = copy.copy(img)
            prof, checker, spl, spl_points = fef.spline_contours(mask_obj=mask_pp,
                                                                 mask_all=mask_all_obj,
                                                                 img=img,
                                                                 checker=check_img)

            # remove profiles containing white background (from sampling beyond the leaf edge)
            mask = prof == 255
            all_white = mask.sum(axis=2) == 3
            cols_drop = np.unique(np.where(all_white != 0)[1])
            contour_img = np.delete(prof, cols_drop, axis=1)
            # plt.imshow(contour_img)

            df_sc = fef.extract_color_profiles(contour_img, task=0, scale=True)
            df_raw = fef.extract_color_profiles(contour_img, task=0, scale=False)
            df = pd.concat([df_sc, df_raw], axis=1, ignore_index=False)

            img_check = checker[1]

            # save data
            prof_name = f'{dirname}/Scans/Profiles/spl_n/{trcname}_iter4.csv'
            Path(os.path.dirname(prof_name)).mkdir(parents=True, exist_ok=True)
            df.to_csv(prof_name, index=False)

            # save checker
            img_check_name = f'{dirname}/Segmented/Overlay/spl_n/{basename}'
            Path(os.path.dirname(img_check_name)).mkdir(parents=True, exist_ok=True)
            imageio.imwrite(img_check_name, img_check)

# initiate class and use functions
def main():
    image_segmentor = ImageSegmentor()
    image_segmentor.iterate_images()


if __name__ == '__main__':
    main()

# ======================================================================================================================