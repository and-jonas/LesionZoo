
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
import imageio
import cv2
from skimage import morphology
from matplotlib import pyplot as plt
import feature_extraction_functions as fef
import utils
import pandas as pd
import numpy as np

# IF WORKING ON REMOTE    <===== !!!
os.environ['R_HOME'] = 'C:/Users/anjonas/Documents/R/R-3.6.2'

import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects import pandas2ri
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
from rpy2.robjects.packages import importr

numpy2ri.activate()
pandas2ri.activate()

model_rds_path = "Z:/Public/Jonas/001_LesionZoo/Output/Models/spl/pls_v4.rds"
model_dep_path = "Z:/Public/Jonas/001_LesionZoo/Output/Models/spl/pls_v4.dep"

# Load R model
model = robjects.r.readRDS(model_rds_path)

# import R's utility package
rutils = rpackages.importr('utils')

# select a mirror for R packages
rutils.chooseCRANmirror(ind=1)  # select the first mirror in the list

# import R base package
base = importr('base')

# # IF WORKING LOCALLY    <===== !!!
# base._libPaths("C:/Users/anjonas/Documents/R3Libs")
print(base._libPaths())

packnames = ('vctrs', 'caret', 'pls', 'segmented', 'nls.multstart', 'tidyverse')

# Selectively install what needs to be install.
# We are fancy, just because we can.
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    rutils.install_packages(StrVector(names_to_install), dependencies=True)

tidyverse = rpackages.importr('tidyverse')
caret = rpackages.importr('caret')
pls = rpackages.importr('pls')
segmented = importr('segmented')
nls_multstart = importr('nls.multstart')

r_source = robjects.r['source']
r_source('C:/Users/anjonas/PycharmProjects/LesionZoo/f_params.R')
r_getparams = robjects.globalenv['get_params']

# ======================================================================================================================


class ImageSegmentor:

    def __init__(self, dir_positives, dir_negatives, dir_model, save_output, file_index):
        self.dir_positives = dir_positives
        self.dir_negatives = dir_negatives
        self.dir_model = dir_model
        self.save_output = save_output
        self.file_index = file_index

    def file_feed(self):

        # get all files and their paths
        files_pos = glob.glob(f'{self.dir_positives}/*.png')
        files_neg = glob.glob(f'{self.dir_negatives}/*.png')
        all_files = files_pos + files_neg

        return all_files

    def segment_image(self, img):
        """
        Segments an image using a pre-trained pixel classification model.
        :param img: The image to be processed.
        :return: The resulting binary segmentation mask.
        """
        print('-segmenting image')

        # load model
        with open(self.dir_model, 'rb') as model:
            model = pickle.load(model)

        # extract pixel features
        color_spaces, descriptors, descriptor_names = fef.get_color_spaces(img)
        descriptors_flatten = descriptors.reshape(-1, descriptors.shape[-1])

        # predict pixel label
        segmented_flatten = model.predict(descriptors_flatten)

        # restore image, return as binary mask
        segmented = segmented_flatten.reshape((descriptors.shape[0], descriptors.shape[1]))
        segmented = np.where(segmented == 'pos', 1, 0)
        segmented = np.where(segmented == 0, 255, 0).astype("uint8")

        return segmented

    def post_process_segmentation_mask(self, img, mask):
        """
        Post-processes the binary segmentation mask resulting from segment_image. Extracts the leaf, removes small
        objects, and smoothes the mask by median filtering, morphological operations, and approximation of contours as
        b-splines.
        :param img: The segmented image
        :param mask: The corresponding binary mask, resulting from segment_image
        :return: A control image with contours, and the post-processed binary mask.
        """
        # reshape image
        img = img[:, :, :3]
        img = np.ascontiguousarray(img, dtype=np.uint8)

        # ==============================================================================================================
        # extract leaf
        # ==============================================================================================================

        # detect leaf (remove white background)
        lower_white = np.array([250, 250, 250], dtype=np.uint8)
        upper_white = np.array([255, 255, 255], dtype=np.uint8)
        mask_leaf = cv2.inRange(img, lower_white, upper_white)  # could also use threshold
        mask_leaf = cv2.bitwise_not(mask_leaf)

        # erode leaf to remove segmentation artifacts along leaf edges
        kernel = np.ones((15, 15), np.uint8)
        mask1_erode = morphology.erosion(mask_leaf, kernel)
        # "shrink" leaf
        mask_leaf = mask * mask1_erode

        # ==============================================================================================================
        # Try to remove central leaf vein and insect damage (grey lesions)
        # ==============================================================================================================

        # threshold values
        lower_grey = np.array([145, 145, 135], dtype=np.uint8)
        upper_grey = np.array([255, 255, 255], dtype=np.uint8)
        # get gray areas
        mask_leaf_cl = cv2.inRange(img, lower_grey, upper_grey)
        # dilate in y-direction
        # this should connect separated segments along leaf vein
        kernel = np.ones((3, 19), np.uint8)
        mask_leaf_cl_dil = morphology.dilation(mask_leaf_cl, kernel)

        # erode leaf mask to keep the leaf edges intact
        kernel = np.ones((35, 35), np.uint8)
        mask2_erode = morphology.erosion(mask1_erode, kernel)
        # "shrink" leaf
        mask_leaf_cl_dil = mask_leaf_cl_dil * mask2_erode

        # keep only strongly elongated objects of certain size
        mask_leaf_cl_dil_filter = utils.filter_objects_size(mask_leaf_cl_dil, 500, "smaller")
        mask_leaf_cl_dil_filter = utils.filter_objects_size(mask_leaf_cl_dil_filter, 10000, "greater")
        mask_leaf_cl_dil_filter2 = utils.filter_object_elongation(mask_leaf_cl_dil_filter, threshold=0.15)

        # dilate kept objects to remove them from mask
        kernel = np.ones((13, 13), np.uint8)
        mask_leaf_cl_dil_filter2 = morphology.dilation(mask_leaf_cl_dil_filter2, kernel)
        # remove from mask
        idx = np.where(mask_leaf_cl_dil_filter2 == 255)
        mask_leaf_pp = copy.copy(mask_leaf)
        mask_leaf_pp[idx] = 0

        # ==============================================================================================================
        # Morphological post-processing
        # ==============================================================================================================

        # median filter to remove noise without affecting edges
        mask_blur = cv2.medianBlur(mask_leaf_pp, 17)

        # remove holes in lesions
        mask_pp = copy.copy(mask_blur)
        _, contour, _ = cv2.findContours(mask_blur, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        for cnt in contour:
            cv2.drawContours(mask_pp, [cnt], 0, 255, -1)

        # remove holes in green leaf tissue (or small lesions)
        mask_pp = utils.filter_objects_size(mask_pp, 2000, "smaller")

        # slight opening
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask_blur = cv2.morphologyEx(mask_pp, cv2.MORPH_CLOSE, kernel)
        # filter again by size
        mask_blur = utils.filter_objects_size(mask_blur, 2000, "smaller")
        # remove holes in lesions AGAIN
        mask_pp = copy.copy(mask_blur)
        _, contour, _ = cv2.findContours(mask_blur, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        for cnt in contour:
            cv2.drawContours(mask_pp, [cnt], 0, 255, -1)

        # ==============================================================================================================
        # Smooth lesion edges and generate output
        # ==============================================================================================================

        # approximate contour through a B-spline to obtain smooth lesion edges
        test_img = copy.copy(img)
        mask_spl = np.zeros(mask_pp.shape).astype("uint8")
        _, contours, _ = cv2.findContours(mask_pp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            coords = utils.spline_approx_contour(contour=[c], interval=45, task="smoothing")
            # create graphical output
            cv2.drawContours(test_img, c, -1, (255, 0, 0), 3)
            test_img[coords] = (0, 0, 255)
            mask_spl[coords] = 255

        _, contours, _ = cv2.findContours(mask_spl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            cv2.fillPoly(mask_spl, pts=[c], color=255)

        # draw ALL contours
        _, contours, _ = cv2.findContours(mask_spl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_out_all_obj = copy.copy(img)
        for cnt in contours:
            cv2.drawContours(img_out_all_obj, [cnt], -1, (0, 0, 255), 3)

        mask_spl = utils.add_image_border(mask_spl, intensity=0)
        mask_leaf_cl_dil_filter2 = utils.add_image_border(mask_leaf_cl_dil_filter2, intensity=0)

        return img_out_all_obj, mask_spl, mask_leaf_cl_dil_filter2

    def iterate_images(self, img_type):

        files = self.file_feed()[self.file_index[0]:self.file_index[1]]

        # files = ["Z:/Public/Jonas/001_LesionZoo/TrainingData_Lesions/Positives/c3_sn108_15_leaf_1.png"]
        # files = ["Z:/Public/Jonas/001_LesionZoo/TrainingData_Lesions/Positives/c3_sn115_14_leaf_1.png"]
        # files = ["D:/EschikonData/c3_collection/Exports/113_2_picture_8_leaf.png"]
        # files = ["D:/LesionZoo/321_1_picture_2_leaf.png"]

        for i, file in enumerate(files):

            print(file)

            # get file basename
            dirname = os.path.dirname(file)
            basename = os.path.basename(file)

            # load image
            img = imageio.imread(file)
            # reshape image
            img = img[:, :, :3]
            img = np.ascontiguousarray(img, dtype=np.uint8)
            # output paths
            mask_name_out = f'{dirname}/Segmented/Mask/{basename}'
            mask_all_name_out = f'{dirname}/Segmented/Mask/allObj/{basename}'
            mask_false_name_out = f'{dirname}/Segmented/Mask/false/{basename}'
            ctrl_output_name = f'{dirname}/Output/V2/{basename}'
            ctrl_cluster_name = f'{dirname}/Segmented/Clusters_loc/{basename}'

            # ==========================================================================================================
            # IMAGE SEGMENTATION
            # ==========================================================================================================

            # if output already exists, load from disk
            if Path(mask_all_name_out).exists() and Path(mask_false_name_out).exists():
                print("Skipping Segmentation")
                # if the purpose is to predict new images
                # load mask containing ALL objects of interest, and the false object mask
                if img_type == "prediction":
                    mask_all_obj = imageio.imread(mask_all_name_out)
                    mask_false = imageio.imread(mask_false_name_out)
                # if the purpose is to "re-predict" the training data
                # load mask containing ONLY the object of interest !
                elif img_type == "training":
                    mask_all_obj = imageio.imread(mask_name_out)

            # if not, process images
            else:
                print(f'processing {i+1}/{len(files)}')
                # predict pixel labels
                mask = self.segment_image(img)
                # post process mask
                img_out_all_obj, mask_all_obj, mask_false = self.post_process_segmentation_mask(
                    img,
                    mask,
                )
                # save masks
                Path(os.path.dirname(mask_all_name_out)).mkdir(parents=True, exist_ok=True)
                imageio.imwrite(mask_all_name_out, mask_all_obj)
                Path(os.path.dirname(mask_false_name_out)).mkdir(parents=True, exist_ok=True)
                imageio.imwrite(mask_false_name_out, mask_false)

            img = utils.add_image_border(img, intensity=255)
            check_img = copy.copy(img)

            # get bounding boxes of all lesions
            rect, check_img = utils.get_bounding_boxes(mask_all_obj, check_img)

            # image for output visualization
            ctrl_output = copy.copy(img)
            # draw all contours
            _, contours, _ = cv2.findContours(mask_all_obj, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                cv2.drawContours(ctrl_output, contour, -1, (0, 0, 0), 1)

            # image for cluster visualization
            ctrl_cluster = copy.copy(img)

            # # # TEMPORARY
            # rect = rect[3:]

            # loop over all contours to process
            for i in range(len(rect)):

                print(f'-lesion {i + 1}/{len(rect)}')

                # extract roi
                empty_mask_all, empty_mask, empty_img, ctr = utils.select_roi(rect=rect[i],
                                                                              img=img,
                                                                              mask=mask_all_obj)

                # extract RGB profiles
                checker = copy.copy(img)
                prof, checker, spl, spl_points = fef.spline_contours(mask_obj=empty_mask,
                                                                     mask_all=empty_mask_all,
                                                                     mask_false=mask_false,
                                                                     img=empty_img,
                                                                     checker=checker)

                dists = utils.dist_to_centroid(spl_points, ctr, scale_factor=5)

                df, col_idx_kept, fig = fef.cluster_profiles(
                    profiles=prof,
                    distances=dists,
                    min_length_profile=60
                )
                # if i == 0:
                #     Path(os.path.dirname(cluster_name)).mkdir(parents=True, exist_ok=True)
                #     fig.figure.savefig(cluster_name, dpi=2400)
                plt.close()

                # if there are no complete profiles, fef.cluster_profiles() returns df = None
                # these lesions are not analyzed
                if df is None:
                    continue

                # generate some colors
                colors = []
                n_clust = len(df['Cluster'].unique())
                for k in range(n_clust):
                    colors.append(utils.random_color())

                # draw all kept contour normals
                for idx in col_idx_kept:
                    row = df.loc[df['row_ind'] == idx]
                    clst = int(row.iloc[0]['Cluster'])
                    cv2.drawContours(ctrl_cluster, spl[0][idx], -1, colors[clst], 1)

                # draw all contours
                _, contours, _ = cv2.findContours(mask_all_obj, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                for contour in contours:
                    cv2.drawContours(ctrl_cluster, contour, -1, (0, 0, 255), 1)

                # ======================================================================================================

                # find breakpoints in cluster labels
                cl = df['Cluster'].tolist()
                diffs = np.ediff1d(cl)
                idx = np.where(diffs != 0)
                idx = [i + 1 for i in idx]
                idx = idx[0].tolist()
                idx = idx + [len(cl)]  # add last element

                # plt.imshow(prof)

                clusters = []
                point_ids = []
                for i in range(len(idx)):
                    # if is the first cluster/segment
                    if i == 0:
                        # if the cluster spans the start/end of the lesion boundary, stitch the two parts together
                        if cl[0] == cl[-1]:
                            row_identifier = list(range(0, idx[i]))+list(range(idx[len(idx)-2], len(cl)))
                        else:
                            row_identifier = list(range(0, idx[i]))
                    # if is the last cluster/segment
                    else:
                        if i == len(idx)-1:
                            if cl[0] != cl[-1]:
                                row_identifier = list(range(idx[i-1], idx[i]))
                            else:
                                continue
                        else:
                            row_identifier = list(range(idx[i-1], idx[i]))

                    # select corresponding color profiles
                    row_subset_identifier = col_idx_kept[row_identifier]
                    point_ids.append(row_subset_identifier)
                    cluster_profile = prof[:, row_subset_identifier, :]
                    clusters.append(cluster_profile)

                # get ordering of columns
                # template = pd.read_csv("Z:/Public/Jonas/001_LesionZoo/TestingData/template_varnames.csv")
                template = pd.read_csv("Z:/Public/Jonas/001_LesionZoo/TestingData/template_varnames_v4.csv")
                cols = template.columns

                # # TEMPORARY
                # clusters = clusters[176:]

                # create prediction for each cluster
                predicted_label = []
                for i, cluster in enumerate(clusters):

                    # last cluster can be single and incomplete pixel line  --> skip
                    if cluster.shape[1] == 1:
                        continue

                    print(f'-----cluster {i + 1}/{len(clusters)}')

                    # extract scaled and raw color profiles
                    df_sc = fef.get_color_profiles(cluster, scale=True, smooth=7, remove_missing=True)
                    df_raw = fef.get_color_profiles(cluster, scale=False, smooth=7, remove_missing=True)
                    df = pd.concat([df_sc, df_raw], axis=1, ignore_index=False)
                    # average profiles
                    df_mean = df.mean(axis=0, skipna=True)
                    df_mean = pd.DataFrame(df_mean)
                    df_mean = df_mean.T
                    # df_mean.to_csv("Z:/Public/Jonas/001_LesionZoo/test.csv", index=False)
                    # get higher-level features
                    try:
                        params = r_getparams(dat=df_mean)
                    except:
                        print("Encountered problem while extracting model parameters!")
                        continue

                    # # get the color profiles and select the relevant columns (used to create a prediction)
                    # # df = fef.get_color_profiles(cluster, scale=True, smooth=7, remove_missing=True)
                    # df_ = df.iloc[:, 1:]
                    # df_ = df_.iloc[:, ::2]
                    # df_ = df_[df_.columns.drop(list(df_.filter(regex='L_Lab|YCC')))]

                    # # average profiles
                    # df_mean = df.mean(axis=0, skipna=True)
                    # df_mean = pd.DataFrame(df_mean)
                    # df_mean = df_mean.T

                    # add to the rest
                    df_mean_ = pd.concat([df_mean.reset_index(drop=True), params.reset_index(drop=True)], axis=1)

                    # reorder columns
                    df_mean = df_mean_[cols]

                    # make prediction
                    pred = robjects.r.predict(model, df_mean)

                    try:
                        pred_lab = pred[0]
                    except IndexError:
                        continue
                    predicted_label.append(pred_lab)

                    # map prediction to contour for graphical evaluation
                    for p in point_ids[i]:
                        coords = tuple([int(spl_points[1][p]), int(spl_points[0][p])])
                        if pred_lab == "neg":
                            cv2.circle(ctrl_output, coords, 1, (0, 0, 255), -1)
                        elif pred_lab == "pos":
                            cv2.circle(ctrl_output, coords, 1, (255, 0, 0), -1)

            Path(os.path.dirname(ctrl_output_name)).mkdir(parents=True, exist_ok=True)
            imageio.imwrite(ctrl_output_name, ctrl_output)

            Path(os.path.dirname(ctrl_cluster_name)).mkdir(parents=True, exist_ok=True)
            imageio.imwrite(ctrl_cluster_name, ctrl_cluster)

            plt.imshow(ctrl_output)