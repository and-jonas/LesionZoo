
# ======================================================================================================================
# Author: Jonas Anderegg, jonas.anderegg@usys.ethz.ch
# Project: LesionZoo
# Date: 15.03.2021
# ======================================================================================================================

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import pickle
import copy
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

import multiprocessing
from multiprocessing import Manager, Process

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
# print(base._libPaths())

packnames = ('vctrs', 'caret', 'pls', 'segmented', 'nls.multstart', 'tidyverse')

# Selectively install what needs to be installed
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

    def __init__(self, dir_to_process, dir_output, dir_model):

        self.dir_to_process = Path(dir_to_process)
        self.dir_model = Path(dir_model)
        # set output paths
        self.path_output = Path(dir_output)
        self.path_mask = self.path_output / "Mask" / "Obj"
        self.path_mask_all = self.path_output / "Mask" / "AllObj"
        self.path_mask_false = self.path_output / "Mask" / "FalseObj"
        self.path_overlay = self.path_output / "Overlay" / "Segmentation"
        self.path_ctrl_output = self.path_output / "Control" / "Final"
        self.path_ctrl_cluster = self.path_output / "Control" / "Cluster"
        self.path_result_cluster = self.path_output / "Control" / "Clusters"
        self.path_num_output_name = self.path_output / "Control" / "csv"

    def prepare_workspace(self):
        """
        Creates all required output directories
        """
        self.path_output.mkdir(parents=True, exist_ok=True)
        self.path_mask.mkdir(parents=True, exist_ok=True)
        self.path_mask_all.mkdir(parents=True, exist_ok=True)
        self.path_mask_false.mkdir(parents=True, exist_ok=True)
        self.path_overlay.mkdir(parents=True, exist_ok=True)
        self.path_ctrl_output.mkdir(parents=True, exist_ok=True)
        self.path_ctrl_cluster.mkdir(parents=True, exist_ok=True)
        self.path_result_cluster.mkdir(parents=True, exist_ok=True)
        self.path_num_output_name.mkdir(parents=True, exist_ok=True)

    def segment_image(self, img):
        """
        Segments an image using a pre-trained pixel classification model.
        :param img: The image to be processed.
        :return: The resulting binary segmentation mask.
        """
        # print('-segmenting image')

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
        mask_leaf_cl_dil_filter = utils.filter_objects_size(mask_leaf_cl_dil_filter, 40000, "greater")
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

        # add borders
        mask_spl = utils.add_image_border(mask_spl, intensity=0)
        mask_leaf_cl_dil_filter2 = utils.add_image_border(mask_leaf_cl_dil_filter2, intensity=0)

        return img_out_all_obj, mask_spl, mask_leaf_cl_dil_filter2

    def process_image(self, work_queue, result):

        for job in iter(work_queue.get, 'STOP'):

            image_name = job['image_name']
            img_name = image_name
            image_path = job['image_path']

            # ==========================================================================================================
            # Segment image
            # ==========================================================================================================

            # check if output exists - load from disk
            check_output_path = self.path_mask_all / (image_name + '.png')
            if check_output_path.exists():
                # print("Image", image_name, "already segmented, skip")
                img = imageio.imread(str(image_path))
                mask_all = imageio.imread(self.path_mask_all / (image_name + '.png'))
                mask_false = imageio.imread(self.path_mask_false / (image_name + '.png'))

            # load, segment and post-process image
            else:
                # load image
                # print("loading masks from disk")
                img = imageio.imread(str(image_path))

                # segment image
                mask = self.segment_image(img=img)

                # post-process mask
                overlay_all, mask_all, mask_false = self.post_process_segmentation_mask(img=img, mask=mask)

                # save output
                imageio.imwrite(self.path_overlay / (image_name + '.png'), overlay_all)
                imageio.imwrite(self.path_mask_all / (image_name + '.png'), mask_all)
                imageio.imwrite(self.path_mask_false / (image_name + '.png'), mask_false)

            img = utils.add_image_border(img, intensity=255)
            check_img = copy.copy(img)

            # ==========================================================================================================
            # Process lesions
            # ==========================================================================================================

            # get bounding boxes of all lesions
            rect, check_img = utils.get_bounding_boxes(mask_all, check_img)

            # image for output visualization
            ctrl_output = copy.copy(img)
            # draw all contours
            _, contours, _ = cv2.findContours(mask_all, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                cv2.drawContours(ctrl_output, contour, -1, (0, 0, 0), 1)

            # image for cluster visualization
            ctrl_cluster = copy.copy(img)

            output = []
            for i in range(len(rect)):

                # print(f'-lesion {i + 1}/{len(rect)}')

                # extract roi
                empty_mask_all, empty_mask, empty_img, ctr = utils.select_roi(rect=rect[i],
                                                                              img=img,
                                                                              mask=mask_all)

                # extract RGB profiles
                checker = copy.copy(img)
                prof, checker, spl, spl_points = fef.spline_contours(mask_obj=empty_mask,
                                                                     mask_all=empty_mask_all,
                                                                     mask_false=mask_false,
                                                                     img=empty_img,
                                                                     checker=checker)

                # initialize output data frame
                x = spl_points[0].astype("int").tolist()
                y = spl_points[1].astype("int").tolist()
                d = {'x': x, 'y': y}
                lesion_edge_coordinates = pd.DataFrame(data=d)
                lesion_edge_coordinates["class"] = "undefined"
                lesion_edge_coordinates["lesion_id"] = i+1
                lesion_edge_coordinates["cluster_id"] = "undefined"

                # extract pixel distances of spline base points from object centroid
                # and scale to attribute weight in clustering
                dists = utils.dist_to_centroid(spl_points, ctr, scale_factor=5)

                df, col_idx_kept, fig = fef.cluster_profiles(
                    profiles=prof,
                    distances=dists,
                    min_length_profile=60
                )
                # save clusters for first lesion for inspection
                if i == 0:
                    try:
                        fig.savefig(self.path_result_cluster / (image_name + '.png'), dpi=2400)
                    except AttributeError:
                        fig.figure.savefig(self.path_result_cluster / (image_name + '.png'), dpi=2400)
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
                _, contours, _ = cv2.findContours(mask_all, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                for contour in contours:
                    cv2.drawContours(ctrl_cluster, contour, -1, (0, 0, 255), 1)

                # ======================================================================================================

                # split full sampled profile into spatial clusters of profiles
                clusters, point_ids = fef.split_spatial_clusters(data=df,
                                                                 profile=prof,
                                                                 column_idx=col_idx_kept)

                # get ordering of columns
                template = pd.read_csv("Z:/Public/Jonas/001_LesionZoo/TestingData/template_varnames_v4.csv")
                cols = template.columns

                # create prediction for each cluster
                predicted_label = []
                for k, cluster in enumerate(clusters):

                    # if cluster consists of very few profiles, skip
                    # labels will be inferred from neighbours or none assigned if spatially isolated
                    if cluster.shape[1] <= 3:
                        continue

                    # print(f'-----cluster {k + 1}/{len(clusters)}')

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
                        # print("Encountered problem while extracting model parameters!")
                        continue

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
                    # and produce numerical output
                    for p in point_ids[k]:
                        coords = tuple([int(spl_points[1][p]), int(spl_points[0][p])])
                        if pred_lab == "neg":
                            cv2.circle(ctrl_output, coords, 1, (0, 0, 255), -1)
                        elif pred_lab == "pos":
                            cv2.circle(ctrl_output, coords, 1, (255, 0, 0), -1)
                        lesion_edge_coordinates.iloc[p, 2] = pred_lab
                        lesion_edge_coordinates.iloc[p, 4] = k+1

                output.append(lesion_edge_coordinates)
            if len(output) == 0:
                data = lesion_edge_coordinates
            else:
                data = pd.concat(output)

            # Save output
            data.to_csv(self.path_num_output_name / (image_name + '.csv'), index=False)
            imageio.imwrite(self.path_ctrl_output / (image_name + '.png'), ctrl_output)
            imageio.imwrite(self.path_ctrl_cluster / (image_name + '.png'), ctrl_cluster)
            result.put(img_name)

    def run(self):

        self.prepare_workspace()
        files = list(self.dir_to_process.glob("*.png"))
        image_paths = {}
        for i, file in enumerate(files):
            image_name = Path(file).stem

            # test if already processed - skip
            final_output_path = self.path_num_output_name / (image_name + '.csv')
            if final_output_path.exists():
                continue
            # otherwise add to jobs list
            else:
                image_path = self.dir_to_process / (image_name + ".png")
                image_paths[image_name] = image_path

        if len(image_paths) > 0:
            # make job and results queue
            m = Manager()
            jobs = m.Queue()
            results = m.Queue()
            processes = []
            # Progress bar counter
            max_jobs = len(image_paths)
            count = 0

            # Build up job queue
            for image_name, image_path in image_paths.items():
                print(image_name, "to queue")
                job = dict()
                job['image_name'] = image_name
                job['image_path'] = image_path
                jobs.put(job)

            # Start processes
            for w in range(multiprocessing.cpu_count() - 1):
                p = Process(target=self.process_image,
                            args=(jobs, results))
                p.daemon = True
                p.start()
                processes.append(p)
                jobs.put('STOP')

            print(str(len(image_paths)) + " jobs started, " + str(multiprocessing.cpu_count() - 1) + " workers")

            # Get results and increment counter along with it
            while count < max_jobs:
                img_names = results.get()
                count += 1
                print("processed " + str(count) + "/" + str(max_jobs))

            for p in processes:
                p.join()

# ======================================================================================================================
# CHECK OUTPUT
# ======================================================================================================================
#
# data = pd.read_csv("D:/LesionZoo/Output/V3/edge_labs/223_1_picture_6_leaf.csv")
#
# for index, row in data.iterrows():
#     coords = tuple([row['y'], row['x']])
#     label = row['class']
#     if label == "neg":
#         cv2.circle(ctrl_output, coords, 1, (0, 0, 255), -1)
#     elif label == "pos":
#         cv2.circle(ctrl_output, coords, 1, (255, 0, 0), -1)
#     elif label == "undefined":
#         cv2.circle(ctrl_output, coords, 1, (0, 0, 0), -1)
#
# plt.imshow(ctrl_output)

# ======================================================================================================================