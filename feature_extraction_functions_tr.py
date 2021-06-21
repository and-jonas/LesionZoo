
import cv2
import numpy as np
import pandas as pd
import copy
from itertools import chain, repeat
from matplotlib import pyplot as plt
import utils_tr as utils_r
import utils as utils
from scipy import ndimage as ndi
from scipy import signal


def get_color_spaces(patch):
    """
    Make color transformations
    :param patch: The image or image patch for which to make the color transformations
    :return: The transformed images, stacked transformed images, and descriptor names
    """
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
    descriptor_names = ['R_RGB', 'G_RGB', 'B_RGB', 'H_HSV', 'S_HSV', 'V_HSV', 'L_Lab', 'a_Lab', 'b_Lab',
                        'L_Luv', 'u_Luv', 'v_Luv', 'Y_YUV', 'U_YUV', 'V_YUV', 'Y_YCC', 'Cb_YCC', 'Cr_YCC',
                        'ExG_RGB', 'ExR_RGB', 'TGI_RGB']

    # Return as tuple
    return (img_RGB, img_HSV, img_Lab, img_Luv, img_YUV, img_YCbCr, ExG, ExR, TGI), descriptors, descriptor_names


def plot_color_profile_clusters(cimg, profiles):
    """
    Plot clustered color profiles
    :param cimg:
    :param profiles:
    :return:
    """
    # transform to ndarray
    matrix = np.asarray(profiles)

    # extract cluster labels of profiles
    labs = matrix[:, -1].astype("int")

    # get number of clusters
    n_clusters = len(np.unique(labs))

    # get the indices of profiles belonging to each cluster
    index = []
    for i in range(n_clusters):
        ind = np.where(labs == i)[0]
        index.append(ind)

    # plot profiles per cluster
    fig, axs = plt.subplots(1, n_clusters, sharex=False, sharey=False)
    fig.subplots_adjust(hspace=.001, wspace=.1)
    axs = axs.ravel()

    for i in range(n_clusters):

        cimg_temp = cimg[:, index[i], :]

        # Show RGB and segmentation mask
        axs[i].imshow(cimg_temp)
        axs[i].set_title(f'Cluster {i}')

    return fig


def extract_normals_pixel_values(img, normals):
    """
    Extracts the pixel values situated on the spline normals.
    :param img: The image, binary mask or edt to process
    :param normals: The normals extracted in cv2 format as resulting from utils.get_spline_normals()
    :return: The "scan", i.e. an image (binary, single-channel 8-bit, or RGB) with stacked extracted profiles
    """
    # check whether is multi-channel image or 2d array
    is_img = utils.is_multi_channel_img(img)

    # For a normal perfectly aligned with the image axes, length equals the number of inward and outward pixels defined
    # utils.get_spline_normals()
    # All normals (differing in "pixel-length" due to varying orientation in space, are interpolated to the same length
    max_length_contour = 76

    # iterate over normals
    profile_list = []
    for k, normal in enumerate(normals):

        # get contour pixel coordinates
        contour_points = utils.flatten_contour_data(normal, asarray=False)

        # extract pixel values
        values = []
        for i, point in enumerate(contour_points):
            x = point[1]
            y = point[0]
            value = img[x, y].tolist()
            values.append(value)
            # split channels (R,G,B)
            # if img is a 3d array:
            if len(img.shape) > 2:
                channels = []
                for channel in range(img.shape[2]):
                    channel = [item[channel] for item in values]
                    channels.append(channel)
            else:
                channels = [values]

        # interpolate pixel values on contours to ensure equal length of all contours
        # for each channel
        interpolated_contours = []
        for channel in channels:
            size = len(channel)
            xloc = np.arange(len(channel))
            new_size = max_length_contour
            new_xloc = np.linspace(0, size, new_size)
            new_data = np.interp(new_xloc, xloc, channel).tolist()
            interpolated_contours.extend(new_data)

        if is_img:
            # create list of arrays
            line_scan = np.zeros([max_length_contour, 1, 3], dtype=np.uint8)
            for i in range(max_length_contour):
                v = interpolated_contours[i::max_length_contour]
                line_scan[i, :] = v
        else:
            line_scan = np.zeros([max_length_contour, 1], dtype=np.uint8)
            for i in range(max_length_contour):
                v = interpolated_contours[i::max_length_contour]
                line_scan[i, :] = v

        profile_list.append(line_scan)

    # stack arrays
    scan = np.hstack(profile_list)

    return scan


def check_color_profiles(color_profiles, dist_profiles, dist_profiles_outer, spline_normals, remove=False):
    """
    Removes spline normals (and corresponding color profiles) that (a) extend into the lesion sphere of the same lesion
    (convexity defects) and replaces values on the inner side of the spline normals that lie beyond the "center" of the
    lesion (i.e. extend too far inwards).
    :param color_profiles: A 3D array (an image), raw color profiles, sampled on the spline normals
    :param dist_profiles: the euclidian distance map
    :param dist_profiles_outer: the euclidian distance map of the inverse binary image
    :param spline_normals: the spline normals in cv2 format
    :param remove: Boolean, whether or not the spline "problematic" spline normals are removed.
    :return: Cleaned color profiles (after removing normals in convexity defects and replacing values of normals
    extending too far inwards) and the cleaned list of spline normals in cv2 format.
    """

    # (1) INWARDS ======================================================================================================

    # Normals end where they extend beyond the object "center"
    # Pixels lying beyond this are extrapolated, using the mean of the last four pixels of the same normal

    # calculate the differences in distance
    dist_profiles = dist_profiles.astype("int32")[:35]
    diff_in = np.diff(dist_profiles, axis=0)

    # identify where profiles extend beyond center for each profile
    ind = []
    for i in range(diff_in.shape[1]):
        result = np.where(diff_in[:, i] > 0)[0]
        if result.size > 0:
            cut_idx = np.max(np.where(diff_in[:, i] > 0))
        else:
            cut_idx = np.nan
        ind.append(cut_idx)

    # for all pixels above the first break in monotoneous increase,
    # replace pixel values with the average of the 4 preceding pixels on the same profile
    color_profiles_ = copy.copy(color_profiles)
    for i in range(color_profiles_.shape[1]):
        if ind[i] is not np.nan:
            color_profiles_[:ind[i], i] = np.mean(color_profiles_[ind[i]:ind[i]+3, i], axis=0)

    # (2) OUTWARDS =====================================================================================================

    dist_profiles_outer = dist_profiles_outer.astype("int32")[35:, ]
    diff_out = np.diff(dist_profiles_outer, axis=0)

    if remove:
        # remove problematic spline normals
        checker_out = np.unique(np.where(diff_out < 0)[1]).tolist()
        spline_normals_clean = [i for j, i in enumerate(spline_normals) if j not in checker_out]
        # remove corresponding color profiles
        color_profiles_ = np.delete(color_profiles_, checker_out, 1)
        return color_profiles_, spline_normals_clean

    else:
        # separate the normals into complete and incomplete
        cols = np.where(diff_out < 0)[1]
        spline_normals_fulllength = [i for j, i in enumerate(spline_normals) if j not in np.unique(cols)]
        spline_normals_redlength = [i for j, i in enumerate(spline_normals) if j in np.unique(cols)]

        # identify where profiles extend into the "sphere" of another near-by lesion
        ind = []
        for i in range(diff_out.shape[1]):
            result = np.where(diff_out[:, i] < 0)[0]
            if result.size > 0:
                cut_idx = np.min(np.where(diff_out[:, i] < 0))
            else:
                cut_idx = np.nan
            ind.append(cut_idx)

        # for all pixels above this intersection
        # replace pixels with white pixels
        for i in range(color_profiles_.shape[1]):
            if ind[i] is not np.nan:
                color_profiles_[35+ind[i]:, i] = (255, 255, 255)

        return color_profiles_, spline_normals_fulllength, spline_normals_redlength


def remove_neighbor_lesions(checked_profiles, dist_profiles_multi, spl_n_clean, check_img, remove=False):
    """
    Removes color profiles that extend into the "incluencing sphere" of close-by other lesions; ALTERNATIVELY,
    the limits of the "sphere of influence" of a lesion is extracted, and color profiles are maintained until this point,
    whereas pixels beyond this point are set to white.
    :param remove: Boolean; Wether "incomplete" color profiles should be removed.
    :param checked_profiles: A 3D array (an image). The color profiles returned by check_color_profiles
    :param dist_profiles_multi: Distance profiles, sampled spline normals (distance map on binary mask with all objects)
    :param spl_n_clean: The maintained spline normals in cv2 format
    :return: A subset of the input color profiles, where profiles exending into near-by lesions have been removed,
    The corresponding spline normals in cv2 format.
    """

    dist_profiles_multi_ = dist_profiles_multi.astype("int32")[35:, ]
    diff_out = np.diff(dist_profiles_multi_, axis=0)

    if remove:
        checker_out = np.unique(np.where(diff_out < 0)[1]).tolist()
        # remove problematic spline normals
        spline_normals_clean = [i for j, i in enumerate(spl_n_clean) if j not in checker_out]
        color_profiles_ = np.delete(checked_profiles, checker_out, 1)
        return color_profiles_, spline_normals_clean

    else:
        # separate the normals into complete and incomplete
        cols = np.where(diff_out < 0)[1]
        spline_normals_fulllength = [i for j, i in enumerate(spl_n_clean) if j not in np.unique(cols)]
        spline_normals_redlength = [i for j, i in enumerate(spl_n_clean) if j in np.unique(cols)]

        # identify where profiles extend into the "sphere" of another near-by lesion
        ind = []
        for i in range(diff_out.shape[1]):
            result = np.where(diff_out[:, i] < 0)[0]
            if result.size > 0:
                cut_idx = np.min(np.where(diff_out[:, i] < 0))
            else:
                cut_idx = np.nan
            ind.append(cut_idx)

        # for all pixels above this intersection
        # replace pixels with white pixels
        color_profiles_ = copy.copy(checked_profiles)
        for i in range(color_profiles_.shape[1]):
            if ind[i] is not np.nan:
                color_profiles_[35+ind[i]:, i] = (255, 255, 255)

        return color_profiles_, spline_normals_fulllength, spline_normals_redlength


def spline_contours(mask_obj, mask_all, img, checker):
    """
    Wrapper function for processing of contours via spline normals
    :param mask_obj: a binary mask containing only the lesion of interest.
    !!IMPORTANT!!: This must be in uint8 (i.e. 0 and 255), otherwise edt produces nonsense output!!
    :param mask_all: a binary mask containing all the segmented objects in the patch
    :param img: the original patch img
    :return: cleaned color profiles from contour normals in cv2 format, an image for evaluation
    """

    checker_filtered = copy.copy(checker)

    check_img = copy.copy(img)
    mask_invert = np.bitwise_not(mask_obj)

    distance = ndi.distance_transform_edt(mask_obj)
    distance_invert = ndi.distance_transform_edt(mask_invert)

    # get contour
    _, contour, _ = cv2.findContours(mask_obj, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # get spline points
    spline_points = utils_r.get_spline_points(contour)

    # get spline normals
    spl_n = utils.get_spline_normals(spline_points[2:])[::6]  # use old NEW points

    # CHECK OUTPUT
    pp2 = (spline_points[2].astype("int"), spline_points[3].astype("int"))  # NEW
    img2 = copy.copy(img)
    img2[pp2] = (0, 255, 0)
    checker[pp2] = (0, 255, 0)
    checker_filtered[pp2] = (0, 255, 0)
    for i in range(len(spl_n)):
        cv2.drawContours(img2, spl_n[i], -1, (255, 0, 0), 1)

    # sample the normals (on image and distance maps)
    color_profiles = extract_normals_pixel_values(img, spl_n)
    dist_profiles = extract_normals_pixel_values(distance, spl_n)
    dist_profiles_outer = extract_normals_pixel_values(distance_invert, spl_n)

    # remove normals that extend into lesion or beyond lesion "center"
    checked_profiles, spl_n_full, spl_n_red = check_color_profiles(
        color_profiles,
        dist_profiles,
        dist_profiles_outer,
        spl_n,
        remove=False
    )

    # remove normals extending into neighbor lesions
    mask_all_invert = np.bitwise_not(mask_all)
    distance_invert_all = ndi.distance_transform_edt(mask_all_invert)
    dist_profiles_multi = extract_normals_pixel_values(distance_invert_all, spl_n)
    final_profiles, spl_n_full, spl_n_red_l = remove_neighbor_lesions(
        checked_profiles,
        dist_profiles_multi,
        spl_n,
        check_img=check_img,
        remove=False
    )

    # create the check image: only complete profiles
    for i in range(len(spl_n_full)):
        cv2.drawContours(checker_filtered, spl_n_full[i], -1, (255, 0, 0), 1)
    # add contour
    _, contours, _ = cv2.findContours(mask_obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        cv2.drawContours(checker_filtered, c, -1, (0, 0, 255), 1)

    # create check image
    for i in range(len(spl_n_full)):
        cv2.drawContours(checker, spl_n_full[i], -1, (255, 0, 0), 1)
    for i in range(len(spl_n_red_l)):
        cv2.drawContours(checker, spl_n_red_l[i], -1, (0, 255, 0), 1)
    for i in range(len(spl_n_red)):
        cv2.drawContours(checker, spl_n_red[i], -1, (0, 122, 0), 1)
    # add contour
    _, contours, _ = cv2.findContours(mask_obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        cv2.drawContours(checker, c, -1, (0, 0, 255), 1)

    return final_profiles, (checker, checker_filtered), (spl_n, spl_n_full, spl_n_red_l, spl_n_red), spline_points


def extract_color_profiles(profile, task, scale, smooth=10):
    """
    Performs color transformation and scales values of each channel in each color space
    :param profile: An RGB image with stacked color profiles
    :param task: If "clustering", only RGB is used and extracted profiles are smoothed, as specified.
    :param scale: Boolean, if True the pixel values in each channel in each color space are scaled to range from 0 to 1.
    Do not use if task is "clustering".
    :param smooth: A numeric, indicating the window length for calculation of the moving average.
    :return: A data frame where each row corresponds to one sampled spline normal
    """
    # get all channels of all color spaces
    _, descriptors, descriptor_names = get_color_spaces(profile)
    descs = cv2.split(descriptors)

    # use only RGB for clustering to reduce computational time
    if task == "clustering":
        descs = descs[:3]
        descriptor_names = descriptor_names[:3]
        # smooth along the y-axis, using moving average
        desc_sm = []
        for desc in descs:
            t = np.ones(smooth)/smooth
            kernel = t.reshape(smooth, 1)
            # 2d-convolve each channel
            smoothed = signal.convolve2d(desc, kernel, mode='same')
            desc_sm.append(smoothed)
        descs = desc_sm

    # get a scaled profile for each channel of each color space
    if scale:
        descriptors_sc = []
        for desc, desc_name in zip(descs, descriptor_names):
            # get the 2- and 98-percentile
            perc_2 = np.percentile(desc, q=2)
            perc_98 = np.percentile(desc, q=98)
            # replace values higher or lower than the percentiles by the percentile value
            # this should be more robust than to scale to the full observed range
            desc = np.where(desc > perc_98, perc_98, desc)
            desc = np.where(desc < perc_2, perc_2, desc)
            # scale to 0...1, with percentiles as max and min values
            desc_sc = (desc - perc_2) / (perc_98 - perc_2)
            descriptors_sc.append(desc_sc)
    else:
        descriptors_sc = descs

    # flatten
    matrix = np.vstack(descriptors_sc)

    # stack color spaces
    matrix_ = matrix.transpose()
    df = pd.DataFrame(matrix_)

    # get the profile depths (variable due to size differences of lesions)
    profile_depth = len(descriptors_sc[0])
    pixel_in = 35
    pixel_out = 41

    # add column names
    if scale:
        name_type = "_sc"
    else:
        name_type = "_raw"
    name_channel = list(chain.from_iterable(zip(*repeat(descriptor_names, profile_depth))))
    name_pixel_position = [i for i in range(-pixel_in, pixel_out, 1)] * len(descs)
    colnames = ["{}_{}".format(a, b) for a, b in zip(name_channel, name_pixel_position)]
    colnames = [name + name_type for name in colnames]
    df.columns = colnames

    return df