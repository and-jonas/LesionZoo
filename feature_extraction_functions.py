
import cv2
import numpy as np
import pandas as pd
import copy
from itertools import chain, repeat
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import utils
from scipy.spatial.distance import cdist
from scipy.spatial import distance
from scipy import ndimage as ndi
from scipy import signal
from scipy.cluster.vq import kmeans


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
    if n_clusters > 1:
        axs = axs.ravel()

    if n_clusters > 1:
        for i in range(n_clusters):
            cimg_temp = cimg[:, index[i], :]
            # Show RGB and segmentation mask
            axs[i].imshow(cimg_temp)
            axs[i].set_title(f'Cluster {i}')
    else:
        fig = plt.imshow(cimg)

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


def remove_neighbor_lesions(checked_profiles, dist_profiles_multi, spl_n_clean, remove=False):
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
    !!IMPORTANT!!: This must be in uint8 (i.e. 0 and 255), otherwise ndi.distance_transform_edt() produces nonsense
    output !!
    :param mask_all: a binary mask containing all the segmented objects in the patch
    :param img: the original patch image
    :param checker: A copy of the (full) image to process.
    :return: cleaned color profiles from contour normals in cv2 format, an image for evaluation
    """

    checker_filtered = copy.copy(checker)

    mask_invert = np.bitwise_not(mask_obj)

    # calculate the euclidian distance transforms on the original and inverted masks
    distance = ndi.distance_transform_edt(mask_obj)
    distance_invert = ndi.distance_transform_edt(mask_invert)

    # get contour
    _, contour, _ = cv2.findContours(mask_obj, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # get spline points
    spline_points = utils.spline_approx_contour(contour, interval=1, task="basepoints")

    # get spline normals
    spl_n = utils.get_spline_normals(spline_points)

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

    dist_profiles_multi = extract_normals_pixel_values(distance_invert_all,
                                                       spl_n)

    final_profiles, spl_n_full, spl_n_red_l = remove_neighbor_lesions(
        checked_profiles,
        dist_profiles_multi,
        spl_n,
        remove=False
    )

    # create the check image: only complete profiles
    for i in range(len(spl_n_full)):
        cv2.drawContours(checker_filtered, spl_n_full[i], -1, (255, 0, 0), 1)
    # add contour
    _, contours, _ = cv2.findContours(mask_obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        cv2.drawContours(checker_filtered, c, -1, (0, 0, 255), 1)

    # create the check image
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

    # fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    # # Show RGB and segmentation mask
    # axs[0].imshow(descs[0])
    # axs[0].set_title('original patch')
    # axs[1].imshow(desc_sm[0])
    # axs[1].set_title('original patch')
    # plt.show(block=True)

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
    name_channel = list(chain.from_iterable(zip(*repeat(descriptor_names, profile_depth))))
    name_pixel_position = [i for i in range(-pixel_in, pixel_out, 1)] * len(descs)
    colnames = ["{}_{}".format(a, b) for a, b in zip(name_channel, name_pixel_position)]
    df.columns = colnames

    return df


def cluster_complete_profiles(data):
    """
    Groups color profiles using k-means clustering. Clustering based on color profiles and distance of the spline normal
    basepoints to the lesion centroid. The "optimum" number of clusters is defined through a stop criterion based on
    within-cluster SS, or by a maximum value, defined through the total number of profiles making up the lesion.
    :param data: A dataFrame with the necessary information for all color profiles (from fef.extract_color_profiles()),
    and the distances between spline normal basepoint and lesion centroid (from  utils.dist_to_centroid())
    :return: The same dataFrame, with the cluster index for the color profiles as an additional column, the cluster
    centroid coordinates (required for assignment of incomplete profiles to clusters).
    """
    print("---clustering profiles...")

    # get number of complete profiles
    n_profs = data.shape[0]

    if n_profs > 20:

        # maximum number of clusters to evaluate
        max_n_clust = int(np.ceil(data.shape[0] / 30))

        # cluster data into various numbers of cluster
        # to speed up, only certain candidate numbers of clusters are evaluated, with increasing spacing for larger
        # numbers of clusters
        if max_n_clust <= 11:
            K = range(1, max_n_clust+1)
        elif max_n_clust <= 30:
            K = chain(range(1, 11), range(12, 20, 2))
        else:
            K = chain(range(1, 11), range(12, 20, 2), range(25, max_n_clust+1, 5))
        KM = [kmeans(data, k) for k in K]
        centroids = [cent for (cent, var) in KM]
        print(f'----making {len(centroids)} clusters')  # get cluster centroids

        # get average within-cluster sum of squares (residual variation)
        D_k = [cdist(data, cent, 'euclidean') for cent in centroids]
        dist = [np.min(D, axis=1) for D in D_k]
        avg_SS = [sum(d) / data.shape[0] for d in dist]

        # get the "optimal" number of clusters
        if any(avg_SS) < 1.0:
            best_n_clust = next(x for x, val in enumerate(avg_SS) if val < 1.0)
        else:
            best_n_clust = max_n_clust

        # perform k-means clustering on profiles, into identified optimal number of clusters
        k_means = KMeans(n_clusters=best_n_clust, random_state=0).fit(data)
        y = k_means.fit_predict(data)
        centroids = k_means.cluster_centers_
        data['Cluster'] = y
        cents = pd.DataFrame(centroids)
        cents['Cluster'] = list(range(best_n_clust))

    else:
        data['Cluster'] = 0
        cents = pd.DataFrame()

    return data, cents


def assign_incomplete_profiles(profile, full_profile, data_complete, data_partial, col_idx_complete, col_idx_partial,
                               centroids):
    """
    Assigns incomplete color profiles (extending into sphere of neighbouring lesions, or extending beyond leaf edge) to
    the closest cluster determined previously
    :param profile: The incomplete color profiles (RGB Image of stacked profiles)
    :param full_profile: The complete color profiles (RGB Image of stacked profiles)
    :param data_complete: The corresponding dataFrame with color and spatial information
    :param data_partial: The corresponding dataFrame with color and spatial information
    :param col_idx_complete: A list of indices of the complete color profiles
    :param col_idx_partial: A list of indices of the incomplete color profiles
    :param centroids: The centroid coordinates in feature space, resulting from fef.cluster_complete_profiles()
    :return: df, all_kept_cols, fig
    """
    # length of a profile in pixels
    prof_len = int((data_complete.shape[1]-2)/3)  # <<-- MAY NEED TO CHANGE THIS ACCORDING TO N CHANNELS USED!

    # if there are no distinct clusters (small objects), assign all partial profiles to the one "cluster"
    if centroids.empty:
        cluster_assign = [0] * data_partial.shape[0]

    # otherwise, assign to the cluster with the nearest centroids
    else:
        mask = profile == 255
        all_white = mask.sum(axis=2) == 3

        # get row indices of first missing pixels
        ind = []
        for i in range(all_white.shape[1]):
            result = np.where(all_white[:, i])[0]
            cut_idx = np.min(np.where(all_white[:, i]))
            ind.append(cut_idx)

        # loop over each incomplete color profile: assign to clusters by nearest neighbor approach
        cluster_assign = []
        for k in range(len(ind)):
            # select data without NAs
            # remove the outermost pixels that are affected by smoothing
            # do not remove innermost, as full profiles are treated in the exact same way.
            alist = []
            for i in range(3):
                list_of_cols = list(range(i * prof_len, i * prof_len + ind[k]))
                alist.extend(list_of_cols)
            data = data_partial.iloc[:, alist]
            spat = data_partial.iloc[:, -2:]
            data = pd.concat([data.reset_index(drop=True), spat], axis=1)

            # use cluster centroids for assignment
            data_full = centroids.iloc[:, alist]
            spat_full = centroids.iloc[:, -3:-1]
            cluster = centroids['Cluster']
            data_full = pd.concat([data_full.reset_index(drop=True), spat_full], axis=1)

            # calculate distances
            euclidean_distances = data_full.apply(lambda row: distance.euclidean(row, data.iloc[k]), axis=1)

            distance_frame = pd.DataFrame(data={"dist": euclidean_distances, "idx": euclidean_distances.index})
            distance_frame.sort_values("dist", inplace=True)

            # assign the same cluster as the nearest (complete) neighbor
            smallest = int(distance_frame.iloc[0]["idx"])
            cl = cluster.iloc[smallest]
            cluster_assign.append(cl)

    data_partial['Cluster'] = cluster_assign

    data_complete['row_ind'] = col_idx_complete
    data_partial['row_ind'] = col_idx_partial

    df = pd.concat([data_complete, data_partial])
    df.sort_values("row_ind", inplace=True)
    df = df.reset_index(drop=True)

    plot_dat = df.iloc[:, :int(data_complete.shape[1]-2)]
    cl = df[['Cluster']]
    plot_dat = pd.concat([plot_dat.reset_index(drop=True), cl], axis=1)

    # plot clustered profiles
    all_kept_cols = np.concatenate([col_idx_complete, col_idx_partial])
    all_kept_cols.sort()
    kept_profiles = full_profile[:, all_kept_cols, :]
    fig = plot_color_profile_clusters(kept_profiles, plot_dat)

    return df, all_kept_cols, fig


def cluster_profiles(profiles, distances, min_length_profile=60):
    """
    Groups similar (and near-by) color profiles by k-means clustering.
    :param profiles: The color profile image resulting from fef.spline_contours()
    :param distances: The distances between the spline normal basepoints and the lesion centroid, resulting from
    utils.dist_to_centroids().
    :param min_length_profile: The mimimum length a color profile must have to be assigned to a cluster.
    :return: A dataframe with the pixel values, the position relative to the centroid, and the cluster index;
    A list of indices, indicating which color profiles were kept; and a figure of the clusters for one lesion.
    """
    # unpack distances
    dx = distances[0]
    dy = distances[1]

    # remove profiles containing white background (from sampling beyond the leaf edge)
    mask = profiles == 255
    all_white = mask.sum(axis=2) == 3
    cols = list(range(all_white.shape[1]))
    cols_drop = np.unique(np.where(all_white != 0)[1])
    cols_keep_complete = np.setdiff1d(cols, cols_drop)
    complete = np.delete(profiles, cols_drop, axis=1)

    # if there are no complete profiles, return None (resulting in lesion not analyzed)
    if complete.shape[1] == 0:
        df = None
        col_idx_kept = None
        fig = None
        return df, col_idx_kept, fig

    else:
        # keep all profiles with a "reasonable" margin (e.g., 25 pixels)
        inv = np.logical_not(all_white)
        cols = np.where(inv)[1]
        rows = np.where(inv)[0]
        idx = np.where(rows < min_length_profile)
        cols_keep = np.unique(np.delete(cols, np.unique(idx)))  # identify profiles at least 65 pixels long
        cols_keep = np.setdiff1d(cols_keep, cols_keep_complete)  # remove the complete profiles (identified above)
        incomplete = profiles[:, cols_keep, :]

        # add distances to centroid
        df_complete = extract_color_profiles(complete, task="clustering", scale=False)
        df_complete['dx'] = dx[cols_keep_complete]
        df_complete['dy'] = dy[cols_keep_complete]

        # cluster complete profiles by KMeans
        df_clusters, centroids = cluster_complete_profiles(data=df_complete)

        # if there are incomplete profiles, assign them to the cluster with the nearest centroid
        if incomplete.shape[1] != 0:

            df_partial = extract_color_profiles(incomplete, task="clustering", scale=False)
            df_partial['dx'] = dx[cols_keep]
            df_partial['dy'] = dy[cols_keep]

            # assign incomplete profiles to clusters by nearest neighbor approach
            df, col_idx_kept, fig = assign_incomplete_profiles(
                profile=incomplete,
                full_profile=profiles,
                data_complete=df_clusters,
                data_partial=df_partial,
                col_idx_complete=cols_keep_complete,
                col_idx_partial=cols_keep,
                centroids=centroids
            )

        else:

            df = df_complete
            col_idx_kept = cols_keep_complete
            fig = plot_color_profile_clusters(complete, df)
            df['row_ind'] = col_idx_kept

        return df, col_idx_kept, fig
