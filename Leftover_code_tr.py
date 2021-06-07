
# from fef_tr


# Get contour coordinates
def get_cont_coordinates(contour, img):

    blank = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(blank, contour, -1, 125, 1)
    # extract coordinates
    coords_cont = np.where(blank == 125)
    return coords_cont


def resize_contour(img, mask_pp, distance_in, distance_out, interval):
    """
    Extracts contours of variable size for lesions
    :param img: The original image of the patch
    :param mask_pp: The post-processed binary mask of the same patch, with only the lesion of interest
    :param distance_in: A numeric. The distance that contours should extend inwards from the original contour
    :param distance_out: A numeric. The distance that contours should extend outwards from the original contour
    :param interval: A numeric. How far contours dist from each other.
    :return: A list of the contours in cv2 format; An image with contours drawn.
    """
    # get pixel range to scan
    distances = range(-distance_in, distance_out+1, interval)

    # checker image to print contours
    img_check = copy.copy(img)
    img_check = np.ascontiguousarray(img_check, dtype=np.uint8)

    # iterate over dilation steps and extract lesion contour
    contours = []
    for dist in distances:
        if dist < 0:
            kernel = np.ones((-dist, -dist), np.uint8)
            # erode image
            mask_post = morphology.erosion(mask_pp, kernel)
            # must ensure that only ONE object is present
            n_comps, output, stats, centroids = cv2.connectedComponentsWithStats(mask_post, connectivity=8)
            sizes = list(stats[:, 4][1:])
            try:
                index = sizes.index(np.max(sizes))
            # if no object is left after erosion
            except ValueError:
                continue
            mask_post = np.uint8(np.where(output == index + 1, 1, 0))
            # extract contour
            _, cnt, _ = cv2.findContours(mask_post, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img_check, cnt, -1, (0, 0, 255), 1)
        elif dist == 0:
            _, cnt, _ = cv2.findContours(mask_pp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img_check, cnt, -1, (255, 0, 0), 1)
        elif dist > 0:
            kernel = np.ones((dist, dist), np.uint8)
            mask_post = cv2.dilate(mask_pp, kernel)
            _, cnt, _ = cv2.findContours(mask_post, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img_check, cnt, -1, (0, 0, 255), 1)
            # # plot the first "shortened" contour (convexity defects)
            # if dist == 39:
            #     cv2.drawContours(img_check, cnt, -1, (0, 255, 0), 1)
        contours.append(cnt)

    return contours, img_check


def post_process_contours(contours, img):
    """
    Should identify "in-folded" regions in contours and split contours on the edge of these regions. Segments should
    then be returned for further processing.
    :param contours: A list of the original contours, in cv2 format
    :param img: the original image of the patch
    :return: So far, a check img and a mask. The check image contains all contours and their split points, if any exist.
    If none exist, only the original contour is drawn. Should later output a list, where each element contains a list
    of contour fragments. These should then be processed via an extended "extract_contour_pixel_values" where segments
    are interpolated separately and finally fused (tbd!).
    """
    # prepare masks
    check_img2 = copy.copy(img)  # image to visualize result
    normals_mask = np.zeros(check_img2.shape[:2]).astype("uint8")  # mask for contour normals at segment endpoints
    mask = np.zeros(check_img2.shape[:2]).astype("uint8")  # for contours

    # set some parameters
    lag = 75  # lag for pair-wise distance calculation
    length_in = 20  # inner extension of normals
    length_out = 80  # outer extension of normals
    elong_in = 60  # elongation of the defect segment towards inner side of defect
    elong_out = 80  # elongation of the defect segment towards outer side of defect
    pair_dist = 40  # distance in contour pixels between points used for pair-wise spatial distance calculation

    # generate some colors
    colors = []
    for k in range(10):
        colors.append(utils.random_color())

    # select outer contours only
    idx_orig = len(contours)-28
    outer_contours = contours[idx_orig:]

    # get outermost contour
    outest_contour = contours[len(contours)-1]
    mask_outest = np.zeros(check_img2.shape[:2]).astype("uint8")  # for outermost contour
    cv2.fillPoly(mask_outest, pts=outest_contour, color=(255, 255, 255))

    # get original contour
    original_contour = contours[idx_orig]
    mask_innest = np.zeros(check_img2.shape[:2]).astype("uint8")  # for original contour
    cv2.fillPoly(mask_innest, pts=original_contour, color=(255, 255, 255))

    # extract outer contour lengths
    contour_lengths = []
    for cont in outer_contours:
        clength = len(cont[0])
        contour_lengths.append(clength)

    # get problematic contours through differences in length of adjacent outer contours
    differences = np.ediff1d(contour_lengths)
    abnormals = np.where(differences < 0)[0]
    problem_contours = [outer_contours[i] for i in abnormals]

    # ==================================================================================================================

    # get a mask with all problematic parts
    for i in range(len(problem_contours)):

        print(i)

        # get contour
        cnt = problem_contours[i]

        point_indices = utils.extract_defect_segments_point_indices(
            contour=cnt,
            lag=lag,
            pair_dist=pair_dist,
            elong_in=elong_in,
            elong_out=elong_out
        )

        defect_contour = utils.extract_defect_contour_points(
            contour=cnt,
            point_indices=point_indices
        )

        #
        # # cv2.drawContours(check_img2, cnt, -1, (0, 255, 0), 1)
        # clen = len(cnt[0])
        #
        # # create point lists with lag for pair-wise distance calculation
        # point_list_x = utils.flatten_contour_data(cnt, asarray=False, as_point_list=True)
        # point_idx = list(range(lag, clen)) + list(range(0, lag))
        # point_list_y = [point_list_x[i] for i in point_idx]
        #
        # # calculate pair-wise distances
        # dist = cdist(point_list_x, point_list_y)
        # # take diagonal only
        # pair_wise_dist = np.diag(dist)
        # # identify close-by points on contour
        # # BOTH points of the pairwise comparison!
        # prob_idx1 = np.where(pair_wise_dist < pair_dist)
        # prob_idx2 = tuple([x+lag for x in prob_idx1])
        # p1 = prob_idx1[0].tolist()
        # p2 = prob_idx2[0].tolist()
        #
        # # extend the segments to both sides
        # separated_p1 = []
        # for k, g in groupby(enumerate(p1), lambda i_x: i_x[0] - i_x[1]):
        #     sep = list(map(itemgetter(1), g))
        #     # extend the segments at both ends
        #     sep_ext = list(range(np.min(sep) - elong_in, np.min(sep))) + sep + list(range(np.max(sep), np.max(sep) + elong_out))
        #     # add to single list
        #     separated_p1.extend(sep_ext)
        #
        # separated_p2 = []
        # for k, g in groupby(enumerate(p2), lambda i_x: i_x[0] - i_x[1]):
        #     sep = list(map(itemgetter(1), g))
        #     # extend the segments at both ends
        #     sep_ext = list(range(np.min(sep) - elong_out, np.min(sep))) + sep + list(range(np.max(sep), np.max(sep) + elong_in))
        #     # add to single list
        #     separated_p2.extend(sep_ext)
        #
        # # merge "partner"-segments
        # unified = tuple(np.sort(separated_p1 + separated_p2))
        # unified = np.unique(unified)
        #
        # try:
        #     # if the problem spreads across the end/beginning of the contour
        #     if max(unified) > len(cnt[0]):
        #         if unified[0] < 0:
        #             # identify where the break is in the contour
        #             x = np.ediff1d(unified)
        #             splitidx = np.where(x > 1)[0][0]
        #             endidx = np.where(unified == len(cnt[0]))[0][0]
        #             # create two segments (one before and one after the break)
        #             seg1 = unified[0:splitidx]
        #             seg1 = [item for item in seg1 if item >= 0]
        #             seg2 = unified[splitidx:endidx].tolist()
        #             # merge segments
        #             unified = seg1 + seg2
        #         else:
        #             u = list(range(unified[0], len(cnt[0])))
        #             ext = list(range(len(unified)-len(u)))
        #             unified = u + ext
        #
        # except ValueError:
        #     continue
        #
        # contour_warps = cnt[0][unified]

        # create check img and mask
        if defect_contour is not None:
            for s in range(len(defect_contour)):
                point = tuple(defect_contour[s][0])
                # cv2.circle(check_img2, point, 1, (255, 0, 0), -1)
                mask[point[1], point[0]] = 255

    # ==================================================================================================================

    # draw original contour
    cv2.drawContours(check_img2, original_contour, -1, (0, 0, 255), 1)
    # make this a point list
    plist = utils.flatten_contour_data(original_contour, asarray=False)

    # dilate mask with defects to get a single object per defect
    kernel = np.ones((15, 15), np.uint8)
    mask_post = cv2.dilate(mask, kernel)

    # ensure that only one object is retained per defect
    if problem_contours:
        mask_post = utils.keep_farthest_object(mask=mask_post, distance_th=120)

    # detect the contour of this object
    _, def_contour, _ = cv2.findContours(mask_post, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # ==================================================================================================================

    # get endpoints on the original contour that are close to the defect object
    # get the perpendicular line
    for cdef in def_contour:

        n_mask1 = np.zeros(check_img2.shape[:2]).astype("uint8")  # for first normal
        n_mask2 = copy.copy(n_mask1)  # for second normal
        # cv2.drawContours(check_img2, cdef, -1, (255, 0, 0), 1)
        # point list of the contour of the defects
        plist_def = utils.flatten_contour_data([cdef], asarray=False)

        # get the endpoints of the defects on the original contour
        endpoints = utils.get_defect_endpoints(point_list_defect=plist_def,
                                               point_list_contour=plist,
                                               check_img=check_img2)

        # cv2.drawContours(check_img2, cc, -1, (0, 255, 0), 1)
        #
        # # draw endpoints
        # for cnt in cc:
        #     # cv2.drawContours(check_img2, [cnt], -1, (255, 255, 255), 2)
        #
        # cv2.circle(check_img2, tuple(endpoints[0]), 1, (255, 0, 0), -1)
        # cv2.circle(check_img2, tuple(endpoints[1]), 1, (255, 0, 0), -1)
        # cv2.circle(check_img2, tuple(endpoints[2]), 1, (255, 0, 0), -1)
        # cv2.circle(check_img2, tuple(endpoints[3]), 1, (255, 0, 0), -1)

        # get endpoints of the normals at each end of the relevant stretch of contour
        normals = utils.get_endpoint_normals(endpoints, length_in=length_in, length_out=length_out)

        # draw normals onto image and empty mask
        # first normal
        cv2.line(check_img2, normals[0], normals[1], (255, 255, 255), 1)
        cv2.line(n_mask1, normals[0], normals[1], 255, 1)
        # second normal
        cv2.line(check_img2, normals[2], normals[3], (255, 255, 255), 1)
        cv2.line(n_mask2, normals[2], normals[3], 255, 1)

        # only keep the part of the normals located inside the scanned band
        band = mask_outest - mask_innest
        n_mask1 = np.bitwise_and(n_mask1, band)
        n_mask2 = np.bitwise_and(n_mask2, band)

        # only keep the largest segment (segment of interest)
        # small segments resulting from overlap with nearby regions are thereby excluded
        # normal 1
        n_mask1 = utils.keep_largest_object(n_mask1)
        # normal 2
        n_mask2 = utils.keep_largest_object(n_mask2)
        # combine
        combined = n_mask1 + n_mask2

        # transfer all normals to one final mask
        normals_mask = normals_mask + combined

    # ==================================================================================================================

    # normals must be shortened in order to avoid overlapping into the same lesion
    # intersection of normals with the filled largest contour
    all_normals = np.bitwise_and(normals_mask, mask_outest)

    # remove small objects
    # these are fragments of the normals that overlap with the lesion on another site
    mask_normals = utils.filter_objects_size(all_normals, size_th=20, dir="smaller")

    # remove large objects
    # these are entire normals, which do not extend through the full scanned range
    mask_normals = utils.filter_objects_size(mask_normals, size_th=95, dir="greater")

    # ==================================================================================================================

    # identify for each contour the start and end points of segments
    # get the segments
    for z, contour in enumerate(outer_contours):
        # empty mask for contour
        mask_intersect = np.zeros(check_img2.shape[:2]).astype("uint8")
        cv2.drawContours(mask_intersect, contour, -1, 255, 2)
        # find intersections of contour normals at endpoints and the z-th outer contour
        intersect = np.logical_and(mask_intersect, mask_normals).astype("uint8")
        # to always obtain an intersection, normals need to be drawn with at least 2px thickness
        # this can result in more than one pixel of intersection
        # this effect must be removed to obtain single endpoints
        _, _, _, centroids = cv2.connectedComponentsWithStats(intersect)
        ctr = np.around(centroids)[1:].astype("int").tolist()
        ctr_f = np.floor(centroids)[1:].astype("int").tolist()
        ctr_c = np.ceil(centroids)[1:].astype("int").tolist()

        # obtain the single endpoints
        endpoints = []
        for i in range(len(ctr)):
            cpoints_list = contour[0].tolist()
            try:
                ep_idx = cpoints_list.index([[ctr[i][0], ctr[i][1]]])
            except ValueError:
                try:
                    ep_idx = cpoints_list.index([[ctr_f[i][0], ctr_f[i][1]]])
                except ValueError:
                    try:
                        ep_idx = cpoints_list.index([[ctr_c[i][0], ctr_c[i][1]]])
                    # if everything fails search for closest point on contour
                    except ValueError:
                        cpoints_list_ = [x[0] for x in cpoints_list]
                        closest_point = utils.get_min_point([ctr_c[i][0], ctr_c[i][1]], cpoints_list_)
                        ep_idx = cpoints_list_.index([closest_point[0], closest_point[1]])

            endpoints.append(ep_idx)
        endpoints = np.sort(endpoints)

        # split the z-th outer contour into segments, using the endpoints
        segments = []
        for i in range(len(endpoints)):
            if not i == len(endpoints)-1:
                l = list(range(endpoints[i], endpoints[i+1]))
                seg = contour[0][l]
            else:
                l = list(range(endpoints[i], len(contour[0]))) + list(range(0, endpoints[0]))
                seg = contour[0][l]
            segments.append(seg)

        for f, seg in enumerate(segments):
            cv2.drawContours(check_img2, seg, -1, colors[f], 1)

    # ==================================================================================================================

    plt.imshow(check_img2)

    # fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
    # # Show RGB and segmentation mask
    # axs[0].imshow(mask_outest)
    # axs[0].set_title('original patch')
    # axs[1].imshow(mask_normals)
    # axs[1].set_title('original patch')
    # axs[2].imshow(all_normals)
    # axs[2].set_title('seg')
    # plt.show(block=True)

    return check_img2, mask


def extract_contour_pixel_values(img, contours):

    # find maximum length of contours
    # all contours will be scaled to this length
    length = []
    for contour in contours:
        l = len(contour[0])
        length.append(l)
    max_length_contour = max(length)

    # iterate over dilation levels
    scan_list = []
    for contour in contours:

        # get contour pixel coordinates
        contour_points = utils.flatten_contour_data(contour, asarray=False)

        # extract pixel values
        values = []
        for i, point in enumerate(contour_points):
            x = point[0]
            y = point[1]
            value = img[x, y].tolist()
            values.append(value)
            # split channels (R,G,B)
            channel1 = [item[0] for item in values]
            channel2 = [item[1] for item in values]
            channel3 = [item[2] for item in values]
            channels = [channel1, channel2, channel3]

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

        # create list of arrays
        line_scan = np.zeros([1, max_length_contour, 3], dtype=np.uint8)
        for i in range(max_length_contour):
            v = interpolated_contours[i::max_length_contour]
            line_scan[:, i] = v
        scan_list .append(line_scan)

    # stack arrays
    scan = np.vstack(scan_list)

    return scan


def extract_clustered_contour_color_profiles(contour_img, slice_by=5):

    # remove profiles containing white background (from sampling beyond the leaf edge)
    mask = contour_img == 255
    all_white = mask.sum(axis=2) == 3
    cols_drop = np.unique(np.where(all_white != 0)[1])
    contour_img = np.delete(contour_img, cols_drop, axis=1)

    # use only every n-th pixel profile
    cimg = contour_img[:, ::slice_by, :]

    plt.imshow(cimg)

    # get all channels of all color spaces
    _, descriptors, descriptor_names = get_color_spaces(cimg)
    descs = cv2.split(descriptors)

    # get a scaled profile for each channel of each color space
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
        desc_sc = (desc-perc_2)/(perc_98-perc_2)
        # # mean over rows
        # row_means = desc_sc.mean(axis=1)
        # # running mean
        # np.convolve(desc_sc, np.ones(3) / 3, mode='valid')
        descriptors_sc.append(desc_sc)

    # flatten
    matrix = np.vstack(descriptors_sc)

    # hstacked color spaces
    df = pd.DataFrame(matrix)

    # vstacked color spaces
    matrix_ = matrix.transpose()
    df = pd.DataFrame(matrix_)

    # get the profile depths (variable due to size differences of lesions)
    profile_depth = len(descriptors_sc[0])
    pixel_in = (profile_depth - 28) * 3
    pixel_out = 81 + 1

    # add column names
    name_channel = list(chain.from_iterable(zip(*repeat(descriptor_names, profile_depth))))
    name_pixel_position = [i for i in range(-pixel_in, pixel_out, 3)] * 21
    colnames = ["{}_{}".format(a, b) for a, b in zip(name_channel, name_pixel_position)]
    df.columns = colnames

    # # exclude profiles that contain white background (leaf edges)
    # df = df.drop(columns=df.columns[((df == 0).mean() > 0.1)], axis=1)

    # perform k-means clustering on profiles
    k_means = KMeans(n_clusters=3, random_state=0).fit(df)
    y = k_means.fit_predict(df)
    df['Cluster'] = y

    fig = plot_color_profile_clusters(cimg, df)

    # # for hstacked color spaces
    # # add row names
    # name_channel = list(chain.from_iterable(zip(*repeat(descriptor_names, 29))))
    # name_pixel_position = [i for i in range(-60, 80+1, 5)] * 21
    # # add row names to df
    # df['channel'] = name_channel
    # df['pos'] = name_pixel_position
    # # rearrange columns
    # cols = df.columns.tolist()
    # cols = cols[-2:] + cols[:-2]
    # df = df[cols]

    return df, fig


