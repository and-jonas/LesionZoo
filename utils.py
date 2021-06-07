import random
import numpy as np
from scipy.spatial.distance import cdist
import math
import cv2
import scipy.interpolate as si
from skimage.draw import line


def most_frequent(input_list):
    """
    Find the most frequent element of a list
    :param input_list: A list to search
    :return: The most frequent element in the list
    """
    counter = 0
    num = input_list[0]
    for i in input_list:
        curr_frequency = input_list.count(i)
        if curr_frequency > counter:
            counter = curr_frequency
            num = i
    return num


def random_color():
    levels = range(32, 256, 32)
    return tuple(random.choice(levels) for _ in range(3))


def sq_distance(x1, x2):
    return sum(map(lambda x: (x[0] - x[1]) ** 2, zip(x1, x2)))


def get_min_point(point, points):
    dists = list(map(lambda x: sq_distance(x, point), points))
    return points[dists.index(min(dists))]


def flatten_contour_data(input_contour, asarray, as_point_list=True):
    """
    Extract contour points from cv2 format into point list
    :param input_contour: The cv2 contour to extract
    :param asarray: Boolean, whether output should be returned as an array
    :param as_point_list: Boolean, whetheer output should be returned as a point list
    :return: array or list containing the contour point coordinate pairs
    """
    xs = []
    ys = []
    for point in input_contour[0]:
        x = point[0][1]
        y = point[0][0]
        xs.append(x)
        ys.append(y)
    if as_point_list:
        point_list = []
        # for a, b in zip(xs, ys):
        for a, b in zip(ys, xs):
            point_list.append([a, b])
            c = point_list
        if asarray:
            c = np.asarray(point_list)
        return c
    else:
        return xs, ys


def keep_central_object(mask):
    """
    Filter objects in a binary mask by centroid position
    :param mask: A binary mask to filter
    :return: A binary mask containing only the central object (by centroid position)
    """
    n_comps, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    ctr_img = centroids[0:1]
    dist = cdist(centroids[1:], ctr_img)
    min_idx = np.argmin(dist)
    lesion_mask = np.uint8(np.where(output == min_idx + 1, 255, 0))

    return lesion_mask


def filter_objects_size(mask, size_th, dir):
    """
    Filter objects in a binary mask by size
    :param mask: A binary mask to filter
    :param size_th: The size threshold used to filter (objects GREATER than the threshold will be kept)
    :return: A binary mask containing only objects greater than the specified threshold
    """
    _, output, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    sizes = stats[1:, -1]
    if dir == "greater":
        idx = (np.where(sizes > size_th)[0] + 1).tolist()
    if dir == "smaller":
        idx = (np.where(sizes < size_th)[0] + 1).tolist()
    out = np.in1d(output, idx).reshape(output.shape)
    cleaned = np.where(out, 0, mask)

    return cleaned

# ======================================================================================================================


def add_image_border(img, intensity):
    """
    Adds a border to images
    :param img: The image to be processed.
    :param intensity: A SINGLE numeric, indicating the pixel values of pixels in the border. 0 for binary mask,
    255 for RGB images.
    :return: The image or mask with added borders.
    """
    new_img = cv2.copyMakeBorder(
        img,
        top=80,
        bottom=80,
        left=80,
        right=80,
        borderType=cv2.BORDER_CONSTANT,
        value=[intensity, intensity, intensity]
    )

    return new_img


def make_point_list(input):

    xs = []
    ys = []
    for point in range(len(input)):
        x = input[point]
        y = input[point]
        xs.append(x)
        ys.append(y)
    point_list = []
    for a, b in zip(xs, ys):
        point_list.append([a, b])
    c = point_list
    return c


def spline_approx_contour(contour, interval, task="smoothing"):
    """
    Approximates lesion edges by b-splines
    :param contour: Contours detected in a binary mask.
    :param interval: A numeric, indicating the distance between contour points to be kept for fitting the b-spline. A
    higher value means more aggressive smoothing.
    :param task: A character vector, either "smoothing" or "basepoints". Affects where the b-spline is evaluated. If
    "smoothing", it is evaluated so often as to obtain a continuous contour. If "basepoints", it is evaluated at the
    desired distances to obtain spline normal base points coordinates.
    :return: x and y coordinates of pixels making up the smoothed contour, OR representing the spline normal base
    points.
    """
    # re-sample contour points
    contour_points = flatten_contour_data(contour, asarray=True)[::interval]
    # find B-Spline representation of contour
    tck, u = si.splprep(contour_points.T, u=None, s=0.0, per=1, quiet=1)
    # evaluate  B-spline
    if task == "smoothing":
        u_new = np.linspace(u.min(), u.max(), len(contour_points)*interval*2)
    elif task == "basepoints":
        u_new = np.linspace(u.min(), u.max(), int(len(contour_points)/4))
    y_new, x_new = si.splev(u_new, tck, der=0)
    # format output
    if task == "smoothing":
        x_new = x_new.astype("uint32")
        y_new = y_new.astype("uint32")
    return x_new, y_new


def get_spline_points(contour, interval):

    # get subset of contour points as point list
    # contour_points = flatten_contour_data(contour, asarray=True)
    contour_points = flatten_contour_data(contour, asarray=True)[::interval]

    x = contour_points[:, 1]
    y = contour_points[:, 0]

    t = range(len(contour_points))
    ipl_t = np.linspace(0.0, len(contour_points) - 3, len(contour_points))

    x_tup = si.splrep(x=t, y=x, xb=t[0], xe=t[0], k=3, per=True)
    y_tup = si.splrep(t, y, xb=t[0], xe=t[0], k=3, per=True)

    x_list = list(x_tup)
    xl = x.tolist()
    x_list[1] = xl + [0.0, 0.0, 0.0, 0.0]

    y_list = list(y_tup)
    yl = y.tolist()
    y_list[1] = yl + [0.0, 0.0, 0.0, 0.0]

    x_i = si.splev(ipl_t, x_list)
    y_i = si.splev(ipl_t, y_list)

    return x_i, y_i


def get_spline_normals(spline_points, length_in=35, length_out=40):
    """
    Gets spline normals (lines) in cv2 format
    :param spline_points: x- and y- coordinates of the spline base points, as returned by spline_approx_contour().
    :param length_in: A numeric, indicating how far splines should extend inwards
    :param length_out: A numeric, indicating how far splines should extend outwards
    :return: A list of the spline normals, each in cv2 format.
    """
    ps = np.vstack((spline_points[1], spline_points[0])).T

    x_i = spline_points[0]
    y_i = spline_points[1]

    # get endpoints of the normals
    endpoints = []
    for i in range(0, len(ps)-1):
        v_x = y_i[i] - y_i[i + 1]
        v_y = x_i[i] - x_i[i + 1]
        mag = math.sqrt(v_x * v_x + v_y * v_y)
        v_x = v_x / mag
        v_y = v_y / mag
        temp = v_x
        v_x = -v_y
        v_y = temp
        A_x = int(y_i[i] + v_x * length_in)
        A_y = int(x_i[i + 1] + v_y * length_in)
        B_x = int(y_i[i] - v_x * length_out)
        B_y = int(x_i[i + 1] - v_y * length_out)
        n = [A_x, A_y], [B_x, B_y]
        endpoints.append(n)

    # get normals (lines) connecting the endpoints
    normals = []
    for i in range(len(endpoints)):
        p1 = endpoints[i][0]
        p2 = endpoints[i][1]
        discrete_line = list(zip(*line(*p1, *p2)))
        discrete_line = [[list(ele)] for ele in discrete_line]
        cc = [np.array(discrete_line, dtype=np.int32)]  # cv2 contour format
        normals.append(cc)

    return normals


def get_bounding_boxes(mask, check_img):
    """
    Get bounding boxes of each maintained lesion in a full leaf image
    :param mask: Binary segmentation mask of the image to process
    :param check_img: A copy of the corresponding image
    :return: Coordinates of the bounding boxes as returned by cv2.boundingRect()
    """
    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    rect_coords = []
    for c in contours:
        # find bounding box of lesions
        x, y, w, h = cv2.boundingRect(c)
        # add buffer
        w = w + 150
        h = h + 150
        x = x - 75
        y = y - 75
        # boxes must not extend beyond the edges of the image
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        # draw bounding boxes for control
        cv2.rectangle(check_img, (x, y), (x + w, y + h), (255, 0, 0), 10)
        coords = x, y, w, h
        rect_coords.append(coords)

    return rect_coords, check_img


def select_roi(rect, img, mask):
    """
    Selects part of an image defined by bounding box coordinates. The selected patch is pasted onto empty masks for
    processing in correct spatial context
    :param rect: bounding box coordinates (x, y, w, h) as returned by cv2.boundingRect()
    :param img: The image to process
    :param mask: The binary mask of the same image
    :return: Roi of masks and img, centroid coordinates of the lesion to process (required for clustering)
    """
    # get the coordinates of the rectangle to process
    x, y, w, h = rect

    # create empty files for processing in spatial context
    empty_img = np.ones(img.shape).astype('int8') * 255
    empty_mask = np.zeros(mask.shape)
    empty_mask_all = np.zeros(mask.shape)

    # filter out the central object (i.e. lesion of interest)
    # isolate the rectangle
    patch_mask_all = mask[y:y + h, x:x + w]

    # select object by size or by centroid position in the patch
    n_comps, output, stats, centroids = cv2.connectedComponentsWithStats(patch_mask_all, connectivity=8)

    # if there is more then one object in the roi, need to select the one of interest
    if n_comps > 2:
        sizes = list(stats[:, 4][1:])
        max_size = np.max(sizes)
        max_idx = np.argmax(sizes)
        del sizes[max_idx]
        sec_max = np.max(sizes)

        # if one object is by far the largest, select it
        if max_size > 7*sec_max:
            lesion_mask = np.uint8(np.where(output == max_idx + 1, 255, 0))
            ctr_obj = [centroids[max_idx + 1][0] + x, centroids[max_idx + 1][1] + y]
        # if not, then select the central object
        else:
            ctr_img = centroids[0:1]
            dist = cdist(centroids[1:], ctr_img)
            min_idx = np.argmin(dist)
            lesion_mask = np.uint8(np.where(output == min_idx + 1, 255, 0))
            ctr_obj = [centroids[min_idx + 1][0] + x, centroids[min_idx + 1][1] + y]

    # if there is only one object, select it
    else:
        lesion_mask = np.uint8(np.where(output == 1, 255, 0))
        ctr_obj = [centroids[1][0] + x, centroids[1][1] + y]

    # paste the patches onto the empty files at the correct position
    empty_img[y:y + h, x:x + w, :] = img[y:y + h, x:x + w, :]
    empty_mask[y:y + h, x:x + w] = lesion_mask
    empty_mask_all[y:y + h, x:x + w] = patch_mask_all

    mask_all = empty_mask_all.astype("uint8")
    mask = empty_mask.astype("uint8")

    return mask_all, mask, empty_img, ctr_obj


def dist_to_centroid(spl_points, ctr, scale_factor=10):
    """
    Calculates the distance in x and y direction (in pixels) between spline basepoints and the lesion centroid.
    :param spl_points: The spline normals basepoints resulting from fef.spline_contours() and specifically,
    utils.spline_approx_contour().
    :param ctr: The centroid coordinates, as returned by utils.select_roi()
    :param scale_factor: A numeric, indicating how strongly the positional context should be weighed when clustering
    color profiles. The distances are scaled to the indicated range.
    :return: distance in x and y direction.
    """
    # distance between spline base points in x and y direction from object centroid
    dx = spl_points[1] - ctr[0]
    dy = spl_points[0] - ctr[1]

    # scale to 0...10
    dx = np.around((dx - np.min(dx)) / (np.max(dx) - np.min(dx)) * scale_factor, 2)
    dy = np.around((dy - min(dy)) / (max(dy) - min(dy)) * scale_factor, 2)

    return dx, dy


def is_multi_channel_img(img):
    """
    Checks whether the supplied image is multi- or single channel (binary mask or edt).
    :param img: The image, binary mask, or edt to process.
    :return: True if image is multi-channel, False if not.
    """
    if len(img.shape) > 2 and img.shape[2] > 1:
        return True
    else:
        return False
