import random
import numpy as np
from scipy.spatial.distance import cdist
import math
import cv2
from itertools import groupby
from operator import itemgetter
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


def extract_defect_segments_point_indices(contour, lag, pair_dist, elong_in, elong_out):
    """
    Get indices of contour points making up defect segments
    :param contour: A cv2 contour
    :param lag: A numeric, the distance in contour pixels between two points for which the pair-wise distance is
    calculated
    :param pair_dist: A numeric, the threshold value below which points are considered to close to each other
    :param elong_in: A numeric, the distance in contour pixels to which the segment is elongated inwards
    (i.e. towards the pair segment)
    :param elong_out: A numeric, the distance in contour pixels to which the segment is elongated outwards
    :return: A list of indices for the contour points located in defect regions
    """
    # contour length
    contour_len = len(contour[0])

    # create point lists with lag for pair-wise distance calculation
    point_list_x = flatten_contour_data(contour, asarray=False, as_point_list=True)
    point_idx = list(range(lag, contour_len)) + list(range(0, lag))
    point_list_y = [point_list_x[i] for i in point_idx]

    # calculate pair-wise distances
    dist = cdist(point_list_x, point_list_y)
    # take diagonal
    pair_wise_dist = np.diag(dist)
    # identify close-by points on contour
    # BOTH points of the pairwise comparison!
    prob_idx1 = np.where(pair_wise_dist < pair_dist)
    prob_idx2 = tuple([x + lag for x in prob_idx1])
    p1 = prob_idx1[0].tolist()
    p2 = prob_idx2[0].tolist()

    # extend the segments to both sides
    # segment 1
    separated_p1 = []
    for k, g in groupby(enumerate(p1), lambda i_x: i_x[0] - i_x[1]):
        sep = list(map(itemgetter(1), g))
        # extend the segments at both ends
        sep_ext = list(range(np.min(sep) - elong_in, np.min(sep))) + sep + list(
            range(np.max(sep), np.max(sep) + elong_out))
        # add to single list
        separated_p1.extend(sep_ext)
    # segment 2
    separated_p2 = []
    for k, g in groupby(enumerate(p2), lambda i_x: i_x[0] - i_x[1]):
        sep = list(map(itemgetter(1), g))
        # extend the segments at both ends
        sep_ext = list(range(np.min(sep) - elong_out, np.min(sep))) + sep + list(
            range(np.max(sep), np.max(sep) + elong_in))
        # add to single list
        separated_p2.extend(sep_ext)

    # merge "partner"-segments
    unified = tuple(np.sort(separated_p1 + separated_p2))
    unified = np.unique(unified)

    return unified


def extract_defect_contour_points(contour, point_indices):

    try:
        # if the problem spreads across the end/beginning of the contour
        if max(point_indices) > len(contour[0]):
            if point_indices[0] < 0:
                # identify where the break is in the contour
                x = np.ediff1d(point_indices)
                split_idx = np.where(x > 1)[0][0]
                end_idx = np.where(point_indices == len(contour[0]))[0][0]
                # create two segments (one before and one after the break)
                seg1 = point_indices[0:split_idx]
                seg1 = [item for item in seg1 if item >= 0]
                seg2 = point_indices[split_idx:end_idx].tolist()
                # merge segments
                point_indices = seg1 + seg2
            else:
                u = list(range(point_indices[0], len(contour[0])))
                ext = list(range(len(point_indices) - len(u)))
                point_indices = u + ext

        contour_warps = contour[0][point_indices]
        return contour_warps

    except ValueError:
        return None




def get_endpoint_normals(endpoints, length_in=20, length_out=80):
    """
    Get the endpoints of the normals at each end of the contour defect
    :param endpoints: A tuple of 4, each element is a list of 2, i.e. the coordinates of the endpoints (2 per side),
    these indicate the direction to which the normal vector is searched
    :param length_in: A numeric, indicating how far into the lesion the normal should point
    :param length_out: A numeric, indicating how far out of the lesion the normal should point
    :return: Endpoints of the normals. A tuple of 4 tuples, each of which is a pair of endpoint coordinates
    """
    # first normal
    v_x = endpoints[0][0] - endpoints[1][0]
    v_y = endpoints[0][1] - endpoints[1][1]
    mag = math.sqrt(v_x * v_x + v_y * v_y)
    v_x = v_x / mag
    v_y = v_y / mag
    temp = v_x
    v_x = -v_y
    v_y = temp
    A_x = int(endpoints[0][0] + v_x * length_in)
    A_y = int(endpoints[1][1] + v_y * length_in)
    B_x = int(endpoints[0][0] - v_x * length_out)
    B_y = int(endpoints[1][1] - v_y * length_out)
    # second normal
    v_x = endpoints[2][0] - endpoints[3][0]
    v_y = endpoints[2][1] - endpoints[3][1]
    mag = math.sqrt(v_x * v_x + v_y * v_y)
    v_x = v_x / mag
    v_y = v_y / mag
    temp = v_x
    v_x = -v_y
    v_y = temp
    C_x = int(endpoints[2][0] + v_x * length_out)
    C_y = int(endpoints[3][1] + v_y * length_out)
    D_x = int(endpoints[2][0] - v_x * length_in)
    D_y = int(endpoints[3][1] - v_y * length_in)

    return tuple([A_x, A_y]), tuple([B_x, B_y]), tuple([C_x, C_y]), tuple([D_x, D_y])


def get_defect_endpoints(point_list_defect, point_list_contour, check_img):
    """
    Extract endpoints of a local convexity defect in the contours
    :param point_list_defect: List of point coordinates of the defect contour
    :param point_list_contour: List of point coordinates of the original object contour
    :param check_img: an image on which the output is drawn
    :return: four endpoints (2 per end), used later to derive the direction of the contour and get a normal vector
    """
    # calculate the distance between points of the defect contour and the original contour
    dist = cdist(point_list_contour, point_list_defect)

    # get minimum distance
    min_dist = np.min(dist)
    # define proximity taking minimum distance into account
    dist_th = min_dist + 20

    # find defect contour points in close proximity to the original contour
    idx = np.unique(np.where(dist < dist_th)[0]).tolist()  # list index of points
    points = [point_list_contour[i] for i in idx]  # list of coordinate pairs
    cc = [np.array(points, dtype=np.int32)]  # cv2 contour format
    # cv2.drawContours(check_img, cc, -1, (0, 255, 0), 1)

    try:
        if idx[0] == 0:
            # identify where the break is in the contour
            x = np.ediff1d(idx)
            endidx = np.where(x > 1)[0][0]
            startidx = idx[endidx + 1]
            endpoint1 = point_list_contour[startidx]
            try:
                endpoint1_ = point_list_contour[startidx + 30]
            # if located before the contour break
            except IndexError:
                endpoint1_ = point_list_contour[len(point_list_contour)-1]
            endpoint2 = point_list_contour[endidx]
            endpoint2_ = cc[0][endidx - 31]
        else:
            # use all contour points between the smallest and greatest index
            start = idx[0]
            stop = idx[-1]
            idx = list(range(start, stop, 1))
            endpoint1 = point_list_contour[idx[0]]
            endpoint1_ = point_list_contour[idx[30]]
            endpoint2 = point_list_contour[idx[-1]]
            endpoint2_ = point_list_contour[idx[-31]]

        return endpoint1, endpoint1_, endpoint2, endpoint2_

    # if there are no points in close proximity to the original contour
    except IndexError:
        return None


def keep_largest_object(mask):
    """
    Filter objects in a binary mask to keep only the largest object
    :param mask: A binary mask to filter
    :return: A binary mask containing only the largest object of the original mask
    """
    _, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    ctr = np.around(centroids)[1:].astype("int")
    sizes = list(stats[:, 4][1:])
    index = sizes.index(np.max(sizes))
    cleaned = np.uint8(np.where(output == index + 1, 1, 0))

    return cleaned


def keep_farthest_object(mask, distance_th):
    """
    Filter objects in a binary mask to keep only the object farthest away from the image center
    :param distance_th: A numeric, the minimum distance between two objects to be considered separate defects
    :param mask: A binary mask to filter
    :return: A binary mask containing only the largest object of the original mask
    """
    _, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    mask_center = np.around(centroids)[0:1].astype("int")

    # calculate distance between centroids
    # and identify close-by objects
    # remove the object that is closer to the center of the image
    ctr = np.around(centroids)[1:].astype("int")
    dist = np.triu(cdist(ctr, ctr))
    subsetter = np.where((dist < distance_th) & (dist > 0))
    size = subsetter[0].size
    if not size == 0:
        idx1 = subsetter[0][0]
        idx2 = subsetter[1][0]
        dist_from_center = cdist(mask_center, ctr).tolist()[0]
        min_dist = np.min([dist_from_center[idx1], dist_from_center[idx2]])
        index = dist_from_center.index(min_dist)
        cleaned = np.uint8(np.where(output == index + 1, 0, mask))
    else:
        cleaned = mask

    return cleaned


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

    new_img = cv2.copyMakeBorder(
        img,
        top=85,
        bottom=85,
        left=85,
        right=85,
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


def get_spline_points(contour, interval=25):

    # get subset of contour points as point list
    contour_points = flatten_contour_data(contour, asarray=True)[::interval]
    x = contour_points[:, 1]
    y = contour_points[:, 0]

    t = range(len(contour_points))
    # ipl_t = np.linspace(0.0, len(contour_points) - 1, len(contour_points)*5)
    ipl_t = np.linspace(0.0, len(contour_points) - 1, len(contour_points)*interval*2)

    x_tup = si.splrep(t, x, k=3)
    y_tup = si.splrep(t, y, k=3)

    # x_tup = si.splrep(t, x, s=0.0, per=1, k=3)
    # y_tup = si.splrep(t, y, s=0.0, per=1, k=3)

    tck, u = si.splprep(contour_points.T, u=None, s=0.0, per=1, quiet=1)
    u_new = np.linspace(u.min(), u.max(), len(contour_points) * interval * 2)
    y_new, x_new = si.splev(u_new, tck, der=0)

    x_list = list(x_tup)
    # xl = x.tolist()
    # x_list[1] = xl + [0.0, 0.0, 0.0, 0.0]

    y_list = list(y_tup)
    # yl = y.tolist()
    # y_list[1] = yl + [0.0, 0.0, 0.0, 0.0]

    x_i = si.splev(ipl_t, x_list)
    y_i = si.splev(ipl_t, y_list)

    return x_i, y_i, x_new, y_new


def get_spline_normals(spline_points, lengthIn=35, LengthOut=40):

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
        A_x = int(y_i[i] + v_x * lengthIn)
        A_y = int(x_i[i + 1] + v_y * lengthIn)
        B_x = int(y_i[i] - v_x * LengthOut)
        B_y = int(x_i[i + 1] - v_y * LengthOut)
        n = [A_x, A_y], [B_x, B_y]
        endpoints.append(n)

    # get normals (lines)
    normals = []
    for i in range(len(endpoints)):
        p1 = endpoints[i][0]
        p2 = endpoints[i][1]
        discrete_line = list(zip(*line(*p1, *p2)))
        discrete_line = [[list(ele)] for ele in discrete_line]
        cc = [np.array(discrete_line, dtype=np.int32)]  # cv2 contour format
        # cv2.drawContours(img, cc, -1, (255, 0, 0), 1)
        normals.append(cc)

    return normals


def is_multi_channel_img(img):
    if len(img.shape) > 2 and img.shape[2] > 1:
        return True
    else:
        return False
