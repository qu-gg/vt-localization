"""

"""
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def compare_trunc_cc(ecgone, ecgtwo):
    """
    Handles shrinking the ecgs of the two given files down to the longest of the two to compare ecgs
    :return: ECG of the two
    """
    # Stacking and dropping zeros to longest one
    stack = np.vstack((ecgone, ecgtwo))
    stack = stack[:, ~np.all(stack == 0, axis=0)]

    # Returning CC between them
    return np.corrcoef(stack[0], stack[1])[0, 1]


def check_corr_coef(site, target, samp_coords, samp_data):
    """
    Function to check the correlation coefficient between the ECGs of two pacing sites. Used to check
    if the highest error next site is within a threshold of correlation.
    :param id: id of the patient
    :param site: the pacing site to grab next
    :param target: the ending target site being predicted for
    :return: correlation coef
    """
    # Find the ECGs/AUCs of the site and target sites
    site_idx, target_idx = None, None
    idx = 0
    for coord in samp_coords:
        if site_idx is not None and target_idx is not None:
            break
        if np.array_equal(np.round(site, 2), np.round(coord, 2)):
            site_idx = idx
        if np.array_equal(np.round(target, 2), np.round(coord, 2)):
            target_idx = idx
        idx += 1

    # Stacking and dropping zeros to longest one
    stack = np.vstack((samp_data[site_idx], samp_data[target_idx]))
    stack = stack[:, ~np.all(stack == 0, axis=0)]

    # Returning CC between them
    return np.corrcoef(stack[0], stack[1])[0, 1]


def get_next_closest_site(data, coords, thres, target, labels):
    """
    Handles grabbing the next closest coord within a euclidean threshold from the overall sampled dataset
    Ignores when the coord is already handled within the training set
    :param thres: euclidean distance to be within
    :param target: site of interest, the one with the highest error in the previous trial
    :param labels: array of already handled labels
    :return: qrs-ints and coordinate of next closest site
    """
    x, y, z = target[0], target[1], target[2]

    for i in range(len(coords)):
        if (coords[i][0] == x and coords[i][1] == y and coords[i][2] == z) or check(coords[i], labels):
            continue
        if np.sqrt((x - coords[i][0])**2 + (y - coords[i][1])**2 + (z - coords[i][2])**2) <= thres:
            return data[i], coords[i]
    return None, None


def get_closest(thres, target, data, labels):
    """
    Handles grabbing the next closest coord within a euclidean threshold from the overall sampled dataset
    :param target: site of interest
    :return: qrs-ints and coordinate of next closest site
    """
    r_index = None

    # Picking a threshold of 15mm in order to make sure points far out aren't picked
    threshold = thres
    for i in range(len(data)):
        if np.array_equal(labels[i], target):
            continue

        distance = get_euclid_error(target, labels[i])
        if distance <= threshold:
            threshold = distance
            r_index = i

    if r_index is None:
        return None, None, None

    return threshold, data[r_index], labels[r_index]


def get_next(thres, x, y, labels, origin, target):
    """
    Handles getting the next site to pace at within a given threshold that is not already within the
    dataset or the target coordinate itself
    :param thres: threshold to be within
    :param x: all x values for a patient
    :param y: all coords for a patient
    :param labels: current training labels
    :param origin: site to check near
    :param target: target site for a given run
    :return: data, label if found
    """
    for idx in range(len(y)):
        if not np.array_equal(origin, y[idx]) and not np.array_equal(target, y[idx]) \
                and not check(y[idx], labels) and get_euclid_error(origin, y[idx]) <= thres:
            return x[idx], y[idx]
    return None, None


def get_segment_dataset(idxs, segs, data, coords):
    indices = []
    for s in range(len(segs)):
        for idx in idxs:
            if np.array_equal(segs[s], idx):
                indices.append(s)
    return data[indices], coords[indices]


def get_euclid_error(s1, s2):
    # Helper function to calculate euclidean distance between two 3D points
    err_x, err_y, err_z = s1[0] - s2[0], s1[1] - s2[1], s1[2] - s2[2]
    euclid = np.sqrt(err_x ** 2 + err_y ** 2 + err_z ** 2)
    return euclid.item()


def check(test, array):
    """
    Simple generator to check whether a 1d array lies within a multidimensional array
    :param test: value to test for
    :param array: multidimensional array to check through
    :return: true iff test in array
    """
    return any(np.array_equal(x, test) for x in array)


def build_target_model(svr_c, train, labels, target, target_coord):
    """
    Helper function to build a model based off a train set and predicting for a target site, calculating
    the resulting euclidean distance between them
    :param train: training set
    :param labels: labels for the train set
    :param target: true, target site being predicted for
    :param target_coord: coord of true site
    :return: euclid dist between pred and actual
    """
    model_x = SVR(gamma='scale', C=svr_c).fit(train, labels[:, 0].ravel())
    model_y = SVR(gamma='scale', C=svr_c).fit(train, labels[:, 1].ravel())
    model_z = SVR(gamma='scale', C=svr_c).fit(train, labels[:, 2].ravel())

    target = np.reshape(target, [1, -1])
    pred_x = model_x.predict(target)
    pred_y = model_y.predict(target)
    pred_z = model_z.predict(target)
    site = [pred_x.item(), pred_y.item(), pred_z.item()]

    euclid = get_euclid_error(site, target_coord)
    return euclid, site


def convert_1016(num):
    """ Function that handles converting 10 seg to 16 seg """
    if num == 1:
        return [1, 2]
    elif num == 2:
        return [3, 4]
    elif num == 3:
        return [5, 6]
    elif num == 4:
        return [7, 8]
    elif num == 5:
        return [9, 10]
    elif num == 6:
        return [11, 12]
    elif num in [7, 8, 9, 10]:
        return [13, 14, 15, 16]


def get_neighbors(num):
    if num == 1:
        return [1, 4]
    elif num == 2:
        return [2, 5]
    elif num == 3:
        return [3, 6]
    elif num == 4:
        return [1, 4, 7, 8]
    elif num == 5:
        return [2, 5, 8, 9]
    elif num == 6:
        return [3, 6, 10]
    elif num in [7, 8, 9, 10]:
        return [4, 5, 6, 7, 8, 9, 10]


def get_num_neighbors(target, coords, thres):
    # Handles getting num neighbors within a threshold for a target site
    number = 0
    for coord in coords:
        if not np.array_equal(target, coord) and get_euclid_error(target, coord) <= thres:
            number += 1
    return number


def graph3d(fignum, preds, tl, cc_euclids, coords, labels):
    """
    Handles graphing all the data for a given runthrough
    :param preds: preds of the model over time
    :param tl: target coord
    :param cc_euclids: euclidean distance between final and tl
    :param coords: full coord set for patient
    :param labels: set of initial training points
    """
    fig = plt.figure(fignum)
    ax = fig.gca(projection='3d')
    ax.set_facecolor('gray')

    c = np.delete(coords, np.where(np.all(tl == coords)), axis=0)
    c = np.delete(c, np.where(np.all(c == labels)), axis=0)

    # Plotting lines between preds
    for idx in range(len(preds) - 1):
        ax.plot(xs=[preds[idx][0], preds[idx + 1][0]], ys=[preds[idx][1], preds[idx + 1][1]],
                zs=[preds[idx][2], preds[idx + 1][2]], color='black')

    # Plotting the line between the target and final pred
    ax.plot(xs=[tl[0], preds[-1][0]], ys=[tl[1], preds[-1][1]], zs=[tl[2], preds[-1][2]])

    # Plotting the inital training set
    ax.scatter(xs=labels[:, 0], ys=labels[:, 1], zs=labels[:, 2], color="green" )

    # Plotting the target site in blue
    ax.scatter(xs=tl[0], ys=tl[1], zs=tl[2], color="red")

    # Plotting the rest of the unused data
    ax.scatter(xs=c[:, 0], ys=c[:, 1], zs=c[:, 2], color="yellow",marker="x")

    # Plotting all of the predictions
    ax.scatter(xs=preds[:, 0], ys=preds[:, 1], zs=preds[:, 2], color="black")


def get_optim_dataset(thres, target, data, coords):
    x, y = [], []
    for i in range(len(data)):
        if not np.array_equal(target, coords[i]) and get_euclid_error(target, coords[i]) <= thres:
            x.append(data[i])
            y.append(coords[i])
    return np.array(x), np.array(y)


def check_cc_success(pred_raw, target_raw):
    """
    Handles checking for the success cases where all 12/12 leads match (>.9) in the ECG file
    :param pred_raw:
    :param target_raw:
    :return:
    """
    pred_raw = np.reshape(pred_raw, [12, -1])
    target_raw = np.reshape(target_raw, [12, -1])
    nums = 0
    for i in range(12):
        ccs = np.corrcoef(pred_raw[i], target_raw[i])[0, 1]
        if ccs > .9:
            nums += 1

    return True if nums == 12 else False

