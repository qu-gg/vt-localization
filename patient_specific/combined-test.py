"""
@file: combined-test.py
@author: Ryan Missel

Combined script that handles running all of the different patient-specific models on the same random initialization
within the segments predicted by the population model. Runs it over every unique pacing site as the target site for a
number of repeated trials.

Prints out averaged metrics at the end for each model, e.g. number of successful runs, average error distance per time
step, etc.
"""
from patient_specific.models.ccmodel import CCModel
from patient_specific.models.rsmodel import RSModel
from patient_specific.models.utilfuncs import *
from patient_specific.confi import *

import pandas as pd
import random
import os

# Ignoring warnings
import warnings
warnings.filterwarnings('ignore')


def get_seg_dataset(row, segs, data, coords):
    """
    Handles grabbing all of the points within the predicted segment and its neighbor
    """
    # Get the labels of the segments right next to the predicted segment
    counts = get_neighbors(row[0])

    # Getting converted segs from 16 segments to the 7 segment model we are using
    nums = []
    for num in counts:
        for seg in convert_1016(num):
            nums.append([seg, row[1]])

    # Grab all of the points within the predicted segment and its neighbors. If there aren't enough to initialize the
    # model with the NUM_POINTS_START param, return and skip it
    x, y = get_segment_dataset(nums, segs, data, coords)
    if x.shape[0] < NUM_POINTS_START + 1:
        return None, None

    # Sample from the segment dataset the number of starting points desired
    samples = random.sample(range(x.shape[0]), NUM_POINTS_START)
    train, labels = x[samples], y[samples]
    return train, labels


def get_random_dataset(data, coords):
    """
    Handles getting the random dataset for the randomly initialized models
    :param target_coord: target coordinate
    :param data: full dataset for a patient
    :param coords: full labels for a patient
    :return: x, y of size 4
    """
    indices = np.random.choice(range(0, data.shape[0]), NUM_POINTS_START, replace=False)
    return data[indices], coords[indices]


def print_model_stats(models, names):
    """
    Simple formatting function for the per step average accuracy of all the models 
    """
    for model, name in zip(models, names):
        print("Model {} ---".format(name))
        for i in range(NUM_STEPS):
            if len(model[i]) == 0:
                continue

            # Step # | Num Active Mean STD | Min Max
            print("Set len %2d | N: %5d %02.2f %02.2f | Min: %3d Max: %3d" %
                  (i + NUM_POINTS_START, len(model[i]), np.mean(model[i]), np.std(model[i]), np.min(model[i]), np.max(model[i])))
        print(" ")


def model_run(model, x, y, train, labels, target, target_coord, target_raw, successes, avg_sites, all_euclids, drop):
    """
    Handles running a single runthrough of a given model on a target site
    Train and labels are the initialized sets to start with (either random or by segment prediction)
    """
    # Run the training loop for the model
    euclids, _, _, success, nsites = model.run(x, y, train, labels, target, target_coord, target_raw)

    # Add to arrays if successful and add the per steps error
    if success:
        successes.append(euclids[-1])
        avg_sites.append(nsites)
    for j in range(len(euclids)):
        all_euclids[j].append(euclids[j])
        if j != 0:
            drop.append(euclids[j] - euclids[j - 1])
    return successes, avg_sites, all_euclids, drop


def main():
    """
    The loop throughout this function is kind of cumbersome, but needed to make it easy to test all models on the
    same given dataset/initialization for all patients

    It loops through each patient a number of times for different initializations and tests each model's runtime on it,
    gathering run statistics and aggregating them into the arrays at the top here for printing at the end
    """
    # Arrays for all metrics
    total_cases = 0
    optim, optim_steps = [], []
    alle, all_points = [], []

    ccri_euclids = [[] for _ in range(NUM_STEPS)]                   # Random init CC arrays
    ccri_successes, ccri_avg_sites = [], []
    ccri_drop = []

    rsri_euclids = [[] for _ in range(NUM_STEPS)]                   # Random init RS arrays
    rsri_successes, rsri_avg_sites = [], []
    rsri_drop = []

    ccsi_euclids = [[] for _ in range(NUM_STEPS)]                   # Segment init CC arrays
    ccsi_successes, ccsi_avg_sites = [], []
    ccsi_drop = []

    rrsi_euclids = [[] for _ in range(NUM_STEPS)]                   # Segment init RS arrays
    rrsi_successes, rrsi_avg_sites = [], []
    rrsi_drop = []

    # Loop through each patient, performing a number of trials
    for folder in os.listdir(DATA_PATH):
        print("------------{}-----------".format(folder))

        # Grabbing data for patient
        path = "{}/{}/".format(DATA_PATH, folder)
        raw = pd.read_csv(path + "raw.csv", header=None).to_numpy()
        data = pd.read_csv(path + "x.csv", header=None).to_numpy()[:, LEADS]
        coords = pd.read_csv(path + "coord.csv", header=None).to_numpy()
        segs = pd.read_csv(path + "y.csv", header=None).to_numpy()
        top = pd.read_csv(path + "top.csv", header=None).to_numpy()

        # Initializing the models to test for patient
        cc_model = CCModel(leads=LEADS, steps=NUM_STEPS, svr_c=SVR_C, cc=CC_THRES, cc_succ=CC_SUCC, mm=15, samp_raw=raw, samp_coords=coords)
        rs_model = RSModel(steps=NUM_STEPS, svr_c=SVR_C, samp_raw=raw, samp_coords=coords, cc_succ=CC_SUCC)

        # Looping through every point to test, n number of times for variance in initialization
        for _ in range(NUM_TRIALS):
            for idx in range(len(data)):
                # The target pacing site and its coordinate
                target, target_coord, target_raw = data[idx], coords[idx], raw[idx]

                # Removing target from dataset for this target
                ridx = np.where(np.all(target_coord == coords, axis=1))
                r = np.delete(raw, ridx, axis=0)
                x = np.delete(data, ridx, axis=0)
                y = np.delete(coords, ridx, axis=0)
                s = np.delete(segs, ridx, axis=0)

                # Get initial datasets for random and segment initializations
                random_x, random_y = get_random_dataset(x, y)
                segment_x, segment_y = get_seg_dataset([top[idx][3], top[idx][1]], s, x, y)
                if segment_x is None:
                    continue

                # Checking for no neighbors
                if NNEIGHBORS and get_num_neighbors(target_coord, y, 15) < 3:
                    continue

                # Increment total cases count
                total_cases += 1

                # If using the full set, test all models on every point
                if FULL_SET:
                    # All model testing
                    all_euclid, all_pred = build_target_model(SVR_C, x, y, target, target_coord)
                    pred_check = get_closest(5, all_pred, raw, coords)[1]
                    if pred_check is not None and check_cc_success(target_raw, pred_check) >= CC_SUCC:
                        all_points.append(len(x))
                        alle.append(all_euclid)

                    # Optim model testing
                    optim_x, optim_y = get_optim_dataset(25, target_coord, x, y)
                    if len(optim_y) > 0:
                        optim_euclid, optim_pred = build_target_model(SVR_C, optim_x, optim_y, target, target_coord)
                        optim_steps.append(len(optim_x))

                        pred_check = get_closest(5, optim_pred, raw, coords)[1]
                        if pred_check is not None and check_cc_success(target_raw, pred_check) >= CC_SUCC:
                            optim.append(optim_euclid)

                    # CCRI Model
                    ccri_successes, ccri_avg_sites, ccri_euclids, ccri_drop = model_run(cc_model, x, y, random_x,
                                                                                        random_y, target, target_coord, target_raw,
                                                                                        ccri_successes, ccri_avg_sites,
                                                                                        ccri_euclids, ccri_drop)

                    # CC SI model testing
                    ccsi_successes, ccsi_avg_sites, ccsi_euclids, ccsi_drop = model_run(cc_model, x, y, segment_x, segment_y,
                                                                                        target, target_coord, target_raw,
                                                                                        ccsi_successes, ccsi_avg_sites,
                                                                                        ccsi_euclids, ccsi_drop)

                    # RS RI model testing
                    rsri_successes, rsri_avg_sites, rsri_euclids, rsri_drop = model_run(rs_model, x, y, random_x, random_y,
                                                                                        target, target_coord, target_raw,
                                                                                        rsri_successes, rsri_avg_sites,
                                                                                        rsri_euclids, rsri_drop)

                    # RS SI model testing
                    rrsi_successes, rrsi_avg_sites, rrsi_euclids, rrsi_drop = model_run(rs_model, x, y, segment_x, segment_y,
                                                                                        target, target_coord, target_raw,
                                                                                        rrsi_successes, rrsi_avg_sites,
                                                                                        rrsi_euclids, rrsi_drop)

                # If not using the full model, only test on the successful targets of the CCSI model
                else:
                    # CC SI model testing
                    euclids, preds, sites, success, nsites = cc_model.run(x, y, segment_x, segment_y, target, target_coord, target_raw)
                    if success:
                        ccsi_avg_sites.append(nsites)
                        ccsi_successes.append(euclids[-1])
                        for j in range(len(euclids)):
                            ccsi_euclids[j].append(euclids[j])
                            if j != 0:
                                ccsi_drop.append(euclids[j] - euclids[j - 1])

                        # All model testing
                        all_euclid, all_pred = build_target_model(SVR_C, x, y, target, target_coord)
                        all_points.append(len(x))
                        alle.append(all_euclid)

                        # Optim model testing
                        optim_x, optim_y = get_optim_dataset(25, target_coord, x, y)
                        optim_euclid, optim_pred = build_target_model(SVR_C, optim_x, optim_y, target, target_coord)
                        optim_steps.append(len(optim_x))
                        optim.append(optim_euclid)

                        # RS RI model testing
                        rsri_successes, rsri_avg_sites, rsri_euclids, rsri_drop = model_run(rs_model, x, y, random_x,
                                                                                            random_y,
                                                                                            target, target_coord,
                                                                                            target_raw,
                                                                                            rsri_successes,
                                                                                            rsri_avg_sites,
                                                                                            rsri_euclids, rsri_drop)

                        # RS SI model testing
                        rrsi_successes, rrsi_avg_sites, rrsi_euclids, rrsi_drop = model_run(rs_model, x, y, segment_x,
                                                                                            segment_y,
                                                                                            target, target_coord,
                                                                                            target_raw,
                                                                                            rrsi_successes,
                                                                                            rrsi_avg_sites,
                                                                                            rrsi_euclids, rrsi_drop)

    # Print the file names used
    print("File set used: ", DATA_PATH)

    # Metric section here
    print("--- Average Success errors ---")
    print("Total number of cases: %d"                            %  total_cases)
    print("All Points: %.2f +- %.2f for %d cases" % (np.mean(alle), np.std(alle), len(alle)))
    print("Optim: %.2f +- %.2f for %d cases"      % (np.mean(optim), np.std(optim), len(optim)))
    print("CC RI: %.2f +- %.2f for %d cases"      % (np.mean(ccri_successes), np.std(ccri_successes), len(ccri_successes)))
    print("RS RI: %.2f +- %.2f for %d cases"      % (np.mean(rsri_successes), np.std(rsri_successes), len(rsri_successes)))
    print("CC SI: %.2f +- %.2f for %d cases"      % (np.mean(ccsi_successes), np.std(ccsi_successes), len(ccsi_successes)))
    print("RS SI: %.2f +- %.2f for %d cases"      % (np.mean(rrsi_successes), np.std(rrsi_successes), len(rrsi_successes)))

    print(" ")

    print("--- Average Num of Sites ---")
    print("All  : %.2f +- %.2f" % (np.mean(all_points), np.std(all_points)))
    print("Optim: %.2f +- %.2f" % (np.mean(optim_steps), np.std(optim_steps)))
    print("CCRI: %.2f +- %.2f" % (np.mean(ccri_avg_sites) + NUM_POINTS_START, np.std(ccri_avg_sites)))
    print("RSRI: %.2f +- %.2f" % (np.mean(rsri_avg_sites) + NUM_POINTS_START, np.std(rsri_avg_sites)))
    print("CCSI: %.2f +- %.2f" % (np.mean(ccsi_avg_sites) + NUM_POINTS_START, np.std(ccsi_avg_sites)))
    print("RRSI: %.2f +- %.2f" % (np.mean(rrsi_avg_sites) + NUM_POINTS_START, np.std(rrsi_avg_sites)))

    print(" ")

    print("--- Average Drop ---")
    print("CCRI: %.2f +- %.2f" % (np.mean(ccri_drop), np.std(ccri_drop)))
    print("RSRI: %.2f +- %.2f" % (np.mean(rsri_drop), np.std(rsri_drop)))
    print("CCSI: %.2f +- %.2f" % (np.mean(ccsi_drop), np.std(ccsi_drop)))
    print("RRSI: %.2f +- %.2f" % (np.mean(rrsi_drop), np.std(rrsi_drop)))

    print(" ")

    print("--- Success Rates ---")
    print("CCRI: %.2f" % (len(ccri_successes) / total_cases))
    print("RSRI: %.2f" % (len(rsri_successes) / total_cases))
    print("CCSI: %.2f" % (len(ccsi_successes) / total_cases))
    print("RRSI: %.2f" % (len(rrsi_successes) / total_cases))

    print(" ")

    print("--- Per Step Averages ---")
    print_model_stats([ccri_euclids, rsri_euclids, ccsi_euclids, rrsi_euclids], ["CC RI", "RS RI", "CC SI", "RS SI"])


if __name__ == '__main__':
    main()
