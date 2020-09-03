"""
@file rsmodel.py
@author Ryan Missel

Class file for the Random Selection model
"""
from patient_specific.models.utilfuncs import *


class RSModel:
    def __init__(self, steps=20, svr_c=5, samp_raw=None, samp_coords=None, cc_succ=.9):
        # Basic hyperparameters of the runtime
        self.num_steps = steps
        self.svr_c = svr_c
        self.cc = cc_succ

        # Sampled data to pull from
        self.samp_coords = samp_coords
        self.samp_raw = samp_raw

    def run(self, x, y, train, labels, target, target_coord, target_raw):
        """
        Handles doing a full run of the random selection model on a single target site
        :param x: constrained dataset for a patient
        :param y: constrained labels for a patient
        :param train: initial training set derived from x
        :param labels: initial label set derived from y
        :param target: AUC of the target site
        :param target_coord: label for the target site
        :return: euclid error, model predictions, and added sites during run
        """
        rs_sites = list()
        rs_euclids = list()
        rs_preds = list()
        success = False
        num_sites = 0

        # Loop, adding another site to the training set at each stage and building a model
        for i in range(self.num_steps):
            euclid, pred = build_target_model(self.svr_c, train, labels, target, target_coord)
            rs_euclids.append(euclid)
            rs_preds.append(pred)

            _, pred_raw, pred_check = get_closest(5, pred, self.samp_raw, self.samp_coords)
            if get_euclid_error(target_coord, pred) <= 5 or (pred_check is not None and check_cc_success(pred_raw, target_raw) >= self.cc):
                success = True
                break

            # Shuffle the labels array and loop through to find any non-picked point
            sample = None
            for samp in np.random.choice(x.shape[0], x.shape[0]):
                if not np.array_equal(y[samp], target_coord) and not check(y[samp], labels):
                    sample = samp

            # If no samples are found
            if sample is None:
                break

            # while True:
            #     sample = np.random.choice(x.shape[0], 1).item()
            #     # print(np.array_equal(y[sample], target_coord), check(y[sample], labels))
            #     if not np.array_equal(y[sample], target_coord) and not check(y[sample], labels):
            #         break
            train = np.vstack((train, x[sample]))
            labels = np.vstack((labels, y[sample]))

            # adding site to list
            rs_sites.append(y[sample])
            num_sites += 1

        return rs_euclids, rs_preds, rs_sites, success, num_sites
