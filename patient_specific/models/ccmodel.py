"""
@file ccmodel.py
@author Ryan Missel

Handles the class and its functions for training and running the CC-Constraint model
"""
from patient_specific.models.utilfuncs import *

# Ignoring warnings
import warnings
warnings.filterwarnings('ignore')


class CCModel:
    def __init__(self, leads=None, svr_c=5, steps=20, mm=15, cc=.75, cc_succ=.9, samp_coords=None, samp_raw=None):
        # Which leads to use
        self.leads = leads if leads is not None else [i for i in range(12)]

        # Hyper params
        self.cc_thres = cc
        self.cc_succ = cc_succ
        self.mm_thres = mm
        self.num_steps = steps
        self.svr_c = svr_c

        # Sampled data to pull from
        self.samp_coords = samp_coords
        self.samp_raw = samp_raw

    def find_interest_site(self, train, labels, target_coord):
        """
        Handles finding the site of interest for the CC model by building a model for each training site and testing
        the CC of that site with the target ECG. Selects the point with the highest error that's within the CC bounds
        :param train: current training set
        :param labels: current label set
        :param target_coord: coordinate of the target site
        :return: the coordinate of the site that meets the criterion
        """
        interest_site = None

        sites = []
        errors = []
        ccs = []
        for i in range(len(train)):
            # Copying data arrays, extracting target site
            x, y = train.copy(), labels.copy()
            target, label = x[i], y[i]
            x = np.delete(x, i, axis=0)
            y = np.delete(y, i, axis=0)

            # Build model, get euclid error back from it
            euclid, pred = build_target_model(self.svr_c, x, y, target, label)
            errors.append(euclid)
            sites.append(label)

            # Get cc for that site and target
            ccs.append(check_corr_coef(label, target_coord, self.samp_coords, self.samp_raw))

        # Sorting both arrays by error distance
        errors, ccs, sites = np.array(errors), np.array(ccs), np.array(sites)
        inds = (-errors).argsort()
        errors, ccs, sites = errors[inds], ccs[inds], sites[inds]

        # Finding the largest error site within the cc thres
        for i in range(len(ccs)):
            if ccs[i] >= self.cc_thres:
                interest_site = sites[i]
                break

        return interest_site

    def run(self, x, y, train, labels, target, target_coord, target_raw):
        """
        Handles doing a full run of the CC model on a single target site
        :param x: dataset for a patient
        :param y: labels for a patient
        :param train: initial training set derived from x
        :param labels: initial label set derived from y
        :param target: AUC of the target site
        :param target_coord: label for the target site
        :return: euclid error, model predictions, and added sites during run
        """
        cc_euclids = list()
        cc_preds = list()
        cc_sites = list()
        success = False
        num_sites = 0

        """ Inital pred """
        # Predicting on target site to test error on target site
        euclid, pred = build_target_model(self.svr_c, train, labels, target, target_coord)

        # Adding to metric arrays
        cc_preds.append(pred)
        cc_euclids.append(euclid)

        _, pred_raw, pred_check = get_closest(5, pred, self.samp_raw, self.samp_coords)
        if pred_check is not None and check_cc_success(pred_raw, target_raw) >= self.cc_succ:
            success = True
            return cc_euclids, cc_preds, cc_sites, success, num_sites

        # Loop that involves finding site with highest localization error, finding closest site, and retraining
        for i in range(self.num_steps):
            # Predicting on target site to test error on target site
            euclid, pred = build_target_model(self.svr_c, train, labels, target, target_coord)

            """ Pred exploitation """
            pred_x, pred_c = get_next(15, x, y, labels, pred, target_coord)
            if pred_x is not None:
                train = np.vstack((train, np.reshape(pred_x, [1, len(self.leads)])))
                labels = np.vstack((labels, np.reshape(pred_c, [1, 3])))
                cc_sites.append(pred_c)

                # Repredicting with pred site if it were added
                euclid, pred = build_target_model(self.svr_c, train, labels, target, target_coord)

                # Adding to metric arrays
                cc_preds.append(pred)
                cc_euclids.append(euclid)

                # Check for successful termination
                _, pred_raw, pred_check = get_closest(5, pred, self.samp_raw, self.samp_coords)
                if pred_check is not None and check_cc_success(pred_raw, target_raw) >= self.cc_succ:
                    success = True
                    break
                num_sites += 1

            """ Input space exploration """
            interest_site = self.find_interest_site(train, labels, target_coord)

            # When there is no new points to add
            if interest_site is None and pred_x is None:
                break

            # Check for sites near interest
            if interest_site is not None:
                # If interest site and predicted site are within 5mm, skip
                if get_euclid_error(interest_site, pred) <= 5:
                    continue

                # Grab next closest site to site of interest within threshold
                next_x, next_c = get_next(self.mm_thres, x, y, labels, interest_site, target_coord)

                # Break if no available sites to go to
                if pred_x is None and next_x is None:
                    break

                # Appending to train sets
                if next_x is not None:
                    train = np.vstack((train, np.reshape(next_x, [1, len(self.leads)])))
                    labels = np.vstack((labels, np.reshape(next_c, [1, 3])))
                    cc_sites.append(next_c)

                    """ Model testing """
                    # Predicting on target site to test error on target site
                    euclid, pred = build_target_model(self.svr_c, train, labels, target, target_coord)

                    # Adding to metric arrays
                    cc_preds.append(pred)
                    cc_euclids.append(euclid)

                    _, pred_raw, pred_check = get_closest(5, pred, self.samp_raw, self.samp_coords)
                    if pred_check is not None and check_cc_success(pred_raw, target_raw) >= self.cc_succ:
                        success = True
                        num_sites = len(train)
                        break
                    num_sites += 1

        return cc_euclids, cc_preds, cc_sites, success, num_sites
