# Global Vars
DATA_PATH = "../data/patient-datasets/"
LEADS = [i for i in range(12)]

# How many random initializations of the models to perform
NUM_TRIALS = 20

# Number of steps to take in a single run
NUM_STEPS = 15

# How many randomly selected points to start with (within the predicted segments)
NUM_POINTS_START = 4

# Hyperparameter of the SVR
SVR_C = 50

# Checks whether to use the full set of data or whether to constrain the testing set to only the successful
# cases of the CCSI model
FULL_SET = False

# Correlation Coefficient Hyperparams
CC_THRES = .75
CC_SUCC = .90

# Force n neighbors within area an area of 15mm around the target site
# This is used to skip out on unrealistic tests
NNEIGHBORS = True
