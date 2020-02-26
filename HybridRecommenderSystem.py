from project2.ContentBasedJose import RecCBSys
from project2.HibridContentCF import HybSys
from project1.RecommSysJose import RecCFSys
import numpy as np


# It does the switching between both systems.
def switching(K, rs_cb, rs_cf):
    # Previously we decided to use the lowest achieved record is 0.625 of mean distance.
    # Using the Pearson correlation, K = 18, test percentage of 5 and using the mean centering.
    ratings_file = 'ratings.txt'
    data_set_rates = np.genfromtxt(ratings_file, int)
    # Get all users
    list_users = np.unique(data_set_rates[:, 0])

    # By default use CF
    use = True

    for user in list_users:

        rates_user_bool = data_set_rates[:, 0] == user
        items_user = data_set_rates[rates_user_bool][:, 1]

        if len(items_user) < K:
            use = False
            break

    # For parameter tuning, we choose the best parameters in the test.
    if use:
        rs_cf.make_prediction(5, True, 'ddsd', 18)
    else:
        rs_cb.make_prediction(5, 'cosine', 3)


# It does the weighting aggregation
def agg_wgt():
    tal = HybSys('ratings.txt', 'userterms.txt', 'movies.txt')
    tal.make_prediction(5, True, 'ddsd', 18)


rs_cb = RecCBSys('ratings.txt', 'userterms.txt', 'movies.txt')
rs_cf = RecCFSys('ratings.txt')

option = input('Introduce the technique used: (Switching (A), Aggregating/Weighting (B)): ')

while option != 'A' or option != 'B':
    option = input('Introduce the technique used: (Switching (A), Aggregating/Weighting (B)): ')

if option == 'A':
    switching(10, rs_cb, rs_cf)
else:
    agg_wgt()
