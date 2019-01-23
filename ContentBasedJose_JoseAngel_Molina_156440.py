"""
@Author:    Jose Angel Molina
@Date:      Nov 2018
@Company:   CIT
"""
import pandas as pd
import numpy as np
import random
import sklearn.metrics as sc1


class RecCBSys:
    """
    Constructor: It assign hyper parameters to the class, reads the dataset and assign the seed
    """

    def __init__(self, file_rates, file_users, file_movies):

        self.k = None
        self.v = None
        self.target_id = None
        # Seed the number of Jose: R00156440
        random.seed(156440)

        # Calculating precision and recall
        self.bias = (0, 0)
        self.conf_matrix = [[0, 0], [0, 0]]

        # Read rates file
        self.data_set_rates = np.genfromtxt(file_rates, int)

        # Read users file
        self.data_set_users = []

        for line in open(file_users, 'r'):
            splitted_line = line.split()

            li = [splitted_line[0]]
            for car in range(4, 23):

                if car == 4:
                    li.append(splitted_line[4][1])
                else:
                    li.append(splitted_line[car][0])

            self.data_set_users.append(li)

        self.data_set_users = np.array(self.data_set_users, int)

        # Read items file
        self.data_set_items = []
        for line in open(file_movies, 'r'):
            splitted_line = line.split('|')

            li = [splitted_line[0]]

            for car in range(4, 23):

                if car == 23:
                    li.append(splitted_line[car][0])
                else:
                    li.append(splitted_line[car])

            self.data_set_items.append(li)

        self.data_set_items = np.array(self.data_set_items, int)

    """
    It generates a prediction
    """
    def make_prediction(self, test_per, sim_func, k):

        # Get all users
        list_things = np.unique(self.data_set_rates[:, 0])

        # Percentage of testing users
        num = round(test_per * len(list_things) / 100)

        # Get the testing dataset
        testing_dataset = list_things[:num]

        # Series of all users
        users_to_predict = pd.Series(testing_dataset)

        # Apply a prediction and return the accuracy for each users od the testing dataset
        users_to_predict = users_to_predict.apply(
            lambda t: self.predict_obt(t, test_per, sim_func, k))

        print('Prediction for a testing percentage of {0}%, K = {1}, a similarity function using {2}'.format(test_per, k, sim_func))

        # Print accuracy of the recommender system
        print('The Recall of the method for like predictions is {0}'.format(
            self.conf_matrix[0][0] / (self.conf_matrix[0][0] + self.conf_matrix[1][0])))
        print('The Precission of the method for like predictions is {0}'.format(
            self.conf_matrix[0][0] / (self.conf_matrix[0][0] + self.conf_matrix[0][1])))

        pred = 'under predictive' if self.bias[0] > self.bias[1] else 'over predictive'

        und_pred = (self.bias[0] * 100) / (self.bias[0] + self.bias[1])
        print(
            'The recommender system tends to be {0}, with a percentage of {1}% for under prediction and {2}% for over prediction'.format(
                pred, round(und_pred, 2), round(100 - und_pred, 2)))

        abs_mean = 0
        sqr_mean = 0
        spearman = 0
        it_nnull = 0
        for seq in users_to_predict:
            if seq is not np.nan:
                it_nnull += 1
                abs_mean += seq[0]
                sqr_mean += seq[1]
                spearman += seq[2]

        abs_mean /= it_nnull
        sqr_mean /= it_nnull
        spearman /= it_nnull

        print('The Mean absolute error is {0}'.format(abs_mean))
        print('The Mean Squared Error is {0}'.format(np.sqrt(sqr_mean)))
        print('The general Spearman correlation for tested users is {0}'.format(spearman))

        print('\n\n')

    def predict_obt(self, user, tes_perc, sim_func, k):

        # Get all items of a given user
        rates_user_bool = self.data_set_rates[:, 0] == user
        items_user = self.data_set_rates[rates_user_bool][:, 1]

        # Calculate the percentage of testing items to predict
        perc = round(tes_perc * len(items_user) / 100)

        # Get items to be predicted randomly
        testing_items = np.random.choice(items_user, perc)

        if len(testing_items) != 0:

            # Dictionary containing the distance for each user-neighbour
            distance = {}

            # Dictionary containing the apropiate consolidation for values of the neighbour
            comps = {}

            # Niceties of user selected
            likes_user_prob = self.data_set_users[:, 0] == user
            likes_user_neig = self.data_set_users[likes_user_prob][0][1:]

            # Items to be compared
            training_items_a = list(set(items_user) - set(testing_items))

            # For all neighbours
            for neig in training_items_a:

                # Get all items of a neighbour
                likes_film_neig_ps = self.data_set_items[:, 0] == neig
                rates_user_neig = self.data_set_items[likes_film_neig_ps][0][1:]

                # Calculate the similarity function
                if sim_func == 'cosine':

                    dis = 0
                    conc = 0
                    for a, b in zip(likes_user_neig, rates_user_neig):
                        if a == b:
                            conc += 1
                        else:
                            dis += 1

                    distance[neig] = (conc - dis) / len(likes_user_neig)
                else:
                    distance[neig] = sc1.jaccard_similarity_score(likes_user_neig, rates_user_neig)

                comps[neig] = self.data_set_rates[(self.data_set_rates[:, 0] == user) & (self.data_set_rates[:,
                                                                                                 1] == neig)][:, 2]

            # Check if the candidate neighbours are less than k
            if len(training_items_a) > k:

                # Get real values of the user
                testing_real_rates = []
                for item in testing_items:
                    rat_a = self.data_set_rates[
                        (self.data_set_rates[:, 1] == item) & (self.data_set_rates[:, 0] == user)]

                    testing_real_rates.append(int(rat_a[:, 2]))

                listed_arr = sorted(distance, key=distance.__getitem__)[:k]

                pred = []

                for pr in range(len(testing_items)):

                    predic_val = 0
                    arrb = 0
                    abj = 0

                    for i in listed_arr:
                        arrb += (distance[i] * comps[i])
                        abj += distance[i]

                    predic_val += (arrb / abj)
                    pred.append(np.round(predic_val))

                # Now, we can make a rank position for the items both actual and predicted.
                rank_pred = {}
                rank_actual = {}
                index = 0
                for a, b, c in zip(pred, testing_real_rates, testing_items):
                    rank_pred[c] = a
                    rank_actual[c] = b
                    index += 1

                # The adjusted rank
                # List of items sorted
                rank_pred = sorted(rank_pred, key=rank_pred.__getitem__)
                rank_actual = sorted(rank_actual, key=rank_actual.__getitem__)

                # Calculate the Spearman rank correlation
                corr = 0
                for itx in testing_items:

                    ind_a = rank_pred.index(itx)
                    ind_b = rank_actual.index(itx)

                    corr += np.square(ind_a - ind_b)

                len_its = len(testing_items)

                if len_its == 1:
                    corr = 1 - corr
                else:
                    corr = 1 - ((6 * corr)/(len_its * (np.square(len_its) - 1)))

                # When it has been calculated the
                self.precission_recall(pred, testing_real_rates)
                self.bias_under_over(pred, testing_real_rates)
                acc_abs = np.mean(np.abs(np.subtract(pred, testing_real_rates)))
                acc_sqr = np.mean(np.square(np.subtract(pred, testing_real_rates)))

                return [acc_abs, acc_sqr, corr]

            else:
                return np.nan

    # Calculate the bias of prediction for the recommender
    def bias_under_over(self, predictions, actual_values):

        for val in range(len(predictions)):

            if predictions[val] != np.NaN:

                if predictions[val] < actual_values[val]:
                    self.bias = (self.bias[0] + 1, self.bias[1])
                elif predictions[val] > actual_values[val]:
                    self.bias = (self.bias[0], self.bias[1] + 1)

    # Calculate both the precision and recall
    def precission_recall(self, predictions, actual_values):

        predictions = list(map(lambda t: 1 if t >= 4 else 0, predictions))
        actual_values = list(map(lambda t: 1 if t >= 4 else 0, actual_values))

        for val in range(len(predictions)):

            if predictions[val] == 1:

                if actual_values[val] == 1:
                    self.conf_matrix[0][0] += 1
                else:
                    self.conf_matrix[1][0] += 1
            else:
                if actual_values[val] == 1:
                    self.conf_matrix[0][1] += 1
                else:
                    self.conf_matrix[1][1] += 1


rsj = RecCBSys('ratings.txt', 'userterms.txt', 'movies.txt')

# For part 2: % of test, Mean-cent/z-scores, Adj-cosine/pearson, knn

#rsj.make_prediction(5, 'cosine', 19)
for perc in range(5, 20, 5):
    for sim in ['jaccard', 'cosine']:
        for k in range(2, 19):
            rsj.make_prediction(perc, sim, k)




