import numpy as np
from misc import citeulike, split_data
import tensorflow as tf
class PUCML_Base():
    def __init__(self, prior=.7, lam=1):
        self.prior = prior
        self.lam = lam

    def fit(self, x, y):
        pass

    def predict(self, x):
        check_is_fitted(self, 'coef_')
        x = check_array(x)
        return np.sign(.1 + np.sign(self._basis(x).dot(self.coef_)))

    def score(self, x, y):
        x_s, x_u = x[y == 1, :], x[y == 0, :]
        f = self.predict
        p_p = self.prior
        p_n = 1 - self.prior

        # SU risk estimator with zero-one loss
        r_s = (np.sign(-f(x_s)) - np.sign(f(x_s))) * p_p / (p_p - p_n)
        r_u = (-p_n * (1 - np.sign(f(x_u))) + p_p * (1 - np.sign(-f(x_u)))) / (p_p - p_n)
        return r_s.mean() + r_u.mean()

    def _basis(self, x):
        # linear basis
        return np.hstack((x, np.ones((len(x), 1))))

def main_algo():
    # get user-item matrix
    user_item_matrix, features = citeulike(tag_occurence_thres=5)
    n_users, n_items = user_item_matrix.shape
    # make feature as dense matrix
    dense_features = features.toarray() + 1E-10
    # get train/valid/test user-item matrices
    train, valid, test = split_data(user_item_matrix)
    # O,U = batch
    # while True:
    #     pass

main_algo()
