import numpy as np
from misc import citeulike, split_data, define_scope
import tensorflow as tf
import argparse

class PUCML_Base():
    def __init__(self,config,features=None):
        # hyper-parameter
        self.beta = config.beta   # tolerance margin
        self.gamma = config.gamma # disconted learning rate

        # dataset parameter
        self.n_users = config.n_users
        self.n_items = config.n_items


        if features is not None:
            self.features = tf.constant(features, dtype=tf.float32)
        else:
            # will be changed to random initialization later
            self.emb_dim = 100
            self.features = tf.Variable(tf.random_normal([self.n_items, self.embed_dim],
                                            stddev=1 / (self.embed_dim ** 0.5), dtype=tf.float32))

    def input_dataset_pipeline(self):
        pass
    def prior_estimation(self):
        # placeholder value 0.5
        return 0.5

    def sample_base_item_pairs(self):
        pass

    def model(self):
        # compute nearest neigbors

        # compute score functions

        # tf.cond for different optimization

def main_algo(config):
    # get user-item matrix
    user_item_matrix, features = citeulike(tag_occurence_thres=5)
    n_users, n_items = user_item_matrix.shape
    # make feature as dense matrix
    dense_features = features.toarray() + 1E-10
    # get train/valid/test user-item matrices
    train, valid, test = split_data(user_item_matrix)

    # add a few stuff to config
    config.n_users = n_users
    config.n_items = n_items

    pucml_learner = PUCML_Base(config,features=dense_features)
    # O,U = batch
    # while True:
    #     pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--beta',
        action   = 'store',
        required = True,
        type     = float,
        default  = 0.5,
        help     = 'tolerance margin')

    parser.add_argument('--gamma',
        action   = 'store',
        required = True,
        type     = float,
        default  = 0.5,
        help     = 'disconted learning rate')

    config = parser.parse_args()
    main_algo(config)
