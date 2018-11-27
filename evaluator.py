"""
The following code is adapted from https://github.com/changun/CollMetric/blob/master/evaluator.py.
"""

import tensorflow as tf
from scipy.sparse import lil_matrix
import toolz
import numpy as np
from tqdm import tqdm

# based on anchor vectors
class RecallEvaluator(object):
    def __init__(self, model, train_user_item_matrix, test_user_item_matrix):
        """
        Create a evaluator for recall@K evaluation
        :param model: the model we are going to evaluate
        :param train_user_item_matrix: the user-item pairs used in the training set. These pairs will be ignored
               in the recall calculation
        :param test_user_item_matrix: the held-out user-item pairs we make prediction against
        """
        self.model = model
        self.train_user_item_matrix = lil_matrix(train_user_item_matrix)
        self.test_user_item_matrix = lil_matrix(test_user_item_matrix)
        n_users = train_user_item_matrix.shape[0]
        self.user_to_test_set = {u: set(self.test_user_item_matrix.rows[u])
                                 for u in range(n_users) if self.test_user_item_matrix.rows[u]}

        if self.train_user_item_matrix is not None:
            self.user_to_train_set = {u: set(self.train_user_item_matrix.rows[u])
                                      for u in range(n_users) if self.train_user_item_matrix.rows[u]}
            self.max_train_count = max(len(row) for row in self.train_user_item_matrix.rows)
        else:
            self.max_train_count = 0

    def eval(self, sess, users, k=50):
        """
        Compute the Top-K recall for a particular user given the predicted scores to items
        :param users: the users to eval the recall
        :param k: compute the recall for the top K items
        :return: recall@K
        """
        # compute the top (K +  Max Number Of Training Items for any user) items for each user

        _, user_tops = sess.run(self.model.top_k,
                                {self.model.score_user_ids: users,self.k:k + self.max_train_count})
        recalls = []
        for user_id, tops in zip(users, user_tops):
            train_set = self.user_to_train_set.get(user_id, set())
            test_set = self.user_to_test_set.get(user_id, set())
            top_n_items = 0
            hits = 0
            for i in tops:
                # ignore item in the training set
                if i in train_set:
                    continue
                elif i in test_set:
                    hits += 1
                top_n_items += 1
                if top_n_items == k:
                    break
            recalls.append(hits / float(len(test_set)))
        return recalls

# use knn_scores
class RecallEvaluator_knn(object):
    def __init__(self, model, train_user_item_matrix, test_user_item_matrix, test_batch_size):
        """
        Create a evaluator for recall@K evaluation
        :param model: the model we are going to evaluate
        :param train_user_item_matrix: the user-item pairs used in the training set. These pairs will be ignored
               in the recall calculation
        :param test_user_item_matrix: the held-out user-item pairs we make prediction against
        """
        self.model = model
        self.test_batch_size = test_batch_size
        self.train_user_item_matrix = lil_matrix(train_user_item_matrix)
        self.test_user_item_matrix = lil_matrix(test_user_item_matrix)
        n_users = train_user_item_matrix.shape[0]
        self.n_items = train_user_item_matrix.shape[1]
        self.user_to_test_set = {u: set(self.test_user_item_matrix.rows[u])
                                 for u in range(n_users) if self.test_user_item_matrix.rows[u]}

        if self.train_user_item_matrix is not None:
            self.user_to_train_set = {u: set(self.train_user_item_matrix.rows[u])
                                      for u in range(n_users) if self.train_user_item_matrix.rows[u]}
            self.max_train_count = max(len(row) for row in self.train_user_item_matrix.rows)
        else:
            self.max_train_count = 0

    def eval(self, sess, users, k=50):
        """
        Compute the Top-K recall for a particular user given the predicted scores to items
        :param users: the users to eval the recall
        :param k: compute the recall for the top K items
        :return: recall@K
        """
        # compute the top (K +  Max Number Of Training Items for any user) items for each user

        top_k_num = k + self.max_train_count
        user_tops = np.zeros((len(users),top_k_num))
        for user_id,user in enumerate(tqdm(users)):
            all_item_scores = np.zeros(self.n_items)
            for items_chunk in toolz.partition_all(self.test_batch_size, np.arange(self.n_items)):
                items = list(items_chunk)
                items.insert(0, user)
                cur_u_i = np.array([items])
                item_scores = sess.run(self.model.item_scores,{self.model.u_i:cur_u_i})
                all_item_scores[cur_u_i[0,1:]] = item_scores[0,:]
                user_tops[user_id,:] = np.argpartition(all_item_scores, -top_k_num)[-top_k_num:]

        recalls = []
        for user_id, tops in zip(users, user_tops):
            train_set = self.user_to_train_set.get(user_id, set())
            test_set = self.user_to_test_set.get(user_id, set())
            top_n_items = 0
            hits = 0
            for i in tops:
                # ignore item in the training set
                if i in train_set:
                    continue
                elif i in test_set:
                    hits += 1
                top_n_items += 1
                if top_n_items == k:
                    break
            recalls.append(hits / float(len(test_set)))
        return recalls
