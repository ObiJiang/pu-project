import numpy as np
from misc import citeulike, split_data
from scipy.sparse import lil_matrix
import tensorflow as tf
import argparse

class PUCML_Base():
    def __init__(self,config,features=None,train=None,valid=None,test=None):
        # hyper-parameter
        self.beta = config.beta   # tolerance margin
        self.gamma = config.gamma # disconted learning rate
        self.batch_size = config.batch_size
        self.n_unlabeled = config.n_unlabeled

        # dataset parameter
        self.n_users = config.n_users
        self.n_items = config.n_items

        # train, valid, test
        self.train = train
        self.valid = valid
        self.test = test

        # preprocess the trainning data
        train_user_item_matrix = lil_matrix(self.train)
        self.train_user_item_pairs = np.asarray(train_user_item_matrix.nonzero()).T
        self.train_user_to_positive_set = {u: set(row) for u, row in enumerate(train_user_item_matrix.rows)}

        # how feature vectors are generated
        if features is not None:
            self.features = tf.constant(features, dtype=tf.float32)
        else:
            # will be changed to random initialization later (done)
            self.emb_dim = 100
            self.features = tf.Variable(tf.random_normal([self.n_items, self.embed_dim],
                                            stddev=1 / (self.embed_dim ** 0.5), dtype=tf.float32))

        self._model()

    def input_dataset_pipeline(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.train_user_item_pairs)
        dataset = dataset.map(self._map, num_parallel_calls=4)
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(1)
        iterator = dataset.make_initializable_iterator()
        return iterator,dataset

    def _map(self,example):
        unlabeled_samples = tf.py_func(self._generate_unlabeled_samples, [example], tf.int64)
        #negative_samples = np.random.randint(0,self.n_items,size=(self.n_unlabeled))

        return example,unlabeled_samples

    def _generate_unlabeled_samples(self,example):
        return np.random.randint(0,self.n_items,size=(self.n_unlabeled))

    def prior_estimation(self):
        # placeholder value 0.5
        return 0.5

    def sample_base_item_pairs(self):
        pass

    def _model(self):
        train_iterator,train_dataset = self.input_dataset_pipeline()

        handle = tf.placeholder(tf.string, shape=[],name='dataset_handle')
        iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)

        p,u = iterator.get_next()

        
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

    pucml_learner = PUCML_Base(config,features=dense_features,train=train,valid=valid,test=test)
    # O,U = batch
    # while True:
    #     pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--beta',
        action   = 'store',
        required = False,
        type     = float,
        default  = 0.5,
        help     = 'tolerance margin')

    parser.add_argument('--gamma',
        action   = 'store',
        required = False,
        type     = float,
        default  = 0.5,
        help     = 'disconted learning rate')

    parser.add_argument('--batch_size',
        action   = 'store',
        required = False,
        type     = int,
        default  = 1000,
        help     = 'batch size')

    parser.add_argument('--n_unlabeled',
        action   = 'store',
        required = False,
        type     = int,
        default  = 20,
        help     = 'number of unlabeled data')

    config = parser.parse_args()
    main_algo(config)
