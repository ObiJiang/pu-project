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
        self.k = config.k
        self.batch_size = config.batch_size
        self.n_unlabeled = config.n_unlabeled
        self.n_subsample_pairs = config.n_subsample_pairs

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

        self.create_varaibles(features)
        self.model()

    def create_varaibles(self,features):
        """ The following are variables used in the model (feature vectors and alpha) """
        # how feature vectors are generated
        if features is not None:
            self.features = tf.constant(self.features, dtype=tf.float32)
        else:
            # will be changed to random initialization later (done)
            self.emb_dim = 30
            self.features = tf.Variable(tf.random_normal([self.n_items, self.emb_dim],
                                            stddev=1 / (self.emb_dim ** 0.5), dtype=tf.float32))

        # subsample base item pairs. It is extremly rare that two items would be the same. If it is the same, it's not a big deal
        vi = tf.constant(np.random.randint(0,self.n_items,size=(self.n_subsample_pairs)),dtype=tf.int32)
        vj = tf.constant(np.random.randint(0,self.n_items,size=(self.n_subsample_pairs)),dtype=tf.int32)
        vi_vj = vi - vj
        self.base_matrices = tf.matmul(tf.expand_dims(tf.gather(self.features,vi_vj),2),
                                       tf.expand_dims(tf.gather(self.features,vi_vj),1)) #(batch,emb_dim,emb_dim)
        # generate alpha for all the users
        self.alpha = tf.Variable(tf.random_normal([self.n_users, self.n_subsample_pairs],
                                        stddev=1 / (self.n_subsample_pairs ** 0.5), dtype=tf.float32))

    def input_dataset_pipeline(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.train_user_item_pairs)
        dataset = dataset.map(self._map, num_parallel_calls=4)
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(1)
        iterator = dataset.make_initializable_iterator()
        return iterator,dataset

    def _map(self,positive_example):
        unlabeled_samples = tf.py_func(self._generate_unlabeled_samples, [positive_example], tf.int32)
        #negative_samples = np.random.randint(0,self.n_items,size=(self.n_unlabeled))

        return tf.concat([positive_example,unlabeled_samples],axis=0)

    def _generate_unlabeled_samples(self,positive_example):
        # TO DO: check
        return np.random.randint(0,self.n_items,size=(self.n_unlabeled),dtype=np.int32)

    def prior_estimation(self):
        # placeholder value 0.5
        return 0.5

    def _find_positive_nn(self,dist_users):
        dist,users = dist_users
        postivie_item_dist = self.train_user_to_positive_set[users]
        return postivie_item_dist,0

    def model(self):
        train_iterator,train_dataset = self.input_dataset_pipeline()

        handle = tf.placeholder(tf.string, shape=[],name='dataset_handle')
        iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)

        p_u = iterator.get_next() # p_u: [user, pos_item, {unlabel_item}]

        # find associated metrices with users in p_u
        alpha_in_batch = tf.gather(self.alpha,p_u[:,0])
        metrics_in_batch = tf.reduce_sum(tf.expand_dims(tf.expand_dims(alpha_in_batch,2),2) * self.base_matrices,
                             axis=1)

        # compute distances
        fea_in_batch = tf.gather(self.features,p_u[:,1:])
        fea_diff_in_batch = tf.expand_dims(fea_in_batch,2) - self.features
        dist_in_batch_part_1 = tf.einsum('bijm,bmn->bijn', fea_diff_in_batch, metrics_in_batch)
        dist_in_batch = tf.negative(tf.einsum('bijn,bijn->bij', fea_diff_in_batch, dist_in_batch_part_1))

        # compute nearest neighbors
        #pnn,_ = tf.map_fn(self._find_positive_nn, (dist_in_batch,p_u[:,0]),dtype=(tf.float32, tf.int32))

        with tf.Session() as sess:
            train_handle = sess.run(train_iterator.string_handle())
            sess.run(train_iterator.initializer)
            sess.run(tf.global_variables_initializer())
            print(sess.run(pnn,feed_dict = {handle: train_handle}))


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

    # without feature vectors
    pucml_learner = PUCML_Base(config,features=None,train=train,valid=valid,test=test)
    # with feature vectors
    #ucml_learner = PUCML_Base(config,features=dense_features,train=train,valid=valid,test=test)
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

    parser.add_argument('--k',
        action   = 'store',
        required = False,
        type     = int,
        default  = 5,
        help     = 'number k in knn')

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

    parser.add_argument('--n_subsample_pairs',
        action   = 'store',
        required = False,
        type     = int,
        default  = 100,
        help     = 'number of base subsample pairs')

    config = parser.parse_args()
    main_algo(config)
