import numpy as np
from misc import citeulike, split_data, AttrDict
from evaluator import RecallEvaluator, RecallEvaluator_knn
from scipy.sparse import lil_matrix
import tensorflow as tf
import toolz
import argparse
from tqdm import tqdm
from prior_est import prior_estimation_data_matrix
from tensorflow.python import debug as tf_debug

""" Import PCA-related stuff from sklearn """
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class PUCML_Base():
    def __init__(self,config,features=None,train=None,valid=None,test=None):
        # hyper-parameter
        self.beta = config.beta   # tolerance margin
        self.gamma = config.gamma # disconted learning rate
        self.prior = config.prior
        self.lr = config.lr
        self.k = config.k
        self.batch_size = config.batch_size
        self.test_batch_size = config.test_batch_size # for testing only see validation for details
        self.n_unlabeled = config.n_unlabeled
        self.n_subsample_pairs = config.n_subsample_pairs
        self.evaluation_loop_num = config.evaluation_loop_num

        # prior
        self.prior = self.prior_estimation()

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
        self.total_num_user_item = len(self.train_user_item_pairs)

        self.train_user_to_positive_set = {u: set(row) for u, row in enumerate(train_user_item_matrix.rows)}

        # creat indices map on postive items
        n_p_elements_per_row = []
        for u, row in enumerate(train_user_item_matrix.rows):
            n_p_elements_per_row.append(len(row))
        self.n_p_elements_per_row_tf = tf.constant(n_p_elements_per_row, dtype=tf.int32)
        self.max_n_p_elem = max(n_p_elements_per_row)

        user_postive_ind_map_numpy = -1*np.ones((self.n_users,self.max_n_p_elem),dtype=np.int32)
        for u, row in enumerate(train_user_item_matrix.rows):
            if row:
                user_postive_ind_map_numpy[u,:len(row)] = np.array(row)
        self.user_postive_ind_map = tf.constant(user_postive_ind_map_numpy, dtype=tf.int32)

        self.create_varaibles(features)
        self.model = self.model()
        self.val_model = self.valuation_model()

    def create_varaibles(self,features):
        """ The following are variables used in the model (feature vectors and alpha) """
        # how feature vectors are generated
        if features is not None:
            self.features = tf.constant(features, dtype=tf.float32)
        else:
            self.emb_dim = 100
            self.features = tf.Variable(tf.random_normal([self.n_items, self.emb_dim],
                                            stddev=1 / (self.emb_dim ** 0.5), dtype=tf.float32))

        self.base_w = tf.random_normal([self.n_subsample_pairs, self.emb_dim],stddev=1 / (self.emb_dim ** 0.5), dtype=tf.float32)

        # generate alpha for all the users
        self.pre_alpha = tf.Variable(tf.random_normal([self.n_users, self.n_subsample_pairs],
                                     stddev=1 / (self.n_subsample_pairs ** 0.5), dtype=tf.float32))
        self.alpha = tf.exp(self.pre_alpha)

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
        return self.prior

    def _find_positive_nn(self,dist_users):
        dist,users = dist_users
        postivie_item_dist = self.train_user_to_positive_set[users]
        return postivie_item_dist,0

    def model(self):
        train_iterator,train_dataset = self.input_dataset_pipeline()

        handle = tf.placeholder(tf.string, shape=[],name='dataset_handle')
        iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)

        p_u = iterator.get_next() # p_u: [user, pos_item, {unlabel_item}]

        """ find associated metrices with users in p_u """
        alpha_in_batch = tf.gather(self.alpha,p_u[:,0])
        weights_in_batch = tf.matmul(alpha_in_batch,self.base_w)

        """ compute score functions """
        fea_in_batch = tf.gather(self.features,p_u[:,1:])
        confidence_scores = tf.matmul(fea_in_batch,tf.expand_dims(weights_in_batch,axis=2))

        p_scores = confidence_scores[:,0]
        u_scores = confidence_scores[:,1:]

        # R_p_plus = tf.reduce_mean(1/(1 + tf.exp(p_scores)))
        # R_p_minus = tf.reduce_mean(1/(1 + tf.exp(-1*p_scores)))
        # P_u_minus = tf.reduce_mean(1/(1 + tf.exp(-1*u_scores)))

        R_p_plus = tf.reduce_mean(-p_scores)
        R_p_minus = tf.reduce_mean(p_scores)
        P_u_minus = tf.reduce_mean(u_scores)

        # R_p_plus = tf.reduce_mean(-1*pnn_dist_sum)
        # R_p_minus = tf.reduce_mean(pnn_dist_sum)
        # P_u_minus = tf.reduce_mean(unn_dist_sum)

        """ define loss and optimization """
        # define two differnt losses and their optimizer
        total_loss = self.prior * R_p_plus + (P_u_minus - self.prior * R_p_minus)# + self.feature_loss#+ tf.nn.l2_loss(self.pre_alpha)
        negative_loss = P_u_minus - self.prior * R_p_minus

        full_opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(total_loss)
        neg_opt = tf.train.AdamOptimizer(learning_rate=self.lr*self.gamma).minimize(-1*negative_loss)

        # tf.cond for different optimization
        # selctive_opt = tf.cond(negative_loss > self.beta, lambda: full_opt, lambda: neg_opt)
        selctive_opt = full_opt

        return AttrDict(locals())  # The magic line.

    # return scores for a couple of users
    def valuation_model(self):
        score_user_ids = tf.placeholder(tf.int32, [None])

        """ find associated metrices with users in p_u """
        alpha_in_batch = tf.gather(self.alpha,score_user_ids)
        weights_in_batch = tf.matmul(alpha_in_batch,self.base_w)

        """ compute score functions """
        item_scores = tf.matmul(weights_in_batch,tf.transpose(self.features))

        return AttrDict(locals())

    def train_main(self):
        model = self.model

        """ Evaluation Set-up """
        val_model = self.val_model
        valid_users = np.random.choice(list(set(self.valid.nonzero()[0])), size=1000, replace=False)
        # validation_recall = RecallEvaluator_knn(val_model, self.train, self.valid, self.test_batch_size)
        validation_recall = RecallEvaluator(val_model, self.train, self.valid)

        """ Config set-up """
        configPro = tf.ConfigProto(allow_soft_placement=True)
        configPro.gpu_options.allow_growth = True

        with tf.Session(config=configPro) as sess:
            #sess = tf_debug.TensorBoardDebugWrapperSession(sess, 'localhost:6064')
            sess.run(tf.global_variables_initializer())

            train_handle = sess.run(model.train_iterator.string_handle())

            while True:
                """ Evaluation recall@k """
                valid_recalls = []
                for user_chunk in toolz.partition_all(100, valid_users):
                    valid_recalls.extend([validation_recall.eval(sess, user_chunk)])
                print("\nRecall on (sampled) validation set: {}".format(np.mean(valid_recalls)))
                # TO DO: early stopping

                """ Trainning model"""
                sess.run(model.train_iterator.initializer)

                losses = []

                for loop_idx in tqdm(range(int(self.total_num_user_item/self.batch_size)), desc="Optimizing..."):
                    _, loss = sess.run((model.selctive_opt, model.total_loss),
                                       feed_dict = {model.handle: train_handle})

                    losses.append(loss)

                print("\nTraining loss {}".format(np.mean(losses)))


def main_algo(config):
    # get user-item matrix
    user_item_matrix, features = citeulike(tag_occurence_thres=5)
    n_users, n_items = user_item_matrix.shape
    # make feature as dense matrix
    dense_features = features.toarray() + 1E-10

    if config.with_feature:
        """ Random Projection based JL lemma """
        jl_dim = 2000
        ori_dim = dense_features.shape[1]
        random_matrix = np.random.normal(size=(ori_dim, jl_dim))*np.sqrt(ori_dim/jl_dim)

        jl_projected_fea = dense_features @ random_matrix

        """ PCA """
        scaler = StandardScaler()
        scaler.fit(jl_projected_fea)
        jl_projected_fea = scaler.transform(jl_projected_fea)

        pca = PCA(.3)
        pca.fit(jl_projected_fea)
        pca_projected_fea = pca.transform(jl_projected_fea)

        print(pca_projected_fea.shape)
        fea = pca_projected_fea/(pca_projected_fea.shape[1])
        print(np.max(fea))
        print(np.min(fea))
        # prior estimation
        prior_estimation_data_matrix(train,fea,config.r_prior_sample)
        return
    else:
        fea = None

    # get train/valid/test user-item matrices
    train, valid, test = split_data(user_item_matrix)

    # add a few stuff to config
    config.n_users = n_users
    config.n_items = n_items

    # without feature vectors
    pucml_learner = PUCML_Base(config,features=fea,train=train,valid=valid,test=test)
    pucml_learner.train_main()


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

    parser.add_argument('--prior',
        action   = 'store',
        required = False,
        type     = float,
        default  = 0.5,
        help     = 'prior distribution')

    parser.add_argument('--lr',
        action   = 'store',
        required = False,
        type     = float,
        default  = 0.001,
        help     = 'learning rate')

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

    parser.add_argument('--test_batch_size',
        action   = 'store',
        required = False,
        type     = int,
        default  = 2,
        help     = 'test batch size')

    parser.add_argument('--evaluation_loop_num',
        action   = 'store',
        required = False,
        type     = int,
        default  = 100,
        help     = 'evaluation loop number')

    parser.add_argument('--n_unlabeled',
        action   = 'store',
        required = False,
        type     = int,
        default  = 20,
        help     = 'number of unlabeled data')

    parser.add_argument('--r_prior_sample',
        action   = 'store',
        required = False,
        type     = int,
        default  = 100,
        help     = 'ratior for sampling unlabeled items for prior estimation')

    parser.add_argument('--n_subsample_pairs',
        action   = 'store',
        required = False,
        type     = int,
        default  = 100,
        help     = 'number of base subsample pairs')

    parser.add_argument('--with_feature',
        action   = 'store',
        required = False,
        type     = bool,
        default  = False,
        help     = 'Flag to determine whether to use features')

    config = parser.parse_args()
    main_algo(config)
