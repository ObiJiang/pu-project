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

# This version anchor will be weighed

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
        self.prior_list_numpy = np.load('./prior_list.npy').astype(np.float32)
        self.prior_list = tf.constant(self.prior_list_numpy)

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
        self.init = tf.global_variables_initializer()
        tf.get_default_graph().finalize()

    def create_varaibles(self,features):
        """ The following are variables used in the model (feature vectors and alpha) """
        # how feature vectors are generated
        if features is not None:
            self.emb_dim = features.shape[1]
            self.features = tf.constant(features, dtype=tf.float32)
        else:
            self.emb_dim = 100
            self.features = tf.Variable(tf.random_normal([self.n_items, self.emb_dim],
                                            stddev=1 / (self.emb_dim ** 0.5), dtype=tf.float32))

        # subsample base item pairs. It is extremly rare that two items would be the same. If it is the same, it's not a big deal
        vi = tf.constant(np.random.randint(0,self.n_items,size=(self.n_subsample_pairs)),dtype=tf.int32)
        vj = tf.constant(np.random.randint(0,self.n_items,size=(self.n_subsample_pairs)),dtype=tf.int32)
        vi_vj = tf.gather(self.features,vi) - tf.gather(self.features,vj)
        self.base_matrices = tf.matmul(tf.expand_dims(vi_vj,2),
                                       tf.expand_dims(vi_vj,1)) #(batch,emb_dim,emb_dim)

        self.pre_alpha = tf.Variable(tf.random_normal([self.n_users, self.n_subsample_pairs],
                                     stddev=1 / (self.n_subsample_pairs ** 0.5), dtype=tf.float32))
        self.alpha = tf.abs(self.pre_alpha)

        # generate alpha for all the users
        # self.pre_alpha = tf.Variable(tf.random_normal([self.n_users, self.emb_dim],
        #                              stddev=1 / (self.emb_dim ** 0.5), dtype=tf.float32))
        # self.alpha = tf.abs(self.pre_alpha)

        self.beta = tf.Variable(tf.random_normal([self.n_users, self.max_n_p_elem],
                                     stddev=1 / (self.emb_dim ** 0.5), dtype=tf.float32))

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

    def model(self):
        tf.set_random_seed(1)
        train_iterator,train_dataset = self.input_dataset_pipeline()

        handle = tf.placeholder(tf.string, shape=[],name='dataset_handle')
        iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)

        p_u = iterator.get_next() # p_u: [user, pos_item, {unlabel_item}]

        """ find associated metrices with users in p_u """
        alpha_in_batch = tf.gather(self.alpha,p_u[:,0])
        beta_in_batch = tf.gather(self.beta,p_u[:,0])
        #metrics_in_batch = tf.linalg.diag(alpha_in_batch)
        metrics_in_batch = tf.reduce_sum(tf.expand_dims(tf.expand_dims(alpha_in_batch,2),2) * self.base_matrices,
                             axis=1)

        """ create anchor vectors for each users """
        user_postive_ind_map_in_batch = tf.gather(self.user_postive_ind_map,p_u[:,0])
        mask_fea_in_batch = tf.expand_dims(tf.cast(tf.sign(user_postive_ind_map_in_batch + 1),tf.float32),axis=2)
        fea_in_batch = tf.gather(self.features,tf.nn.relu(user_postive_ind_map_in_batch))
        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     train_handle = sess.run(train_iterator.string_handle())
        #     sess.run(train_iterator.initializer)
        anchor_vectors = tf.reduce_sum(tf.expand_dims(beta_in_batch,axis=2)*mask_fea_in_batch * fea_in_batch,axis=1)/(tf.reduce_sum(mask_fea_in_batch,axis=1))

        """ compute distances """
        fea_in_batch = tf.gather(self.features,p_u[:,1:])
        fea_diff_in_batch = tf.expand_dims(anchor_vectors,axis=1) - fea_in_batch
        dist_in_batch_part_1 = tf.einsum('bim,bmn->bin', fea_diff_in_batch, metrics_in_batch)
        confidence_scores = tf.exp(tf.negative(tf.einsum('bin,bin->bi', fea_diff_in_batch, dist_in_batch_part_1)))

        """ compute score functions """
        p_scores = confidence_scores[:,0]
        u_scores = confidence_scores[:,1:]

        prior_in_batch =  tf.gather(self.prior_list,p_u[:,0])
        R_p_plus = tf.reduce_mean(-1*tf.log(0.001+p_scores)*prior_in_batch)
        R_p_minus = tf.reduce_mean(-1*tf.log(0.001+1-p_scores)*prior_in_batch)
        P_u_minus = tf.reduce_mean(-1*tf.log(0.001+1-u_scores))

        """ define loss and optimization """
        # define two differnt losses and their optimizer
        # total_loss = self.prior * R_p_plus + (P_u_minus - self.prior * R_p_minus)+ tf.nn.l2_loss(self.alpha)# + self.feature_loss#
        total_loss =  R_p_plus + (P_u_minus - R_p_minus) # + self.feature_loss
        negative_loss = P_u_minus - R_p_minus

        full_opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(total_loss)
        neg_opt = tf.train.AdamOptimizer(learning_rate=self.lr*self.gamma).minimize(-1*negative_loss)

        # tf.cond for different optimization
        #selctive_opt = tf.cond(negative_loss > self.beta, lambda: full_opt, lambda: neg_opt)
        selctive_opt = full_opt

        return AttrDict(locals())  # The magic line.

    # return scores for a couple of users
    def valuation_model(self):
        score_user_ids = tf.placeholder(tf.int32, [None])
        k = tf.placeholder(tf.int32)

        """ find associated metrices with users in score_user_ids """
        alpha_in_batch = tf.gather(self.alpha,score_user_ids)
        #metrics_in_batch = tf.linalg.diag(alpha_in_batch)
        metrics_in_batch = tf.reduce_sum(tf.expand_dims(tf.expand_dims(alpha_in_batch,2),2) * self.base_matrices,
                             axis=1)
        beta_in_batch = tf.gather(self.beta,score_user_ids)

        """ create anchor vectors for each users """
        user_postive_ind_map_in_batch = tf.gather(self.user_postive_ind_map,score_user_ids)
        mask_fea_in_batch = tf.expand_dims(tf.cast(tf.sign(user_postive_ind_map_in_batch + 1),tf.float32),axis=2)
        fea_in_batch = tf.gather(self.features,tf.nn.relu(user_postive_ind_map_in_batch))
        anchor_vectors = tf.reduce_sum(tf.expand_dims(beta_in_batch,axis=2)*mask_fea_in_batch * fea_in_batch,axis=1)/(tf.reduce_sum(mask_fea_in_batch,axis=1))

        """ caculate distance to anchor_vectors """
        fea_diff_in_batch = tf.expand_dims(anchor_vectors,axis=1) - self.features
        dist_in_batch_part_1 = tf.einsum('bim,bmn->bin', fea_diff_in_batch, metrics_in_batch)
        item_scores = tf.negative(tf.einsum('bin,bin->bi', fea_diff_in_batch, dist_in_batch_part_1))

        top_k = tf.nn.top_k(item_scores, k)
        return AttrDict(locals())

    def train_main(self):
        model = self.model

        """ Evaluation Set-up """
        val_model = self.val_model
        valid_users = np.random.choice(list(set(self.valid.nonzero()[0])), size=1000, replace=False)
        train_users = np.random.choice(list(set(self.train.nonzero()[0])), size=1000, replace=False)
        # validation_recall = RecallEvaluator_knn(val_model, self.train, self.valid, self.test_batch_size)
        validation_recall = RecallEvaluator(val_model, self.train, self.valid)
        train_recall = RecallEvaluator(val_model, self.valid, self.train)

        """ Config set-up """
        configPro = tf.ConfigProto(allow_soft_placement=True)
        configPro.gpu_options.allow_growth = True

        sess = tf.Session(config=configPro)
        sess.run(self.init)

        train_handle = sess.run(model.train_iterator.string_handle())
        epoch_idx = 0
        while True:
            # TO DO: early stopping
            """ Trainning model"""
            sess.run(model.train_iterator.initializer)

            losses = []

            for loop_idx in tqdm(range(int(self.total_num_user_item/self.batch_size)), desc="Optimizing..."):
                _, loss = sess.run((model.selctive_opt, model.total_loss),
                                   feed_dict = {model.handle: train_handle})

                losses.append(loss)
                # print("\nTraining loss {}".format(np.mean(losses)))

            if epoch_idx%self.evaluation_loop_num == 0:
                """ Evaluation recall@k """
                valid_recalls = []
                for user_chunk in toolz.partition_all(10, valid_users):
                    valid_recalls.extend([validation_recall.eval(sess, user_chunk)])
                print("Recall on (sampled) validation set: {}".format(np.mean(valid_recalls)))

                """ Evaluation recall@k """
                train_recalls = []
                for user_chunk in toolz.partition_all(100, train_users):
                    train_recalls.extend([train_recall.eval(sess, user_chunk)])
                print("Recall on (sampled) training set: {}".format(np.mean(train_recalls)))
            epoch_idx += 1


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

        fea = pca_projected_fea/(pca_projected_fea.shape[1])
    else:
        fea = None

    # get train/valid/test user-item matrices
    train, valid, test = split_data(user_item_matrix)

    # prior estimation
    #prior_estimation_data_matrix(train,fea,config.r_prior_sample)

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
