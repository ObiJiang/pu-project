import numpy as np
from scipy.sparse import lil_matrix
from tqdm import tqdm
""" Prior Estimation for single user """
def PEPriorEst_Per_User(xp, xu, sigma_list = None, lambda_list=None, kfolds=5):
    n_p,d = xp.shape
    n_u,d = xu.shape

    # Set the default sigma_list
    if sigma_list == None:
        x_pu = np.concatenate((xp,xu),axis=0)
        med_dist = CalculateMedian(x_pu,x_pu)
        sigma_list = np.linspace(0.2, 5.0, num=10)*med_dist

    if lambda_list == None:
        lambda_list = np.logspace(-3,1,num = 9)

    # shuffle the data
    np.random.shuffle(xp)
    np.random.shuffle(xu)

    # split the data
    cv_split_xp = np.floor(np.arange(0,n_p)*kfolds/n_p)
    cv_split_xu = np.floor(np.arange(0,n_u)*kfolds/n_u)

    cv_scores = np.zeros((len(sigma_list),len(lambda_list)))

    for k in range(kfolds):
        # get training and test datasets
        idx_xp_tr = (cv_split_xp!=k)
        idx_xp_te = (cv_split_xp==k)

        idx_xu_tr = (cv_split_xu!=k)
        idx_xu_te = (cv_split_xu==k)

        np_tr = np.sum(idx_xp_tr)
        np_te = np.sum(idx_xp_te)
        nu_tr = np.sum(idx_xu_tr)
        nu_te = np.sum(idx_xu_te)

        p1_tr = np_tr/(np_tr+nu_tr)
        p1_te = np_te/(np_te+nu_te)

        xp_tr = xp[idx_xp_tr,:]
        xp_te = xp[idx_xp_te,:]

        xu_tr = xu[idx_xu_tr,:]
        xu_te = xu[idx_xu_te,:]

        x_ce = xp_tr
        for sigma_idx, sigma_val in enumerate(sigma_list):

            # Calculate Phi first
            Phi1_tr = GaussBasis(xp_tr, x_ce, sigma_val)
            Phi1_te = GaussBasis(xp_te, x_ce, sigma_val)

            Phi2_tr = GaussBasis(xu_tr, x_ce, sigma_val)
            Phi2_te = GaussBasis(xu_te, x_ce, sigma_val)

            # Calculate H and h
            h_tr = np.mean(Phi1_tr, axis=1)
            h_te = np.mean(Phi1_te, axis=1)

            H_tr = (p1_tr/np_tr)*Phi1_tr @ Phi1_tr.T + ((1-p1_tr)/nu_tr)*Phi2_tr @ Phi2_tr.T
            H_te = (p1_te/np_te)*Phi1_te @ Phi1_te.T + ((1-p1_te)/nu_te)*Phi2_te @ Phi2_te.T

            for lambda_idx, lambda_val in enumerate(lambda_list):
                alpha = np.linalg.solve(H_tr + lambda_val* np.eye(np_tr), h_tr)

                # calculate the score
                score = 1/2*alpha.T @ H_te @ alpha - alpha.T @ h_te

                cv_scores[sigma_idx, lambda_idx] = cv_scores[sigma_idx, lambda_idx] + score

    sigma_chosen_ind = np.argmin(np.min(cv_scores,axis=1))
    lambda_chosen_ind = np.argmin(np.min(cv_scores,axis=0))

    sigma_chosen = sigma_list[sigma_chosen_ind]
    lambda_chosen = lambda_list[lambda_chosen_ind]

    x_ce = xp

    Phi1 = GaussBasis(xp, x_ce, sigma_chosen)
    Phi2 = GaussBasis(xu, x_ce, sigma_chosen)

    p1 = n_p/(n_p+n_u);

    h = np.mean(Phi1, axis=1)
    H = (p1/n_p) * Phi1 @ Phi1.T + (1-p1)/n_u * Phi2 @ Phi2.T

    # calculate the density ratio
    alpha = np.linalg.solve(H + lambda_chosen*np.eye(n_p), h)

    # calculate the prior
    prior = 1/(2*alpha.T @ h - alpha.T @ H @alpha)

    prior = max(prior, p1)
    prior = min(prior, 1)

    return prior

def GaussBasis(z,c,sigma):
    Phi_tmp = -CalculateDist(z,c)/2
    Phi= np.exp(Phi_tmp/(sigma**2))
    return Phi.T

def CalculateMedian(X,Y):
    nx,d = X.shape
    ny,d = Y.shape
    X2 = np.sum(X**2,axis=1,keepdims=True)
    Y2 = np.sum(Y**2,axis=1,keepdims=True)

    XC_dist2 = np.tile(Y2.T, (nx,1))+np.tile(X2,(1,ny))-2*X @ Y.T
    median = np.sqrt(np.median(XC_dist2))

    return median

def CalculateDist(X,Y):
    nx,d = X.shape
    ny,d = Y.shape
    X2 = np.sum(X**2,axis=1,keepdims=True)
    Y2 = np.sum(Y**2,axis=1,keepdims=True)

    XC_dist2 = np.tile(Y2.T, (nx,1))+np.tile(X2,(1,ny))-2*X @ Y.T

    return XC_dist2

""" Prior Estimation Wrapper"""
def prior_estimation_data_matrix(train,features,r_prior_sample):
    train_user_item_matrix = lil_matrix(train)
    priorList = []
    for u, row in enumerate(tqdm(train_user_item_matrix.rows)):
        if row:
            # labled data
            row = np.array(row)
            xp = features[row,:]

            # unlabeled data
            mask = np.ones(train.shape[1],dtype=bool)
            mask[row] = False
            xu = features[mask]
            mask = np.random.choice(train.shape[1]-len(row),np.min([r_prior_sample*len(row),train.shape[1]-len(row)]),replace=False)
            xu = xu[mask]
            prior = PEPriorEst_Per_User(xp,xu)
            priorList.append(prior)

    np.save('./prior_list.npy',np.array(priorList))
    print("Prior (Sampled 1:{}) {}".format(r_prior_sample,np.mean(priorList)))

if __name__ == '__main__':
    test_prior = 0.5 # biased
    dim = 50
    num_samples = 100
    prior_seq = np.random.binomial(1, test_prior, num_samples)

    n_p = np.sum(prior_seq == 1)
    n_u = num_samples - n_p

    xp = np.random.normal(0, 0.001, (n_p,dim))
    xm = np.random.normal(3, 0.001, (n_u,dim))

    prior = PEPriorEst_Per_User(xp,xm)
    print(prior)
