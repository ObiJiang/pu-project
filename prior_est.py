import numpy as np

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

        xi_tr = xu[idx_xu_tr,:]
        xu_te = xu[idx_xu_te,:]

        x_ce = xp_tr
        for sigma_idx, sigma_val in enumerate(sigma_list):

            # Calculate Phi first
            Phi1_tr = GaussBasis(xp_tr, x_ce, sigma_val)
			Phi1_te = GaussBasis(xp_te, x_ce, sigma_val)

			Phi2_tr = GaussBasis(xu_tr, x_ce, sigma_val)
			Phi2_te = GaussBasis(xu_te, x_ce, sigma_val

            for lambda_idx, lambda_val in enumerate(lambda_list):

    return 0.3

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

if __name__ == '__main__':
    test_prior = 0.3 # biased
    dim = 50
    num_samples = 100
    prior_seq = np.random.binomial(1, test_prior, num_samples)

    n_p = np.sum(prior_seq == 1)
    n_u = num_samples - n_p

    xp = np.random.normal(0, 0.1, (n_p,dim))
    xm = np.random.normal(5, 0.1, (n_u,dim))

    PEPriorEst_Per_User(xp,xm)
