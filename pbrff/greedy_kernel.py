import numpy as np
import pickle

from math import ceil, sqrt

from scipy.special import logsumexp

from sklearn.utils import check_random_state
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC, SVC
        
class GreedyKernelLearner(object):
    '''
    Greedy Kernel learner class
    '''
    def __init__(self, dataset, C_range, gamma, N, random_state=42):
        self.dataset = dataset
        self.n, self.d = self.dataset['X_train'].shape
        self.gamma = gamma
        self.N = N
        self.sigma = 1. / sqrt(2 * self.gamma)
        self.C_range = C_range
        self.loss = None
        self.random_state = check_random_state(random_state)
         
    def sample_omega(self):
        self.omega = self.random_state.randn(self.d, self.N) / self.sigma
        
    def compute_loss(self):
        cos_values = np.sum(np.einsum('ij,i->ij', self.transform_cos(self.omega, self.dataset['X_train']), self.dataset['y_train']), axis=0)
        sin_values = np.sum(np.einsum('ij,i->ij', self.transform_sin(self.omega, self.dataset['X_train']), self.dataset['y_train']), axis=0)
        
        self.loss = 1/(self.n*(self.n-1)) * (cos_values ** 2 + sin_values ** 2)
        self.loss = (1 - self.loss) / 2 - 1/(self.n -1)
        
    def learn_rff(self, D):
        w = self.omega[:, :D]
        
        transformed_X_train = self.transform_sincos(w, self.dataset['X_train'], D)
        transformed_X_valid = self.transform_sincos(w, self.dataset['X_valid'], D)
        transformed_X_test = self.transform_sincos(w, self.dataset['X_test'], D)

        C_search = []
        for C in self.C_range:
            clf = LinearSVC(C=C, random_state=self.random_state)
            clf.fit(transformed_X_train, self.dataset['y_train'])
            err = 1 - accuracy_score(self.dataset['y_valid'], clf.predict(transformed_X_valid))
            C_search.append((err, C, clf))

        val_err, C, clf = sorted(C_search, key=lambda x: x[0])[0]
        train_err = 1 - accuracy_score(self.dataset['y_train'], clf.predict(transformed_X_train))
        y_pred = clf.predict(transformed_X_test)
        test_err = 1 - accuracy_score(self.dataset['y_test'], y_pred)
        f1 = f1_score(self.dataset['y_test'], y_pred)

        return dict([("dataset", self.dataset['name']), ("exp", 'greedy'), ("algo", 'RFF'), ("C", C), ("D", D), ("N", self.N), \
                    ("gamma", self.gamma), ("train_error", train_err), ("val_error", val_err), ("test_error", test_err), ("f1", f1)])
                     
    def compute_pb_q(self, beta):
        if self.loss is None:
            self.compute_loss()
        t = sqrt(self.n) * beta
        
        self.beta = beta
        self.pb_q = -t*self.loss - logsumexp(-t*self.loss)
        self.pb_q = np.exp(self.pb_q)
        
    def learn_pbrff(self, D):
        kernel_features = self.omega[:, self.random_state.choice(self.omega.shape[1], D, replace=True, p=self.pb_q)]
        
        transformed_X_train = self.transform_sincos(kernel_features, self.dataset['X_train'], D)
        transformed_X_valid = self.transform_sincos(kernel_features, self.dataset['X_valid'], D)
        transformed_X_test = self.transform_sincos(kernel_features, self.dataset['X_test'], D)

        C_search = []
        for C in self.C_range:
            clf = LinearSVC(C=C, random_state=self.random_state)
            clf.fit(transformed_X_train, self.dataset['y_train'])
            err = 1 - accuracy_score(self.dataset['y_valid'], clf.predict(transformed_X_valid))
            C_search.append((err, C, clf))

        val_err, C, clf = sorted(C_search, key=lambda x: x[0])[0]
        train_err = 1 - accuracy_score(self.dataset['y_train'], clf.predict(transformed_X_train))
        y_pred = clf.predict(transformed_X_test)
        test_err = 1 - accuracy_score(self.dataset['y_test'], y_pred)
        f1 = f1_score(self.dataset['y_test'], y_pred)
        
        return dict([("dataset", self.dataset['name']), ("exp", 'greedy'), ("algo", 'PBRFF'), ("C", C), ("D", D), ("N", self.N), \
                    ("gamma", self.gamma), ("beta", self.beta), ("train_error", train_err), ("val_error", val_err), \
                    ("test_error", test_err), ("f1", f1)])
                     
    def compute_ok_q(self, rho):
        self.rho = rho
        v = 2 * self.loss -1
        u = np.ones(self.N) * 1 / self.N
        acc = 1e-8
        
        def project_onto_simplex(w, B):
            z = -np.sort(-w)
            sv = np.cumsum(z)
            rho = np.argwhere(z > np.divide((sv - B), np.arange(1, len(w) + 1)))[-1][0]
            theta = (sv[rho] - B) / (rho + 1)
            q = w - theta
            q[q<0]=0
            
            return q
        
        duality_gap = np.inf
    
        max_lambda = np.inf
        min_lambda = 0
        
        x = project_onto_simplex(u, 1)
    
        if (np.linalg.norm(u-x) ** 2 > rho):
            raise RuntimeError("Problem is not feasible")
        
        start_lambda = 1
        while(np.isinf(max_lambda)):
            x = project_onto_simplex(u - v / start_lambda, 1)
            lam_grad = 0.5 * np.linalg.norm(x - u) ** 2 - self.rho/2;
            if (lam_grad < 0):
                max_lambda = start_lambda;
            else:
                start_lambda = start_lambda * 2;
                
        while (max_lambda - min_lambda > acc * start_lambda):
            lambda_value = (min_lambda + max_lambda) / 2
            x = project_onto_simplex(u - v / lambda_value, 1);
            lam_grad = 0.5 * np.linalg.norm(x - u) ** 2 - self.rho/2;
            if (lam_grad < 0):
            # Then lambda is too large, so decrease max_lambda
                max_lambda = lambda_value
            else:
                min_lambda = lambda_value
        
        self.ok_q = x
        
    def learn_okrff(self, D):
        kernel_features = self.omega[:, self.random_state.choice(self.omega.shape[1], D, replace=True, p=self.ok_q)]
        
        transformed_X_train = self.transform_sincos(kernel_features, self.dataset['X_train'], D)
        transformed_X_valid = self.transform_sincos(kernel_features, self.dataset['X_valid'], D)
        transformed_X_test = self.transform_sincos(kernel_features, self.dataset['X_test'], D)

        C_search = []
        for C in self.C_range:
            clf = LinearSVC(C=C, random_state=self.random_state)
            clf.fit(transformed_X_train, self.dataset['y_train'])
            err = 1 - accuracy_score(self.dataset['y_valid'], clf.predict(transformed_X_valid))
            C_search.append((err, C, clf))

        val_err, C, clf = sorted(C_search, key=lambda x: x[0])[0]
        train_err = 1 - accuracy_score(self.dataset['y_train'], clf.predict(transformed_X_train))
        y_pred = clf.predict(transformed_X_test)
        test_err = 1 - accuracy_score(self.dataset['y_test'], y_pred)
        f1 = f1_score(self.dataset['y_test'], y_pred)
        
        return dict([("dataset", self.dataset['name']), ("exp", 'greedy'), ("algo", 'OKRFF'), ("C", C), ("D", D), ("N", self.N), \
                    ("gamma", self.gamma), ("rho", self.rho), ("train_error", train_err), ("val_error", val_err), \
                    ("test_error", test_err), ("f1", f1)])
        
    def transform_sincos(self, w, X, D):
        WX = np.dot(X, w)
        return np.hstack((np.cos(WX), np.sin(WX))) / np.sqrt(D)
        
    def transform_cos(self, w, X):
        WX = np.dot (X, w)
        return np.cos(WX)
        
    def transform_sin(self, w, X):
        WX = np.dot (X, w)
        return np.sin(WX)

def compute_greedy_kernel(args, greedy_kernel_learner_file, gamma, D_range, random_state):
    tmp_results = []
    
    with open(greedy_kernel_learner_file, 'rb') as in_file:
        greedy_kernel_learner = pickle.load(in_file)
        
    if args["algo"] == "rff":
        print("Processing: rff")
        for D in D_range:
           tmp_results.append(greedy_kernel_learner.learn_rff(D))
            
    elif args["algo"] == "pbrff":
        print(f"Processing: pbrff with beta: {args['param']}")
        greedy_kernel_learner.compute_pb_q(beta=args['param'])
        for D in D_range:
            tmp_results.append(greedy_kernel_learner.learn_pbrff(D))
        
    elif args["algo"] == "okrff":
        print(f"Processing: okrff with rho: {args['param']}")
        greedy_kernel_learner.compute_ok_q(rho=args['param'])
        for D in D_range:
            tmp_results.append(greedy_kernel_learner.learn_okrff(D))
    
    with open(args["output_file"], 'wb') as out_file:
            pickle.dump(tmp_results, out_file, protocol=4)
            
    return args["algo"]




