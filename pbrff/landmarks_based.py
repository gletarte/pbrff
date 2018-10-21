import numpy as np
import pickle
import time

from math import ceil, sqrt

from scipy.special import logsumexp
from scipy.spatial.distance import cdist

from sklearn.utils import check_random_state
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC, SVC

from pbrff.landmarks_selector import LandmarksSelector
        
class LandmarksBasedLearner(object):
    '''
    Landmarks-Based learner class
    '''
    def __init__(self, dataset, C_range, gamma, landmarks_selection_method, random_state=42):
        self.dataset = dataset
        self.n, self.d = self.dataset['X_train'].shape
        self.gamma = gamma
        self.sigma = 1. / sqrt(2 * self.gamma)
        self.C_range = C_range
        self.landmarks_selection_method = landmarks_selection_method
        self.random_state = check_random_state(random_state)
        
    def select_landmarks(self, percentage_landmarks):
        self.percentage_landmarks = percentage_landmarks
        nb_landmarks_per_label = int(ceil(len(self.dataset['y_train'])*percentage_landmarks) / len(np.unique(self.dataset['y_train'])))
        self.nb_landmarks = nb_landmarks_per_label * len(np.unique(self.dataset['y_train']))
        landmarks_selector = LandmarksSelector(nb_landmarks_per_label, self.landmarks_selection_method, random_state=self.random_state)
        self.landmarks_X, self.landmarks_y = landmarks_selector.fit(self.dataset['X_train'], self.dataset['y_train'])
        
    def compute_q(self, beta):
        self.beta = beta
        # Computing t hp
        t = sqrt(self.n) * self.beta
        
        # Q
        self.q = -t*self.loss - logsumexp(-t*self.loss, axis=1).reshape(-1, 1)
        self.q = np.exp(self.q)
        
    def compute_loss(self, D):
        self.D = D
        
        # Randomly sampling omega
        self.omega = self.random_state.randn(len(self.landmarks_y), self.d, self.D) / (1. / sqrt(2 * self.gamma))
        
        loss = []
        for i in range(self.nb_landmarks):
            X_diff = self.dataset['X_train'] - self.landmarks_X[i]
            
            WX = self.transform_cos(self.omega[i], X_diff)
            diff_y = -np.ones(self.n)
            diff_y[(self.dataset['y_train'] == self.landmarks_y[i])] = 1
            
            L = diff_y @ WX
            L = -1 * L / (self.n - 1)
            L = (L + 1) / 2
            loss.append(L)
        self.loss = np.array(loss)
            
    def pb_transform(self, X):
        new_X = []
        for i in range(self.nb_landmarks):
            X_diff = X - self.landmarks_X[i]
            WX = self.transform_cos(self.omega[i], X_diff)
            new_X.append(np.sum(WX * self.q[i], 1))
        return np.array(new_X).T
        
    def learn_pb(self):
        transformed_X_train = self.pb_transform(self.dataset['X_train'])
        transformed_X_valid = self.pb_transform(self.dataset['X_valid'])
        transformed_X_test = self.pb_transform(self.dataset['X_test'])

        C_search = []
        for C in self.C_range:
            clf = LinearSVC(C=C, random_state=self.random_state)
            clf.fit(transformed_X_train, self.dataset['y_train'])
            err = 1 - accuracy_score(self.dataset['y_valid'], clf.predict(transformed_X_valid))
            C_search.append((err, C, clf))
        
        mean_max_q = np.mean(np.max(self.q, axis=1))
        val_err, C, clf = sorted(C_search, key=lambda x: x[0])[0]
        train_err = 1 - accuracy_score(self.dataset['y_train'], clf.predict(transformed_X_train))
        y_pred = clf.predict(transformed_X_test)
        test_err = 1 - accuracy_score(self.dataset['y_test'], y_pred)
        f1 = f1_score(self.dataset['y_test'], y_pred)
        
        return dict([("dataset", self.dataset['name']), ("exp", 'landmarks'), ("algo", 'PB'), ("method", self.landmarks_selection_method), \
                    ("C", C), ("D", self.D), ("nb_landmarks", self.nb_landmarks), ("perc_landmarks", self.percentage_landmarks), \
                    ("gamma", self.gamma), ("beta", self.beta), ("train_error", train_err), ("val_error", val_err), ("test_error", test_err), \
                    ("f1", f1), ("mean_max_q",  mean_max_q)])
                     
    def rbf_transform(self, X):
        return np.exp(-self.gamma * cdist(X, self.landmarks_X, 'sqeuclidean'))
        
    def learn_rbf(self):
        transformed_X_train = self.rbf_transform(self.dataset['X_train'])
        transformed_X_valid = self.rbf_transform(self.dataset['X_valid'])
        transformed_X_test = self.rbf_transform(self.dataset['X_test'])

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
        
        return dict([("dataset", self.dataset['name']), ("exp", 'landmarks'), ("algo", 'RBF'), ("method", self.landmarks_selection_method), \
                    ("C", C), ("nb_landmarks", self.nb_landmarks), ("perc_landmarks", self.percentage_landmarks), ("gamma", self.gamma),\
                    ("train_error", train_err), ("val_error", val_err), ("test_error", test_err), ("f1", f1)])
        
    def transform_sincos(self, w, X, D):
        WX = np.dot(X, w)
        return np.hstack((np.cos(WX), np.sin(WX))) / np.sqrt(D)
        
    def transform_cos(self, w, X):
        WX = np.dot (X, w)
        return np.cos(WX)
        
    def transform_sin(self, w, X):
        WX = np.dot (X, w)
        return np.sin(WX)
        
def compute_landmarks_selection(args, dataset, C_range, gamma, random_state):
    landmarks_based_learner = LandmarksBasedLearner(dataset, C_range, gamma, args['method'], random_state)
    landmarks_based_learner.select_landmarks(args['percentage_landmarks'])
    
    print(f"Processing: {100*args['percentage_landmarks']}% landmarks {args['method']} selection")
    with open(args['output_file'], 'wb') as out_file:
            pickle.dump(landmarks_based_learner, out_file, protocol=4)
    
    return args['method'] 

def compute_landmarks_based(args, beta_range):
    tmp_results = []
    
    with open(args["input_file"], 'rb') as in_file:
        landmarks_based_learner = pickle.load(in_file)
        
    if args["algo"] == "rbf":
        print(f"Processing: rff with {100*args['percentage_landmarks']}% landmarks {args['method']}")
        tmp_results.append(landmarks_based_learner.learn_rbf())
            
    elif args["algo"] == "pb":
        print(f"Processing: pb with {args['D']} features, {100*args['percentage_landmarks']}% landmarks {args['method']}")
        landmarks_based_learner.compute_loss(args['D'])
        for beta in beta_range:
            landmarks_based_learner.compute_q(beta)
            tmp_results.append(landmarks_based_learner.learn_pb())

    with open(args["output_file"], 'wb') as out_file:
        pickle.dump(tmp_results, out_file, protocol=4)
            
    return args["algo"]




