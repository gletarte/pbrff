from math import ceil, sqrt

import pickle
import numpy as np

from scipy.special import logsumexp
from scipy.spatial.distance import cdist

from sklearn.utils import check_random_state
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC

from pbrff.landmarks_selector import LandmarksSelector

class LandmarksBasedLearner(object):
    """Landmarks-Based learner class

    Parameters
    ----------
    dataset : dict
        The dataset as a dictionnary with the following keys:
        X_train, X_valid, X_test, y_train, y_valid, y_test, name.

    C_range : list
        C values range to search from.

    gamma : float
        Gamma value.

    landmarks_selection_method : str
        The landmarks selection method from: {random, clustering}.

    random_state: None, int or instance of RandomState.
        Information about the random state to be used.


    Attributes
    ----------
    dataset : dict
        The dataset as a dictionnary with the following keys:
        X_train, X_valid, X_test, y_train, y_valid, y_test, name.

    n : int
        Number of samples in the training set.

    d : int
        Number of features in the dataset.

    C_range : list
        C values range to search from.

    gamma : float
        Gamma value.

    sigma: float
        Sigma value computed using gamma value.

    landmarks_selection_method : str
        The landmarks selection method from: {random, clustering}.

    random_state: instance of RandomState.
        Random state for all random operations.

    percentage_landmarks : float
            Number of landmarks as a percentage of training set examples.

    n_landmarks : int
            Number of landmarks.

    landmarks_X : array, shape = [n_landmarks, d]
        Landmarks.

    landmarks_y : array, shape = [n_landmarks]
        Target labels of the landmarks.

    beta : float
        Beta value.

    Q : array, shape = [n_landmarks, D]
        Pseudo-posterior distribution.

    D : int
        Number of features per landmarks.

    Omega : array, shape = [n_landmarks, d, D]
        Randomly sampled Omega.

    loss : array, shape = [n_landmarks, d]
        Empirical loss.

    """
    def __init__(self, dataset, C_range, gamma, landmarks_selection_method, random_state=42):
        self.dataset = dataset
        self.n, self.d = self.dataset['X_train'].shape
        self.C_range = C_range
        self.gamma = gamma
        self.sigma = 1. / sqrt(2 * self.gamma)
        self.landmarks_selection_method = landmarks_selection_method
        self.random_state = check_random_state(random_state)

    def select_landmarks(self, percentage_landmarks):
        """Select landmarks from a dataset using LandmarksSelector.

        Parameters
        ----------
        percentage_landmarks : float
            Number of landmarks to select as the percentage of training set samples.

        """
        self.percentage_landmarks = percentage_landmarks
        n_landmarks_per_label = int(ceil(len(self.dataset['y_train'])*percentage_landmarks) / len(np.unique(self.dataset['y_train'])))
        self.n_landmarks = n_landmarks_per_label * len(np.unique(self.dataset['y_train']))
        landmarks_selector = LandmarksSelector(n_landmarks_per_label, self.landmarks_selection_method, random_state=self.random_state)
        self.landmarks_X, self.landmarks_y = landmarks_selector.fit(self.dataset['X_train'], self.dataset['y_train'])

    def compute_Q(self, beta):
        """Compute Q distribution according to beta value.

        Parameters
        ----------
        beta : float
            Beta value

        """
        self.beta = beta
        # Computing t
        t = self.beta * sqrt(self.n)

        # Computing Q
        self.Q = -t*self.loss - logsumexp(-t*self.loss, axis=1).reshape(-1, 1)
        self.Q = np.exp(self.Q)

    def transform_cos(self, omega, delta):
        """Hypothesis computation: h_omega(delta)

        Parameters
        ----------
        omega : array, shape = [d, D]
            omega values.

        delta : array, shape = [n, d]
            Pairwise distance.

        Returns
        -------
        hypothesis : array, shape = [n, D]
            Hypothesis.

        """
        return np.cos(np.dot(delta, omega))

    def compute_loss(self, D):
        """Compute loss for a given number of features per landmarks.

        Parameters
        ----------
        D : int
            Number of features per landmarks.

        """
        self.D = D

        # Randomly sampling Omega
        self.Omega = self.random_state.randn(self.n_landmarks, self.d, self.D) / (1. / sqrt(2 * self.gamma))

        loss = []
        # Computing loss for each landmarks
        for i in range(self.n_landmarks):
            transformed_X = self.transform_cos(self.Omega[i], self.dataset['X_train'] - self.landmarks_X[i])

            lambda_y = -np.ones(self.n)
            lambda_y[(self.dataset['y_train'] == self.landmarks_y[i])] = 1

            landmark_loss = lambda_y @ transformed_X

            # For the clustering method, landmarks are not sampled directly from dataset
            if self.landmarks_selection_method == "clustering":
                landmark_loss = landmark_loss / (self.n)

            # For the random method, case where X_i == landmark needs to be substract
            elif self.landmarks_selection_method == "random":
                landmark_loss = (landmark_loss - 1) / (self.n - 1)

            landmark_loss = (1 - landmark_loss) / 2
            loss.append(landmark_loss)
        self.loss = np.array(loss)

    def pb_mapping(self, X):
        """PAC-Bayesian landmarks-based mapping of X according to computed pseudo-posterior Q distribution.

        Parameters
        ----------
        X : array, shape = [n_samples, n_features]
            The dataset examples.

        Returns
        -------
        mapped_X : array, shape = [n_samples, n_landmarks]
            The dataset examples mapped in the landmarks-based representation.

        """
        mapped_X = []
        for i in range(self.n_landmarks):
            transformed_X = self.transform_cos(self.Omega[i], X - self.landmarks_X[i])
            mapped_X.append(np.sum(transformed_X* self.Q[i], 1))
        return np.array(mapped_X).T

    def learn_pb(self):
        """ Learn using PAC-Bayesion landmarks-based mappping

        Returns
        -------
        results : dict
            Relevant metrics and informations

        """
        transformed_X_train = self.pb_mapping(self.dataset['X_train'])
        transformed_X_valid = self.pb_mapping(self.dataset['X_valid'])
        transformed_X_test = self.pb_mapping(self.dataset['X_test'])

        # C search using a validation set
        C_search = []
        for C in self.C_range:
            clf = LinearSVC(C=C, random_state=self.random_state)
            clf.fit(transformed_X_train, self.dataset['y_train'])
            err = 1 - accuracy_score(self.dataset['y_valid'], clf.predict(transformed_X_valid))
            C_search.append((err, C, clf))

        # Computing relevant metrics
        mean_max_q = np.mean(np.max(self.Q, axis=1))
        val_err, C, clf = sorted(C_search, key=lambda x: x[0])[0]
        train_err = 1 - accuracy_score(self.dataset['y_train'], clf.predict(transformed_X_train))
        y_pred = clf.predict(transformed_X_test)
        test_err = 1 - accuracy_score(self.dataset['y_test'], y_pred)
        f1 = f1_score(self.dataset['y_test'], y_pred)

        return dict([("dataset", self.dataset['name']), ("exp", 'landmarks'), ("algo", 'PB'), ("method", self.landmarks_selection_method), \
                    ("C", C), ("D", self.D), ("n_landmarks", self.n_landmarks), ("perc_landmarks", self.percentage_landmarks), \
                    ("gamma", self.gamma), ("beta", self.beta), ("train_error", train_err), ("val_error", val_err), ("test_error", test_err), \
                    ("f1", f1), ("mean_max_q", mean_max_q)])

    def rbf_mapping(self, X):
        """RBF landmarks-based mapping of X.

        Parameters
        ----------
        X : array, shape = [n_samples, n_features]
            The dataset examples.

        Returns
        -------
        mapped_X : array, shape = [n_samples, n_landmarks]
            The dataset examples mapped in the landmarks-based representation.

        """
        return np.exp(-self.gamma * cdist(X, self.landmarks_X, 'sqeuclidean'))

    def learn_rbf(self):
        """ Learn using PAC-Bayesion landmarks-based mappping

        Returns
        -------
        results : dict
            Relevant metrics and informations

        """
        transformed_X_train = self.rbf_mapping(self.dataset['X_train'])
        transformed_X_valid = self.rbf_mapping(self.dataset['X_valid'])
        transformed_X_test = self.rbf_mapping(self.dataset['X_test'])

        # C search using a validation set
        C_search = []
        for C in self.C_range:
            clf = LinearSVC(C=C, random_state=self.random_state)
            clf.fit(transformed_X_train, self.dataset['y_train'])
            err = 1 - accuracy_score(self.dataset['y_valid'], clf.predict(transformed_X_valid))
            C_search.append((err, C, clf))

        # Computing relevant metrics
        val_err, C, clf = sorted(C_search, key=lambda x: x[0])[0]
        train_err = 1 - accuracy_score(self.dataset['y_train'], clf.predict(transformed_X_train))
        y_pred = clf.predict(transformed_X_test)
        test_err = 1 - accuracy_score(self.dataset['y_test'], y_pred)
        f1 = f1_score(self.dataset['y_test'], y_pred)

        return dict([("dataset", self.dataset['name']), ("exp", 'landmarks'), ("algo", 'RBF'), ("method", self.landmarks_selection_method), \
                    ("C", C), ("n_landmarks", self.n_landmarks), ("perc_landmarks", self.percentage_landmarks), ("gamma", self.gamma),\
                    ("train_error", train_err), ("val_error", val_err), ("test_error", test_err), ("f1", f1)])


def compute_landmarks_selection(args, dataset, C_range, gamma, random_state):
    """
    Landmarks selection function for parallel processing
    """
    landmarks_based_learner = LandmarksBasedLearner(dataset, C_range, gamma, args['method'], random_state)
    landmarks_based_learner.select_landmarks(args['percentage_landmarks'])

    print(f"Processing: {100*args['percentage_landmarks']}% landmarks {args['method']} selection")
    with open(args['output_file'], 'wb') as out_file:
        pickle.dump(landmarks_based_learner, out_file, protocol=4)

    return args['method']

def compute_landmarks_based(args, beta_range):
    """
    Landmarks-based learning function for parallel processing
    """
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
            landmarks_based_learner.compute_Q(beta)
            tmp_results.append(landmarks_based_learner.learn_pb())

    with open(args["output_file"], 'wb') as out_file:
        pickle.dump(tmp_results, out_file, protocol=4)

    return args["algo"]
