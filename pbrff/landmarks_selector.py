import numpy as np

from sklearn.cluster import KMeans
from sklearn.utils import check_random_state

class LandmarksSelector(object):
    '''
    Landmarks selector class
    '''

    def __init__(self, nb_landmarks_per_label=10, method="clustering", random_state=None):
        self.nb_landmarks = nb_landmarks_per_label
        self.method = method
        self.random_state = check_random_state(random_state)

    def fit(self, X, y):
        labels = np.unique(y)
        landmarks_X = np.zeros((0, np.shape(X)[1]))
        landmarks_y = np.zeros(0)

        nb_landmarks_per_label = {l:self.nb_landmarks for l in labels}
        labels_dist = {l:np.sum(y == l) for l in labels}
        under_labels = [l for l, n in labels_dist.items() if n < self.nb_landmarks]
        if under_labels:
            nb_landmarks_per_label[under_labels[0]] = labels_dist[under_labels[0]]
            nb_landmarks_per_label[list(set(labels) - set(under_labels))[0]] += self.nb_landmarks - labels_dist[under_labels[0]]
            
        for label in labels:
            mask = y == label
            if self.method == "clustering":
                clustering = KMeans(nb_landmarks_per_label[label], random_state=self.random_state)
                clustering.fit(X[mask, :])
                new_landmarks_X = clustering.cluster_centers_
            elif self.method == "random":
                X_label = X[mask, :]
                new_landmarks_X = X_label[self.random_state.choice(X_label.shape[0], nb_landmarks_per_label[label], replace=False), :]

            landmarks_X = np.vstack((landmarks_X, new_landmarks_X))
            landmarks_y = np.concatenate((landmarks_y, label*np.ones(nb_landmarks_per_label[label])))

        assert(np.shape(landmarks_X)[0] == len(landmarks_y))
        return landmarks_X, landmarks_y
