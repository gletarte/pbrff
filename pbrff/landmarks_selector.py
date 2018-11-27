import numpy as np

from sklearn.cluster import KMeans
from sklearn.utils import check_random_state

class LandmarksSelector(object):
    """Landmarks selector class.

    Parameters
    ----------
    nb_landmarks_per_label: int
        The number of landmarks to select per labels in the dataset.

    method: str
        The landmarks selection method from: {random, clustering}.

    random_state: None, int or instance of RandomState.
        Information about the random state to be used.


    Attributes
    ----------
    nb_landmarks_per_label: int
        The number of landmarks to select per labels in the dataset.

    method: str
        The landmarks selection method from: {random, clustering}.

    random_state: instance of RandomState.
        Random state for all random operations.

    """

    def __init__(self, n_landmarks_per_label=10, method="random", random_state=None):
        self.n_landmarks_per_label = n_landmarks_per_label
        self.method = method
        self.random_state = check_random_state(random_state)

    def fit(self, X, y):
        """Select landmarks from a dataset.

        Parameters
        ----------
        X: array, shape = [n_samples, n_features]
            The dataset samples.

        y: array, shape = [n_sample]
            The target labels

        Returns
        -------

        landmarks_X: array, shape = [n_landmarks, n_features]
            Selected landmarks.

        landmarks_y: array, shape = [n_landmarks]
            Target labels of the selected landmarks.

        """
        labels = np.unique(y)
        landmarks_X = np.zeros((0, X.shape[1]))
        landmarks_y = np.zeros(0)

        # Defining actual number of landmarks per label it is possible to select according to number of
        # samples per label present in the dataset.
        n_landmarks = {l:self.n_landmarks_per_label for l in labels}
        labels_counts = {l:np.sum(y == l) for l in labels}
        label_lacking_samples = [l for l, n in labels_counts.items() if n < self.n_landmarks_per_label]
        if label_lacking_samples:
            n_landmarks[label_lacking_samples[0]] = labels_counts[label_lacking_samples[0]]
            n_landmarks[list(set(labels) - set(label_lacking_samples))[0]] += self.n_landmarks_per_label - labels_counts[label_lacking_samples[0]]

        # Selecting landmarks for each target label
        for label in labels:
            mask = (y == label)

            # Clustering selection using KMeans algorithm
            if self.method == "clustering":
                clustering = KMeans(n_landmarks[label], random_state=self.random_state)
                clustering.fit(X[mask, :])
                new_landmarks_X = clustering.cluster_centers_

            # Random selection approach
            elif self.method == "random":
                X_label = X[mask, :]
                new_landmarks_X = X_label[self.random_state.choice(X_label.shape[0], n_landmarks[label], replace=False), :]
            else:
                raise Exception(f'Unknown selection method: {self.method}')

            landmarks_X = np.vstack((landmarks_X, new_landmarks_X))
            landmarks_y = np.concatenate((landmarks_y, label*np.ones(n_landmarks[label])))

        assert(np.shape(landmarks_X)[0] == len(landmarks_y))
        return landmarks_X, landmarks_y
