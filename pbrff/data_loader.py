from functools import partial
from os.path import join, abspath, dirname

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_breast_cancer, load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state


DATA_ROOT_PATH = join(dirname(abspath(__file__)), "..", "data")

class DataLoader(object):
    '''
    Data Loader class
    '''

    def __init__(self, data_path=DATA_ROOT_PATH, test_size=0.25, random_state=42):
        self.data_path = data_path
        self.random_state = check_random_state(random_state)
        self.test_size = test_size

    def load(self, dataset):
        dataset_loaders = {'adult': self._load_adult,
                           'breast': self._load_breast,
                           'farm': self._load_farm,
                           'ads': self._load_ads,
                           'mnist17': partial(self._load_mnist, low=1, high=7),
                           'mnist49': partial(self._load_mnist, low=4, high=9),
                           'mnist56': partial(self._load_mnist, low=5, high=6)}

        if dataset not in dataset_loaders.keys():
            raise RuntimeError(f"Invalid dataset {dataset}")

        return dataset_loaders[dataset]()

    def _load_adult(self):
        features = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
        categorical_features = [f for i, f in enumerate(features) if i in [1, 3, 5, 6, 7, 8, 9, 13]]

        # Train split
        df = pd.read_csv(join(self.data_path, "adult.data"), sep=",", header=None)

        le = LabelEncoder()
        y_train = df.iloc[:, -1]
        y_train = le.fit_transform(y_train)

        X_train = df.iloc[:, :-1]
        X_train = X_train.rename(columns={i:f for i, f in enumerate(features)})
        X_train = pd.get_dummies(X_train, columns=categorical_features)
        X_train = X_train.drop([c for c in X_train.columns.values if '_?' in c], axis=1)

        # Test split
        df = pd.read_csv(join(self.data_path, "adult.test"), sep=",", header=None)
        y_test = df.iloc[:, -1]
        y_test = le.fit_transform(y_test)

        X_test = df.iloc[:, :-1]
        X_test = X_test.rename(columns={i:f for i, f in enumerate(features)})
        X_test = pd.get_dummies(X_test, columns=categorical_features)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        return np.ascontiguousarray(X_train), np.ascontiguousarray(X_test), 2*y_train-1, 2*y_test-1

    def _load_breast(self):
        breast = load_breast_cancer()
        return train_test_split(breast.data, 2*breast.target-1, test_size=self.test_size, random_state=self.random_state)

    def _load_farm(self):
        X, y = load_svmlight_file(join(self.data_path, "farm-ads-vect"))
        y = y.astype('int32')
        return train_test_split(X.toarray(), y, test_size=self.test_size, random_state=self.random_state)

    def _load_ads(self):
        df = pd.read_csv(join(self.data_path, "ad.data"), sep=",", header=None)

        le = LabelEncoder()
        y = df.iloc[:, -1]
        y = 2*le.fit_transform(y)-1

        # We use all but the first 4 features which are sometimes missing in the data.
        X = df.iloc[:, 4:-1]

        return train_test_split(X.values, y, test_size=self.test_size, random_state=self.random_state)

    def _load_mnist(self, low, high):
        X_low = np.loadtxt(join(self.data_path, "mnist", f"mnist_{low}")) / 255
        y_low = -1 * np.ones(X_low.shape[0])

        X_high = np.loadtxt(join(self.data_path, "mnist", f"mnist_{high}")) / 255
        y_high = np.ones(X_high.shape[0])

        X = np.vstack((X_low, X_high))
        y = np.hstack((y_low, y_high))

        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
