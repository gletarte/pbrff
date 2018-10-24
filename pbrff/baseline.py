import pickle
import time
import numpy as np

from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

def learn_svm(dataset, C_range, gamma_range, output_file, n_cpu, random_state):
    """Learn a SVM baseline.

    Using a validation set, hyperparamters C and gamma are selected
    from given ranges through grid search.

    Parameters
    ----------
    dataset : dict
        The dataset as a dictionnary with the following keys:
        X_train, X_valid, X_test, y_train, y_valid, y_test, name.

    C_range : list
        C values range to search from (SVM's penalty parameter).

    gamma_range : list
        Gamma values range to search from (RBF kernel's bandwidth paramter).

    output_file : str
        File path to save results with pickle

    n_cpu : int
        The number of CPUs to use during the grid search.

    random_state : instance of RandomState
        Random state for all random operations.

    """
    print("Computing SVM baseline")

    # Defining the validation set for GridSearchCV
    X = np.concatenate((dataset['X_train'], dataset['X_valid']))
    y = np.concatenate((dataset['y_train'], dataset['y_valid']))
    valid_fold = np.zeros(len(y))
    valid_fold[:len(dataset['y_train'])] = -1
    valid_split = PredefinedSplit(test_fold=valid_fold)

    # Grid search using a validation set
    param_grid = [{'C': C_range, 'gamma': gamma_range}]
    gs = GridSearchCV(SVC(kernel='rbf', random_state=random_state),
                      param_grid=param_grid,
                      n_jobs=n_cpu,
                      cv=valid_split,
                      refit=False)
    gs.fit(X, y)

    # SVM classifier training using best hps values selected
    clf = SVC(kernel='rbf', gamma=gs.best_params_['gamma'], C=gs.best_params_['C'], random_state=random_state)
    start_time = time.time()
    clf.fit(dataset['X_train'], dataset['y_train'])
    train_time = (time.time() - start_time) * 1000

    # Computing relevant metrics
    test_err = 1 - accuracy_score(dataset['y_test'], clf.predict(dataset['X_test']))
    f1 = f1_score(dataset['y_test'], clf.predict(dataset['X_test']))
    val_err = 1 - accuracy_score(dataset['y_valid'], clf.predict(dataset['X_valid']))
    train_err = 1 - accuracy_score(dataset['y_train'], clf.predict(dataset['X_train']))

    # Logging metrics and informations
    results = [dict([("dataset", dataset['name']), ("exp", 'baseline'), ("algo", 'SVM'),\
                    ("C", gs.best_params_['C']), ("gamma", gs.best_params_['gamma']), ("time", train_time),\
                    ("train_error", train_err), ("val_error", val_err), ("test_error", test_err), ("f1", f1)])]

    with open(output_file, 'wb') as out_file:
        pickle.dump(results, out_file, protocol=4)
