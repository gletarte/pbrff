import pickle
import numpy as np

from sklearn.model_selection import PredefinedSplit, GridSearchCV, ParameterGrid
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, f1_score
        
def learn_svm(dataset, C, gamma, output_file, n_cpu, random_state):
    print("Computing SVM")
    
    # Defining the validation set for GridSearch
    X = np.concatenate((dataset['X_train'], dataset['X_valid']))
    y = np.concatenate((dataset['y_train'], dataset['y_valid']))
    valid_fold = np.zeros(len(y))
    valid_fold[:len(dataset['y_train'])] = -1
    valid_split = PredefinedSplit(test_fold=valid_fold)
    
    param_grid = [{'C': C, 'gamma': gamma}]
    gs = GridSearchCV(SVC(kernel='rbf', random_state=random_state),
                        param_grid=param_grid,
                        n_jobs=n_cpu,
                        cv=valid_split,
                        refit=False)
    gs.fit(X, y)
    clf = SVC(kernel='rbf', gamma=gs.best_params_['gamma'], C=gs.best_params_['C'], random_state=random_state)
    clf.fit(dataset['X_train'], dataset['y_train'])
    
    test_err = 1 - accuracy_score(dataset['y_test'], clf.predict(dataset['X_test']))
    f1 = f1_score(dataset['y_test'], clf.predict(dataset['X_test']))
    val_err = 1 - accuracy_score(dataset['y_valid'], clf.predict(dataset['X_valid']))
    train_err = 1 - accuracy_score(dataset['y_train'], clf.predict(dataset['X_train']))

    results = [dict([("dataset", dataset['name']), ("exp", 'baseline'), ("algo", 'SVM'),\
                    ("C", gs.best_params_['C']), ("gamma", gs.best_params_['gamma']),\
                    ("train_error", val_err), ("val_error", val_err), ("test_error", test_err), ("f1", f1)])]
    print(results)
    with open(output_file, 'wb') as out_file:
        pickle.dump(results, out_file)
