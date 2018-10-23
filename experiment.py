import argparse
import multiprocessing
import os
import pickle
import numpy as np

from os.path import join, abspath, dirname, exists
from os import makedirs


from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split, ParameterGrid

from pbrff.data_loader import DataLoader
from pbrff.baseline import learn_svm
from pbrff.greedy_kernel import GreedyKernelLearner, compute_greedy_kernel
from pbrff.landmarks_based import LandmarksBasedLearner, compute_landmarks_selection, compute_landmarks_based

import multiprocessing
from multiprocessing import Pool
from functools import partial

RESULTS_PATH = os.environ.get('PBRFF_RESULTS_DIR', join(dirname(abspath(__file__)), "results"))

def main():
    parser = argparse.ArgumentParser(description="PAC-Bayes RFF Experiment")
    parser.add_argument('-d', '--dataset', type=str, default="breast")
    parser.add_argument('-e', '--experiments', type=str, nargs='+', default=["landmarks_based"])
    parser.add_argument('-l', '--landmarks-method', type=str, nargs='+', default=["random"])
    parser.add_argument('-n', '--n-cpu', type=int, default=-1)
    args = parser.parse_args()
    
    # Setting random seed for repeatability
    random_seed = 42
    random_state = check_random_state(random_seed)
    
    # Number of CPU for parallel computing
    if args.n_cpu == -1:
        n_cpu = multiprocessing.cpu_count()
    else:
        n_cpu = args.n_cpu
    print(f"Running on {n_cpu} cpus")
    
    # Preparing output paths
    paths = {'cache': join(RESULTS_PATH, "cache", args.dataset),
             'baseline': join(RESULTS_PATH, "baseline", args.dataset),
             'greedy_kernel': join(RESULTS_PATH, "greedy_kernel", args.dataset)}
    paths.update({f'landmarks_based_{l}':  join(RESULTS_PATH, "landmarks_based", l, args.dataset) for l in args.landmarks_method})
    
    for path_name, path in paths.items():
        if (not exists(path)): makedirs(path)

    # Loading dataset
    dataloader = DataLoader(random_state=random_state)
    X_train, X_test, y_train, y_test = dataloader.load(args.dataset)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state)
    dataset = {'name': args.dataset,
               'X_train': X_train, 'X_valid': X_valid, 'X_test': X_test,
               'y_train': y_train, 'y_valid': y_valid, 'y_test': y_test}
    
    # HPs for landmarks-based and greedy kernel learning experiments
    hps = {'gamma': np.logspace(-7, 2, 10),
           'C': np.logspace(-5, 4, 10),
           'beta': np.logspace(-3, 3, 7),
           'landmarks_percentage': [0.01, 0.05, 0.1, 0.15, 0.20, 0.25],
           'landmarks_D': [8, 16, 32, 64, 128],
           'rho': [1.0, 0.1, 0.01, 0.001, 0.0001],
           'tuning_rho': np.logspace(-4, 0, 20),
           'tuning_beta': np.logspace(1, 3, 20),
           'tuning_epsilon': 1e-10,
           'greedy_kernel_N': 20000,
           'greedy_kernel_D': [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 225, 250, 275,\
                               300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1250, 1500, 1750, 2000, 2500, 3000,\
                               3500, 4000, 4500, 5000]}
    
    ### Experiments ###
    
    # Baseline (SVM)
    svm_file = join(paths['baseline'], "svm.pkl")
    if not(exists(svm_file)):
        learn_svm(dataset=dataset,
                  C_range=hps['C'],
                  gamma_range=hps['gamma'],
                  output_file=svm_file,
                  n_cpu=n_cpu,
                  random_state=random_state)
                  
    with open(svm_file, 'rb') as in_file:
        svm_results = pickle.load(in_file)
    
    gamma = svm_results[0]["gamma"]
    
    # Landmarks-based learning
    if "landmarks_based" in args.experiments:
        
        # Initializing landmarks-based learners by selecting landmarks according to methods
        param_grid = ParameterGrid([{'method': args.landmarks_method, 'percentage_landmarks': hps['landmarks_percentage']}])
        param_grid = list(param_grid)
        
        random_state.shuffle(param_grid)
        results_files = {join(paths['cache'], f"{p['method']}_landmarks_based_learner_{100*p['percentage_landmarks']}.pkl"): p \
                                                                                                            for p in param_grid}
        results_to_compute = [dict({"output_file":f}, **p) for f, p in results_files.items() if not(exists(f))]
        
        if results_to_compute:
            parallel_func = partial(compute_landmarks_selection, 
                                    dataset=dataset,
                                    C_range=hps['C'],
                                    gamma=gamma,
                                    random_state=random_state)
                                
            computed_results = list(Pool(processes=n_cpu).imap(parallel_func, results_to_compute))
            
        # Learning
        param_grid = ParameterGrid([{'algo': ['pb'], 'D': hps['landmarks_D'], 'method': args.landmarks_method, \
                                                                                       'percentage_landmarks': hps['landmarks_percentage']},
                                    {'algo': ['rbf'], 'method': args.landmarks_method, 'percentage_landmarks': hps['landmarks_percentage']}])
        param_grid = list(param_grid)
        random_state.shuffle(param_grid)
        results_files = {join(paths[f"landmarks_based_{p['method']}"], f"{p['algo']}_{100*p['percentage_landmarks']}" \
                                                        + (f"_{p['D']}.pkl" if 'D' in p else ".pkl")): p for p in param_grid}
                                                        
        results_to_compute = [dict({"output_file":f, "input_file": join(paths['cache'], \
                                    f"{p['method']}_landmarks_based_learner_{100*p['percentage_landmarks']}.pkl")}, **p) \
                                                                                    for f, p in results_files.items() if not(exists(f))]
        if results_to_compute:
            parallel_func = partial(compute_landmarks_based, 
                                    beta_range=hps['beta'])
                                
            computed_results = list(Pool(processes=n_cpu).imap(parallel_func, results_to_compute))
    
    # Greedy Kernel Learning
    if "greedy_kernel" in args.experiments:
        
        # Initializing greedy kernel learner
        greedy_kernel_learner_cache_file = join(paths['cache'], "greedy_kernel_learner.pkl")
        if not exists(greedy_kernel_learner_cache_file):
            greedy_kernel_learner = GreedyKernelLearner(dataset, hps['C'], gamma, hps['greedy_kernel_N'], random_state, hps['tuning_epsilon'])
            greedy_kernel_learner.sample_omega()
            greedy_kernel_learner.compute_loss()
        
            with open(greedy_kernel_learner_cache_file, 'wb') as out_file:
                pickle.dump(greedy_kernel_learner, out_file, protocol=4)
            
        param_grid = ParameterGrid([{'algo': ["tpbrff"], 'param': hps['tuning_beta']},
                                    {'algo': ["tokrff"], 'param': hps['tuning_rho']},
                                    #{'algo': ["pbrff"], 'param': hps['beta']},
                                    #{'algo': ["okrff"], 'param': hps['rho']},  
                                    {'algo': ["rff"]}])
                                    
        param_grid = list(param_grid)
        random_state.shuffle(param_grid)
        results_files = {join(paths['greedy_kernel'], f"{p['algo']}" + (f"_{p['param']}.pkl" if 'param' in p else ".pkl")): p \
                                                                                                            for p in param_grid}
        results_to_compute = [dict({"output_file":f}, **p) for f, p in results_files.items() if not(exists(f))]
        
        if results_to_compute:
            parallel_func = partial(compute_greedy_kernel, 
                                    greedy_kernel_learner_file=greedy_kernel_learner_cache_file,
                                    gamma=gamma,
                                    D_range=hps['greedy_kernel_D'], 
                                    random_state=random_state)
                                
            computed_results = list(Pool(processes=n_cpu).imap(parallel_func, results_to_compute))
    
    print("### DONE ###")

if __name__ == '__main__':
    main()
