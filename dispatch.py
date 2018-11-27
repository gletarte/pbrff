import os
from os.path import join, abspath, dirname, exists
from os import makedirs
from subprocess import call

RESULTS_PATH = os.environ.get('PBRFF_RESULTS_DIR', join(dirname(abspath(__file__)), "results"))
PROJECT_ROOT = dirname(abspath(__file__))
    
def launch_slurm_experiment(dataset, experiments, landmarks_method, n_cpu, time, dispatch_path ):
    exp_file = join(dispatch_path, f"{dataset}__" + "__".join(experiments))
                        
    submission_script = ""
    submission_script += f"#!/bin/bash\n"
    submission_script += f"#SBATCH --account=def-laviolet\n"
    submission_script += f"#SBATCH --nodes=1\n" 
    submission_script += f"#SBATCH --time={time}:00:00\n" 
    submission_script += f"#SBATCH --output={exp_file + '.out'}\n\n" 
    submission_script += f"cd $HOME/dev/git/pbrff\n" 
    submission_script += f"date\n" 
    submission_script += f"python experiment.py -d {dataset} -e {' '.join(experiments)} -l {' '.join(landmarks_method)} -n {n_cpu} "

    submission_path = exp_file + ".sh"
    with open(submission_path, 'w') as out_file:
        out_file.write(submission_script)
        
    call(["sbatch", submission_path])

def main():
    datasets = ["breast", "ads", "adult", "mnist17", "mnist49", "mnist56"]
    experiments = ["greedy_kernel"]
    landmarks_method = ["random"]
    n_cpu = 40
    time = 6
    
    dispatch_path = join(RESULTS_PATH, "dispatch")
    if not exists(dispatch_path): makedirs(dispatch_path)
    
    for dataset in datasets:
        print(f"Launching {dataset}")
        launch_slurm_experiment(dataset, experiments, landmarks_method, n_cpu, time, dispatch_path)
    
    print("### DONE ###")

if __name__ == '__main__':
    main()
