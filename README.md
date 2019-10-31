# Pseudo-Bayesian Learning with Kernel Fourier Transform as Prior
This python code has been used to conduct the experiments
presented in Section 6 of the following paper:

> Gaël Letarte, Emilie Morvant, Pascal Germain.
> Pseudo-Bayesian Learning with Kernel Fourier Transform as Prior
http://proceedings.mlr.press/v89/letarte19a.html

## Content
* ``experiment.py`` contains the code used to launch experiments and save the results in the ``results`` folder.
* ``pbrff.ipynb`` is a _jupyter notebook_ to process the ``results`` and produce relevant figures.
* ``pbrff/landmarks_based.py`` and ``pbrff/landmarks_selector.py`` implement algorithms used for **Landmarks-Based Learning** experiments (section 6.1).
* ``pbrff/greedy_kernel.py`` implements algorithms used for **Greedy Kernel Learning** experiments (section 6.2).
* ``pbrff/data_loader.py`` contains the code to load the datasets (located in ``data`` folder) used in the experiments.

## Launching an experiment
In order to launch an experiment, launch ``experiment.py`` 
```zsh
python experiment.py
```
with the following arguments:
* **-d**, **--datasets** with the dataset name to process from {"breast", "ads", "adult", "farm", "mnist17", "mnist49", "mnist56"}.
* **-e**, **--experiments** with either "landmarks_based", "greedy_kernel" or both.
* **-l**, **--landmarks-method** with either "random" or "clustering" to select the landmarks selection method for the _landmarks_based_ experiment.
* **-n**, **--n-cpu** with the desired number of cpus to be used or "-1" to use all available.

Here is an example:
```zsh
python experiment.py -d breast -e landmarks_based greedy_kernel -l random -n -1
```

Of note, to change the various parameters explored in the experiments, modify the values in ``experiment.py`` _hps_ dictionnary.

## BiBTeX
```
@inproceedings{letarte2019pseudo,
  title={Pseudo-Bayesian Learning with Kernel Fourier Transform as Prior},
  author={Letarte, Ga{\"e}l and Morvant, Emilie and Germain, Pascal},
  booktitle={The 22nd International Conference on Artificial Intelligence and Statistics},
  pages={768--776},
  year={2019}
}
```
