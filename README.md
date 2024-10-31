# Geometric Methods for High-Dimensional Data Analysis

The official repository for the homework assignments of the lecture \<Geometric Methods for High-Dimensional Data Analysis> instructed by Frank C. Park in Seoul National University.


### Table of Contents

[Instructions for Settings](#instruction-for-settings)  
[Homework 1](#homework-1)


### TA Contacts

- Byeongdo Lim (bdlim@robotics.snu.ac.kr, *Head TA*)
- Jungbin Lim (jblim97@robotics.snu.ac.kr)
- Jongmin Kim (jmkim@robotics.snu.ac.kr)


## Instruction for Settings
### Install Anaconda3 and Jupyter Notebook

For those unfamiliar with environment setup, Anaconda3 installation is recommended.

Download and install Anaconda3 from the following link: [https://repo.anaconda.com/archive/](https://repo.anaconda.com/archive/).

After Anaconda3 installation, jupyter is automatically installed and you can run jupyter notebook with the following command.
```shell
jupyter notebook
```


### Guidelines for Anaconda3 and Jupyter Notebook

Here's some commands for conda environment.
```shell
conda create -n {name} python=3.11  # Create conda environment
conda activate {name}               # Activate conda environment
conda deactivate {name}             # Deactivate conda environment
```

If you want to use the created conda environment in jupyter notebook, you need to register the kernel. The command to register is following.
```shell
pip install ipykernel
python -m ipykernel install --user --name {name} --display-name "{name-in-jupyter}"
```
{name} and {name-in-jupyter} can be replaced by your choices.


### Environment
The codes are executed in the following environment.
- python 3.9
- numpy
- matplotlib
- scikit-learn
- *pytorch*
- *torchvision*

To setup the environment except *pytorch* and *torchvision*, run the following script in the command line.
```
pip install -r requirements.txt
```
Install PyTorch and torchvision from the following link: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).

## Homework 1
Follow the instructions in the ``HW1-1.ipynb`` and ``HW1-2.ipynb`` files. After you complete and run the HW ipython files, upload them to eTL.

## Homework 3
Follow the instructions in the ``HW3-1.ipynb`` and ``HW3-2.ipynb`` files. After you complete and run the HW ipython files, upload them to eTL.
You need to set up a separate environment for HW3-2. Please follow the instructions carefully.
