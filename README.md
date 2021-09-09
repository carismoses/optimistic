# optimistic

This repo implements a method for learning classifiers on top of optimistic action models to use for planning. Currently it only works in a simple toy domain called ordered blocks. It leverages PDDLStream for task and motion planning (TAMP), and pb_robot for integration with the Panda robot. It can be run in pyBullet or on a real Panda robot.

## Installation

This repo was developed using Python 3.9.6.

1. Clone this repo and install requirements
```
git clone git@github.com:carismoses/optimistic.git
cd optimistic
xargs -n1 python3 -m pip install < requirements.txt
```
2. Install [`pb_robot`](https://github.com/mike-n-7/pb_robot) using the instructions below, not the ones in the repo's README.
This installs dependecies, clones the repo and compliles the IKFask library for the panda
   ```
   pip3 install numpy pybullet recordclass catkin_pkg IPython networkx scipy numpy-quaternion
   pip3 install git+https://github.com/mike-n-7/tsr.git
   cd ~
   git clone git@github.com/mike-n-7/pb_robot
   cd pb_robot/src/pb_robot/ikfast/franka_panda
   python3 setup.py build
   ```
3. Install [`pddlstream`](https://github.com/caelan/pddlstream)  using the instructions found there
4. Generate a symlink from pddlstream and pb_robot to this directory
```
cd ~/optimistic
ln -s ~/pddlstream/pddlstream .  # this assumes you installed it to your home directory
ln -s ~/pb_robot/src/pb_robot .  # this assumes you installed it to your home directory
```

## Run

### Train classifier ###
```
python3 -m experiments.collect_data_and_train --data-collection-mode <DCM> --domain <D> --domain-args <DA> --exp-name <EN>
```

```<EN>``` (required) saves results to ```logs/experiments/<EN>```, ```<D>``` (default: ```'ordered_blocks'```) is the desired domain, and ```<DA>``` (required) are the relevant domain arguments. ```<DCM>``` (required) can be any of the following data collection methods

- ```'random-goals-opt'```: sample random goals, plan to achieve them using the optimistic model, train on collected data. Use random rollouts when not plan is found
- ```'random-goals-learned'```: sample random goals, plan to achieve them using the current learned model, train on collected data. Use random rollouts when not plan is found
- ```'random-actions'```: sample random actions

### Evaluate Performance ###

#### Model Accuracy ####
The following command will calculate the accuracy of the learned models in test domains varying from 2-8 blocks. To generate a test dataset, run the following command separately for ```<NB>``` equal to 2 through 8

```
python3 -m experiments.collect_data_and_train --data-collection-mode random-actions --exp-name <EN> --domain-args <NB> --train False
```
and write the paths in ```domains/ordered_blocks/test_datasets.py```. Store your paths from the training runs in ```domains/ordered_blocks/results_paths.py```. Then run the following command which will save plots and data to ```logs/model_accuracu/<EN>```.

```
python3 -m evaluate.accuracy_vary_test_blocks --exp-name <EN>
```

## ToDos
1. Integrate with pb_robot and PDDLStream instead of using my planning code and object types
2. Render in pyBullet
3. Develop tools world in pyBullet
4. Develop tools world actions in PDDLStream
5. Collect data on when optimistic model fails
6. Learn classifier from data
