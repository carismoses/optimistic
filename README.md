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
2. Install [`pb_robot`](https://github.com/carismoses/pb_robot) using the instructions below, not the ones in the repo's README.
This installs dependecies, clones the repo and compliles the IKFask library for the panda (see Troubleshooting below if the build command fails)
   ```
   cd ~
   git clone git@github.com:carismoses/pb_robot.git
   cd pb_robot/src/pb_robot/ikfast/franka_panda
   python3 setup.py build
   ```
3. Install [`pddlstream`](https://github.com/carismoses/pddlstream)  using the instructions found there
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
- ```'curriculum-goals-learned'```: goals increase in complexity over time (more blocks and higher heights). Use random rollouts when not plan is found
- ```'curriculum-goals-learned-new'```: goals increase in complexity over time (higher heights always consider all blocks). Use random rollouts when not plan is found
  - optionally add the ```--curriculum-max-t <MT>``` argument to perform curriculum learning until ```<MT>``` then do random goal sampling afterwards

### Evaluate Performance ###

#### Model Accuracy across Domains (Generalization) ####

The following command will calculate the accuracy of the learned models in test domains varying from 2-8 blocks. Store your paths from the training runs in ```domains/ordered_blocks/results_paths.py```. Then run the following command which will save plots and data to ```logs/model_accuracu/<EN>```.

```
python3 -m evaluate.accuracy_vary_test_blocks --exp-name <EN>
```

#### Model Accuracy over Training Time ####

The following command will calculate the accuracy of the learned models over the course of training for domain sizes of 2-8 blocks (separate plot for each). Store your paths from the training runs in ```domains/ordered_blocks/results_paths.py```. Then run the following command which will save plots and data to ```logs/model_accuracu/<EN>```.

```
python3 -m evaluate.accuracy_vary_time_steps --exp-name <EN>
```

## ToDos
1. Integrate with pb_robot and PDDLStream instead of using my planning code and object types
2. Render in pyBullet
3. Develop tools world in pyBullet
4. Develop tools world actions in PDDLStream
5. Collect data on when optimistic model fails
6. Learn classifier from data

## Troubleshooting

This may be useful for setting up the repo using a different Python version. If building
pb_robot with `python setup.py build` failed with the following error:

```./ikfast.h:41:10: fatal error: 'python3.6/Python.h' file not found```

The compiler can't find the appropriate python header. The solution is to first locate the header:

```
$ find /usr/local/Cellar/ -name Python.h
/usr/local/Cellar//python/3.7.7/Frameworks/Python.framework/Versions/3.7/include/python3.7m/Python.h
/usr/local/Cellar//python@3.8/3.8.2/Frameworks/Python.framework/Versions/3.8/include/python3.8/Python.h
```

which prints the python include directories. I wanted to use 3.7, so then I set the environment variable

```export CPLUS_INCLUDE_PATH=/usr/local/Cellar//python/3.7.7/Frameworks/Python.framework/Versions/3.7/include/```

and finally modify `pb_robot/src/pb_robot/ikfast/ikfast.h` by changing

```
#include "python3.6/Python.h" -> #include "python3.7m/Python.h"
```
