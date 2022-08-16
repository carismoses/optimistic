# optimistic

This repo implements the code for the workshop paper [Learning to Plan with Optimistic Action Models](http://people.csail.mit.edu/cmm386/publications/ICRA_WS2022_final.pdf), which is a method for learning action models to be used within a task and motion planner (see paper for more details). This code performs experiments in a tool-use domain. It leverages PDDLStream for task and motion planning (TAMP), and pb_robot for integration with a Panda robot run in a pyBullet simulator.

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

### Collect Data and Train Classifiers ###
This will iteratively have the robot take actions and train action models on the collected data. See paper for the details of each data collection method.
```
python3 -m experiments.collect_data_and_train --data-collection-mode <DCM> --exp-name <EN> --max-actions <MA> --vis
```

```--vis``` is an optional argument that allows you to visualize the actions the robot is taking in pyBullet.
```<EN>``` (required) saves results to ```logs/experiments/<EN>```.
```<MA>``` is an integer number of actions to have the robot perform and train on.
```<DCM>``` (required) can be any of the following data collection methods:

- ```'random-actions'```: sample and execute random actions (Random Actions method in paper)  
- ```'random-goals-opt'```: sample random goals, plan to achieve them using the optimistic model, execute found plan. Use random rollouts when no plan is found (Random Goals method in paper)
- ```'sequential-plans'```: sample and execute actions found using the Sequential method (see paper)
- ```'sequential-goals'```: sample and execute actions found using the Sequential Goals method (see paper)

### Evaluating Model Accuracy ###

```evaluate/model_accuracy.py``` is used to calculate the accuracy of trained models. The ```all_models_path``` variable contains the paths to the experiments to be evaluated and the ```test_dataset_path``` is the path to the experiment to be used as the test dataset.

### Planning with Learned Models ###

To plan with the learned action models we developed a skeleton-based planner (see paper for details). The code for using this planner is in ```evaluate/plan_success.py```.

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
