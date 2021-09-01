# optimistic

This repo implements a method for learning classifiers on top of optimistic action models to use for planning. Currently it only works in a simple toy domain called ordered blocks.

## Installation

This repo was developed using Python 3.9.6.

Clone repo and install requirements
```
git clone git@github.com:carismoses/optimistic.git
cd optimistic
xargs -n1 python3 -m pip install < requirements.txt
```

## Run

### Train classifier ###
```
python3 -m experiments.collect_data_and_train --data-collection-mode <DCM> --exp-name <EN> --num-blocks <NB>
```

```<EN>``` (required) saves results to ```experiments/logs/<EN>```, ```<NB>``` (default: 4) is the number of blocks in the training domain, and ```<DCM>``` (required) can be any of the following data collection methods

- ```'random-goals-opt'```: sample random goals, plan to achieve them using the optimistic model, train on collected data. Use random rollouts when not plan is found
- ```'random-goals-learned'```: sample random goals, plan to achieve them using the current learned model, train on collected data. Use random rollouts when not plan is found
- ```'random-actions'```: sample random actions

### Evaluate Performance ###

#### Model Accuracy ####
The following command will calculate the accuracy of the learned models in test domains varying from 2-8 blocks. To generate a test dataset, run the following command separately for ```<NB>``` equal to 2 through 8

```
python3 -m experiments.collect_data_and_train --data-collection-mode random-actions --exp-name <EN> --num-blocks <NB>
```
and write the paths in ```domains/ordered_blocks/test_datasets.py```. Store your paths from the training runs in ```domains/ordered_blocks/results_paths.py```. Then run the following command

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
