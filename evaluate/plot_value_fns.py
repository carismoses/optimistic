from sklearn.neighbors import NearestNeighbors
import numpy as np

import pb_robot

from domains.tools.primitives import contact_approach_fn
from learning.utils import model_forward
from experiments.strategies import bald

def get_model_accuracy_fn(ensembles, ret):
    def model_accuracy_fn(world, action, obj, xv, yv, grasp=None):
        if 'move_contact' in action:
            # plot all model accuracies as if block is being pushed from init pose
            # (might have different performance from diff initial poses but this
            # is the easiest way to visualize model accuracy in 2D)
            init_xy = world.init_objs_pos_xy[obj]
            x = [*init_xy, xv, yv, *grasp]
        else:
            x = [xv, yv]
        if ret == 'mean':
            return model_forward(ensembles, np.array(x), action, obj, single_batch=True).mean().squeeze()
        elif ret == 'std':
            return model_forward(ensembles, np.array(x), action, obj, single_batch=True).std().squeeze()
    return model_accuracy_fn


def get_seq_fn(ensembles):
    def seq_fn(world, action, obj, xv, yv, grasp=None):
        predictions = model_forward(ensembles, np.array([xv, yv]), action, obj, single_batch=True).squeeze()
        mean_prediction = predictions.mean()
        return mean_prediction*bald(predictions)
    return seq_fn


def get_planable_fn(goals, planabilities):
    def planable_fn(world, cont, xv, yv, grasp=None):
        ## dist parameters ##
        max_n_n = 5 # max number of nearest neighors to use in calculation
        ##

        n_points = len(goals)
        if n_points == 0:
            return 0
        n_n = min(n_points, max_n_n)
        goals_vec = np.array([goal[2].pose[0][:2] for goal in goals])
        nbrs = NearestNeighbors(n_neighbors=n_n, algorithm='ball_tree').fit(np.array(goals_vec))
        distances, indices = nbrs.kneighbors(np.array([[xv, yv]]))
        return distances.squeeze().mean()
    return planable_fn
