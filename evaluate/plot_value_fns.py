from sklearn.neighbors import NearestNeighbors
import numpy as np

import pb_robot

from domains.tools.primitives import contact_approach_fn
from learning.utils import model_forward
from experiments.strategies import bald

def get_model_accuracy_fn(model, ret):
    def model_accuracy_fn(world, cont, xv, yv):
        init_state = world.get_init_state()
        init_pose = world.get_obj_pose_from_state(world.objects['yellow_block'], init_state)

        pose = ((xv, yv, init_pose[0][2]), init_pose[1])
        goal_pose = pb_robot.vobj.BodyPose(world.objects['yellow_block'], pose)

        # NOTE this will generate approach configurations that might
        # not actually be able to follow a push path (due to kinematic constraints)
        tool_approach = contact_approach_fn(world.objects['tool'],
                                               world.objects['yellow_block'],
                                               world.obj_init_poses['yellow_block'],
                                               goal_pose,
                                               cont)
        vof, vef, va = world.get_model_inputs(tool_approach, goal_pose)

        if ret == 'mean':
            return model_forward(model, [vof, vef, va], single_batch=True).mean().squeeze()
        elif ret == 'std':
            return model_forward(model, [vof, vef, va], single_batch=True).std().squeeze()
    return model_accuracy_fn


def get_seq_fn(model):
    def seq_fn(world, cont, xv, yv):
        init_state = world.get_init_state()
        init_pose = world.get_obj_pose_from_state(world.objects['yellow_block'], init_state)

        pose = ((xv, yv, init_pose[0][2]), init_pose[1])
        goal_pose = pb_robot.vobj.BodyPose(world.objects['yellow_block'], pose)

        # NOTE this will generate approach configurations that might
        # not actually be able to follow a push path (due to kinematic constraints)
        tool_approach = contact_approach_fn(world.objects['tool'],
                                               world.objects['yellow_block'],
                                               world.obj_init_poses['yellow_block'],
                                               goal_pose,
                                               cont)
        vof, vef, va = world.get_model_inputs(tool_approach, goal_pose)
        predictions = model_forward(model, [vof, vef, va], single_batch=True)
        mean_prediction = predictions.mean()
        return mean_prediction*bald(predictions)

    return seq_fn


def get_planable_fn(goals, planabilities):
    def planable_fn(world, cont, xv, yv):
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
