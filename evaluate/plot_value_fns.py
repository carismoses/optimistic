import pb_robot

from domains.tools.primitives import contact_approach_fn
from learning.utils import model_forward

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


#def bald(world, cont, xv, yv):

#def dist_action():

#def dist_state():
