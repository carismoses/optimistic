import numpy as np

import pb_robot

from learning.utils import model_forward

def get_trust_contact_model(world, logger, planning_model_i=None):
    def test(obj1, obj2, pose1, pose2, cont):
        ## Calculate angle between contact frame and push direction
        # get cont axis in world
        block_world = pb_robot.geometry.tform_from_pose(pose1.pose)
        cont_tform = pb_robot.geometry.tform_from_pose(cont.rel_pose)
        tool_w_tform = block_world@cont_tform
        cont_w_tform = np.dot(tool_w_tform, np.linalg.inv(cont.tool_in_cont_tform))
        valid_push_cont_frame = (1., 0., 0.)
        valid_push_w_frame = np.dot(cont_w_tform[:3,:3], valid_push_cont_frame[:3])[:2]
        unit_valid_push_w_frame = valid_push_w_frame/np.linalg.norm(valid_push_w_frame)

        # push direction in world
        push_w_frame = np.subtract(pose2.pose[0][:2], pose1.pose[0][:2])
        unit_push_w_frame = push_w_frame/np.linalg.norm(push_w_frame)

        # angle between vectors
        dot_product = np.dot(unit_valid_push_w_frame, unit_push_w_frame)
        angle = np.arccos(dot_product)

        # yellow can be pushed if cont axis aligns with push direction
        if obj2 == world.objects['yellow_block']:
            if angle < world.approx_valid_push_angle:
                return True
            return False
        # blue can be pushed if cont and push direction align AND push direction
        # is x-axis in world
        if obj2 == world.objects['blue_block']:
            angle_from_x_world = np.arccos(np.dot((1,0), unit_push_w_frame))
            if angle < world.approx_valid_push_angle and \
                    angle_from_x_world < world.approx_valid_push_angle:
                return True
            return False
        #model = logger.load_trans_model(i=planning_model_i)
        #x = world.pred_args_to_vec(obj1, obj2, pose1, pose2, cont)
        #trust_model = model_forward(cont.type, model, x, single_batch=True).mean().round().squeeze()
        #return trust_model
    return test

def get_trust_pick_model(world, logger, planning_model_i=None):
    def test(obj_top, pick_pose):
        #model = logger.load_trans_model(i=planning_model_i)
        #x = world.pred_args_to_vec(obj1, obj2, pose1, pose2, cont)
        #trust_model = model_forward(cont.type, model, x, single_batch=True).mean().round().squeeze()
        #return trust_model
        if obj_top == world.objects['yellow_block']:
            if np.linalg.norm(pick_pose.pose[0][:2]) < world.valid_pick_yellow_radius:
                return True
            return False
        if obj_top == world.objects['blue_block']:
            if world.block_in_tunnel(pick_pose.pose[0][:2]):
                return False
            return True
    return test
