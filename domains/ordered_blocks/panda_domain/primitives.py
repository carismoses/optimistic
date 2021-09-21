import numpy as np

import pb_robot
from pb_robot.tsrs.panda_box import ComputePrePose


DEBUG_FAILURE = False


def get_free_motion_gen(robot, fixed=[]):
    def fn(conf1, conf2, fluents=[]):
        obstacles = assign_fluent_state(fluents)
        fluent_names = [o.get_name() for o in obstacles]
        for o in fixed:
            if o.get_name() not in fluent_names:
                obstacles.append(o)

        path = robot.arm.birrt.PlanToConfiguration(robot.arm, conf1.configuration, conf2.configuration, obstacles=obstacles)

        if path is None:
            if DEBUG_FAILURE: input('Free motion failed')
            return None
        command = [pb_robot.vobj.JointSpacePath(robot.arm, path)]
        return (command,)
    return fn


def get_holding_motion_gen(robot, fixed=[]):
    def fn(conf1, conf2, block, grasp, fluents=[]):
        obstacles = assign_fluent_state(fluents)
        fluent_names = [o.get_name() for o in obstacles]
        for o in fixed:
            if o.get_name() not in fluent_names:
                obstacles.append(o)

        old_q = robot.arm.GetJointValues()
        orig_pose = block.get_base_link_pose()
        robot.arm.SetJointValues(conf1.configuration)
        robot.arm.Grab(block, grasp.grasp_objF)

        path = robot.arm.birrt.PlanToConfiguration(robot.arm, conf1.configuration, conf2.configuration, obstacles=obstacles)

        robot.arm.Release(block)
        block.set_base_link_pose(orig_pose)
        robot.arm.SetJointValues(old_q)

        if path is None:
            if DEBUG_FAILURE: input('Holding motion failed')
            return None
        command = [pb_robot.vobj.JointSpacePath(robot.arm, path)]
        return (command,)
    return fn


def get_ik_fn(robot, fixed=[], num_attempts=4, approach_frame='gripper', backoff_frame='global'):
    def fn(block, pose, grasp, return_grasp_q=False, check_robust=False):
        obstacles = fixed + [block]
        obj_worldF = pb_robot.geometry.tform_from_pose(pose.pose)
        grasp_worldF = np.dot(obj_worldF, grasp.grasp_objF)
        grasp_worldR = grasp_worldF[:3,:3]

        e_x, e_y, e_z = np.eye(3) # basis vectors

        # The x-axis of the gripper points toward the camera
        # The y-axis of the gripper points along the plane of the hand
        # The z-axis of the gripper points forward

        is_top_grasp = grasp_worldR[:,2].dot(-e_z) > 0.999
        is_upside_down_grasp = grasp_worldR[:,2].dot(e_z) > 0.001
        is_gripper_sideways = np.abs(grasp_worldR[:,1].dot(e_z)) > 0.999
        is_camera_down = grasp_worldR[:,0].dot(-e_z) > 0.999
        is_wrist_too_low = grasp_worldF[2,3] < 0.088/2 + 0.005


        if is_gripper_sideways:
            return None
        if is_upside_down_grasp:
            return None
        if is_camera_down:# and approach_frame == 'gripper':
            return None

        # the gripper is too close to the ground. the wrist of the arm is 88mm
        # in diameter, and it is the widest part of the hand. Include a 5mm
        # clearance
        if not is_top_grasp and is_wrist_too_low:
            return None
        # If the block/gripper is in the storage area, don't use low grasps.
        if grasp_worldF[0,3] < 0.2 and grasp_worldF[2,3] < 0.1:
            return None

        if approach_frame == 'gripper':
            approach_tform = ComputePrePose(grasp_worldF, [0, 0, -0.1], approach_frame)
        elif approach_frame == 'global':
            approach_tform = ComputePrePose(grasp_worldF, [0, 0, 0.1], approach_frame) # Was -0.125
        else:
            raise NotImplementedError()

        if backoff_frame == 'gripper':
            backoff_tform = ComputePrePose(grasp_worldF, [0, 0, -0.1], backoff_frame)
        elif backoff_frame == 'global':
            backoff_tform = ComputePrePose(grasp_worldF, [0, 0, 0.1], backoff_frame) # Was -0.125
        else:
            raise NotImplementedError()

        for ax in range(num_attempts):
            q_grasp = robot.arm.ComputeIK(grasp_worldF)
            if (q_grasp is None):
                if DEBUG_FAILURE: input('No Grasp IK')
                continue
            if not robot.arm.IsCollisionFree(q_grasp, obstacles=obstacles, debug=DEBUG_FAILURE):
                if DEBUG_FAILURE: input('Grasp collision')
                continue

            q_approach = robot.arm.ComputeIK(approach_tform, seed_q=q_grasp)
            if (q_approach is None):
                if DEBUG_FAILURE: input('No approach IK')
                continue
            if not robot.arm.IsCollisionFree(q_approach, obstacles=obstacles, debug=DEBUG_FAILURE):
                if DEBUG_FAILURE: input('Approach motion collision')
                continue
            conf_approach = pb_robot.vobj.BodyConf(robot, q_approach)

            # Only recompute the backoff if it's different from the approach.
            if approach_frame == backoff_frame:
                q_backoff = q_approach
            else:
                q_backoff = robot.arm.ComputeIK(backoff_tform, seed_q=q_grasp)
                if (q_backoff is None):
                    if DEBUG_FAILURE: input('No backoff IK')
                    continue
                if not robot.arm.IsCollisionFree(q_backoff, obstacles=obstacles, debug=DEBUG_FAILURE):
                    if DEBUG_FAILURE: input('Backoff motion collision')
                    continue
            conf_backoff = pb_robot.vobj.BodyConf(robot, q_backoff)

            path_approach = robot.arm.snap.PlanToConfiguration(robot.arm, q_approach, q_grasp, obstacles=obstacles)
            if path_approach is None:
                if DEBUG_FAILURE: input('Approach motion failed')
                continue
            if backoff_frame == 'global':
                path_backoff = robot.arm.snap.PlanToConfiguration(robot.arm, q_grasp, q_backoff, obstacles=obstacles, check_upwards=True)
            else:
                path_backoff = robot.arm.snap.PlanToConfiguration(robot.arm, q_grasp, q_backoff, obstacles=obstacles, check_upwards=False)
            if path_backoff is None:
                if DEBUG_FAILURE: input('Backoff motion failed')
                continue

            # If the grasp is valid, check that it is robust (i.e., also valid under pose estimation error).
            if check_robust:
                for _ in range(10):
                    x, y, z = pose.pose[0]
                    new_pose = ((x + np.random.randn()*0.02, y + np.random.randn()*0.02, z), pose.pose[1])
                    new_pose = pb_robot.vobj.BodyPose(block, new_pose)
                    valid = fn(block, pose, grasp, check_robust=False)
                    if not valid:
                        print('Grasp not robust')
                        print(x - new_pose.pose[0][0], y - new_pose.pose[0][1])
                        return None

            command = [pb_robot.vobj.MoveToTouch(robot.arm, q_approach, q_grasp, grasp, block),
                       grasp,
                       pb_robot.vobj.MoveFromTouch(robot.arm, q_backoff)]

            if return_grasp_q:
                return (pb_robot.vobj.BodyConf(robot, q_grasp),)
            return (conf_approach, conf_backoff, command)
        return None
    return fn


def get_pose_gen_block(fixed=[]):
    def fn(top_block, bottom_block, bottom_block_pose):
        """
        @param rel_pose: A homogeneous transformation matrix.
        """
        # NOTE: this assumes we want all blocks at the same orientation always (when not held)
        bottom_block_tform = pb_robot.geometry.tform_from_pose(bottom_block_pose.pose)
        rel_z_pose = bottom_block.get_dimensions()[2]/2+top_block.get_dimensions()[2]/2
        rel_pose = ((0., 0., rel_z_pose), (0., 0., 0., 1.))
        top_block_tform = bottom_block_tform@rel_pose
        top_block_pose = pb_robot.geometry.tform_from_pose(top_block_tform)
        top_block_pose = pb_robot.vobj.BodyPose(top_block, top_block_pose)
        return (top_block_pose,)
    return fn


def get_grasp_gen(robot, add_slanted_grasps=True, add_orthogonal_grasps=True):
    # add_slanted_grasps = True
    # I opt to use TSR to define grasp sets but you could replace this
    # with your favorite grasp generator
    def gen(block):
        grasp_tsr = pb_robot.tsrs.panda_box.grasp(block,
            add_slanted_grasps=add_slanted_grasps, add_orthogonal_grasps=add_orthogonal_grasps)
        grasps = []

        for sampled_tsr in grasp_tsr:
            grasp_worldF = sampled_tsr.sample()
            grasp_objF = np.dot(np.linalg.inv(block.get_base_link_transform()), grasp_worldF)
            block_grasp = pb_robot.vobj.BodyGrasp(block, grasp_objF, robot.arm)
            grasps.append((block_grasp,))
        return grasps
    return gen


def assign_fluent_state(fluents):
    obstacles = []
    for fluent in fluents:
        name, args = fluent[0], fluent[1:]
        if name == 'atpose':
            o, p = args
            obstacles.append(o)
            o.set_base_link_pose(p.pose)
        else:
            raise ValueError(name)
    return obstacles
